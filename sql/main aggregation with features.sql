-- ============================================================================
-- CASINO ANALYTICS: DAILY AGGREGATIONS WITH LAG & MOVING FEATURES
-- ============================================================================
-- This SQL computes all features needed for time series prediction:
-- 1. Base daily aggregations (original)
-- 2. Lag features (1, 7, 30 days)
-- 3. Moving averages (7, 30 days)
-- 4. Moving standard deviations (7, 30 days)
-- 5. Growth rates
-- 6. Temporal features
-- ============================================================================

CREATE OR REPLACE TABLE casino_ctg.log_db.daily_aggregations_features AS

WITH 
-- ============================================================================
-- STEP 1: Player First Appearances (Original)
-- ============================================================================
player_first_appearances AS (
    SELECT 
        player_id, 
        MIN(CAST(timestamp AS DATE)) as first_play_date
    FROM casino_ctg.log_db.transactions_bronze
    GROUP BY player_id
),

-- ============================================================================
-- STEP 2: Daily Platform Growth (Original)
-- ============================================================================
daily_platform_growth AS (
    SELECT 
        first_play_date AS date,
        COUNT(player_id) as new_players_today,
        SUM(COUNT(player_id)) OVER (
            ORDER BY first_play_date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as cumulative_player_base
    FROM player_first_appearances
    GROUP BY first_play_date
),

-- ============================================================================
-- STEP 3: Base Daily Aggregation (Original)
-- ============================================================================
base_daily AS (
    SELECT 
        -- Temporal Dimensions
        CAST(t.timestamp AS DATE) AS date,
        t.game_id,

        -- Growth Variable
        COALESCE(pg.cumulative_player_base, 0) AS cumulative_player_base, 

        -- Target KPIs
        COUNT(DISTINCT t.player_id) AS daily_active_users,
        SUM(t.bet_amount) AS gross_daily_revenue,
        SUM(t.win_amount) AS daily_payout,
        SUM(-t.net_result) AS daily_ggr,

        -- Volume Metrics
        COUNT(t.transaction_id) AS total_bets,
        COUNT(DISTINCT t.player_id) AS active_players,
        SUM(t.bet_amount) AS total_wagered,
        SUM(t.win_amount) AS total_won,
        SUM(t.net_result) AS gross_gaming_revenue,
        COUNT(DISTINCT t.session_id) AS total_sessions,

        -- Derived Metrics
        CASE 
            WHEN SUM(t.bet_amount) = 0 THEN 0 
            ELSE (-SUM(t.net_result) / SUM(t.bet_amount)) * 100 
        END AS hold_percentage,

        CASE 
            WHEN COUNT(t.transaction_id) = 0 THEN 0 
            ELSE SUM(t.bet_amount) / COUNT(t.transaction_id) 
        END AS avg_bet_size,

        CASE 
            WHEN COUNT(DISTINCT t.player_id) = 0 THEN 0 
            ELSE COUNT(t.transaction_id) / COUNT(DISTINCT t.player_id) 
        END AS bets_per_player,

        -- Game Metadata
        m.type AS game_type,
        m.rtp AS theoretical_rtp,
        m.volatility,
        m.visual_engagement

    FROM casino_ctg.log_db.transactions_bronze t
    INNER JOIN casino_ctg.log_db.games_bronze m ON t.game_id = m.game_id
    LEFT JOIN daily_platform_growth pg ON CAST(t.timestamp AS DATE) = pg.date

    GROUP BY 
        CAST(t.timestamp AS DATE), 
        t.game_id,
        pg.cumulative_player_base,
        m.type,
        m.rtp,
        m.volatility,
        m.visual_engagement
),

-- ============================================================================
-- STEP 4: Add All Window Features (LAGs, Moving Averages, Std Dev)
-- ============================================================================
with_window_features AS (
    SELECT 
        b.*,
        
        -- =================================================================
        -- TEMPORAL FEATURES
        -- =================================================================
        DATEDIFF(date, (SELECT MIN(date) FROM base_daily)) AS days_from_launch,
        YEAR(date) AS year,
        MONTH(date) AS month,
        DAY(date) AS day_of_month,
        DAYOFWEEK(date) AS day_of_week,  -- 1=Sunday, 7=Saturday
        WEEKOFYEAR(date) AS week_of_year,
        CASE WHEN DAYOFWEEK(date) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
        CASE WHEN DAY(date) IN (1, 2, 3) THEN 1 ELSE 0 END AS is_month_start,
        CASE WHEN DAY(date) IN (14, 15, 16, 17) THEN 1 ELSE 0 END AS is_month_mid,
        
        -- =================================================================
        -- LAG FEATURES: GGR (Partitioned by game_id!)
        -- =================================================================
        LAG(daily_ggr, 1) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS ggr_lag1,
        
        LAG(daily_ggr, 7) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS ggr_lag7,
        
        LAG(daily_ggr, 30) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS ggr_lag30,
        
        -- =================================================================
        -- LAG FEATURES: Total Wagered (Revenue)
        -- =================================================================
        LAG(total_wagered, 1) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS revenue_lag1,
        
        LAG(total_wagered, 7) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS revenue_lag7,
        
        LAG(total_wagered, 30) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS revenue_lag30,
        
        -- =================================================================
        -- LAG FEATURES: Daily Active Users
        -- =================================================================
        LAG(daily_active_users, 1) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS dau_lag1,
        
        LAG(daily_active_users, 7) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS dau_lag7,
        
        -- =================================================================
        -- LAG FEATURES: Total Bets
        -- =================================================================
        LAG(total_bets, 7) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS bets_lag7,
        
        -- =================================================================
        -- LAG FEATURES: Cumulative Player Base
        -- =================================================================
        LAG(cumulative_player_base, 7) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS player_base_lag7,
        
        -- =================================================================
        -- MOVING AVERAGES: GGR (7-day and 30-day)
        -- =================================================================
        AVG(daily_ggr) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ggr_ma7,
        
        AVG(daily_ggr) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS ggr_ma30,
        
        -- =================================================================
        -- MOVING AVERAGES: Total Wagered (Revenue)
        -- =================================================================
        AVG(total_wagered) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS revenue_ma7,
        
        AVG(total_wagered) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS revenue_ma30,
        
        -- =================================================================
        -- MOVING AVERAGES: DAU
        -- =================================================================
        AVG(daily_active_users) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS dau_ma7,
        
        -- =================================================================
        -- MOVING AVERAGES: Total Bets
        -- =================================================================
        AVG(total_bets) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS bets_ma7,
        
        -- =================================================================
        -- MOVING STANDARD DEVIATION: GGR (Volatility)
        -- =================================================================
        STDDEV(daily_ggr) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ggr_std7,
        
        STDDEV(daily_ggr) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS ggr_std30,
        
        -- =================================================================
        -- MOVING STANDARD DEVIATION: Revenue
        -- =================================================================
        STDDEV(total_wagered) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS revenue_std7,
        
        -- =================================================================
        -- MOVING SUM: For weekly/monthly totals
        -- =================================================================
        SUM(daily_ggr) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ggr_sum7,
        
        SUM(total_wagered) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS revenue_sum7,
        
        -- =================================================================
        -- MIN/MAX: For range features
        -- =================================================================
        MIN(daily_ggr) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ggr_min7,
        
        MAX(daily_ggr) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ggr_max7
        
    FROM base_daily b
)

-- ============================================================================
-- STEP 5: Final Output with Growth Rates
-- ============================================================================
SELECT 
    w.*,
    
    -- =================================================================
    -- GROWTH RATES (Computed from lag features)
    -- =================================================================
    CASE 
        WHEN ggr_lag7 IS NULL OR ggr_lag7 = 0 THEN 0 
        ELSE (daily_ggr - ggr_lag7) / ABS(ggr_lag7) 
    END AS ggr_growth_7d,
    
    CASE 
        WHEN ggr_lag30 IS NULL OR ggr_lag30 = 0 THEN 0 
        ELSE (daily_ggr - ggr_lag30) / ABS(ggr_lag30) 
    END AS ggr_growth_30d,
    
    CASE 
        WHEN revenue_lag7 IS NULL OR revenue_lag7 = 0 THEN 0 
        ELSE (total_wagered - revenue_lag7) / revenue_lag7 
    END AS revenue_growth_7d,
    
    CASE 
        WHEN dau_lag7 IS NULL OR dau_lag7 = 0 THEN 0 
        ELSE (daily_active_users - dau_lag7) / CAST(dau_lag7 AS DOUBLE) 
    END AS dau_growth_7d,
    
    CASE 
        WHEN player_base_lag7 IS NULL OR player_base_lag7 = 0 THEN 0 
        ELSE (cumulative_player_base - player_base_lag7) / CAST(player_base_lag7 AS DOUBLE) 
    END AS player_growth_7d,
    
    -- =================================================================
    -- INTERACTION FEATURES
    -- =================================================================
    CASE 
        WHEN daily_active_users = 0 THEN 0 
        ELSE daily_ggr / daily_active_users 
    END AS ggr_per_player,
    
    CASE 
        WHEN daily_active_users = 0 THEN 0 
        ELSE total_wagered / daily_active_users 
    END AS revenue_per_player,
    
    CASE 
        WHEN total_sessions = 0 THEN 0 
        ELSE total_bets / CAST(total_sessions AS DOUBLE) 
    END AS bets_per_session,
    
    CASE 
        WHEN theoretical_rtp = 0 THEN 0 
        ELSE hold_percentage / (100 * (1 - theoretical_rtp)) 
    END AS hold_vs_expected,
    
    -- =================================================================
    -- VOLATILITY RATIO (Current vs 7-day avg)
    -- =================================================================
    CASE 
        WHEN ggr_ma7 IS NULL OR ggr_ma7 = 0 THEN 0 
        ELSE daily_ggr / ggr_ma7 
    END AS ggr_vs_ma7_ratio,
    
    -- =================================================================
    -- RANGE FEATURES
    -- =================================================================
    COALESCE(ggr_max7 - ggr_min7, 0) AS ggr_range7

FROM with_window_features w

ORDER BY 
    date ASC, 
    game_id ASC;


-- ============================================================================
-- VERIFICATION QUERY: Check the features
-- ============================================================================
-- SELECT 
--     date, 
--     game_id,
--     daily_ggr,
--     ggr_lag1,
--     ggr_lag7,
--     ggr_ma7,
--     ggr_std7,
--     ggr_growth_7d
-- FROM casino_ctg.log_db.daily_aggregations_features
-- WHERE game_id = 'G001'
-- ORDER BY date
-- LIMIT 50;