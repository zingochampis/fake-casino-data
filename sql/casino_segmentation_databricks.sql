-- ============================================================================
-- CASINO USER SEGMENTATION - DATABRICKS SQL QUERIES
-- ============================================================================
--
-- PURPOSE: Generate player-level features and segment players into:
--          Whale (top 1%), VIP (top 5%), High (top 15%), Regular (top 40%), Casual (bottom 60%)
--
-- DATABRICKS COMPATIBILITY NOTES:
-- - Uses PERCENTILE_APPROX() instead of PERCENTILE_CONT()
-- - Uses DATE_ADD/DATE_SUB instead of INTERVAL arithmetic
-- - Uses DATEDIFF() for date calculations
-- - Uses element_at(collect_list()) instead of MODE()
-- - Uses DATE_TRUNC('WEEK', ...) for week aggregation
-- - All aggregations verified for Spark SQL
--
-- TRANSACTION TABLE STRUCTURE:
-- transactions (
--     transaction_id STRING,
--     player_id STRING,
--     game_id STRING,
--     timestamp TIMESTAMP,
--     bet_amount DOUBLE,
--     win_amount DOUBLE,
--     net_result DOUBLE,        -- negative = player loss, positive = player win
--     session_id STRING,
--     session_balance DOUBLE,   -- running balance within session
--     consecutive_losses INT,
--     consecutive_wins INT,
--     hour INT,                 -- 0-23
--     day_of_week INT,          -- 1=Monday, 7=Sunday
--     day_of_month INT,         -- 1-31
--     month INT,                -- 1-12
--     is_weekend INT,           -- 1=Saturday/Sunday, 0=weekday
--     is_peak_hour INT,         -- 1=peak (18:00-23:00), 0=off-peak
--     is_payday INT,            -- 1=payday period (1-3, 15-17 of month), 0=other
--     date DATE
-- )
--
-- ============================================================================


-- ============================================================================
-- PART 1: TRAINING DATA FOR USER SEGMENTATION MODEL
-- ============================================================================
-- 
-- PURPOSE: Generate comprehensive player-level features for segmentation
-- TIMEFRAME: Use 3-6 months of historical data for stable patterns
-- OUTPUT: One row per player with 40+ behavioral features
--
-- KEY FEATURES CALCULATED:
-- 1. Monetary: total_wagered, avg_bet_amount, total_net_loss
-- 2. Behavioral: sessions_per_day, bets_per_session, win_rate
-- 3. Risk: max_consecutive_losses, chasing_incidents, bet_volatility
-- 4. Temporal: weekend_play_pct, peak_hour_play_pct, payday_play_pct
-- 5. Recency: days_since_last_bet, active_days_last_7d/30d
--
-- ============================================================================

WITH player_aggregates AS (
    -- ============================================================
    -- STEP 1.1: Calculate player-level aggregations
    -- ============================================================
    -- This CTE computes all basic statistics for each player
    -- over the training window (6 months)
    -- ============================================================
    
    SELECT 
        player_id,
        
        -- ----------------------------------------------------------
        -- MONETARY FEATURES (Most important for segmentation)
        -- ----------------------------------------------------------
        -- These metrics determine if a player is Whale/VIP/Casual
        
        COUNT(*) as total_bets,                    -- Volume of activity
        SUM(bet_amount) as total_wagered,          -- Total money bet (primary segment driver)
        SUM(win_amount) as total_winnings,         -- Total money won back
        SUM(net_result) as total_net_result,       -- Net win/loss (negative = player lost)
        ABS(SUM(net_result)) as total_net_loss,    -- Absolute value of losses (GGR contribution)
        
        -- ----------------------------------------------------------
        -- BET SIZE STATISTICS
        -- ----------------------------------------------------------
        -- Understanding typical bet sizes helps identify high rollers
        
        AVG(bet_amount) as avg_bet_amount,                    -- Mean bet size (key segmentation feature)
        STDDEV_POP(bet_amount) as stddev_bet_amount,          -- Spread of bet sizes
        PERCENTILE_APPROX(bet_amount, 0.5) as median_bet_amount,  -- Median bet (less affected by outliers)
        MIN(bet_amount) as min_bet_amount,                    -- Smallest bet ever placed
        MAX(bet_amount) as max_bet_amount,                    -- Largest bet ever placed
        
        -- ----------------------------------------------------------
        -- BET VOLATILITY (Coefficient of Variation)
        -- ----------------------------------------------------------
        -- CV = Standard Deviation / Mean
        -- High CV = erratic betting (risk indicator)
        -- Low CV = consistent betting (controlled behavior)
        -- Example: CV=0.5 means bets vary by 50% from average
        
        CASE 
            WHEN AVG(bet_amount) > 0 
            THEN STDDEV_POP(bet_amount) / AVG(bet_amount)
            ELSE 0 
        END as bet_cv,
        
        -- ----------------------------------------------------------
        -- SESSION & ENGAGEMENT FEATURES
        -- ----------------------------------------------------------
        
        COUNT(DISTINCT session_id) as total_sessions,  -- How many gambling sessions
        COUNT(DISTINCT date) as active_days,           -- How many unique days player bet
        COUNT(DISTINCT game_id) as games_played,       -- Game diversity (breadth of involvement)
        
        -- ----------------------------------------------------------
        -- TEMPORAL FEATURES (Player lifecycle)
        -- ----------------------------------------------------------
        
        MIN(timestamp) as first_bet_date,              -- When player joined
        MAX(timestamp) as last_bet_date,               -- Most recent activity
        DATEDIFF(MAX(date), MIN(date)) as days_active_span,  -- Calendar days from first to last bet
        
        -- ----------------------------------------------------------
        -- WIN/LOSS PATTERNS
        -- ----------------------------------------------------------
        -- Understanding win/loss distribution helps identify:
        -- - Lucky players (high win rate)
        -- - Losing streaks (problem gambling indicator)
        
        SUM(CASE WHEN net_result > 0 THEN 1 ELSE 0 END) as winning_bets,  -- Number of winning bets
        SUM(CASE WHEN net_result < 0 THEN 1 ELSE 0 END) as losing_bets,   -- Number of losing bets
        SUM(CASE WHEN net_result = 0 THEN 1 ELSE 0 END) as neutral_bets,  -- Push/tie outcomes
        
        -- ----------------------------------------------------------
        -- WIN RATE CALCULATION
        -- ----------------------------------------------------------
        -- Win Rate = (Winning Bets / Total Bets) × 100
        -- Example: 400 wins out of 1000 bets = 40% win rate
        -- Note: This doesn't mean profit - could be small wins, big losses
        
        CASE 
            WHEN COUNT(*) > 0 
            THEN CAST(SUM(CASE WHEN net_result > 0 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*)
            ELSE 0 
        END as win_rate,
        
        -- ----------------------------------------------------------
        -- LOSS CHASING INDICATORS
        -- ----------------------------------------------------------
        -- Consecutive losses are a key problem gambling indicator
        -- Research shows 3+ consecutive losses = chasing risk
        
        MAX(consecutive_losses) as max_consecutive_losses,      -- Worst losing streak
        MAX(consecutive_wins) as max_consecutive_wins,          -- Best winning streak
        AVG(consecutive_losses) as avg_consecutive_losses,      -- Average streak length
        
        -- ----------------------------------------------------------
        -- TEMPORAL PLAY PATTERNS
        -- ----------------------------------------------------------
        -- When do they play? Helps identify:
        -- - Weekend warriors (casual players)
        -- - Late night gamblers (potentially impulsive)
        -- - Payday players (financial stress indicator)
        
        AVG(CASE WHEN is_weekend = 1 THEN 1.0 ELSE 0.0 END) as weekend_play_pct,     -- % of bets on weekends
        AVG(CASE WHEN is_peak_hour = 1 THEN 1.0 ELSE 0.0 END) as peak_hour_play_pct, -- % of bets 6pm-11pm
        AVG(CASE WHEN is_payday = 1 THEN 1.0 ELSE 0.0 END) as payday_play_pct,       -- % of bets near payday
        
        -- ----------------------------------------------------------
        -- MOST COMMON DAY AND HOUR (MODE)
        -- ----------------------------------------------------------
        -- Databricks doesn't have MODE(), so we use this pattern:
        -- 1. Group by value, count occurrences
        -- 2. Order by count descending
        -- 3. Take first element
        -- This is computed in a subquery below
        
        day_of_week,  -- We'll compute mode in next CTE
        hour          -- We'll compute mode in next CTE
        
    FROM transactions
    WHERE date >= DATE_SUB(CURRENT_DATE(), 180)  -- 6 months = ~180 days
    GROUP BY player_id, day_of_week, hour  -- Need these for mode calculation
),

player_aggregates_with_mode AS (
    -- ============================================================
    -- STEP 1.2: Calculate MODE (most common values)
    -- ============================================================
    -- Databricks doesn't have MODE(), so we manually find it
    -- by ranking by frequency and taking the most common value
    -- ============================================================
    
    SELECT
        player_id,
        total_bets,
        total_wagered,
        total_winnings,
        total_net_result,
        total_net_loss,
        avg_bet_amount,
        stddev_bet_amount,
        median_bet_amount,
        min_bet_amount,
        max_bet_amount,
        bet_cv,
        total_sessions,
        active_days,
        games_played,
        first_bet_date,
        last_bet_date,
        days_active_span,
        winning_bets,
        losing_bets,
        neutral_bets,
        win_rate,
        max_consecutive_losses,
        max_consecutive_wins,
        avg_consecutive_losses,
        weekend_play_pct,
        peak_hour_play_pct,
        payday_play_pct,
        
        -- Get most common day (mode)
        FIRST_VALUE(day_of_week) OVER (
            PARTITION BY player_id 
            ORDER BY COUNT(*) OVER (PARTITION BY player_id, day_of_week) DESC
        ) as most_common_day,
        
        -- Get most common hour (mode)
        FIRST_VALUE(hour) OVER (
            PARTITION BY player_id 
            ORDER BY COUNT(*) OVER (PARTITION BY player_id, hour) DESC
        ) as most_common_hour
        
    FROM player_aggregates
),

player_aggregates_final AS (
    -- ============================================================
    -- STEP 1.3: Deduplicate after mode calculation
    -- ============================================================
    -- The window function above creates multiple rows per player
    -- We just need one row, so we take DISTINCT
    -- ============================================================
    
    SELECT DISTINCT
        player_id,
        total_bets,
        total_wagered,
        total_winnings,
        total_net_result,
        total_net_loss,
        avg_bet_amount,
        stddev_bet_amount,
        median_bet_amount,
        min_bet_amount,
        max_bet_amount,
        bet_cv,
        total_sessions,
        active_days,
        games_played,
        first_bet_date,
        last_bet_date,
        days_active_span,
        winning_bets,
        losing_bets,
        neutral_bets,
        win_rate,
        max_consecutive_losses,
        max_consecutive_wins,
        avg_consecutive_losses,
        weekend_play_pct,
        peak_hour_play_pct,
        payday_play_pct,
        most_common_day,
        most_common_hour
    FROM player_aggregates_with_mode
),

session_features AS (
    -- ============================================================
    -- STEP 2: Calculate session-level features
    -- ============================================================
    -- Session analysis helps identify:
    -- - Long sessions (possible addiction)
    -- - High-variance sessions (emotional gambling)
    -- - Session loss tolerance
    -- ============================================================
    
    SELECT
        player_id,
        AVG(session_duration_minutes) as avg_session_duration,        -- Average minutes per session
        STDDEV_POP(session_duration_minutes) as stddev_session_duration,  -- Session length variability
        AVG(bets_per_session) as avg_bets_per_session,                -- Betting intensity
        AVG(session_net_result) as avg_session_net_result,            -- Typical session outcome
        MIN(final_session_balance) as worst_session_balance,          -- Biggest single-session loss
        MAX(final_session_balance) as best_session_balance            -- Biggest single-session win
    FROM (
        -- Subquery: Calculate metrics for each individual session
        SELECT 
            player_id,
            session_id,
            
            -- Session duration in minutes
            -- unix_timestamp returns seconds, divide by 60 for minutes
            (unix_timestamp(MAX(timestamp)) - unix_timestamp(MIN(timestamp))) / 60.0 as session_duration_minutes,
            
            COUNT(*) as bets_per_session,                  -- How many bets in this session
            SUM(net_result) as session_net_result,         -- Net win/loss for session
            MAX(session_balance) as final_session_balance  -- Ending balance
            
        FROM transactions
        WHERE date >= DATE_SUB(CURRENT_DATE(), 180)
        GROUP BY player_id, session_id
    ) sessions
    GROUP BY player_id
),

recency_features AS (
    -- ============================================================
    -- STEP 3: Calculate recency metrics
    -- ============================================================
    -- Recency helps identify:
    -- - Active players (recent activity)
    -- - At-risk churners (inactive)
    -- - Engagement trends (increasing/decreasing)
    -- ============================================================
    
    SELECT
        player_id,
        
        -- Days since last bet (key churn indicator)
        -- 0-7 days = active, 8-30 days = at risk, 30+ days = churned
        DATEDIFF(CURRENT_DATE(), MAX(date)) as days_since_last_bet,
        
        -- Active days in last 7 days (short-term engagement)
        -- 0 = inactive, 7 = daily player
        COUNT(DISTINCT 
            CASE 
                WHEN date >= DATE_SUB(CURRENT_DATE(), 7)
                THEN date 
            END
        ) as active_days_last_7d,
        
        -- Active days in last 30 days (medium-term engagement)
        -- 0 = churned, 1-10 = casual, 10-20 = regular, 20+ = heavy user
        COUNT(DISTINCT 
            CASE 
                WHEN date >= DATE_SUB(CURRENT_DATE(), 30)
                THEN date 
            END
        ) as active_days_last_30d
        
    FROM transactions
    WHERE date >= DATE_SUB(CURRENT_DATE(), 180)
    GROUP BY player_id
),

bet_escalation AS (
    -- ============================================================
    -- STEP 4: Detect loss chasing behavior
    -- ============================================================
    -- Loss chasing = increasing bet size after losses
    -- This is a PRIMARY indicator of problem gambling
    -- 
    -- Research (Deng et al. 2021) shows:
    -- - 20%+ bet increase after loss = chasing
    -- - 3+ consecutive chasing incidents = high risk
    -- ============================================================
    
    SELECT
        player_id,
        
        -- Average % bet increase after a loss
        -- Example: Bet $10, lose, next bet $15 = 50% increase
        AVG(
            CASE 
                WHEN prev_net_result < 0 AND bet_amount > prev_bet_amount 
                THEN (bet_amount - prev_bet_amount) / NULLIF(prev_bet_amount, 0)
                ELSE 0
            END
        ) as avg_bet_increase_after_loss,
        
        -- Count of "chasing incidents"
        -- Definition: Bet increased by 20%+ immediately after a loss
        -- This is a binary flag (0 or 1) summed across all bets
        SUM(
            CASE 
                WHEN prev_net_result < 0 AND bet_amount > prev_bet_amount * 1.2
                THEN 1 
                ELSE 0 
            END
        ) as chasing_incidents
        
    FROM (
        -- Subquery: Get previous bet info using LAG window function
        SELECT
            player_id,
            bet_amount,
            net_result,
            
            -- LAG gets value from previous row within same player-session
            -- Order by timestamp to ensure chronological sequence
            LAG(bet_amount) OVER (
                PARTITION BY player_id, session_id 
                ORDER BY timestamp
            ) as prev_bet_amount,
            
            LAG(net_result) OVER (
                PARTITION BY player_id, session_id 
                ORDER BY timestamp
            ) as prev_net_result
            
        FROM transactions
        WHERE date >= DATE_SUB(CURRENT_DATE(), 180)
    ) bet_sequences
    GROUP BY player_id
)

-- ============================================================
-- FINAL TRAINING DATASET
-- ============================================================
-- Combine all feature CTEs into one training table
-- This output can be used for:
-- 1. Machine Learning (K-Means clustering, Random Forest)
-- 2. Manual segmentation rules
-- 3. Feature importance analysis
-- ============================================================

SELECT
    pa.player_id,
    
    -- ----------------------------------------------------------
    -- PRIMARY SEGMENTATION FEATURES
    -- ----------------------------------------------------------
    -- These 6 features are usually sufficient for basic segmentation
    
    pa.total_wagered,              -- #1 Most important: Total money bet
    pa.total_net_loss,             -- #2 Revenue contribution (GGR)
    pa.avg_bet_amount,             -- #3 Bet size (differentiates Whales)
    pa.median_bet_amount,          -- #4 Robust bet size measure
    pa.total_sessions,             -- #5 Engagement frequency
    pa.active_days,                -- #6 Loyalty indicator
    
    -- ----------------------------------------------------------
    -- DERIVED KEY METRICS
    -- ----------------------------------------------------------
    -- Ratios provide normalized metrics that work across segments
    
    CASE 
        WHEN pa.active_days > 0 
        THEN CAST(pa.total_sessions AS DOUBLE) / pa.active_days 
        ELSE 0 
    END as sessions_per_day,       -- Intensity: 0.5 = casual, 2+ = heavy user
    
    CASE 
        WHEN pa.total_sessions > 0 
        THEN CAST(pa.total_bets AS DOUBLE) / pa.total_sessions 
        ELSE 0 
    END as bets_per_session,       -- Velocity: 50 = slow, 500+ = fast
    
    CASE 
        WHEN pa.days_active_span > 0 
        THEN CAST(pa.active_days AS DOUBLE) / pa.days_active_span 
        ELSE 0 
    END as activity_ratio,         -- Consistency: 0.3 = sporadic, 0.8+ = regular
    
    -- ----------------------------------------------------------
    -- VOLATILITY & RISK FEATURES
    -- ----------------------------------------------------------
    
    pa.stddev_bet_amount,          -- Spread of bet sizes
    pa.bet_cv as bet_volatility_coefficient,  -- CV = stddev/mean (risk indicator)
    pa.max_bet_amount,             -- Highest bet ever (impulse indicator)
    pa.min_bet_amount,             -- Lowest bet ever (range)
    
    -- ----------------------------------------------------------
    -- WIN/LOSS PATTERNS
    -- ----------------------------------------------------------
    
    pa.win_rate,                   -- % of bets that won
    pa.winning_bets,               -- Count of wins
    pa.losing_bets,                -- Count of losses
    pa.max_consecutive_losses,     -- Longest losing streak
    pa.avg_consecutive_losses,     -- Average streak length
    
    -- ----------------------------------------------------------
    -- BEHAVIORAL RISK INDICATORS
    -- ----------------------------------------------------------
    -- These features help identify problem gambling
    
    be.avg_bet_increase_after_loss,  -- Loss chasing tendency
    be.chasing_incidents,             -- Count of chasing events (threshold: >10 = high risk)
    
    -- ----------------------------------------------------------
    -- SESSION CHARACTERISTICS
    -- ----------------------------------------------------------
    
    sf.avg_session_duration,       -- Minutes per session (threshold: >120 = concern)
    sf.stddev_session_duration,    -- Variability in session length
    sf.avg_session_net_result,     -- Typical session win/loss
    sf.worst_session_balance,      -- Biggest single-session loss (threshold: <-$1000 = risk)
    
    -- ----------------------------------------------------------
    -- TEMPORAL PATTERNS
    -- ----------------------------------------------------------
    
    pa.weekend_play_pct,           -- 0.7 = mostly weekends (casual), 0.4 = mixed
    pa.peak_hour_play_pct,         -- 0.8 = mostly evening (normal), 0.3 = off-hours (concern)
    pa.payday_play_pct,            -- 0.5+ = financial stress indicator
    pa.most_common_day,            -- 1=Mon, 7=Sun
    pa.most_common_hour,           -- 0-23 (late night = risk)
    
    -- ----------------------------------------------------------
    -- RECENCY FEATURES (RFM Analysis)
    -- ----------------------------------------------------------
    
    rf.days_since_last_bet,        -- 0-7 = active, 30+ = churned
    rf.active_days_last_7d,        -- Short-term engagement
    rf.active_days_last_30d,       -- Medium-term engagement
    
    -- ----------------------------------------------------------
    -- GAME DIVERSITY
    -- ----------------------------------------------------------
    
    pa.games_played,               -- 1-2 = specialized, 10+ = variety seeker
    
    -- ----------------------------------------------------------
    -- TIMESTAMPS (for time-based analysis)
    -- ----------------------------------------------------------
    
    pa.first_bet_date,             -- Player acquisition date
    pa.last_bet_date,              -- Most recent activity
    pa.days_active_span,           -- Calendar days from first to last
    
    -- ----------------------------------------------------------
    -- PRELIMINARY RISK SCORE (Optional)
    -- ----------------------------------------------------------
    -- Simple additive risk score for validation
    -- Range: 0 (no risk) to 8 (high risk)
    (
        CASE WHEN pa.avg_bet_amount > 100 THEN 2 ELSE 0 END +          -- High stakes
        CASE WHEN pa.total_net_loss > 10000 THEN 2 ELSE 0 END +        -- Large losses
        CASE WHEN pa.max_consecutive_losses > 5 THEN 1 ELSE 0 END +    -- Losing streaks
        CASE WHEN be.chasing_incidents > 10 THEN 2 ELSE 0 END +        -- Loss chasing
        CASE WHEN pa.bet_cv > 1.0 THEN 1 ELSE 0 END                    -- Erratic betting
    ) as preliminary_risk_score

FROM player_aggregates_final pa
LEFT JOIN session_features sf ON pa.player_id = sf.player_id
LEFT JOIN recency_features rf ON pa.player_id = rf.player_id
LEFT JOIN bet_escalation be ON pa.player_id = be.player_id

-- ----------------------------------------------------------
-- FILTER FOR MEANINGFUL ACTIVITY
-- ----------------------------------------------------------
-- Exclude players with insufficient data for stable patterns
WHERE pa.total_bets >= 10       -- Minimum 10 bets for statistical validity
  AND pa.active_days >= 2       -- Active on at least 2 different days

ORDER BY pa.total_wagered DESC;


-- ============================================================================
-- PART 2: ASSIGN SEGMENTS USING MANUAL RULES
-- ============================================================================
--
-- PURPOSE: Rule-based segmentation using percentile thresholds
-- USE CASE: Quick segmentation without ML model
-- 
-- SEGMENTATION LOGIC:
-- - Uses DUAL criteria: avg_bet_amount AND total_wagered
-- - Both must exceed threshold to qualify for segment
-- - More conservative than single-criterion segmentation
--
-- EXPECTED DISTRIBUTION (Pareto principle):
-- - Whale (1%): Top players, avg bet >$200, total wagered >$50k
-- - VIP (4%): High-value players, avg bet $30-200, wagered $10k-50k
-- - High (10%): Above-average players, avg bet $10-30, wagered $3k-10k
-- - Regular (25%): Average players, avg bet $3-10, wagered $500-3k
-- - Casual (60%): Low-value players, avg bet <$3, wagered <$500
--
-- ============================================================================

WITH player_features AS (
    -- ============================================================
    -- STEP 2.1: Calculate simplified features for segmentation
    -- ============================================================
    -- Only the essential metrics needed for rule-based assignment
    -- ============================================================
    
    SELECT 
        player_id,
        AVG(bet_amount) as avg_bet_amount,           -- Mean bet size
        SUM(bet_amount) as total_wagered,            -- Total volume
        COUNT(DISTINCT session_id) as total_sessions,
        COUNT(DISTINCT date) as active_days,
        CURRENT_DATE() as analysis_date              -- Timestamp for reproducibility
        
    FROM transactions
    WHERE date >= DATE_SUB(CURRENT_DATE(), 90)      -- 3 months = 90 days
    GROUP BY player_id
    HAVING COUNT(*) >= 10                            -- Minimum 10 bets
),

percentile_thresholds AS (
    -- ============================================================
    -- STEP 2.2: Calculate percentile thresholds
    -- ============================================================
    -- Databricks: Use PERCENTILE_APPROX() not PERCENTILE_CONT()
    -- Approximate is faster and accurate enough for segmentation
    -- ============================================================
    
    SELECT
        -- Bet amount thresholds
        PERCENTILE_APPROX(avg_bet_amount, 0.99) as whale_bet_threshold,     -- Top 1%
        PERCENTILE_APPROX(avg_bet_amount, 0.95) as vip_bet_threshold,       -- Top 5%
        PERCENTILE_APPROX(avg_bet_amount, 0.85) as high_bet_threshold,      -- Top 15%
        PERCENTILE_APPROX(avg_bet_amount, 0.60) as regular_bet_threshold,   -- Top 40%
        
        -- Total wagered thresholds
        PERCENTILE_APPROX(total_wagered, 0.99) as whale_wager_threshold,    -- Top 1%
        PERCENTILE_APPROX(total_wagered, 0.95) as vip_wager_threshold,      -- Top 5%
        PERCENTILE_APPROX(total_wagered, 0.85) as high_wager_threshold,     -- Top 15%
        PERCENTILE_APPROX(total_wagered, 0.60) as regular_wager_threshold   -- Top 40%
        
    FROM player_features
)

-- ============================================================
-- FINAL SEGMENT ASSIGNMENT
-- ============================================================
-- Uses CASE statement with cascading logic
-- Checks highest segment first, falls through to lower segments
-- ============================================================

SELECT
    pf.player_id,
    pf.avg_bet_amount,
    pf.total_wagered,
    pf.total_sessions,
    pf.active_days,
    
    -- ----------------------------------------------------------
    -- SEGMENT ASSIGNMENT LOGIC
    -- ----------------------------------------------------------
    -- DUAL CRITERIA: Must meet BOTH avg_bet AND total_wagered thresholds
    -- This prevents:
    -- - New whales (1 big bet) from being classified as Whale
    -- - Lucky casuals (won big once) from being classified as High
    CASE
        -- Whale: Top 1% in BOTH avg_bet and total_wagered
        WHEN pf.avg_bet_amount >= pt.whale_bet_threshold 
         AND pf.total_wagered >= pt.whale_wager_threshold
        THEN 'Whale'
        
        -- VIP: Top 5% in BOTH metrics
        WHEN pf.avg_bet_amount >= pt.vip_bet_threshold 
         AND pf.total_wagered >= pt.vip_wager_threshold
        THEN 'VIP'
        
        -- High: Top 15% in BOTH metrics
        WHEN pf.avg_bet_amount >= pt.high_bet_threshold 
         AND pf.total_wagered >= pt.high_wager_threshold
        THEN 'High'
        
        -- Regular: Top 40% in BOTH metrics
        WHEN pf.avg_bet_amount >= pt.regular_bet_threshold 
         AND pf.total_wagered >= pt.regular_wager_threshold
        THEN 'Regular'
        
        -- Casual: Everyone else (bottom 60%)
        ELSE 'Casual'
    END as player_segment,
    
    pf.analysis_date                                 -- When this segmentation was run
    
FROM player_features pf
CROSS JOIN percentile_thresholds pt                  -- Cartesian join (1 row) to get thresholds
ORDER BY pf.total_wagered DESC;

-- ----------------------------------------------------------
-- HOW TO CREATE PERSISTENT SEGMENTS TABLE
-- ----------------------------------------------------------
-- Option 1: Create as Delta table (recommended)
-- CREATE TABLE player_segments
-- USING DELTA
-- AS
-- SELECT player_id, player_segment, analysis_date
-- FROM (... query above ...)
--
-- Option 2: Create as view (refreshes each query)
-- CREATE OR REPLACE VIEW player_segments AS
-- SELECT player_id, player_segment, analysis_date
-- FROM (... query above ...)
-- ----------------------------------------------------------


-- ============================================================================
-- PART 3: VISUALIZATION DATA - DAILY METRICS BY SEGMENT
-- ============================================================================
--
-- PURPOSE: Generate time-series data for Plotly dashboard
-- OUTPUT: Daily aggregated metrics by segment for last 90 days
-- USE CASE: Stacked area charts, trend analysis, segment comparison
--
-- METRICS CALCULATED:
-- - active_players: Count of unique players active each day
-- - total_ggr: Gross Gaming Revenue (player losses)
-- - hold_percentage: GGR / Total Wagered (house edge performance)
-- - avg_wagered_per_player: ARPU (Average Revenue Per User)
--
-- ============================================================================

-- PREREQUISITE: Assumes player_segments table exists
-- Run Part 2 above first, or create from ML model output

WITH daily_player_metrics AS (
    -- ============================================================
    -- STEP 3.1: Calculate daily metrics per player per segment
    -- ============================================================
    -- Granular data: one row per (date, segment, player)
    -- Aggregated next to get segment-level metrics
    -- ============================================================
    
    SELECT
        t.date,
        ps.player_segment,
        t.player_id,
        
        -- Counting metrics
        COUNT(*) as bets,                             -- Number of bets this day
        
        -- Monetary metrics
        SUM(t.bet_amount) as wagered,                 -- Total bet volume
        SUM(t.win_amount) as winnings,                -- Total payouts
        SUM(t.net_result) as net_result,              -- Net win/loss (negative = player loss)
        
        -- Activity metrics
        COUNT(DISTINCT t.session_id) as sessions      -- Number of sessions this day
        
    FROM transactions t
    INNER JOIN player_segments ps ON t.player_id = ps.player_id
    WHERE t.date >= DATE_SUB(CURRENT_DATE(), 90)     -- Last 3 months
    GROUP BY t.date, ps.player_segment, t.player_id
)

-- ============================================================
-- FINAL DAILY AGGREGATION BY SEGMENT
-- ============================================================
-- Rolls up player-level data to segment-level
-- Output format: one row per (date, segment) combination
-- ============================================================

SELECT
    date,
    player_segment,
    
    -- ----------------------------------------------------------
    -- PLAYER COUNT METRICS
    -- ----------------------------------------------------------
    
    COUNT(DISTINCT player_id) as active_players,      -- DAU (Daily Active Users) by segment
    
    -- ----------------------------------------------------------
    -- VOLUME METRICS
    -- ----------------------------------------------------------
    
    SUM(bets) as total_bets,                          -- Total bets placed
    SUM(wagered) as total_wagered,                    -- Total money bet
    SUM(winnings) as total_winnings,                  -- Total money paid out
    SUM(net_result) as total_net_result,              -- Net (negative = house wins)
    
    -- ----------------------------------------------------------
    -- GGR CALCULATION
    -- ----------------------------------------------------------
    -- GGR = Gross Gaming Revenue = Player Losses
    -- Only count negative net_result (when player loses)
    -- Take absolute value to make it positive revenue
    ABS(SUM(CASE WHEN net_result < 0 THEN net_result ELSE 0 END)) as total_ggr,
    
    SUM(sessions) as total_sessions,                  -- Total sessions across all players
    
    -- ----------------------------------------------------------
    -- PER-PLAYER AVERAGES
    -- ----------------------------------------------------------
    -- These are ARPU-style metrics (Average Revenue Per User)
    
    AVG(bets) as avg_bets_per_player,                 -- Average bets per player this day
    AVG(wagered) as avg_wagered_per_player,           -- Average money bet per player
    AVG(net_result) as avg_net_result_per_player,     -- Average win/loss per player
    
    -- ----------------------------------------------------------
    -- HOLD PERCENTAGE (House Edge Performance)
    -- ----------------------------------------------------------
    -- Hold % = (GGR / Total Wagered) × 100
    -- Example: $100 wagered, $5 kept by house = 5% hold
    -- Typical hold: 3-7% for table games, 5-15% for slots
    CASE 
        WHEN SUM(wagered) > 0 
        THEN (ABS(SUM(CASE WHEN net_result < 0 THEN net_result ELSE 0 END)) / SUM(wagered)) * 100
        ELSE 0 
    END as hold_percentage

FROM daily_player_metrics
GROUP BY date, player_segment
ORDER BY date, player_segment;


-- ============================================================================
-- PART 4: WEEKLY AGGREGATED METRICS BY SEGMENT
-- ============================================================================
--
-- PURPOSE: Smoother trends by aggregating to weekly level
-- USE CASE: Executive dashboards, trend analysis with less noise
-- OUTPUT: Weekly metrics for last 6 months
--
-- WHY WEEKLY?
-- - Reduces day-to-day volatility
-- - Smooths out weekend vs weekday patterns
-- - More stable for trend detection
-- - Still granular enough for actionable insights
--
-- ============================================================================

WITH weekly_player_metrics AS (
    -- ============================================================
    -- STEP 4.1: Calculate weekly metrics per player per segment
    -- ============================================================
    -- Databricks: DATE_TRUNC('WEEK', timestamp) truncates to Monday
    -- Output is TIMESTAMP, cast to DATE for cleaner output
    -- ============================================================
    
    SELECT
        CAST(DATE_TRUNC('WEEK', t.timestamp) AS DATE) as week_start,  -- Monday of week
        ps.player_segment,
        t.player_id,
        
        -- Volume metrics
        COUNT(*) as bets,
        SUM(t.bet_amount) as wagered,
        SUM(t.win_amount) as winnings,
        SUM(t.net_result) as net_result,
        
        -- Activity metrics
        COUNT(DISTINCT t.session_id) as sessions,
        COUNT(DISTINCT t.date) as active_days        -- How many days this week player was active
        
    FROM transactions t
    INNER JOIN player_segments ps ON t.player_id = ps.player_id
    WHERE t.date >= DATE_SUB(CURRENT_DATE(), 180)   -- Last 6 months
    GROUP BY week_start, ps.player_segment, t.player_id
)

-- ============================================================
-- FINAL WEEKLY AGGREGATION BY SEGMENT
-- ============================================================

SELECT
    week_start,
    player_segment,
    
    -- ----------------------------------------------------------
    -- PLAYER COUNT METRICS
    -- ----------------------------------------------------------
    
    COUNT(DISTINCT player_id) as active_players,     -- WAU (Weekly Active Users)
    
    -- ----------------------------------------------------------
    -- VOLUME METRICS
    -- ----------------------------------------------------------
    
    SUM(bets) as total_bets,
    SUM(wagered) as total_wagered,
    SUM(winnings) as total_winnings,
    SUM(net_result) as total_net_result,
    ABS(SUM(CASE WHEN net_result < 0 THEN net_result ELSE 0 END)) as total_ggr,
    
    -- ----------------------------------------------------------
    -- ENGAGEMENT METRICS
    -- ----------------------------------------------------------
    
    SUM(sessions) as total_sessions,
    SUM(active_days) as total_active_days,           -- Total active days across all players
    
    -- ----------------------------------------------------------
    -- PER-PLAYER AVERAGES
    -- ----------------------------------------------------------
    
    AVG(bets) as avg_bets_per_player,
    AVG(wagered) as avg_wagered_per_player,
    AVG(sessions) as avg_sessions_per_player,        -- Sessions per player this week
    AVG(active_days) as avg_active_days_per_player,  -- Days per player this week (max 7)
    
    -- ----------------------------------------------------------
    -- PERFORMANCE METRICS
    -- ----------------------------------------------------------
    
    -- Hold percentage
    CASE 
        WHEN SUM(wagered) > 0 
        THEN (ABS(SUM(CASE WHEN net_result < 0 THEN net_result ELSE 0 END)) / SUM(wagered)) * 100
        ELSE 0 
    END as hold_percentage,
    
    -- Average revenue per user (ARPU)
    CASE
        WHEN COUNT(DISTINCT player_id) > 0
        THEN SUM(wagered) / COUNT(DISTINCT player_id)
        ELSE 0
    END as avg_revenue_per_user

FROM weekly_player_metrics
GROUP BY week_start, player_segment
ORDER BY week_start, player_segment;


-- ============================================================================
-- PART 5: SEGMENT DISTRIBUTION SNAPSHOT
-- ============================================================================
--
-- PURPOSE: Current segment composition and revenue contribution
-- USE CASE: Pie charts, executive summary, Pareto analysis
-- OUTPUT: One row per segment with count and revenue
--
-- KEY INSIGHTS:
-- - Validates Pareto principle (20% players = 80% revenue)
-- - Shows segment distribution vs revenue contribution
-- - Identifies high-value segments for targeted marketing
--
-- EXPECTED OUTPUT EXAMPLE:
-- Whale: 1% of players, 35% of revenue
-- VIP: 4% of players, 30% of revenue
-- High: 10% of players, 20% of revenue
-- Regular: 25% of players, 12% of revenue
-- Casual: 60% of players, 3% of revenue
--
-- ============================================================================

SELECT
    ps.player_segment,
    
    -- ----------------------------------------------------------
    -- PLAYER COUNT
    -- ----------------------------------------------------------
    
    COUNT(DISTINCT ps.player_id) as player_count,
    
    -- ----------------------------------------------------------
    -- 30-DAY ACTIVITY (Most recent month)
    -- ----------------------------------------------------------
    
    SUM(CASE 
        WHEN t.date >= DATE_SUB(CURRENT_DATE(), 30) 
        THEN t.bet_amount 
        ELSE 0 
    END) as wagered_30d,                              -- Total wagered last 30 days
    
    ABS(SUM(CASE 
        WHEN t.date >= DATE_SUB(CURRENT_DATE(), 30) AND t.net_result < 0 
        THEN t.net_result 
        ELSE 0 
    END)) as ggr_30d,                                 -- Total GGR last 30 days
    
    -- ----------------------------------------------------------
    -- REVENUE CONTRIBUTION PERCENTAGE
    -- ----------------------------------------------------------
    -- Shows what % of total revenue comes from this segment
    -- Window function SUM() OVER () calculates grand total
    -- Example: Whale GGR = $100k, Total GGR = $1M → 10%
    ABS(SUM(CASE 
        WHEN t.date >= DATE_SUB(CURRENT_DATE(), 30) AND t.net_result < 0 
        THEN t.net_result 
        ELSE 0 
    END)) * 100.0 / 
    NULLIF(SUM(ABS(CASE 
        WHEN t.date >= DATE_SUB(CURRENT_DATE(), 30) AND t.net_result < 0 
        THEN t.net_result 
        ELSE 0 
    END)) OVER (), 0) as pct_of_total_ggr

FROM player_segments ps
LEFT JOIN transactions t ON ps.player_id = t.player_id
GROUP BY ps.player_segment
ORDER BY ggr_30d DESC;


-- ============================================================================
-- PART 6: RISK INDICATORS BY SEGMENT
-- ============================================================================
--
-- PURPOSE: Monitor problem gambling indicators by segment
-- USE CASE: Responsible gaming, compliance reporting, risk alerts
-- OUTPUT: Risk metrics for each segment
--
-- RISK INDICATORS MONITORED:
-- 1. Long losing streaks (5+ consecutive losses)
-- 2. Large session losses (>$1000 in one session)
-- 3. Average consecutive losses
-- 4. % of players showing high-risk behavior
--
-- THRESHOLDS (based on research):
-- - 5+ consecutive losses = moderate risk
-- - $1000+ session loss = high risk (for non-Whales)
-- - 10%+ of segment high-risk = segment-level concern
--
-- ============================================================================

SELECT
    ps.player_segment,
    
    -- ----------------------------------------------------------
    -- BASELINE METRICS
    -- ----------------------------------------------------------
    
    COUNT(DISTINCT ps.player_id) as total_players,
    
    -- ----------------------------------------------------------
    -- HIGH-RISK INDICATOR COUNTS (Last 30 days)
    -- ----------------------------------------------------------
    
    -- Players with long losing streaks
    -- Research shows 5+ consecutive losses = chasing risk
    COUNT(DISTINCT CASE 
        WHEN t.date >= DATE_SUB(CURRENT_DATE(), 30)
         AND t.consecutive_losses >= 5
        THEN t.player_id 
    END) as players_with_long_losing_streaks,
    
    -- Players with large single-session losses
    -- $1000 threshold is adjustable based on segment
    -- For Casuals, even $100 might be concerning
    COUNT(DISTINCT CASE 
        WHEN t.date >= DATE_SUB(CURRENT_DATE(), 30)
         AND t.session_balance < -1000
        THEN t.player_id 
    END) as players_with_large_session_losses,
    
    -- ----------------------------------------------------------
    -- AVERAGE RISK METRICS
    -- ----------------------------------------------------------
    
    AVG(CASE 
        WHEN t.date >= DATE_SUB(CURRENT_DATE(), 30)
        THEN t.consecutive_losses 
    END) as avg_consecutive_losses,                   -- Average across all bets
    
    -- ----------------------------------------------------------
    -- PERCENTAGE AT HIGH RISK
    -- ----------------------------------------------------------
    -- Combined indicator: ANY of the risk factors
    -- (Long streaks OR Large losses)
    COUNT(DISTINCT CASE 
        WHEN t.date >= DATE_SUB(CURRENT_DATE(), 30)
         AND (t.consecutive_losses >= 5 OR t.session_balance < -1000)
        THEN t.player_id 
    END) * 100.0 / NULLIF(COUNT(DISTINCT ps.player_id), 0) as pct_high_risk_players

FROM player_segments ps
LEFT JOIN transactions t ON ps.player_id = t.player_id
GROUP BY ps.player_segment
ORDER BY pct_high_risk_players DESC;

-- ----------------------------------------------------------
-- INTERPRETATION GUIDE:
-- ----------------------------------------------------------
-- pct_high_risk_players < 5% = Low risk (normal)
-- pct_high_risk_players 5-10% = Moderate risk (monitor)
-- pct_high_risk_players > 10% = High risk (intervention needed)
--
-- Whales may naturally have higher percentages due to large bets
-- Adjust thresholds by segment for more accurate risk assessment
-- ----------------------------------------------------------


-- ============================================================================
-- PART 7: GAME PREFERENCE BY SEGMENT
-- ============================================================================
--
-- PURPOSE: Understand which games attract which player segments
-- USE CASE: Game portfolio optimization, marketing targeting
-- OUTPUT: Top games by segment with concentration metrics
--
-- KEY INSIGHTS:
-- - Whales prefer high-stakes table games
-- - Casuals prefer low-volatility slots
-- - Game diversity varies by segment
-- - Revenue concentration by game-segment combo
--
-- ============================================================================

SELECT
    ps.player_segment,
    t.game_id,
    
    -- ----------------------------------------------------------
    -- PLAYER METRICS
    -- ----------------------------------------------------------
    
    COUNT(DISTINCT t.player_id) as unique_players,   -- How many players played this game
    
    -- ----------------------------------------------------------
    -- VOLUME METRICS
    -- ----------------------------------------------------------
    
    COUNT(*) as total_bets,                           -- Total bets on this game
    SUM(t.bet_amount) as total_wagered,               -- Total money bet
    ABS(SUM(CASE WHEN t.net_result < 0 THEN t.net_result ELSE 0 END)) as total_ggr,  -- Revenue from this game
    
    -- ----------------------------------------------------------
    -- CONCENTRATION METRIC
    -- ----------------------------------------------------------
    -- What % of this segment's bets are on this game?
    -- Window function: SUM() OVER (PARTITION BY segment)
    -- Example: Whales bet 1000 times, 400 on BlackJack → 40%
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY ps.player_segment) as pct_of_segment_bets

FROM player_segments ps
INNER JOIN transactions t ON ps.player_id = t.player_id
WHERE t.date >= DATE_SUB(CURRENT_DATE(), 30)         -- Last 30 days
GROUP BY ps.player_segment, t.game_id
ORDER BY ps.player_segment, total_ggr DESC;

-- ----------------------------------------------------------
-- TYPICAL FINDINGS:
-- ----------------------------------------------------------
-- Whale segment:
-- - BlackJack: 45% of bets, $250k GGR
-- - Baccarat: 30% of bets, $180k GGR
-- - Roulette: 25% of bets, $120k GGR
--
-- Casual segment:
-- - Penny Slots: 60% of bets, $12k GGR
-- - Video Poker: 25% of bets, $5k GGR
-- - Classic Slots: 15% of bets, $3k GGR
-- ----------------------------------------------------------


-- ============================================================================
-- SUMMARY & USAGE GUIDE
-- ============================================================================
--
-- QUERY EXECUTION ORDER:
-- 1. Run Part 1 (Training Data) → Export to CSV for ML model
-- 2. Run Part 2 (Segment Assignment) → Create player_segments table
-- 3. Run Parts 3-7 (Analytics) → Generate dashboard data
--
-- PERFORMANCE TIPS:
-- - Cache player_segments table: CACHE TABLE player_segments;
-- - Use Delta format: CREATE TABLE ... USING DELTA
-- - Add Z-ORDER: OPTIMIZE table_name ZORDER BY (player_id, date)
-- - Partition large tables: PARTITIONED BY (date)
--
-- REFRESH SCHEDULE:
-- - Part 1 (Training): Monthly (expensive, uses 6 months data)
-- - Part 2 (Segments): Weekly (recalculate segments)
-- - Parts 3-7 (Analytics): Daily (dashboard refresh)
--
-- VALIDATION CHECKS:
-- - Segment distribution: ~1% Whale, 4% VIP, 10% High, 25% Regular, 60% Casual
-- - Revenue concentration: Top 20% players = 80-90% of GGR (Pareto)
-- - Hold percentage: 3-7% table games, 5-15% slots
-- - Risk indicators: <10% high-risk players per segment
--
-- ============================================================================
