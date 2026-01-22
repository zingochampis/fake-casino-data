%sql
DROP TABLE IF EXISTS segmentation_training_data;
CREATE TABLE segmentation_training_data AS 

WITH data_date_range AS (
    -- Get the actual date range from the data
    SELECT 
        MAX(CAST(date AS DATE)) as max_date,
        MIN(CAST(date AS DATE)) as min_date
    FROM casino_ctg.log_db.transactions_bronze
),

player_base AS (
    SELECT 
        player_id,
        
        COUNT(*) as total_bets,
        SUM(CAST(bet_amount AS DOUBLE)) as total_wagered,
        SUM(CAST(win_amount AS DOUBLE)) as total_winnings,
        SUM(CAST(net_result AS DOUBLE)) as total_net_result,
        ABS(SUM(CAST(net_result AS DOUBLE))) as total_net_loss,
        
        AVG(CAST(bet_amount AS DOUBLE)) as avg_bet_amount,
        STDDEV_POP(CAST(bet_amount AS DOUBLE)) as stddev_bet_amount,
        PERCENTILE_APPROX(CAST(bet_amount AS DOUBLE), 0.5) as median_bet_amount,
        MIN(CAST(bet_amount AS DOUBLE)) as min_bet_amount,
        MAX(CAST(bet_amount AS DOUBLE)) as max_bet_amount,
        
        CASE 
            WHEN AVG(CAST(bet_amount AS DOUBLE)) > 0 
            THEN STDDEV_POP(CAST(bet_amount AS DOUBLE)) / AVG(CAST(bet_amount AS DOUBLE))
            ELSE 0 
        END as bet_cv,
        
        COUNT(DISTINCT session_id) as total_sessions,
        COUNT(DISTINCT CAST(date AS DATE)) as active_days,
        COUNT(DISTINCT game_id) as games_played,
        
        MIN(timestamp) as first_bet_date,
        MAX(timestamp) as last_bet_date,
        DATEDIFF(MAX(CAST(date AS DATE)), MIN(CAST(date AS DATE))) as days_active_span,
        
        SUM(CASE WHEN CAST(net_result AS DOUBLE) > 0 THEN 1 ELSE 0 END) as winning_bets,
        SUM(CASE WHEN CAST(net_result AS DOUBLE) < 0 THEN 1 ELSE 0 END) as losing_bets,
        SUM(CASE WHEN CAST(net_result AS DOUBLE) = 0 THEN 1 ELSE 0 END) as neutral_bets,
        
        CASE 
            WHEN COUNT(*) > 0 
            THEN CAST(SUM(CASE WHEN CAST(net_result AS DOUBLE) > 0 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*)
            ELSE 0 
        END as win_rate,
        
        MAX(CAST(consecutive_losses AS INT)) as max_consecutive_losses,
        MAX(CAST(consecutive_wins AS INT)) as max_consecutive_wins,
        AVG(CAST(consecutive_losses AS INT)) as avg_consecutive_losses,
        
        AVG(CASE WHEN CAST(is_weekend AS INT) = 1 THEN 1.0 ELSE 0.0 END) as weekend_play_pct,
        AVG(CASE WHEN CAST(is_peak_hour AS INT) = 1 THEN 1.0 ELSE 0.0 END) as peak_hour_play_pct,
        AVG(CASE WHEN CAST(is_payday AS INT) = 1 THEN 1.0 ELSE 0.0 END) as payday_play_pct
        
    FROM casino_ctg.log_db.transactions_bronze
    GROUP BY player_id
),

player_mode_day AS (
    SELECT 
        player_id,
        CAST(day_of_week AS INT) as most_common_day
    FROM (
        SELECT 
            player_id,
            day_of_week,
            COUNT(*) as cnt,
            ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY COUNT(*) DESC) as rn
        FROM casino_ctg.log_db.transactions_bronze
        GROUP BY player_id, day_of_week
    ) ranked
    WHERE rn = 1
),

player_mode_hour AS (
    SELECT 
        player_id,
        CAST(hour AS INT) as most_common_hour
    FROM (
        SELECT 
            player_id,
            hour,
            COUNT(*) as cnt,
            ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY COUNT(*) DESC) as rn
        FROM casino_ctg.log_db.transactions_bronze
        GROUP BY player_id, hour
    ) ranked
    WHERE rn = 1
),

session_level AS (
    SELECT 
        player_id,
        session_id,
        (unix_timestamp(MAX(timestamp)) - unix_timestamp(MIN(timestamp))) / 60.0 as session_duration_minutes,
        COUNT(*) as bets_per_session,
        SUM(CAST(net_result AS DOUBLE)) as session_net_result,
        MIN(CAST(session_balance AS DOUBLE)) as min_session_balance
    FROM casino_ctg.log_db.transactions_bronze
    GROUP BY player_id, session_id
),

session_features AS (
    SELECT
        player_id,
        AVG(session_duration_minutes) as avg_session_duration,
        STDDEV_POP(session_duration_minutes) as stddev_session_duration,
        AVG(bets_per_session) as avg_bets_per_session,
        AVG(session_net_result) as avg_session_net_result,
        MIN(min_session_balance) as worst_session_balance,
        MAX(session_net_result) as best_session_result
    FROM session_level
    GROUP BY player_id
),

recency_features AS (
    SELECT
        t.player_id,
        -- Days since last bet (relative to data end date)
        DATEDIFF(dr.max_date, MAX(CAST(t.date AS DATE))) as days_since_last_bet,
        -- Active days in last 7 days of data
        COUNT(DISTINCT CASE 
            WHEN CAST(t.date AS DATE) >= DATE_SUB(dr.max_date, 7) 
            THEN CAST(t.date AS DATE) 
        END) as active_days_last_7d,
        -- Active days in last 30 days of data
        COUNT(DISTINCT CASE 
            WHEN CAST(t.date AS DATE) >= DATE_SUB(dr.max_date, 30) 
            THEN CAST(t.date AS DATE) 
        END) as active_days_last_30d,
        -- Also capture: what % of the data period was the player active?
        -- Useful for understanding engagement relative to data window
        CAST(COUNT(DISTINCT CAST(t.date AS DATE)) AS DOUBLE) / 
            NULLIF(DATEDIFF(dr.max_date, dr.min_date) + 1, 0) as pct_days_active_in_period
    FROM casino_ctg.log_db.transactions_bronze t
    CROSS JOIN data_date_range dr
    GROUP BY t.player_id, dr.max_date, dr.min_date
),

bet_escalation AS (
    SELECT
        player_id,
        AVG(CASE 
            WHEN prev_net_result < 0 AND bet_amount > prev_bet_amount 
            THEN (bet_amount - prev_bet_amount) / NULLIF(prev_bet_amount, 0)
            ELSE 0
        END) as avg_bet_increase_after_loss,
        SUM(CASE 
            WHEN prev_net_result < 0 AND bet_amount > prev_bet_amount * 1.2
            THEN 1 ELSE 0 
        END) as chasing_incidents
    FROM (
        SELECT
            player_id,
            CAST(bet_amount AS DOUBLE) as bet_amount,
            LAG(CAST(bet_amount AS DOUBLE)) OVER (PARTITION BY player_id, session_id ORDER BY timestamp) as prev_bet_amount,
            LAG(CAST(net_result AS DOUBLE)) OVER (PARTITION BY player_id, session_id ORDER BY timestamp) as prev_net_result
        FROM casino_ctg.log_db.transactions_bronze
    ) bet_sequences
    GROUP BY player_id
)

SELECT
    pb.player_id,
    
    pb.total_wagered,
    pb.total_net_loss,
    pb.avg_bet_amount,
    pb.median_bet_amount,
    pb.total_sessions,
    pb.active_days,
    
    CASE WHEN pb.active_days > 0 THEN CAST(pb.total_sessions AS DOUBLE) / pb.active_days ELSE 0 END as sessions_per_day,
    CASE WHEN pb.total_sessions > 0 THEN CAST(pb.total_bets AS DOUBLE) / pb.total_sessions ELSE 0 END as bets_per_session,
    CASE WHEN pb.days_active_span > 0 THEN CAST(pb.active_days AS DOUBLE) / pb.days_active_span ELSE 0 END as activity_ratio,
    
    pb.stddev_bet_amount,
    pb.bet_cv as bet_volatility_coefficient,
    pb.max_bet_amount,
    pb.min_bet_amount,
    
    pb.win_rate,
    pb.winning_bets,
    pb.losing_bets,
    pb.max_consecutive_losses,
    pb.avg_consecutive_losses,
    
    be.avg_bet_increase_after_loss,
    be.chasing_incidents,
    
    sf.avg_session_duration,
    sf.stddev_session_duration,
    sf.avg_session_net_result,
    sf.worst_session_balance,
    
    pb.weekend_play_pct,
    pb.peak_hour_play_pct,
    pb.payday_play_pct,
    md.most_common_day,
    mh.most_common_hour,
    
    rf.days_since_last_bet,
    rf.active_days_last_7d,
    rf.active_days_last_30d,
    rf.pct_days_active_in_period,
    
    pb.games_played,
    
    pb.first_bet_date,
    pb.last_bet_date,
    pb.days_active_span,
    
    (
        CASE WHEN pb.avg_bet_amount > 100 THEN 2 ELSE 0 END +
        CASE WHEN pb.total_net_loss > 10000 THEN 2 ELSE 0 END +
        CASE WHEN pb.max_consecutive_losses > 5 THEN 1 ELSE 0 END +
        CASE WHEN be.chasing_incidents > 10 THEN 2 ELSE 0 END +
        CASE WHEN pb.bet_cv > 1.0 THEN 1 ELSE 0 END
    ) as preliminary_risk_score

FROM player_base pb
LEFT JOIN player_mode_day md ON pb.player_id = md.player_id
LEFT JOIN player_mode_hour mh ON pb.player_id = mh.player_id
LEFT JOIN session_features sf ON pb.player_id = sf.player_id
LEFT JOIN recency_features rf ON pb.player_id = rf.player_id
LEFT JOIN bet_escalation be ON pb.player_id = be.player_id

WHERE pb.total_bets >= 10
  AND pb.active_days >= 2

ORDER BY pb.total_wagered DESC;
