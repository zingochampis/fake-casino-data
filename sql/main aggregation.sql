
  CREATE OR REPLACE TABLE casino_ctg.log_db.daily_aggregations AS
  WITH 
  -- 1. Identify when each player first appeared (The "Activation Date")
  player_first_appearances AS (
      SELECT 
          player_id, 
          MIN(CAST(timestamp AS DATE)) as first_play_date
      FROM casino_ctg.log_db.transactions_bronze
      GROUP BY player_id
  ),
 
  -- 2. Count new players per day and calculate the Running Total (Cumulative Growth)
  daily_platform_growth AS (
      SELECT 
          first_play_date AS date,
          COUNT(player_id) as new_players_today,
          -- Window function to sum all new players up to the current row's date
          SUM(COUNT(player_id)) OVER (
              ORDER BY first_play_date 
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          ) as cumulative_player_base
      FROM player_first_appearances
      GROUP BY first_play_date
  )
 
  -- 3. Main Aggregation
  SELECT 
      -- Temporal Dimensions
      CAST(t.timestamp AS DATE) AS date,
      t.game_id,
 
      -- Growth Variable (Explains the "Ramp Up")
      COALESCE(pg.cumulative_player_base, 0) AS cumulative_player_base, 
 
      -- Target KPIs
      COUNT(DISTINCT t.player_id) AS daily_active_users,
      SUM(t.bet_amount) AS gross_daily_revenue,
      SUM(t.win_amount) AS daily_payout,
      SUM(-t.net_result) AS daily_ggr,
 
      -- Generator Auditing Columns (For CSV comparison)
      COUNT(t.transaction_id) AS total_bets,
      COUNT(DISTINCT t.player_id) AS active_players,
      SUM(t.bet_amount) AS total_wagered,
      SUM(t.win_amount) AS total_won,
      SUM(t.net_result) AS gross_gaming_revenue, -- (Negative)
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
 
      -- Metadata Features
      m.type AS game_type,
      m.rtp AS theoretical_rtp,
      m.volatility,
      m.visual_engagement
 
  FROM casino_ctg.log_db.transactions_bronze t
  INNER JOIN casino_ctg.log_db.games_bronze m ON t.game_id = m.game_id
  -- Join the daily growth stats
  LEFT JOIN daily_platform_growth pg ON CAST(t.timestamp AS DATE) = pg.date
 
  GROUP BY 
      CAST(t.timestamp AS DATE), 
      t.game_id,
      pg.cumulative_player_base,
      m.type,
      m.rtp,
      m.volatility,
      m.visual_engagement
 
  ORDER BY 
      date ASC, 
      t.game_id ASC;

# COMMAND ----------

  --%sql
  --select * from casino_ctg.log_db.daily_aggregations



