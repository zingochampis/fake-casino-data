"""
Casino User Segmentation - Data Pipeline
Connects SQL queries to Plotly visualizations for real transaction data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Database connection parameters (adjust for your setup)
DB_CONFIG = {
    'host': 'your_host',
    'database': 'casino_db',
    'user': 'your_user',
    'password': 'your_password'
}

# Alternatively, if using CSV files:
CSV_PATH = '/mnt/user-data/uploads/'  # Adjust to your path


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_transactions_from_csv(months=['2024-01'], path=CSV_PATH):
    """
    Load transaction data from CSV files
    
    Args:
        months: List of month identifiers like ['2024-01', '2024-02']
        path: Path to CSV files
    
    Returns:
        DataFrame with combined transactions
    """
    dfs = []
    for month in months:
        filename = f'{path}transactions_{month}.csv'
        try:
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = pd.to_datetime(df['date'])
            dfs.append(df)
            print(f"Loaded {len(df):,} transactions from {month}")
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal transactions loaded: {len(combined):,}")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"Unique players: {combined['player_id'].nunique():,}")
        return combined
    else:
        raise ValueError("No transaction files found")


def load_from_database(query, connection_params):
    """
    Load data from database using SQL query
    
    Args:
        query: SQL query string
        connection_params: Database connection parameters
    
    Returns:
        DataFrame with query results
    """
    import psycopg2  # or your database connector
    
    conn = psycopg2.connect(**connection_params)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# ============================================================================
# FEATURE ENGINEERING (PYTHON IMPLEMENTATION OF SQL LOGIC)
# ============================================================================

def calculate_player_features(transactions_df):
    """
    Calculate player-level features for segmentation
    Python implementation of the SQL training query
    
    Args:
        transactions_df: DataFrame with transaction data
    
    Returns:
        DataFrame with player features (one row per player)
    """
    print("\nCalculating player features...")
    
    # Basic aggregations
    player_agg = transactions_df.groupby('player_id').agg({
        'transaction_id': 'count',
        'bet_amount': ['sum', 'mean', 'std', 'median', 'min', 'max'],
        'win_amount': 'sum',
        'net_result': 'sum',
        'session_id': 'nunique',
        'date': 'nunique',
        'game_id': 'nunique',
        'timestamp': ['min', 'max'],
        'consecutive_losses': 'max',
        'consecutive_wins': 'max',
        'is_weekend': 'mean',
        'is_peak_hour': 'mean',
        'is_payday': 'mean'
    })
    
    # Flatten multi-level columns
    player_agg.columns = ['_'.join(col).strip('_') for col in player_agg.columns]
    player_agg = player_agg.rename(columns={
        'transaction_id_count': 'total_bets',
        'bet_amount_sum': 'total_wagered',
        'bet_amount_mean': 'avg_bet_amount',
        'bet_amount_std': 'stddev_bet_amount',
        'bet_amount_median': 'median_bet_amount',
        'bet_amount_min': 'min_bet_amount',
        'bet_amount_max': 'max_bet_amount',
        'win_amount_sum': 'total_winnings',
        'net_result_sum': 'total_net_result',
        'session_id_nunique': 'total_sessions',
        'date_nunique': 'active_days',
        'game_id_nunique': 'games_played',
        'timestamp_min': 'first_bet_date',
        'timestamp_max': 'last_bet_date',
        'consecutive_losses_max': 'max_consecutive_losses',
        'consecutive_wins_max': 'max_consecutive_wins',
        'is_weekend_mean': 'weekend_play_pct',
        'is_peak_hour_mean': 'peak_hour_play_pct',
        'is_payday_mean': 'payday_play_pct'
    })
    
    # Calculate derived features
    player_agg['total_net_loss'] = player_agg['total_net_result'].abs()
    player_agg['bet_cv'] = player_agg['stddev_bet_amount'] / player_agg['avg_bet_amount'].replace(0, np.nan)
    player_agg['sessions_per_day'] = player_agg['total_sessions'] / player_agg['active_days'].replace(0, 1)
    player_agg['bets_per_session'] = player_agg['total_bets'] / player_agg['total_sessions'].replace(0, 1)
    player_agg['days_active_span'] = (player_agg['last_bet_date'] - player_agg['first_bet_date']).dt.days + 1
    player_agg['activity_ratio'] = player_agg['active_days'] / player_agg['days_active_span'].replace(0, 1)
    
    # Win/loss patterns
    win_loss = transactions_df.groupby('player_id').apply(
        lambda x: pd.Series({
            'winning_bets': (x['net_result'] > 0).sum(),
            'losing_bets': (x['net_result'] < 0).sum(),
            'neutral_bets': (x['net_result'] == 0).sum()
        })
    )
    player_agg = player_agg.join(win_loss)
    player_agg['win_rate'] = player_agg['winning_bets'] / player_agg['total_bets']
    
    # Session features
    session_agg = transactions_df.groupby(['player_id', 'session_id']).agg({
        'timestamp': ['min', 'max'],
        'transaction_id': 'count',
        'net_result': 'sum',
        'session_balance': 'last'
    })
    session_agg.columns = ['_'.join(col).strip('_') for col in session_agg.columns]
    session_agg['session_duration'] = (
        session_agg['timestamp_max'] - session_agg['timestamp_min']
    ).dt.total_seconds() / 60  # minutes
    
    session_features = session_agg.groupby('player_id').agg({
        'session_duration': ['mean', 'std'],
        'transaction_id_count': 'mean',
        'net_result_sum': 'mean',
        'session_balance_last': ['min', 'max']
    })
    session_features.columns = [
        'avg_session_duration', 'stddev_session_duration',
        'avg_bets_per_session', 'avg_session_net_result',
        'worst_session_balance', 'best_session_balance'
    ]
    player_agg = player_agg.join(session_features)
    
    # Recency features
    latest_date = transactions_df['date'].max()
    player_agg['days_since_last_bet'] = (latest_date - player_agg['last_bet_date'].dt.date).dt.days
    
    # Loss chasing indicators
    chasing = transactions_df.sort_values(['player_id', 'session_id', 'timestamp']).groupby(['player_id', 'session_id']).apply(
        lambda x: pd.Series({
            'chasing_incidents': (
                (x['net_result'].shift(1) < 0) & 
                (x['bet_amount'] > x['bet_amount'].shift(1) * 1.2)
            ).sum()
        })
    ).groupby('player_id').sum()
    player_agg = player_agg.join(chasing)
    
    # Fill NaN values
    player_agg = player_agg.fillna(0)
    
    print(f"Features calculated for {len(player_agg):,} players")
    return player_agg.reset_index()


# ============================================================================
# SEGMENTATION MODELS
# ============================================================================

def segment_players_kmeans(features_df, n_clusters=5):
    """
    Segment players using K-Means clustering
    
    Args:
        features_df: DataFrame with player features
        n_clusters: Number of segments (default 5: Whale, VIP, High, Regular, Casual)
    
    Returns:
        DataFrame with added 'cluster' column
    """
    print(f"\nPerforming K-Means clustering with {n_clusters} segments...")
    
    # Select key features for clustering
    feature_cols = [
        'avg_bet_amount', 'total_wagered', 'total_net_loss',
        'sessions_per_day', 'bets_per_session', 'bet_cv',
        'win_rate', 'max_consecutive_losses'
    ]
    
    X = features_df[feature_cols].copy()
    
    # Handle any remaining NaN or inf values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Assign segment names based on average bet amount ranking
    cluster_avg_bet = features_df.groupby('cluster')['avg_bet_amount'].mean().sort_values(ascending=False)
    segment_names = ['Whale', 'VIP', 'High', 'Regular', 'Casual']
    cluster_to_segment = {cluster: segment_names[i] for i, cluster in enumerate(cluster_avg_bet.index)}
    
    features_df['player_segment'] = features_df['cluster'].map(cluster_to_segment)
    
    print("\nSegment distribution:")
    print(features_df['player_segment'].value_counts().sort_index())
    print("\nAverage bet by segment:")
    print(features_df.groupby('player_segment')['avg_bet_amount'].mean().sort_values(ascending=False))
    
    return features_df


def segment_players_percentile(features_df):
    """
    Segment players using percentile-based rules
    Python implementation of SQL percentile segmentation
    
    Args:
        features_df: DataFrame with player features
    
    Returns:
        DataFrame with added 'player_segment' column
    """
    print("\nSegmenting players using percentile rules...")
    
    # Calculate percentile thresholds
    whale_bet_threshold = features_df['avg_bet_amount'].quantile(0.99)
    vip_bet_threshold = features_df['avg_bet_amount'].quantile(0.95)
    high_bet_threshold = features_df['avg_bet_amount'].quantile(0.85)
    regular_bet_threshold = features_df['avg_bet_amount'].quantile(0.60)
    
    whale_wager_threshold = features_df['total_wagered'].quantile(0.99)
    vip_wager_threshold = features_df['total_wagered'].quantile(0.95)
    high_wager_threshold = features_df['total_wagered'].quantile(0.85)
    regular_wager_threshold = features_df['total_wagered'].quantile(0.60)
    
    # Assign segments
    def assign_segment(row):
        if (row['avg_bet_amount'] >= whale_bet_threshold and 
            row['total_wagered'] >= whale_wager_threshold):
            return 'Whale'
        elif (row['avg_bet_amount'] >= vip_bet_threshold and 
              row['total_wagered'] >= vip_wager_threshold):
            return 'VIP'
        elif (row['avg_bet_amount'] >= high_bet_threshold and 
              row['total_wagered'] >= high_wager_threshold):
            return 'High'
        elif (row['avg_bet_amount'] >= regular_bet_threshold and 
              row['total_wagered'] >= regular_wager_threshold):
            return 'Regular'
        else:
            return 'Casual'
    
    features_df['player_segment'] = features_df.apply(assign_segment, axis=1)
    
    print("\nSegment distribution:")
    print(features_df['player_segment'].value_counts())
    print("\nThresholds used:")
    print(f"Whale: avg_bet >= ${whale_bet_threshold:.2f}, total_wagered >= ${whale_wager_threshold:.2f}")
    print(f"VIP: avg_bet >= ${vip_bet_threshold:.2f}, total_wagered >= ${vip_wager_threshold:.2f}")
    
    return features_df


# ============================================================================
# VISUALIZATION DATA PREPARATION
# ============================================================================

def prepare_daily_metrics(transactions_df, segments_df):
    """
    Prepare daily metrics by segment for time-series visualization
    Python implementation of SQL visualization query
    
    Args:
        transactions_df: DataFrame with transaction data
        segments_df: DataFrame with player_id and player_segment
    
    Returns:
        DataFrame with daily metrics by segment
    """
    print("\nPreparing daily metrics for visualization...")
    
    # Merge transactions with segments
    df = transactions_df.merge(segments_df[['player_id', 'player_segment']], on='player_id', how='inner')
    
    # Daily aggregation
    daily_metrics = df.groupby(['date', 'player_segment']).agg({
        'player_id': 'nunique',
        'transaction_id': 'count',
        'bet_amount': 'sum',
        'win_amount': 'sum',
        'net_result': 'sum',
        'session_id': 'nunique'
    }).reset_index()
    
    daily_metrics.columns = [
        'date', 'player_segment', 'active_players', 'total_bets',
        'total_wagered', 'total_winnings', 'total_net_result', 'total_sessions'
    ]
    
    # Calculate GGR (Gross Gaming Revenue = absolute value of negative net results)
    daily_metrics['total_ggr'] = daily_metrics['total_net_result'].apply(lambda x: abs(x) if x < 0 else 0)
    
    # Calculate hold percentage
    daily_metrics['hold_percentage'] = (
        daily_metrics['total_ggr'] / daily_metrics['total_wagered'].replace(0, np.nan) * 100
    ).fillna(0)
    
    # Per-player averages
    daily_metrics['avg_wagered_per_player'] = (
        daily_metrics['total_wagered'] / daily_metrics['active_players'].replace(0, 1)
    )
    
    return daily_metrics


def prepare_weekly_metrics(transactions_df, segments_df):
    """
    Prepare weekly metrics for smoother trend visualization
    
    Args:
        transactions_df: DataFrame with transaction data
        segments_df: DataFrame with player_id and player_segment
    
    Returns:
        DataFrame with weekly metrics by segment
    """
    # Merge and add week column
    df = transactions_df.merge(segments_df[['player_id', 'player_segment']], on='player_id', how='inner')
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='d')
    
    # Weekly aggregation
    weekly_metrics = df.groupby(['week_start', 'player_segment']).agg({
        'player_id': 'nunique',
        'transaction_id': 'count',
        'bet_amount': 'sum',
        'win_amount': 'sum',
        'net_result': 'sum',
        'session_id': 'nunique'
    }).reset_index()
    
    weekly_metrics.columns = [
        'week_start', 'player_segment', 'active_players', 'total_bets',
        'total_wagered', 'total_winnings', 'total_net_result', 'total_sessions'
    ]
    
    weekly_metrics['total_ggr'] = weekly_metrics['total_net_result'].apply(lambda x: abs(x) if x < 0 else 0)
    weekly_metrics['hold_percentage'] = (
        weekly_metrics['total_ggr'] / weekly_metrics['total_wagered'].replace(0, np.nan) * 100
    ).fillna(0)
    
    return weekly_metrics


# ============================================================================
# PLOTLY VISUALIZATIONS
# ============================================================================

def create_dashboard(daily_metrics, segments_df):
    """
    Create interactive Plotly dashboard with multiple visualizations
    
    Args:
        daily_metrics: DataFrame with daily metrics by segment
        segments_df: DataFrame with player segments
    
    Returns:
        Plotly Figure object
    """
    print("\nCreating dashboard visualizations...")
    
    # Define color scheme for segments
    segment_colors = {
        'Whale': '#8B0000',    # Dark red
        'VIP': '#FF4500',      # Orange red
        'High': '#FFD700',     # Gold
        'Regular': '#90EE90',  # Light green
        'Casual': '#87CEEB'    # Sky blue
    }
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Daily GGR by Segment',
            'Segment Distribution',
            'Active Players Over Time',
            'Revenue Concentration',
            'Hold Percentage by Segment',
            'Avg Wagered per Player'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'pie'}],
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'scatter'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 1. Daily GGR by Segment (Stacked Area)
    for segment in ['Whale', 'VIP', 'High', 'Regular', 'Casual']:
        segment_data = daily_metrics[daily_metrics['player_segment'] == segment]
        fig.add_trace(
            go.Scatter(
                x=segment_data['date'],
                y=segment_data['total_ggr'],
                name=segment,
                mode='lines',
                stackgroup='one',
                fillcolor=segment_colors.get(segment),
                line=dict(width=0.5, color=segment_colors.get(segment)),
                hovertemplate='%{y:$,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Segment Distribution (Pie)
    segment_counts = segments_df['player_segment'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            marker=dict(colors=[segment_colors.get(seg) for seg in segment_counts.index]),
            hovertemplate='%{label}: %{value} players<br>%{percent}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Active Players Over Time
    for segment in ['Whale', 'VIP', 'High', 'Regular', 'Casual']:
        segment_data = daily_metrics[daily_metrics['player_segment'] == segment]
        fig.add_trace(
            go.Scatter(
                x=segment_data['date'],
                y=segment_data['active_players'],
                name=segment,
                mode='lines',
                line=dict(color=segment_colors.get(segment)),
                showlegend=False,
                hovertemplate='%{y} players<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 4. Revenue Concentration (Bar chart of total GGR by segment)
    total_ggr_by_segment = daily_metrics.groupby('player_segment')['total_ggr'].sum().reset_index()
    total_ggr_by_segment = total_ggr_by_segment.sort_values('total_ggr', ascending=False)
    fig.add_trace(
        go.Bar(
            x=total_ggr_by_segment['player_segment'],
            y=total_ggr_by_segment['total_ggr'],
            marker=dict(color=[segment_colors.get(seg) for seg in total_ggr_by_segment['player_segment']]),
            showlegend=False,
            hovertemplate='%{x}: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 5. Hold Percentage Over Time
    for segment in ['Whale', 'VIP', 'High', 'Regular', 'Casual']:
        segment_data = daily_metrics[daily_metrics['player_segment'] == segment]
        fig.add_trace(
            go.Scatter(
                x=segment_data['date'],
                y=segment_data['hold_percentage'],
                name=segment,
                mode='lines',
                line=dict(color=segment_colors.get(segment)),
                showlegend=False,
                hovertemplate='%{y:.1f}%<extra></extra>'
            ),
            row=3, col=1
        )
    
    # 6. Avg Wagered per Player
    for segment in ['Whale', 'VIP', 'High', 'Regular', 'Casual']:
        segment_data = daily_metrics[daily_metrics['player_segment'] == segment]
        fig.add_trace(
            go.Scatter(
                x=segment_data['date'],
                y=segment_data['avg_wagered_per_player'],
                name=segment,
                mode='lines',
                line=dict(color=segment_colors.get(segment)),
                showlegend=False,
                hovertemplate='$%{y:,.0f}<extra></extra>'
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Casino Player Segmentation Dashboard',
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#1e3a8a')
        ),
        height=1200,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_xaxes(title_text='Date', row=3, col=2)
    fig.update_yaxes(title_text='GGR ($)', row=1, col=1)
    fig.update_yaxes(title_text='Players', row=2, col=1)
    fig.update_yaxes(title_text='GGR ($)', row=2, col=2)
    fig.update_yaxes(title_text='Hold %', row=3, col=1)
    fig.update_yaxes(title_text='Avg Wagered ($)', row=3, col=2)
    
    return fig


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline:
    1. Load transaction data
    2. Calculate player features
    3. Segment players
    4. Prepare visualization data
    5. Create dashboard
    """
    print("=" * 80)
    print("CASINO USER SEGMENTATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[Step 1] Loading transaction data...")
    # Option A: From CSV files
    transactions = load_transactions_from_csv(['2024-01'])  # Add more months as needed
    
    # Option B: From database (uncomment if using database)
    # with open('casino_segmentation_queries.sql', 'r') as f:
    #     training_query = f.read().split('-- ============================================================================')[1]
    # transactions = load_from_database(training_query, DB_CONFIG)
    
    # Step 2: Calculate features
    print("\n[Step 2] Calculating player features...")
    player_features = calculate_player_features(transactions)
    
    # Filter for meaningful activity
    player_features = player_features[
        (player_features['total_bets'] >= 10) &
        (player_features['active_days'] >= 2)
    ]
    print(f"Filtered to {len(player_features):,} players with meaningful activity")
    
    # Step 3: Segment players
    print("\n[Step 3] Segmenting players...")
    # Option A: K-Means clustering
    # segments = segment_players_kmeans(player_features, n_clusters=5)
    
    # Option B: Percentile-based rules (recommended for interpretability)
    segments = segment_players_percentile(player_features)
    
    # Step 4: Prepare visualization data
    print("\n[Step 4] Preparing visualization data...")
    daily_metrics = prepare_daily_metrics(transactions, segments)
    
    # Step 5: Create dashboard
    print("\n[Step 5] Creating dashboard...")
    dashboard = create_dashboard(daily_metrics, segments)
    
    # Save outputs
    print("\n[Step 6] Saving outputs...")
    segments[['player_id', 'player_segment']].to_csv('/mnt/user-data/outputs/player_segments.csv', index=False)
    player_features.to_csv('/mnt/user-data/outputs/player_features.csv', index=False)
    daily_metrics.to_csv('/mnt/user-data/outputs/daily_metrics.csv', index=False)
    dashboard.write_html('/mnt/user-data/outputs/segmentation_dashboard.html')
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nOutputs saved:")
    print("  - player_segments.csv: Player ID to segment mapping")
    print("  - player_features.csv: Full feature set for each player")
    print("  - daily_metrics.csv: Daily aggregated metrics by segment")
    print("  - segmentation_dashboard.html: Interactive Plotly dashboard")
    
    return dashboard, segments, daily_metrics


if __name__ == "__main__":
    dashboard, segments, daily_metrics = main()
    
    # Display summary statistics
    print("\n" + "=" * 80)
    print("SEGMENT SUMMARY")
    print("=" * 80)
    
    summary = segments.groupby('player_segment').agg({
        'player_id': 'count',
        'total_wagered': 'sum',
        'total_net_loss': 'sum',
        'avg_bet_amount': 'mean',
        'sessions_per_day': 'mean'
    }).round(2)
    summary.columns = ['Players', 'Total Wagered', 'Total GGR', 'Avg Bet', 'Sessions/Day']
    print(summary)
    
    # Calculate Pareto percentages
    print("\n" + "=" * 80)
    print("PARETO ANALYSIS (Revenue Concentration)")
    print("=" * 80)
    total_ggr = segments['total_net_loss'].sum()
    for segment in ['Whale', 'VIP', 'High', 'Regular', 'Casual']:
        segment_ggr = segments[segments['player_segment'] == segment]['total_net_loss'].sum()
        pct_players = (segments['player_segment'] == segment).sum() / len(segments) * 100
        pct_revenue = segment_ggr / total_ggr * 100
        print(f"{segment:8s}: {pct_players:5.1f}% of players → {pct_revenue:5.1f}% of revenue")
