"""
Casino Data Analytics Dashboard
Interactive Plotly Dash app for exploring aggregated gaming data and transaction patterns.

Run with: python app2_analytics_dashboard.py
Then open: http://127.0.0.1:8051

Required data files in same directory:
- main_aggregation.csv
- transactions_2024-01.csv
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "🎰 Casino Analytics Dashboard"

# Color scheme
COLORS = {
    'whale': '#e53e3e',
    'vip': '#ed8936', 
    'high': '#ecc94b',
    'regular': '#48bb78',
    'casual': '#4299e1',
    'background': '#0f1419',
    'card': '#1a1f2e',
    'card_hover': '#252d3d',
    'text': '#e2e8f0',
    'text_muted': '#8892a0',
    'accent': '#667eea',
    'accent2': '#764ba2',
    'success': '#38a169',
    'warning': '#d69e2e',
    'danger': '#e53e3e',
    'slot': '#9f7aea',
    'table': '#38b2ac'
}

GAME_COLORS = {
    'G001': '#9f7aea',  # SLOT_FORTUNE
    'G002': '#38b2ac',  # BLACKJACK_LIVE
    'G003': '#f6ad55',  # ROULETTE_EURO
    'G004': '#fc8181'   # SLOT_CLASSIC
}

GAME_NAMES = {
    'G001': 'SLOT_FORTUNE',
    'G002': 'BLACKJACK_LIVE', 
    'G003': 'ROULETTE_EURO',
    'G004': 'SLOT_CLASSIC'
}

# ============================================
# DATA LOADING
# ============================================

def load_data():
    """Load the CSV files"""
    # Try multiple possible locations
    possible_paths = [
        ('main_aggregation.csv', 'transactions_2024-01.csv'),
        (f'C:\\Users\\DmitriApassov\\df\\Desktop\\POC\\main_aggregation.csv', f'C:\\Users\\DmitriApassov\\df\\Desktop\\POC\\transactions_2024-01.csv'),
    ]
    
    agg_df = None
    trans_df = None
    
    for agg_path, trans_path in possible_paths:
        try:
            if os.path.exists(agg_path):
                agg_df = pd.read_csv(agg_path)
                agg_df['date'] = pd.to_datetime(agg_df['date'])
                print(f"✓ Loaded aggregation data: {len(agg_df)} rows")
            if os.path.exists(trans_path):
                trans_df = pd.read_csv(trans_path)
                trans_df['timestamp'] = pd.to_datetime(trans_df['timestamp'])
                trans_df['date'] = pd.to_datetime(trans_df['date'])
                print(f"✓ Loaded transaction data: {len(trans_df)} rows")
            if agg_df is not None:
                break
        except Exception as e:
            print(f"Warning: {e}")
            continue
    
    # Generate sample data if files not found
    if agg_df is None:
        print("⚠ Creating sample aggregation data...")
        agg_df = generate_sample_aggregation()
    if trans_df is None:
        print("⚠ Creating sample transaction data...")
        trans_df = generate_sample_transactions()
    
    return agg_df, trans_df

def generate_sample_aggregation():
    """Generate sample data if CSV not found"""
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    games = ['G001', 'G002', 'G003', 'G004']
    
    data = []
    for date in dates:
        for game in games:
            data.append({
                'date': date,
                'game_id': game,
                'daily_active_users': np.random.randint(50, 500),
                'daily_ggr': np.random.lognormal(10, 1.5),
                'total_bets': np.random.randint(500, 5000),
                'total_wagered': np.random.lognormal(12, 1),
                'hold_percentage': np.random.uniform(-20, 60),
                'avg_bet_size': np.random.lognormal(2, 1),
                'game_type': 'slot' if game in ['G001', 'G004'] else 'table',
                'theoretical_rtp': 0.96 if game == 'G001' else 0.995 if game == 'G002' else 0.973 if game == 'G003' else 0.94,
                'volatility': 'high' if game == 'G001' else 'low' if game == 'G002' else 'medium'
            })
    
    return pd.DataFrame(data)

def generate_sample_transactions():
    """Generate sample transaction data if CSV not found"""
    np.random.seed(42)
    n = 10000
    
    data = {
        'transaction_id': [f'T{i:08d}' for i in range(n)],
        'player_id': [f'P{np.random.randint(1, 500):06d}' for _ in range(n)],
        'game_id': np.random.choice(['G001', 'G002', 'G003', 'G004'], n, p=[0.4, 0.2, 0.15, 0.25]),
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='2min'),
        'bet_amount': np.random.lognormal(2, 1.5, n),
        'win_amount': np.random.lognormal(2, 2, n) * np.random.binomial(1, 0.3, n),
        'hour': np.random.choice(range(24), n, p=[0.02]*10 + [0.04]*4 + [0.06]*4 + [0.08]*4 + [0.04]*2),
        'day_of_week': np.random.choice(range(7), n, p=[0.10, 0.10, 0.12, 0.13, 0.20, 0.20, 0.15]),
        'consecutive_losses': np.random.geometric(0.4, n) - 1,
        'is_weekend': np.random.binomial(1, 0.35, n),
        'is_peak_hour': np.random.binomial(1, 0.25, n)
    }
    
    df = pd.DataFrame(data)
    df['net_result'] = df['win_amount'] - df['bet_amount']
    df['date'] = df['timestamp'].dt.date
    
    return df

# Load data
print("\n" + "="*60)
print("Loading data...")
print("="*60)
AGG_DF, TRANS_DF = load_data()

# ============================================
# HELPER FUNCTIONS
# ============================================

def format_currency(value):
    """Format number as currency"""
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.2f}"

def format_number(value):
    """Format large numbers"""
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:.0f}"

def create_kpi_card(title, value, subtitle="", icon="📊", color=COLORS['accent']):
    """Create a styled KPI card"""
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize': '2rem'}),
            html.Div([
                html.P(title, style={'margin': '0', 'fontSize': '0.85rem', 'opacity': '0.8'}),
                html.H3(value, style={'margin': '0.25rem 0', 'fontSize': '1.6rem', 'fontWeight': '700'}),
                html.P(subtitle, style={'margin': '0', 'fontSize': '0.75rem', 'opacity': '0.6'})
            ], style={'marginLeft': '1rem'})
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={
        'backgroundColor': COLORS['card'],
        'padding': '1.25rem',
        'borderRadius': '12px',
        'borderLeft': f'4px solid {color}',
        'color': COLORS['text'],
        'flex': '1',
        'minWidth': '200px'
    })

# ============================================
# LAYOUT
# ============================================

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("🎰 Casino Analytics Dashboard", 
                    style={'margin': '0', 'fontSize': '2rem', 'fontWeight': '700'}),
            html.P("Real-time gaming performance metrics and player behavior analysis",
                   style={'margin': '0.5rem 0 0 0', 'opacity': '0.8'})
        ]),
        html.Div([
            html.Label("Date Range:", style={'marginRight': '0.5rem', 'opacity': '0.8'}),
            dcc.DatePickerRange(
                id='date-range',
                start_date=AGG_DF['date'].min(),
                end_date=AGG_DF['date'].max(),
                display_format='MMM D, YYYY',
                style={'fontSize': '0.9rem'}
            )
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={
        'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, #1a365d 100%)',
        'color': COLORS['text'],
        'padding': '1.5rem 2rem',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'flexWrap': 'wrap',
        'gap': '1rem'
    }),
    
    # Main content
    html.Div([
        # KPI Cards
        html.Div(id='kpi-cards', style={
            'display': 'flex',
            'gap': '1rem',
            'marginBottom': '1.5rem',
            'flexWrap': 'wrap'
        }),
        
        # Tabs
        dcc.Tabs(id='main-tabs', value='tab-overview', children=[
            dcc.Tab(label='📈 Overview', value='tab-overview',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 
                          'border': 'none', 'padding': '12px 20px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 
                                  'border': 'none', 'padding': '12px 20px', 'borderRadius': '8px 8px 0 0'}),
            dcc.Tab(label='🎮 Game Analysis', value='tab-games',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 
                          'border': 'none', 'padding': '12px 20px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 
                                  'border': 'none', 'padding': '12px 20px', 'borderRadius': '8px 8px 0 0'}),
            dcc.Tab(label='👥 Player Behavior', value='tab-players',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 
                          'border': 'none', 'padding': '12px 20px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 
                                  'border': 'none', 'padding': '12px 20px', 'borderRadius': '8px 8px 0 0'}),
            dcc.Tab(label='⏰ Temporal Patterns', value='tab-temporal',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 
                          'border': 'none', 'padding': '12px 20px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 
                                  'border': 'none', 'padding': '12px 20px', 'borderRadius': '8px 8px 0 0'}),
            dcc.Tab(label='📊 Distributions', value='tab-distributions',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 
                          'border': 'none', 'padding': '12px 20px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 
                                  'border': 'none', 'padding': '12px 20px', 'borderRadius': '8px 8px 0 0'}),
        ], style={'marginBottom': '0'}),
        
        # Tab content
        html.Div(id='tab-content', style={
            'backgroundColor': COLORS['card'],
            'borderRadius': '0 12px 12px 12px',
            'padding': '1.5rem',
            'minHeight': '500px'
        })
        
    ], style={
        'maxWidth': '1600px',
        'margin': '0 auto',
        'padding': '1.5rem'
    }),
    
    # Footer
    html.Div([
        html.P("Casino Gaming Data Generator - Analytics Dashboard | Data generated using research-based statistical models",
               style={'margin': '0', 'opacity': '0.7', 'fontSize': '0.85rem'})
    ], style={
        'backgroundColor': COLORS['card'],
        'color': COLORS['text'],
        'padding': '1rem 2rem',
        'textAlign': 'center',
        'marginTop': '1rem'
    })
    
], style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'fontFamily': "'Segoe UI', system-ui, -apple-system, sans-serif"
})

# ============================================
# CALLBACKS
# ============================================

@app.callback(
    [Output('kpi-cards', 'children'),
     Output('tab-content', 'children')],
    [Input('main-tabs', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_dashboard(tab, start_date, end_date):
    # Filter data by date range
    mask = (AGG_DF['date'] >= start_date) & (AGG_DF['date'] <= end_date)
    filtered_agg = AGG_DF[mask]
    
    # Calculate KPIs
    total_ggr = -filtered_agg['gross_gaming_revenue'].sum() if 'gross_gaming_revenue' in filtered_agg.columns else filtered_agg['daily_ggr'].sum()
    total_wagered = filtered_agg['total_wagered'].sum()
    total_bets = filtered_agg['total_bets'].sum()
    avg_hold = filtered_agg['hold_percentage'].mean()
    unique_days = filtered_agg['date'].nunique()
    avg_daily_users = filtered_agg['daily_active_users'].mean() if 'daily_active_users' in filtered_agg.columns else 100
    
    # KPI cards
    kpi_cards = [
        create_kpi_card("Gross Gaming Revenue", format_currency(total_ggr), 
                       f"{unique_days} days", "💰", COLORS['success']),
        create_kpi_card("Total Wagered", format_currency(total_wagered),
                       f"Avg: {format_currency(total_wagered/max(unique_days,1))}/day", "🎲", COLORS['accent']),
        create_kpi_card("Total Bets", format_number(total_bets),
                       f"Avg: {format_number(total_bets/max(unique_days,1))}/day", "📊", COLORS['warning']),
        create_kpi_card("Avg Hold %", f"{avg_hold:.1f}%",
                       "House edge realized", "📈", COLORS['danger'] if avg_hold < 0 else COLORS['success']),
        create_kpi_card("Daily Active Users", f"{avg_daily_users:.0f}",
                       "Average per day", "👥", COLORS['vip'])
    ]
    
    # Generate tab content
    if tab == 'tab-overview':
        content = render_overview_tab(filtered_agg)
    elif tab == 'tab-games':
        content = render_games_tab(filtered_agg)
    elif tab == 'tab-players':
        content = render_players_tab(TRANS_DF)
    elif tab == 'tab-temporal':
        content = render_temporal_tab(TRANS_DF, filtered_agg)
    elif tab == 'tab-distributions':
        content = render_distributions_tab(TRANS_DF)
    else:
        content = html.P("Select a tab")
    
    return kpi_cards, content

def render_overview_tab(df):
    """Overview dashboard"""
    # Daily GGR time series
    daily_ggr = df.groupby('date').agg({
        'gross_gaming_revenue': 'sum' if 'gross_gaming_revenue' in df.columns else 'daily_ggr',
        'total_wagered': 'sum',
        'total_bets': 'sum',
        'daily_active_users': 'sum' if 'daily_active_users' in df.columns else lambda x: 100
    }).reset_index()
    
    if 'gross_gaming_revenue' in df.columns:
        daily_ggr['ggr'] = -daily_ggr['gross_gaming_revenue']
    else:
        daily_ggr['ggr'] = daily_ggr.get('daily_ggr', np.random.lognormal(10, 1, len(daily_ggr)))
    
    # GGR time series
    fig_ggr = go.Figure()
    fig_ggr.add_trace(go.Scatter(
        x=daily_ggr['date'], y=daily_ggr['ggr'],
        mode='lines+markers',
        name='Daily GGR',
        line=dict(color=COLORS['success'], width=2),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor='rgba(56, 161, 105, 0.2)'
    ))
    
    # Add 7-day moving average
    daily_ggr['ma7'] = daily_ggr['ggr'].rolling(7).mean()
    fig_ggr.add_trace(go.Scatter(
        x=daily_ggr['date'], y=daily_ggr['ma7'],
        mode='lines',
        name='7-Day MA',
        line=dict(color=COLORS['warning'], width=3, dash='dash')
    ))
    
    fig_ggr.update_layout(
        title={'text': '📈 Daily Gross Gaming Revenue (GGR)', 'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='Date',
        yaxis_title='GGR ($)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    fig_ggr.update_xaxes(gridcolor='#2d3748', showgrid=True)
    fig_ggr.update_yaxes(gridcolor='#2d3748', showgrid=True)
    
    # Game breakdown pie
    game_totals = df.groupby('game_id').agg({
        'total_wagered': 'sum'
    }).reset_index()
    game_totals['game_name'] = game_totals['game_id'].map(GAME_NAMES)
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=game_totals['game_name'],
        values=game_totals['total_wagered'],
        hole=0.5,
        marker_colors=[GAME_COLORS[g] for g in game_totals['game_id']],
        textinfo='label+percent',
        textfont_size=11
    )])
    fig_pie.update_layout(
        title={'text': '🎮 Wagering by Game', 'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        showlegend=False
    )
    
    # Hold percentage by game over time
    fig_hold = go.Figure()
    for game_id in df['game_id'].unique():
        game_data = df[df['game_id'] == game_id].groupby('date')['hold_percentage'].mean().reset_index()
        fig_hold.add_trace(go.Scatter(
            x=game_data['date'], y=game_data['hold_percentage'],
            mode='lines',
            name=GAME_NAMES.get(game_id, game_id),
            line=dict(color=GAME_COLORS.get(game_id, COLORS['accent']), width=2)
        ))
    
    # Add theoretical line at 0
    fig_hold.add_hline(y=0, line_dash="dash", line_color=COLORS['text_muted'], opacity=0.5)
    
    fig_hold.update_layout(
        title={'text': '📊 Hold Percentage by Game', 'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='Date',
        yaxis_title='Hold %',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )
    fig_hold.update_xaxes(gridcolor='#2d3748')
    fig_hold.update_yaxes(gridcolor='#2d3748')
    
    return html.Div([
        dcc.Graph(figure=fig_ggr),
        html.Div([
            html.Div([dcc.Graph(figure=fig_pie)], style={'flex': '1'}),
            html.Div([dcc.Graph(figure=fig_hold)], style={'flex': '2'})
        ], style={'display': 'flex', 'gap': '1rem', 'flexWrap': 'wrap'})
    ])

def render_games_tab(df):
    """Game-level analysis"""
    # Game performance comparison
    game_stats = df.groupby('game_id').agg({
        'total_wagered': 'sum',
        'total_bets': 'sum',
        'hold_percentage': 'mean',
        'avg_bet_size': 'mean',
        'daily_active_users': 'mean' if 'daily_active_users' in df.columns else lambda x: 100
    }).reset_index()
    game_stats['game_name'] = game_stats['game_id'].map(GAME_NAMES)
    
    # Bar chart comparison
    fig_compare = make_subplots(rows=2, cols=2,
                                subplot_titles=['Total Wagered', 'Total Bets', 'Avg Hold %', 'Avg Bet Size'])
    
    colors = [GAME_COLORS[g] for g in game_stats['game_id']]
    
    fig_compare.add_trace(go.Bar(x=game_stats['game_name'], y=game_stats['total_wagered'],
                                 marker_color=colors, showlegend=False), row=1, col=1)
    fig_compare.add_trace(go.Bar(x=game_stats['game_name'], y=game_stats['total_bets'],
                                 marker_color=colors, showlegend=False), row=1, col=2)
    fig_compare.add_trace(go.Bar(x=game_stats['game_name'], y=game_stats['hold_percentage'],
                                 marker_color=colors, showlegend=False), row=2, col=1)
    fig_compare.add_trace(go.Bar(x=game_stats['game_name'], y=game_stats['avg_bet_size'],
                                 marker_color=colors, showlegend=False), row=2, col=2)
    
    fig_compare.update_layout(
        title={'text': '🎮 Game Performance Comparison', 'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=500
    )
    fig_compare.update_xaxes(gridcolor='#2d3748')
    fig_compare.update_yaxes(gridcolor='#2d3748')
    
    # Slots vs Tables comparison
    if 'game_type' in df.columns:
        type_stats = df.groupby('game_type').agg({
            'total_wagered': 'sum',
            'total_bets': 'sum',
            'hold_percentage': 'mean'
        }).reset_index()
        
        fig_types = go.Figure()
        fig_types.add_trace(go.Bar(
            x=['Slots', 'Tables'],
            y=[type_stats[type_stats['game_type']=='slot']['total_wagered'].sum(),
               type_stats[type_stats['game_type']=='table']['total_wagered'].sum()],
            marker_color=[COLORS['slot'], COLORS['table']],
            text=['Slots', 'Tables'],
            textposition='inside'
        ))
        fig_types.update_layout(
            title={'text': '🎰 Slots vs 🃏 Tables - Total Wagered', 'font': {'size': 18, 'color': COLORS['text']}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            height=300,
            yaxis_title='Total Wagered ($)'
        )
        fig_types.update_xaxes(gridcolor='#2d3748')
        fig_types.update_yaxes(gridcolor='#2d3748')
    else:
        fig_types = go.Figure()
        fig_types.add_annotation(text="Game type data not available", showarrow=False)
    
    # Game volatility scatter
    fig_scatter = go.Figure()
    for _, row in game_stats.iterrows():
        fig_scatter.add_trace(go.Scatter(
            x=[row['total_bets']],
            y=[row['hold_percentage']],
            mode='markers+text',
            marker=dict(size=row['total_wagered']/game_stats['total_wagered'].max()*80 + 20,
                       color=GAME_COLORS.get(row['game_id'], COLORS['accent']),
                       opacity=0.7),
            text=[row['game_name']],
            textposition='top center',
            name=row['game_name'],
            hovertemplate=f"<b>{row['game_name']}</b><br>Bets: {row['total_bets']:,.0f}<br>Hold: {row['hold_percentage']:.1f}%<extra></extra>"
        ))
    
    fig_scatter.update_layout(
        title={'text': '📊 Game Performance Matrix (size = wagered amount)', 
               'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400,
        xaxis_title='Total Bets',
        yaxis_title='Average Hold %',
        showlegend=False
    )
    fig_scatter.update_xaxes(gridcolor='#2d3748')
    fig_scatter.update_yaxes(gridcolor='#2d3748')
    
    return html.Div([
        dcc.Graph(figure=fig_compare),
        html.Div([
            html.Div([dcc.Graph(figure=fig_types)], style={'flex': '1'}),
            html.Div([dcc.Graph(figure=fig_scatter)], style={'flex': '2'})
        ], style={'display': 'flex', 'gap': '1rem', 'flexWrap': 'wrap'})
    ])

def render_players_tab(df):
    """Player behavior analysis"""
    # Top players by wagering
    player_stats = df.groupby('player_id').agg({
        'bet_amount': ['sum', 'mean', 'count'],
        'net_result': 'sum',
        'win_amount': 'sum'
    }).reset_index()
    player_stats.columns = ['player_id', 'total_wagered', 'avg_bet', 'num_bets', 'net_result', 'total_won']
    player_stats = player_stats.sort_values('total_wagered', ascending=False).head(20)
    
    fig_top_players = go.Figure(data=[go.Bar(
        x=player_stats['player_id'],
        y=player_stats['total_wagered'],
        marker_color=COLORS['accent'],
        text=[f'${v:,.0f}' for v in player_stats['total_wagered']],
        textposition='outside'
    )])
    fig_top_players.update_layout(
        title={'text': '🏆 Top 20 Players by Total Wagered', 'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='Player ID',
        yaxis_title='Total Wagered ($)'
    )
    fig_top_players.update_xaxes(gridcolor='#2d3748', tickangle=45)
    fig_top_players.update_yaxes(gridcolor='#2d3748')
    
    # Loss chasing analysis
    if 'consecutive_losses' in df.columns:
        loss_chase = df.groupby('consecutive_losses').agg({
            'bet_amount': 'mean',
            'transaction_id': 'count'
        }).reset_index()
        loss_chase.columns = ['consecutive_losses', 'avg_bet', 'count']
        loss_chase = loss_chase[loss_chase['consecutive_losses'] <= 10]
        
        fig_chase = make_subplots(specs=[[{"secondary_y": True}]])
        fig_chase.add_trace(go.Bar(
            x=loss_chase['consecutive_losses'],
            y=loss_chase['avg_bet'],
            name='Avg Bet Size',
            marker_color=COLORS['accent']
        ), secondary_y=False)
        fig_chase.add_trace(go.Scatter(
            x=loss_chase['consecutive_losses'],
            y=loss_chase['count'],
            name='Frequency',
            mode='lines+markers',
            line=dict(color=COLORS['warning'], width=2)
        ), secondary_y=True)
        
        fig_chase.update_layout(
            title={'text': '🧠 Loss Chasing Behavior Analysis', 'font': {'size': 18, 'color': COLORS['text']}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            height=350,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        fig_chase.update_xaxes(title_text='Consecutive Losses', gridcolor='#2d3748')
        fig_chase.update_yaxes(title_text='Average Bet ($)', secondary_y=False, gridcolor='#2d3748')
        fig_chase.update_yaxes(title_text='Frequency', secondary_y=True)
    else:
        fig_chase = go.Figure()
        fig_chase.add_annotation(text="Loss chasing data not available", showarrow=False)
    
    # Player value distribution (Pareto check)
    all_players = df.groupby('player_id')['bet_amount'].sum().sort_values(ascending=False).reset_index()
    all_players['cumulative_pct'] = all_players['bet_amount'].cumsum() / all_players['bet_amount'].sum() * 100
    all_players['player_pct'] = (np.arange(len(all_players)) + 1) / len(all_players) * 100
    
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Scatter(
        x=all_players['player_pct'],
        y=all_players['cumulative_pct'],
        mode='lines',
        name='Actual',
        line=dict(color=COLORS['success'], width=3),
        fill='tozeroy',
        fillcolor='rgba(56, 161, 105, 0.2)'
    ))
    # Add 80/20 reference line
    fig_pareto.add_trace(go.Scatter(
        x=[0, 20, 20], y=[0, 0, 80],
        mode='lines',
        name='80/20 Rule',
        line=dict(color=COLORS['danger'], width=2, dash='dash')
    ))
    fig_pareto.add_trace(go.Scatter(
        x=[20, 100], y=[80, 100],
        mode='lines',
        showlegend=False,
        line=dict(color=COLORS['danger'], width=2, dash='dash')
    ))
    
    # Calculate actual 20% contribution
    top_20_pct = all_players[all_players['player_pct'] <= 20]['cumulative_pct'].max()
    
    fig_pareto.update_layout(
        title={'text': f'📊 Pareto Distribution: Top 20% = {top_20_pct:.1f}% of Wagering', 
               'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='% of Players (ranked by value)',
        yaxis_title='Cumulative % of Wagering'
    )
    fig_pareto.update_xaxes(gridcolor='#2d3748')
    fig_pareto.update_yaxes(gridcolor='#2d3748')
    
    return html.Div([
        dcc.Graph(figure=fig_top_players),
        html.Div([
            html.Div([dcc.Graph(figure=fig_chase)], style={'flex': '1'}),
            html.Div([dcc.Graph(figure=fig_pareto)], style={'flex': '1'})
        ], style={'display': 'flex', 'gap': '1rem', 'flexWrap': 'wrap'})
    ])

def render_temporal_tab(trans_df, agg_df):
    """Temporal patterns analysis"""
    # Hour of day distribution
    if 'hour' in trans_df.columns:
        hourly = trans_df.groupby('hour').agg({
            'bet_amount': ['sum', 'count', 'mean']
        }).reset_index()
        hourly.columns = ['hour', 'total_wagered', 'num_bets', 'avg_bet']
        
        fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hourly.add_trace(go.Bar(
            x=hourly['hour'],
            y=hourly['num_bets'],
            name='Number of Bets',
            marker_color=COLORS['accent'],
            opacity=0.7
        ), secondary_y=False)
        fig_hourly.add_trace(go.Scatter(
            x=hourly['hour'],
            y=hourly['avg_bet'],
            name='Avg Bet Size',
            mode='lines+markers',
            line=dict(color=COLORS['warning'], width=3)
        ), secondary_y=True)
        
        fig_hourly.update_layout(
            title={'text': '⏰ Betting Activity by Hour of Day', 'font': {'size': 18, 'color': COLORS['text']}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            height=350,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        fig_hourly.update_xaxes(title_text='Hour (0-23)', gridcolor='#2d3748', dtick=2)
        fig_hourly.update_yaxes(title_text='Number of Bets', secondary_y=False, gridcolor='#2d3748')
        fig_hourly.update_yaxes(title_text='Avg Bet ($)', secondary_y=True)
    else:
        fig_hourly = go.Figure()
    
    # Day of week distribution
    if 'day_of_week' in trans_df.columns:
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily = trans_df.groupby('day_of_week')['bet_amount'].agg(['sum', 'count', 'mean']).reset_index()
        daily.columns = ['day_of_week', 'total_wagered', 'num_bets', 'avg_bet']
        
        colors = [COLORS['text_muted'] if d < 4 else COLORS['warning'] if d == 4 else COLORS['success'] 
                  for d in daily['day_of_week']]
        
        fig_daily = go.Figure(data=[go.Bar(
            x=[days[d] for d in daily['day_of_week']],
            y=daily['total_wagered'],
            marker_color=colors,
            text=[f'{v/1000:.0f}K' for v in daily['total_wagered']],
            textposition='outside'
        )])
        fig_daily.update_layout(
            title={'text': '📅 Wagering by Day of Week', 'font': {'size': 18, 'color': COLORS['text']}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            height=350,
            xaxis_title='Day of Week',
            yaxis_title='Total Wagered ($)'
        )
        fig_daily.update_xaxes(gridcolor='#2d3748')
        fig_daily.update_yaxes(gridcolor='#2d3748')
    else:
        fig_daily = go.Figure()
    
    # Weekend vs Weekday
    if 'is_weekend' in trans_df.columns:
        weekend_stats = trans_df.groupby('is_weekend').agg({
            'bet_amount': ['sum', 'mean', 'count']
        }).reset_index()
        weekend_stats.columns = ['is_weekend', 'total', 'avg', 'count']
        
        fig_weekend = go.Figure(data=[go.Pie(
            labels=['Weekday', 'Weekend'],
            values=weekend_stats['total'],
            hole=0.6,
            marker_colors=[COLORS['text_muted'], COLORS['success']],
            textinfo='label+percent'
        )])
        fig_weekend.update_layout(
            title={'text': '📊 Weekend vs Weekday Wagering', 'font': {'size': 18, 'color': COLORS['text']}},
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            height=300
        )
    else:
        fig_weekend = go.Figure()
    
    # Peak hours analysis
    if 'is_peak_hour' in trans_df.columns:
        peak_stats = trans_df.groupby('is_peak_hour')['bet_amount'].agg(['sum', 'mean']).reset_index()
        peak_stats.columns = ['is_peak', 'total', 'avg']
        
        fig_peak = go.Figure(data=[go.Bar(
            x=['Off-Peak (0-19h)', 'Peak (20-23h)'],
            y=peak_stats['avg'],
            marker_color=[COLORS['text_muted'], COLORS['danger']],
            text=[f'${v:.2f}' for v in peak_stats['avg']],
            textposition='outside'
        )])
        fig_peak.update_layout(
            title={'text': '🌙 Average Bet: Peak vs Off-Peak Hours', 'font': {'size': 18, 'color': COLORS['text']}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            height=300,
            yaxis_title='Average Bet ($)'
        )
        fig_peak.update_xaxes(gridcolor='#2d3748')
        fig_peak.update_yaxes(gridcolor='#2d3748')
    else:
        fig_peak = go.Figure()
    
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=fig_hourly)], style={'flex': '2'}),
            html.Div([dcc.Graph(figure=fig_daily)], style={'flex': '1'})
        ], style={'display': 'flex', 'gap': '1rem', 'flexWrap': 'wrap'}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_weekend)], style={'flex': '1'}),
            html.Div([dcc.Graph(figure=fig_peak)], style={'flex': '1'})
        ], style={'display': 'flex', 'gap': '1rem', 'flexWrap': 'wrap'})
    ])

def render_distributions_tab(df):
    """Statistical distributions from actual data"""
    # Bet amount distribution
    fig_bets = go.Figure()
    bet_data = df['bet_amount'].clip(upper=df['bet_amount'].quantile(0.99))
    
    fig_bets.add_trace(go.Histogram(
        x=bet_data,
        nbinsx=100,
        marker_color=COLORS['accent'],
        opacity=0.7,
        name='Actual'
    ))
    
    # Add log-normal fit line
    mu, sigma = np.log(bet_data).mean(), np.log(bet_data).std()
    x_fit = np.linspace(bet_data.min(), bet_data.max(), 100)
    from scipy import stats
    y_fit = stats.lognorm.pdf(x_fit, sigma, scale=np.exp(mu)) * len(bet_data) * (bet_data.max() - bet_data.min()) / 100
    
    fig_bets.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode='lines',
        name=f'Log-Normal Fit (μ={mu:.2f}, σ={sigma:.2f})',
        line=dict(color=COLORS['warning'], width=3)
    ))
    
    fig_bets.update_layout(
        title={'text': '💰 Bet Amount Distribution with Log-Normal Fit', 
               'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='Bet Amount ($)',
        yaxis_title='Frequency',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    fig_bets.update_xaxes(gridcolor='#2d3748')
    fig_bets.update_yaxes(gridcolor='#2d3748')
    
    # Win multiplier distribution
    wins = df[df['win_amount'] > 0].copy()
    if len(wins) > 0:
        wins['multiplier'] = wins['win_amount'] / wins['bet_amount']
        mult_data = wins['multiplier'].clip(upper=wins['multiplier'].quantile(0.99))
        
        fig_mult = go.Figure()
        fig_mult.add_trace(go.Histogram(
            x=mult_data,
            nbinsx=50,
            marker_color=COLORS['success'],
            opacity=0.7
        ))
        fig_mult.update_layout(
            title={'text': '🎯 Win Multiplier Distribution', 'font': {'size': 18, 'color': COLORS['text']}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            height=350,
            xaxis_title='Win Multiplier (Win/Bet)',
            yaxis_title='Frequency'
        )
        fig_mult.update_xaxes(gridcolor='#2d3748')
        fig_mult.update_yaxes(gridcolor='#2d3748')
    else:
        fig_mult = go.Figure()
    
    # Net result distribution
    fig_net = go.Figure()
    net_data = df['net_result'].clip(lower=df['net_result'].quantile(0.01), 
                                      upper=df['net_result'].quantile(0.99))
    
    fig_net.add_trace(go.Histogram(
        x=net_data,
        nbinsx=100,
        marker_color=[COLORS['danger'] if x < 0 else COLORS['success'] for x in np.linspace(net_data.min(), net_data.max(), 100)],
        opacity=0.7
    ))
    fig_net.add_vline(x=0, line_dash="solid", line_color="white", line_width=2)
    fig_net.add_vline(x=net_data.mean(), line_dash="dash", line_color=COLORS['warning'],
                      annotation_text=f"Mean: ${net_data.mean():.2f}")
    
    fig_net.update_layout(
        title={'text': '📉 Net Result Distribution (Win - Bet)', 
               'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='Net Result ($)',
        yaxis_title='Frequency'
    )
    fig_net.update_xaxes(gridcolor='#2d3748')
    fig_net.update_yaxes(gridcolor='#2d3748')
    
    # Summary stats box
    stats_box = html.Div([
        html.H4("📊 Distribution Summary Statistics", style={'color': COLORS['accent'], 'marginBottom': '1rem'}),
        html.Div([
            html.Div([
                html.P("Bet Amount", style={'fontWeight': 'bold', 'marginBottom': '0.5rem'}),
                html.P(f"Mean: ${df['bet_amount'].mean():.2f}"),
                html.P(f"Median: ${df['bet_amount'].median():.2f}"),
                html.P(f"Std Dev: ${df['bet_amount'].std():.2f}"),
                html.P(f"Skewness: {df['bet_amount'].skew():.2f}")
            ], style={'flex': '1', 'padding': '1rem', 'backgroundColor': COLORS['background'], 'borderRadius': '8px'}),
            html.Div([
                html.P("Win Rate", style={'fontWeight': 'bold', 'marginBottom': '0.5rem'}),
                html.P(f"Hit Rate: {(df['win_amount'] > 0).mean()*100:.1f}%"),
                html.P(f"Avg Win: ${df[df['win_amount'] > 0]['win_amount'].mean():.2f}" if len(df[df['win_amount'] > 0]) > 0 else "N/A"),
                html.P(f"Max Win: ${df['win_amount'].max():.2f}"),
                html.P(f"House Edge: {-df['net_result'].sum()/df['bet_amount'].sum()*100:.2f}%")
            ], style={'flex': '1', 'padding': '1rem', 'backgroundColor': COLORS['background'], 'borderRadius': '8px'}),
            html.Div([
                html.P("Net Result", style={'fontWeight': 'bold', 'marginBottom': '0.5rem'}),
                html.P(f"Mean: ${df['net_result'].mean():.2f}"),
                html.P(f"Total GGR: ${-df['net_result'].sum():,.0f}"),
                html.P(f"Win Sessions: {(df['net_result'] > 0).sum():,}"),
                html.P(f"Loss Sessions: {(df['net_result'] < 0).sum():,}")
            ], style={'flex': '1', 'padding': '1rem', 'backgroundColor': COLORS['background'], 'borderRadius': '8px'})
        ], style={'display': 'flex', 'gap': '1rem', 'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['card_hover'], 'padding': '1rem', 'borderRadius': '8px', 'marginBottom': '1rem'})
    
    return html.Div([
        stats_box,
        html.Div([
            html.Div([dcc.Graph(figure=fig_bets)], style={'flex': '1'}),
            html.Div([dcc.Graph(figure=fig_mult)], style={'flex': '1'})
        ], style={'display': 'flex', 'gap': '1rem', 'flexWrap': 'wrap'}),
        dcc.Graph(figure=fig_net)
    ])

# ============================================
# RUN THE APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎰 Casino Analytics Dashboard")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser to: http://127.0.0.1:8051")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8051)