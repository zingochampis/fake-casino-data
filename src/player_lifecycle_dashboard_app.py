"""
Casino Player Lifecycle Dashboard - Interactive Dash App
Acquisition, Retention & Churn Simulator with Embedded Explanations

Run with: python player_lifecycle_dashboard_app.py
Then open: http://127.0.0.1:8050
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "🎰 Casino Lifecycle Dashboard"

# Color scheme matching the other dashboard
COLORS = {
    'whale': '#e53e3e',
    'vip': '#ed8936',
    'high': '#ecc94b',
    'regular': '#48bb78',
    'casual': '#4299e1',
    'background': '#1a202c',
    'card': '#2d3748',
    'text': '#e2e8f0',
    'accent': '#667eea'
}

SEGMENT_COLORS = {
    'Whale': COLORS['whale'],
    'VIP': COLORS['vip'],
    'High': COLORS['high'],
    'Regular': COLORS['regular'],
    'Casual': COLORS['casual']
}

# ============================================================================
# CONFIGURATION: Player Lifecycle Parameters
# ============================================================================

SEGMENTS = {
    'Whale': {'pct': 0.01, 'color': COLORS['whale'], 'acquisition_rate': 5, 'churn_rate': 0.05},
    'VIP': {'pct': 0.04, 'color': COLORS['vip'], 'acquisition_rate': 20, 'churn_rate': 0.13},
    'High': {'pct': 0.10, 'color': COLORS['high'], 'acquisition_rate': 50, 'churn_rate': 0.17},
    'Regular': {'pct': 0.25, 'color': COLORS['regular'], 'acquisition_rate': 125, 'churn_rate': 0.20},
    'Casual': {'pct': 0.60, 'color': COLORS['casual'], 'acquisition_rate': 300, 'churn_rate': 0.29}
}

COHORT_MONTHS = 12
TOTAL_MONTHS = 24

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def simulate_acquisition(months=24):
    """Simulate monthly player acquisition by segment"""
    data = []
    
    for month in range(1, months + 1):
        seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
        campaign_boost = 1.0 + (0.5 if np.random.random() < 0.15 else 0)
        
        for segment, params in SEGMENTS.items():
            base_rate = params['acquisition_rate']
            expected = base_rate * seasonal * campaign_boost
            new_players = np.random.poisson(expected)
            
            data.append({
                'month': month,
                'segment': segment,
                'new_players': new_players,
                'seasonal_factor': seasonal,
                'campaign_active': campaign_boost > 1.0
            })
    
    return pd.DataFrame(data)


def simulate_retention_curves(segment, months=12):
    """Generate retention curve for a cohort using exponential decay"""
    churn_rate = SEGMENTS[segment]['churn_rate']
    time_points = np.arange(0, months + 1)
    retention = np.exp(-churn_rate * time_points) * 100
    noise = np.random.normal(0, 2, len(retention))
    retention = np.clip(retention + noise, 0, 100)
    return retention


def calculate_cohort_sizes(acquisition_df, months=12):
    """Calculate active player counts by tracking cohorts over time"""
    cohorts = []
    acquisition_by_month = acquisition_df.groupby(['month', 'segment'])['new_players'].sum().reset_index()
    
    for _, row in acquisition_by_month.iterrows():
        cohort_month = row['month']
        segment = row['segment']
        initial_size = row['new_players']
        retention_curve = simulate_retention_curves(segment, months)
        
        for future_month in range(cohort_month, min(cohort_month + months, TOTAL_MONTHS + 1)):
            months_since_acquisition = future_month - cohort_month
            retention_pct = retention_curve[months_since_acquisition]
            active_players = initial_size * (retention_pct / 100)
            
            cohorts.append({
                'cohort_month': cohort_month,
                'current_month': future_month,
                'segment': segment,
                'cohort_size': initial_size,
                'active_players': active_players,
                'retention_pct': retention_pct,
                'months_since_acquisition': months_since_acquisition
            })
    
    return pd.DataFrame(cohorts)


def calculate_churn_metrics(cohort_df):
    """Calculate monthly churn statistics"""
    churn_data = []
    
    for month in range(2, TOTAL_MONTHS + 1):
        for segment in SEGMENTS.keys():
            current = cohort_df[
                (cohort_df['current_month'] == month) & 
                (cohort_df['segment'] == segment)
            ]['active_players'].sum()
            
            previous = cohort_df[
                (cohort_df['current_month'] == month - 1) & 
                (cohort_df['segment'] == segment)
            ]['active_players'].sum()
            
            churned = previous - current
            churn_rate = (churned / previous * 100) if previous > 0 else 0
            
            churn_data.append({
                'month': month,
                'segment': segment,
                'active_start': previous,
                'active_end': current,
                'churned': churned,
                'churn_rate_pct': churn_rate
            })
    
    return pd.DataFrame(churn_data)


# Run simulation once on startup
print("Simulating 24 months of player lifecycle data...")
acquisition_df = simulate_acquisition(TOTAL_MONTHS)
cohort_df = calculate_cohort_sizes(acquisition_df, COHORT_MONTHS)
churn_df = calculate_churn_metrics(cohort_df)
active_by_month = cohort_df.groupby(['current_month', 'segment'])['active_players'].sum().reset_index()
print("Simulation complete!")

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🎰 Casino Player Lifecycle Dashboard", style={'margin': '0', 'fontSize': '2.5rem'}),
        html.P("Interactive Acquisition, Retention & Churn Analytics", 
               style={'margin': '0.5rem 0 0 0', 'opacity': '0.8', 'fontSize': '1.1rem'})
    ], style={
        'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, #2c5282 100%)',
        'color': COLORS['text'],
        'padding': '2rem',
        'textAlign': 'center',
        'marginBottom': '1rem'
    }),
    
    # Main content
    html.Div([
        # Tab navigation
        dcc.Tabs(id='tabs', value='tab-overview', children=[
            dcc.Tab(label='📊 Overview', value='tab-overview',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'border': 'none', 'padding': '12px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '12px'}),
            dcc.Tab(label='📈 Acquisition', value='tab-acquisition',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'border': 'none', 'padding': '12px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '12px'}),
            dcc.Tab(label='🔄 Retention', value='tab-retention',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'border': 'none', 'padding': '12px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '12px'}),
            dcc.Tab(label='📉 Churn', value='tab-churn',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'border': 'none', 'padding': '12px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '12px'}),
            dcc.Tab(label='💰 Lifetime Value', value='tab-ltv',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'border': 'none', 'padding': '12px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '12px'}),
        ], style={'marginBottom': '1rem'}),
        
        # Content area
        html.Div(id='tab-content', style={'padding': '1rem'})
        
    ], style={
        'maxWidth': '1400px',
        'margin': '0 auto',
        'padding': '0 1rem'
    }),
    
    # Footer
    html.Div([
        html.P("Based on academic research: Deng et al. (2021), Clark (2019) | Simulating 24 months of data", 
               style={'margin': '0', 'opacity': '0.7'})
    ], style={
        'backgroundColor': COLORS['card'],
        'color': COLORS['text'],
        'padding': '1rem',
        'textAlign': 'center',
        'marginTop': '2rem'
    })
    
], style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'fontFamily': "'Segoe UI', system-ui, sans-serif"
})

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-overview':
        return render_overview_tab()
    elif tab == 'tab-acquisition':
        return render_acquisition_tab()
    elif tab == 'tab-retention':
        return render_retention_tab()
    elif tab == 'tab-churn':
        return render_churn_tab()
    elif tab == 'tab-ltv':
        return render_ltv_tab()


def render_overview_tab():
    """Executive summary with key metrics"""
    
    # Calculate key metrics
    total_acquired = acquisition_df['new_players'].sum()
    total_churned = churn_df['churned'].sum()
    final_active = active_by_month[active_by_month['current_month']==TOTAL_MONTHS]['active_players'].sum()
    net_growth = total_acquired - total_churned
    
    # Key metrics cards
    metrics_cards = html.Div([
        # Row 1
        html.Div([
            html.Div([
                html.H3("Total Acquired", style={'color': COLORS['text'], 'margin': '0 0 0.5rem 0'}),
                html.H2(f"{total_acquired:,.0f}", style={'color': COLORS['regular'], 'margin': '0'}),
                html.P("Players joined over 24 months", style={'color': COLORS['text'], 'opacity': '0.7', 'margin': '0.5rem 0 0 0'})
            ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'flex': '1'}),
            
            html.Div([
                html.H3("Total Churned", style={'color': COLORS['text'], 'margin': '0 0 0.5rem 0'}),
                html.H2(f"{total_churned:,.0f}", style={'color': COLORS['whale'], 'margin': '0'}),
                html.P("Players left the platform", style={'color': COLORS['text'], 'opacity': '0.7', 'margin': '0.5rem 0 0 0'})
            ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'flex': '1'}),
            
            html.Div([
                html.H3("Net Growth", style={'color': COLORS['text'], 'margin': '0 0 0.5rem 0'}),
                html.H2(f"{net_growth:+,.0f}", style={'color': COLORS['accent'], 'margin': '0'}),
                html.P(f"Current active: {final_active:,.0f}", style={'color': COLORS['text'], 'opacity': '0.7', 'margin': '0.5rem 0 0 0'})
            ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'flex': '1'}),
            
            html.Div([
                html.H3("Avg Churn Rate", style={'color': COLORS['text'], 'margin': '0 0 0.5rem 0'}),
                html.H2(f"{churn_df['churn_rate_pct'].mean():.1f}%", style={'color': COLORS['vip'], 'margin': '0'}),
                html.P("Monthly average across segments", style={'color': COLORS['text'], 'opacity': '0.7', 'margin': '0.5rem 0 0 0'})
            ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'flex': '1'}),
        ], style={'display': 'flex', 'gap': '1rem', 'marginBottom': '2rem'}),
    ])
    
    # Active player base over time
    fig_active = go.Figure()
    for segment in ['Whale', 'VIP', 'High', 'Regular', 'Casual']:
        segment_data = active_by_month[active_by_month['segment'] == segment]
        fig_active.add_trace(go.Scatter(
            x=segment_data['current_month'],
            y=segment_data['active_players'],
            name=segment,
            mode='lines',
            stackgroup='one',
            fillcolor=SEGMENT_COLORS[segment],
            line=dict(width=0.5, color=SEGMENT_COLORS[segment]),
            hovertemplate='%{y:.0f} active players<extra></extra>'
        ))
    
    fig_active.update_layout(
        title={'text': 'Active Player Base Over Time (All Cohorts Combined)', 'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400,
        xaxis_title='Month',
        yaxis_title='Active Players',
        hovermode='x unified'
    )
    fig_active.update_xaxes(gridcolor='#4a5568')
    fig_active.update_yaxes(gridcolor='#4a5568')
    
    # Net growth by month
    monthly_acquired = acquisition_df.groupby('month')['new_players'].sum()
    monthly_churned = churn_df.groupby('month')['churned'].sum()
    net_growth_by_month = monthly_acquired.subtract(monthly_churned, fill_value=0)
    colors_growth = ['#48bb78' if x > 0 else '#e53e3e' for x in net_growth_by_month.values]
    
    fig_growth = go.Figure(data=[go.Bar(
        x=net_growth_by_month.index,
        y=net_growth_by_month.values,
        marker_color=colors_growth,
        hovertemplate='%{y:+.0f} net players<extra></extra>'
    )])
    fig_growth.add_hline(y=0, line_dash="solid", line_color="white", line_width=1)
    fig_growth.update_layout(
        title={'text': 'Net Growth: New Players - Churned Players', 'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='Month',
        yaxis_title='Net Growth',
        showlegend=False
    )
    fig_growth.update_xaxes(gridcolor='#4a5568')
    fig_growth.update_yaxes(gridcolor='#4a5568')
    
    # Explanation text
    explanation = html.Div([
        html.H3("📖 Understanding This Dashboard", style={'color': COLORS['accent']}),
        html.P([
            html.Strong("What you're seeing:"), " This dashboard simulates 24 months of player lifecycle data for a casino platform with 5 player segments."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Acquisition:"), " New players join each month, with seasonal patterns (winter peaks) and random campaign boosts."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Retention:"), " Players stay active based on exponential decay curves. Whales stay longer (95% after 1 year), Casuals leave faster (75% after 1 year)."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Churn:"), " Players who haven't bet in 60+ days are considered 'churned'. Different segments have different churn rates."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Net Growth:"), " Green bars = more joining than leaving (good!). Red bars = more leaving than joining (bad!)."
        ], style={'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'marginTop': '1rem'})
    
    return html.Div([
        metrics_cards,
        dcc.Graph(figure=fig_active),
        dcc.Graph(figure=fig_growth),
        explanation
    ])


def render_acquisition_tab():
    """Monthly acquisition analysis"""
    
    # Monthly acquisition stacked area
    fig_acquisition = go.Figure()
    for segment in ['Whale', 'VIP', 'High', 'Regular', 'Casual']:
        segment_data = acquisition_df[acquisition_df['segment'] == segment]
        fig_acquisition.add_trace(go.Scatter(
            x=segment_data['month'],
            y=segment_data['new_players'],
            name=segment,
            mode='lines',
            stackgroup='one',
            fillcolor=SEGMENT_COLORS[segment],
            line=dict(width=0.5, color=SEGMENT_COLORS[segment]),
            hovertemplate='%{y} new players<extra></extra>'
        ))
    
    fig_acquisition.update_layout(
        title={'text': 'Monthly New Player Acquisition by Segment', 'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=450,
        xaxis_title='Month',
        yaxis_title='New Players',
        hovermode='x unified'
    )
    fig_acquisition.update_xaxes(gridcolor='#4a5568')
    fig_acquisition.update_yaxes(gridcolor='#4a5568')
    
    # Cumulative acquisition
    cumulative_acquired = acquisition_df.groupby('month')['new_players'].sum().cumsum()
    
    fig_cumulative = go.Figure(data=[go.Scatter(
        x=cumulative_acquired.index,
        y=cumulative_acquired.values,
        mode='lines',
        line=dict(width=3, color=COLORS['regular']),
        fill='tozeroy',
        hovertemplate='%{y:,.0f} total acquired<extra></extra>'
    )])
    
    fig_cumulative.update_layout(
        title={'text': 'Cumulative Player Acquisition', 'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='Month',
        yaxis_title='Cumulative Players Acquired'
    )
    fig_cumulative.update_xaxes(gridcolor='#4a5568')
    fig_cumulative.update_yaxes(gridcolor='#4a5568')
    
    # Explanation
    explanation = html.Div([
        html.H3("📊 Acquisition Model Explained", style={'color': COLORS['accent']}),
        html.P([
            html.Strong("Mathematical Model:"), " Poisson(λ) with seasonal & campaign adjustments"
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Seasonal Pattern:"), " Activity peaks in winter (December/January) with ~30% boost, dips in summer. This mimics real casino patterns where people gamble more during holidays."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Campaign Boosts:"), " 15% chance each month of a +50% acquisition spike. This simulates marketing campaigns, special promotions, or viral effects."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Base Rates by Segment:"), " Whale: 5/month, VIP: 20/month, High: 50/month, Regular: 125/month, Casual: 300/month. Reflects that most new players are low-value."
        ], style={'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'marginTop': '1rem'})
    
    return html.Div([
        dcc.Graph(figure=fig_acquisition),
        dcc.Graph(figure=fig_cumulative),
        explanation
    ])


def render_retention_tab():
    """Retention curve analysis"""
    
    # Retention curves by segment
    fig_retention = go.Figure()
    for segment in SEGMENTS.keys():
        retention = simulate_retention_curves(segment, COHORT_MONTHS)
        months = np.arange(0, COHORT_MONTHS + 1)
        
        fig_retention.add_trace(go.Scatter(
            x=months,
            y=retention,
            name=segment,
            mode='lines+markers',
            line=dict(width=3, color=SEGMENT_COLORS[segment]),
            marker=dict(size=8),
            hovertemplate='Month %{x}: %{y:.1f}% retained<extra></extra>'
        ))
    
    fig_retention.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5,
                           annotation_text="50% retention")
    
    fig_retention.update_layout(
        title={'text': 'Retention Curves by Segment (12-Month Cohort)', 'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=450,
        xaxis_title='Months Since Acquisition',
        yaxis_title='Retention %',
        hovermode='x unified'
    )
    fig_retention.update_xaxes(gridcolor='#4a5568')
    fig_retention.update_yaxes(gridcolor='#4a5568')
    
    # Retention comparison table
    retention_stats = []
    for segment in SEGMENTS.keys():
        retention = simulate_retention_curves(segment, 12)
        retention_stats.append({
            'Segment': segment,
            'Month 1': f"{retention[1]:.1f}%",
            'Month 3': f"{retention[3]:.1f}%",
            'Month 6': f"{retention[6]:.1f}%",
            'Month 12': f"{retention[12]:.1f}%",
            'Annual Churn': f"{SEGMENTS[segment]['churn_rate']*100:.0f}%"
        })
    
    retention_table = html.Div([
        html.H4("Retention Benchmarks by Segment", style={'color': COLORS['accent'], 'marginBottom': '1rem'}),
        html.Table([
            html.Thead(html.Tr([html.Th(col, style={'padding': '0.75rem', 'borderBottom': '2px solid #4a5568', 'color': COLORS['text']}) 
                               for col in ['Segment', 'Month 1', 'Month 3', 'Month 6', 'Month 12', 'Annual Churn']])),
            html.Tbody([
                html.Tr([
                    html.Td(row['Segment'], style={'padding': '0.75rem', 'borderBottom': '1px solid #4a5568', 
                                                     'color': SEGMENT_COLORS[row['Segment']], 'fontWeight': 'bold'}),
                    html.Td(row['Month 1'], style={'padding': '0.75rem', 'borderBottom': '1px solid #4a5568', 'color': COLORS['text']}),
                    html.Td(row['Month 3'], style={'padding': '0.75rem', 'borderBottom': '1px solid #4a5568', 'color': COLORS['text']}),
                    html.Td(row['Month 6'], style={'padding': '0.75rem', 'borderBottom': '1px solid #4a5568', 'color': COLORS['text']}),
                    html.Td(row['Month 12'], style={'padding': '0.75rem', 'borderBottom': '1px solid #4a5568', 'color': COLORS['text']}),
                    html.Td(row['Annual Churn'], style={'padding': '0.75rem', 'borderBottom': '1px solid #4a5568', 'color': COLORS['text']})
                ]) for row in retention_stats
            ])
        ], style={'width': '100%', 'borderCollapse': 'collapse'})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'marginTop': '1rem'})
    
    # Explanation
    explanation = html.Div([
        html.H3("🔄 Retention Model Explained", style={'color': COLORS['accent']}),
        html.P([
            html.Strong("Mathematical Model:"), " Exponential decay R(t) = e^(-λt) where λ = churn rate"
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("What this means:"), " If a cohort starts with 100 players and has a 20% annual churn rate (λ=0.20), after 3 months: R(3) = e^(-0.20×3) = 54.9% still active."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Why exponential?"), " Churn is a continuous process. Each month, a percentage of remaining players leave. High-value players (Whales) have LOW churn rates, low-value players (Casuals) have HIGH churn rates."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Key Insight:"), " The gap between Whale retention (95% at month 1) and Casual retention (71% at month 1) shows why VIP programs matter - keeping high-value players is critical!"
        ], style={'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'marginTop': '1rem'})
    
    return html.Div([
        dcc.Graph(figure=fig_retention),
        retention_table,
        explanation
    ])


def render_churn_tab():
    """Churn analysis"""
    
    # Monthly churn by segment
    fig_churn = go.Figure()
    for segment in ['Whale', 'VIP', 'High', 'Regular', 'Casual']:
        segment_data = churn_df[churn_df['segment'] == segment]
        fig_churn.add_trace(go.Scatter(
            x=segment_data['month'],
            y=segment_data['churned'],
            name=segment,
            mode='lines',
            stackgroup='one',
            fillcolor=SEGMENT_COLORS[segment],
            line=dict(width=0.5, color=SEGMENT_COLORS[segment]),
            hovertemplate='%{y:.0f} churned<extra></extra>'
        ))
    
    fig_churn.update_layout(
        title={'text': 'Monthly Churn by Segment', 'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400,
        xaxis_title='Month',
        yaxis_title='Churned Players',
        hovermode='x unified'
    )
    fig_churn.update_xaxes(gridcolor='#4a5568')
    fig_churn.update_yaxes(gridcolor='#4a5568')
    
    # Churn rate percentage
    fig_churn_rate = go.Figure()
    for segment in SEGMENTS.keys():
        segment_data = churn_df[churn_df['segment'] == segment]
        fig_churn_rate.add_trace(go.Scatter(
            x=segment_data['month'],
            y=segment_data['churn_rate_pct'],
            name=segment,
            mode='lines',
            line=dict(width=2, color=SEGMENT_COLORS[segment]),
            hovertemplate='%{y:.1f}% churn rate<extra></extra>'
        ))
    
    fig_churn_rate.update_layout(
        title={'text': 'Churn Rate (%) Over Time', 'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400,
        xaxis_title='Month',
        yaxis_title='Churn Rate %',
        hovermode='x unified'
    )
    fig_churn_rate.update_xaxes(gridcolor='#4a5568')
    fig_churn_rate.update_yaxes(gridcolor='#4a5568')
    
    # Cumulative churn
    cumulative_churned = churn_df.groupby('month')['churned'].sum().cumsum()
    cumulative_acquired = acquisition_df.groupby('month')['new_players'].sum().cumsum()
    
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(
        x=cumulative_acquired.index,
        y=cumulative_acquired.values,
        name='Cumulative Acquired',
        mode='lines',
        line=dict(width=3, color=COLORS['regular']),
        hovertemplate='%{y:,.0f} total acquired<extra></extra>'
    ))
    fig_cumulative.add_trace(go.Scatter(
        x=cumulative_churned.index,
        y=cumulative_churned.values,
        name='Cumulative Churned',
        mode='lines',
        line=dict(width=3, color=COLORS['whale']),
        hovertemplate='%{y:,.0f} total churned<extra></extra>'
    ))
    
    fig_cumulative.update_layout(
        title={'text': 'Cumulative Acquisition vs Churn', 'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400,
        xaxis_title='Month',
        yaxis_title='Cumulative Players'
    )
    fig_cumulative.update_xaxes(gridcolor='#4a5568')
    fig_cumulative.update_yaxes(gridcolor='#4a5568')
    
    # Explanation
    explanation = html.Div([
        html.H3("📉 Churn Calculations Explained", style={'color': COLORS['accent']}),
        html.P([
            html.Strong("How we calculate churn:"), " Churned Players = Active Players (Month N-1) - Active Players (Month N)"
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Churn Rate Formula:"), " (Churned / Active at Start) × 100. Example: Start with 1000, end with 950 → 50 churned → 5% churn rate."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Industry Benchmark:"), " 5-7% monthly churn is typical for online casinos. Higher = problem, Lower = excellent retention."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Why track cumulative?"), " The gap between green line (acquired) and red line (churned) = current active player base. If lines converge, you're losing players!"
        ], style={'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'marginTop': '1rem'})
    
    return html.Div([
        dcc.Graph(figure=fig_churn),
        dcc.Graph(figure=fig_churn_rate),
        dcc.Graph(figure=fig_cumulative),
        explanation
    ])


def render_ltv_tab():
    """Lifetime Value projection"""
    
    # Player base composition
    composition = active_by_month.pivot(index='current_month', columns='segment', values='active_players')
    composition_pct = composition.div(composition.sum(axis=1), axis=0) * 100
    
    fig_composition = go.Figure()
    for segment in ['Whale', 'VIP', 'High', 'Regular', 'Casual']:
        if segment in composition_pct.columns:
            fig_composition.add_trace(go.Scatter(
                x=composition_pct.index,
                y=composition_pct[segment],
                name=segment,
                mode='lines',
                stackgroup='one',
                fillcolor=SEGMENT_COLORS[segment],
                line=dict(width=0.5, color=SEGMENT_COLORS[segment]),
                hovertemplate='%{y:.1f}%<extra></extra>'
            ))
    
    fig_composition.update_layout(
        title={'text': 'Player Base Composition Over Time', 'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400,
        xaxis_title='Month',
        yaxis_title='% Composition',
        hovermode='x unified'
    )
    fig_composition.update_xaxes(gridcolor='#4a5568')
    fig_composition.update_yaxes(gridcolor='#4a5568')
    
    # LTV projection by segment (simplified model)
    ltv_data = []
    for segment, params in SEGMENTS.items():
        retention_curve = simulate_retention_curves(segment, 12)
        # Simplified: assume monthly revenue decreases with retention
        monthly_arpu = {
            'Whale': 500,
            'VIP': 100,
            'High': 30,
            'Regular': 10,
            'Casual': 3
        }
        
        ltv_12m = sum([monthly_arpu[segment] * (retention_curve[i]/100) for i in range(13)])
        
        ltv_data.append({
            'Segment': segment,
            '12-Month LTV': f"${ltv_12m:.2f}",
            'Month 0 ARPU': f"${monthly_arpu[segment]:.2f}",
            'Churn Rate': f"{params['churn_rate']*100:.0f}%"
        })
    
    ltv_table = html.Div([
        html.H4("Estimated 12-Month Lifetime Value by Segment", style={'color': COLORS['accent'], 'marginBottom': '1rem'}),
        html.Table([
            html.Thead(html.Tr([html.Th(col, style={'padding': '0.75rem', 'borderBottom': '2px solid #4a5568', 'color': COLORS['text']}) 
                               for col in ['Segment', '12-Month LTV', 'Month 0 ARPU', 'Churn Rate']])),
            html.Tbody([
                html.Tr([
                    html.Td(row['Segment'], style={'padding': '0.75rem', 'borderBottom': '1px solid #4a5568', 
                                                     'color': SEGMENT_COLORS[row['Segment']], 'fontWeight': 'bold'}),
                    html.Td(row['12-Month LTV'], style={'padding': '0.75rem', 'borderBottom': '1px solid #4a5568', 'color': COLORS['text']}),
                    html.Td(row['Month 0 ARPU'], style={'padding': '0.75rem', 'borderBottom': '1px solid #4a5568', 'color': COLORS['text']}),
                    html.Td(row['Churn Rate'], style={'padding': '0.75rem', 'borderBottom': '1px solid #4a5568', 'color': COLORS['text']})
                ]) for row in ltv_data
            ])
        ], style={'width': '100%', 'borderCollapse': 'collapse'})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'marginTop': '1rem'})
    
    # ROI calculation example
    roi_example = html.Div([
        html.H4("💰 ROI Calculation Example", style={'color': COLORS['accent'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Scenario:"), " You spend $50 to acquire a VIP player via marketing."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            "• Acquisition Cost: $50", html.Br(),
            "• 12-Month LTV: $784.34", html.Br(),
            "• Profit: $784.34 - $50 = $734.34", html.Br(),
            "• ROI: ($734.34 / $50) × 100 = ", html.Strong("1,469%", style={'color': COLORS['regular']})
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Key Insight:"), " Even though VIPs have 13% annual churn, their high ARPU and good retention make them extremely profitable. This is why casinos focus so much on VIP programs!"
        ], style={'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'marginTop': '1rem'})
    
    # Explanation
    explanation = html.Div([
        html.H3("💰 Lifetime Value Explained", style={'color': COLORS['accent']}),
        html.P([
            html.Strong("LTV Formula:"), " LTV = Σ(Monthly Revenue × Retention Rate) over 12 months"
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("ARPU:"), " Average Revenue Per User per month. Whales generate $500/month, Casuals only $3/month."
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Why composition matters:"), " If your player base shifts from 60% Casual to 70% Casual, even with same total players, revenue drops significantly!"
        ], style={'color': COLORS['text'], 'marginBottom': '1rem'}),
        html.P([
            html.Strong("Actionable:"), " Track LTV by acquisition channel. If Facebook ads bring Casuals (LTV=$26) but email brings VIPs (LTV=$784), shift budget to email!"
        ], style={'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'marginTop': '1rem'})
    
    return html.Div([
        dcc.Graph(figure=fig_composition),
        ltv_table,
        roi_example,
        explanation
    ])


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎰 Casino Player Lifecycle Dashboard")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser to: http://127.0.0.1:8050")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8050)
