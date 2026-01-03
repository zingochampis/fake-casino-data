"""
Casino Data Generator - Mathematical Foundations Visualizer
Interactive Plotly Dash app showing the statistical distributions used in data generation.

Run with: python app1_math_visualizer.py
Then open: http://127.0.0.1:8050
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "🎰 Casino Generator - Mathematical Foundations"

# Color scheme
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

SEGMENT_COLORS = [COLORS['whale'], COLORS['vip'], COLORS['high'], COLORS['regular'], COLORS['casual']]

# ============================================
# DISTRIBUTION GENERATION FUNCTIONS
# ============================================

def generate_lognormal_samples(mu, sigma, n=10000):
    """Generate log-normal samples for bet size distributions"""
    return np.random.lognormal(mu, sigma, n)

def generate_pareto_samples(alpha, scale=1, n=10000):
    """Generate Pareto samples for win multipliers"""
    return (np.random.pareto(alpha, n) + 1) * scale

def generate_beta_samples(a, b, n=10000):
    """Generate Beta samples for risk tolerance/churn"""
    return np.random.beta(a, b, n)

def generate_gamma_samples(shape, scale, n=10000):
    """Generate Gamma samples for session frequency"""
    return np.random.gamma(shape, scale, n)

def generate_exponential_samples(scale, n=10000):
    """Generate Exponential samples for inter-bet timing"""
    return np.random.exponential(scale, n)

# ============================================
# LAYOUT
# ============================================

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🎰 Casino Data Generator", style={'margin': '0', 'fontSize': '2.5rem'}),
        html.P("Interactive Mathematical Foundations Visualizer", 
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
        dcc.Tabs(id='tabs', value='tab-segments', children=[
            dcc.Tab(label='📊 Player Segments', value='tab-segments',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'border': 'none', 'padding': '12px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '12px'}),
            dcc.Tab(label='💰 Bet Sizes (Log-Normal)', value='tab-bets',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'border': 'none', 'padding': '12px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '12px'}),
            dcc.Tab(label='🎯 Win Multipliers (Pareto)', value='tab-wins',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'border': 'none', 'padding': '12px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '12px'}),
            dcc.Tab(label='🧠 Risk & Churn (Beta)', value='tab-beta',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'border': 'none', 'padding': '12px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '12px'}),
            dcc.Tab(label='⏱️ Timing (Exponential)', value='tab-timing',
                   style={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'border': 'none', 'padding': '12px'},
                   selected_style={'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '12px'}),
            dcc.Tab(label='📅 Temporal Patterns', value='tab-temporal',
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
        html.P("Based on academic research: Deng et al. (2021), Clark (2019), Omike (2022)", 
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

# ============================================
# CALLBACKS
# ============================================

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-segments':
        return render_segments_tab()
    elif tab == 'tab-bets':
        return render_bets_tab()
    elif tab == 'tab-wins':
        return render_wins_tab()
    elif tab == 'tab-beta':
        return render_beta_tab()
    elif tab == 'tab-timing':
        return render_timing_tab()
    elif tab == 'tab-temporal':
        return render_temporal_tab()

def render_segments_tab():
    """Player segmentation overview"""
    # Segment distribution pie chart
    segments = ['Whale (1%)', 'VIP (4%)', 'High (10%)', 'Regular (25%)', 'Casual (60%)']
    values = [1, 4, 10, 25, 60]
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=segments,
        values=values,
        hole=0.5,
        marker_colors=SEGMENT_COLORS,
        textinfo='label+percent',
        textfont_size=12
    )])
    fig_pie.update_layout(
        title={'text': 'Player Segment Distribution (Pareto-Inspired)', 'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        showlegend=False,
        height=400
    )
    
    # Revenue contribution (estimated)
    fig_revenue = go.Figure(data=[go.Bar(
        x=['Whale', 'VIP', 'High', 'Regular', 'Casual'],
        y=[35, 30, 20, 12, 3],
        marker_color=SEGMENT_COLORS,
        text=['35%', '30%', '20%', '12%', '3%'],
        textposition='outside'
    )])
    fig_revenue.update_layout(
        title={'text': 'Estimated Revenue Contribution by Segment', 'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        yaxis_title='% of Total Revenue',
        xaxis_title='Player Segment',
        height=400
    )
    fig_revenue.update_xaxes(gridcolor='#4a5568')
    fig_revenue.update_yaxes(gridcolor='#4a5568')
    
    # Segment characteristics table
    table_data = pd.DataFrame({
        'Segment': ['Whale', 'VIP', 'High', 'Regular', 'Casual'],
        'Population': ['1% (100)', '4% (400)', '10% (1,000)', '25% (2,500)', '60% (6,000)'],
        'Avg Bet': ['~$245', '~$33', '~$10', '~$2.70', '~$1.00'],
        'Sessions/Week': ['~8', '~5', '~3', '~1', '~0.3'],
        'Session Duration': ['150 min', '90 min', '60 min', '40 min', '20 min'],
        'Annual Churn': ['~5%', '~5%', '~17%', '~20%', '~29%']
    })
    
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=list(table_data.columns),
            fill_color=COLORS['accent'],
            align='center',
            font=dict(color='white', size=14)
        ),
        cells=dict(
            values=[table_data[col] for col in table_data.columns],
            fill_color=[[COLORS['whale'], COLORS['vip'], COLORS['high'], COLORS['regular'], COLORS['casual']]],
            align='center',
            font=dict(color='white', size=12),
            height=35
        )
    )])
    fig_table.update_layout(
        title={'text': 'Segment Characteristics Summary', 'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    return html.Div([
        html.Div([
            html.Div([
                html.H3("📊 The Pareto Principle in Gaming", style={'color': COLORS['text'], 'marginBottom': '1rem'}),
                html.P("Research shows the top 20% of casino players generate 80-92% of revenue. "
                       "This generator uses a 5-segment model to capture this distribution.",
                       style={'color': COLORS['text'], 'opacity': '0.9', 'lineHeight': '1.6'}),
                html.Code("# From Deng et al. (2021): Top 20% = 92% of bets, 90% of net losses",
                         style={'backgroundColor': COLORS['card'], 'padding': '0.5rem', 'borderRadius': '4px',
                                'color': '#68d391', 'display': 'block', 'marginTop': '1rem'})
            ], style={'backgroundColor': COLORS['card'], 'padding': '1.5rem', 'borderRadius': '8px', 'marginBottom': '1rem'})
        ]),
        html.Div([
            html.Div([dcc.Graph(figure=fig_pie)], style={'flex': '1'}),
            html.Div([dcc.Graph(figure=fig_revenue)], style={'flex': '1'})
        ], style={'display': 'flex', 'gap': '1rem', 'flexWrap': 'wrap'}),
        dcc.Graph(figure=fig_table)
    ])

def render_bets_tab():
    """Log-normal bet size distributions"""
    np.random.seed(42)
    
    # Generate samples for each segment
    segments_params = {
        'Whale': (5.5, 1.2),
        'VIP': (3.5, 1.0),
        'High': (2.3, 0.8),
        'Regular': (1.0, 0.6),
        'Casual': (0, 0.5)
    }
    
    fig = make_subplots(rows=2, cols=3, 
                        subplot_titles=['Whale (μ=5.5, σ=1.2)', 'VIP (μ=3.5, σ=1.0)', 
                                       'High (μ=2.3, σ=0.8)', 'Regular (μ=1.0, σ=0.6)',
                                       'Casual (μ=0, σ=0.5)', 'All Segments Combined'],
                        specs=[[{}, {}, {}], [{}, {}, {}]])
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    all_samples = []
    
    for i, (name, (mu, sigma)) in enumerate(segments_params.items()):
        samples = generate_lognormal_samples(mu, sigma, 5000)
        samples = np.clip(samples, 0, 500)  # Clip for visualization
        all_samples.extend(samples)
        
        row, col = positions[i]
        fig.add_trace(
            go.Histogram(x=samples, nbinsx=50, name=name, 
                        marker_color=SEGMENT_COLORS[i], opacity=0.8,
                        showlegend=True),
            row=row, col=col
        )
        
        # Add median line
        median = np.exp(mu)
        fig.add_vline(x=median, line_dash="dash", line_color="white", 
                     annotation_text=f"Median: ${median:.0f}", row=row, col=col)
    
    # Combined histogram
    fig.add_trace(
        go.Histogram(x=all_samples, nbinsx=100, name='Combined',
                    marker_color=COLORS['accent'], opacity=0.7),
        row=2, col=3
    )
    
    fig.update_layout(
        title={'text': 'Bet Size Distributions by Segment (Log-Normal)', 
               'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=700,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    fig.update_xaxes(gridcolor='#4a5568', title_text='Bet Amount ($)')
    fig.update_yaxes(gridcolor='#4a5568', title_text='Frequency')
    
    # Formula explanation
    formula_fig = go.Figure()
    formula_fig.add_annotation(
        x=0.5, y=0.5,
        text="<b>Log-Normal Distribution</b><br><br>" +
             "X ~ LogNormal(μ, σ) means log(X) ~ Normal(μ, σ)<br><br>" +
             "E[X] = e^(μ + σ²/2)  |  Median = e^μ<br><br>" +
             "<i>Why Log-Normal?</i> Captures right-skewed betting patterns<br>" +
             "with heavy tails for high-roller outliers.",
        showarrow=False,
        font=dict(size=16, color=COLORS['text']),
        align='center'
    )
    formula_fig.update_layout(
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        height=200,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return html.Div([
        html.Div([
            dcc.Graph(figure=formula_fig)
        ], style={'marginBottom': '1rem'}),
        dcc.Graph(figure=fig)
    ])

def render_wins_tab():
    """Pareto distribution for win multipliers"""
    np.random.seed(42)
    
    # Regular wins (α=2.5) vs Bonus wins (α=1.5)
    regular_wins = generate_pareto_samples(2.5, scale=1, n=10000)
    bonus_wins = generate_pareto_samples(1.5, scale=10, n=10000)
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Regular Wins (α=2.5)', 'Bonus Wins (α=1.5)'])
    
    # Regular wins
    fig.add_trace(
        go.Histogram(x=np.clip(regular_wins, 0, 20), nbinsx=50, 
                    marker_color=COLORS['regular'], name='Regular'),
        row=1, col=1
    )
    
    # Bonus wins
    fig.add_trace(
        go.Histogram(x=np.clip(bonus_wins, 0, 200), nbinsx=50,
                    marker_color=COLORS['whale'], name='Bonus'),
        row=1, col=2
    )
    
    fig.update_layout(
        title={'text': 'Win Multiplier Distributions (Pareto)', 
               'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400,
        showlegend=True
    )
    fig.update_xaxes(gridcolor='#4a5568', title_text='Multiplier')
    fig.update_yaxes(gridcolor='#4a5568', title_text='Frequency')
    
    # Survival function (P(X > x))
    x = np.linspace(1, 50, 500)
    survival_25 = (1/x)**2.5
    survival_15 = (1/x)**1.5
    
    fig_survival = go.Figure()
    fig_survival.add_trace(go.Scatter(x=x, y=survival_25, mode='lines', 
                                      name='α=2.5 (Regular)', line=dict(width=3, color=COLORS['regular'])))
    fig_survival.add_trace(go.Scatter(x=x, y=survival_15, mode='lines',
                                      name='α=1.5 (Bonus)', line=dict(width=3, color=COLORS['whale'])))
    fig_survival.update_layout(
        title={'text': 'Survival Function P(X > x) - Probability of Large Wins', 
               'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='Multiplier (x)',
        yaxis_title='P(Win > x)',
        yaxis_type='log'
    )
    fig_survival.update_xaxes(gridcolor='#4a5568')
    fig_survival.update_yaxes(gridcolor='#4a5568')
    
    # Code snippet
    code_block = html.Pre([
        html.Code("""# Slot Win Multiplier Generation
if np.random.random() < game['bonus_frequency']:  # 0.8% chance
    # Bonus: Heavy tail for jackpots
    multiplier = np.random.pareto(1.5) * 10
    multiplier = min(multiplier, game['max_multiplier'])
else:
    # Regular win: Most are small
    multiplier = np.random.pareto(2.5) + 1
    multiplier = min(multiplier, game['max_multiplier'] * 0.1)""",
                 style={'color': '#68d391'})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1rem', 'borderRadius': '8px',
              'overflow': 'auto', 'fontSize': '0.9rem'})
    
    return html.Div([
        html.Div([
            html.H3("🎯 Power Law Distribution for Wins", style={'color': COLORS['text']}),
            html.P("The Pareto distribution creates the characteristic 'many small wins, rare big wins' pattern. "
                   "Lower α values produce heavier tails (more large wins).",
                   style={'color': COLORS['text'], 'opacity': '0.9'})
        ], style={'backgroundColor': COLORS['card'], 'padding': '1rem', 'borderRadius': '8px', 'marginBottom': '1rem'}),
        dcc.Graph(figure=fig),
        dcc.Graph(figure=fig_survival),
        code_block
    ])

def render_beta_tab():
    """Beta distributions for risk tolerance and churn"""
    np.random.seed(42)
    
    # Risk tolerance by segment
    risk_params = {
        'Whale': (8, 2),
        'VIP': (6, 4),
        'High': (5, 5),
        'Regular': (4, 6),
        'Casual': (2, 8)
    }
    
    fig_risk = go.Figure()
    x = np.linspace(0, 1, 200)
    
    for i, (name, (a, b)) in enumerate(risk_params.items()):
        y = stats.beta.pdf(x, a, b)
        fig_risk.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name,
                                      line=dict(width=3, color=SEGMENT_COLORS[i])))
    
    fig_risk.update_layout(
        title={'text': 'Risk Tolerance Distribution by Segment (Beta)', 
               'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400,
        xaxis_title='Risk Tolerance (0=Risk Averse, 1=Risk Seeking)',
        yaxis_title='Probability Density'
    )
    fig_risk.update_xaxes(gridcolor='#4a5568')
    fig_risk.update_yaxes(gridcolor='#4a5568')
    
    # Churn probability
    churn_params = {
        'Whale': (1, 20),
        'VIP': (1, 20),
        'High': (2, 8),
        'Regular': (2, 8),
        'Casual': (2, 5)
    }
    
    fig_churn = go.Figure()
    x = np.linspace(0, 0.6, 200)
    
    for i, (name, (a, b)) in enumerate(churn_params.items()):
        y = stats.beta.pdf(x, a, b)
        fig_churn.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name,
                                       line=dict(width=3, color=SEGMENT_COLORS[i])))
    
    fig_churn.update_layout(
        title={'text': 'Annual Churn Probability Distribution (Beta)', 
               'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400,
        xaxis_title='Annual Churn Probability',
        yaxis_title='Probability Density'
    )
    fig_churn.update_xaxes(gridcolor='#4a5568')
    fig_churn.update_yaxes(gridcolor='#4a5568')
    
    # Beta distribution explanation
    explanation = html.Div([
        html.H4("Why Beta Distribution?", style={'color': COLORS['accent']}),
        html.Ul([
            html.Li("Naturally bounded between 0 and 1 (perfect for probabilities)"),
            html.Li("Flexible shape: can be U-shaped, uniform, or unimodal"),
            html.Li("α > β → right-skewed (higher values more likely)"),
            html.Li("α < β → left-skewed (lower values more likely)"),
            html.Li("Mode = (α-1)/(α+β-2) for α,β > 1")
        ], style={'color': COLORS['text'], 'lineHeight': '1.8'})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1rem', 'borderRadius': '8px', 'marginBottom': '1rem'})
    
    return html.Div([
        explanation,
        dcc.Graph(figure=fig_risk),
        dcc.Graph(figure=fig_churn)
    ])

def render_timing_tab():
    """Exponential distribution for inter-bet timing"""
    np.random.seed(42)
    
    # Generate samples
    slot_times = generate_exponential_samples(5, 5000)  # 5 seconds mean
    table_times = generate_exponential_samples(30, 5000)  # 30 seconds mean
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Slot Machines (λ=5 sec)', 'Table Games (λ=30 sec)'])
    
    fig.add_trace(
        go.Histogram(x=np.clip(slot_times, 0, 30), nbinsx=50,
                    marker_color=COLORS['accent'], name='Slots'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=np.clip(table_times, 0, 120), nbinsx=50,
                    marker_color=COLORS['vip'], name='Tables'),
        row=1, col=2
    )
    
    fig.update_layout(
        title={'text': 'Inter-Bet Time Distribution (Exponential)', 
               'font': {'size': 20, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400
    )
    fig.update_xaxes(gridcolor='#4a5568', title_text='Seconds between bets')
    fig.update_yaxes(gridcolor='#4a5568', title_text='Frequency')
    
    # Theoretical curves
    x_slot = np.linspace(0, 30, 200)
    x_table = np.linspace(0, 120, 200)
    
    fig_theory = go.Figure()
    fig_theory.add_trace(go.Scatter(x=x_slot, y=stats.expon.pdf(x_slot, scale=5),
                                    mode='lines', name='Slots (λ=5s)', 
                                    line=dict(width=3, color=COLORS['accent'])))
    fig_theory.add_trace(go.Scatter(x=x_table, y=stats.expon.pdf(x_table, scale=30),
                                    mode='lines', name='Tables (λ=30s)',
                                    line=dict(width=3, color=COLORS['vip'])))
    
    fig_theory.update_layout(
        title={'text': 'Theoretical PDF: f(x) = (1/λ)e^(-x/λ)', 
               'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='Time (seconds)',
        yaxis_title='Probability Density'
    )
    fig_theory.update_xaxes(gridcolor='#4a5568')
    fig_theory.update_yaxes(gridcolor='#4a5568')
    
    # Betting rate calculation
    rate_info = html.Div([
        html.H4("⚡ Implied Betting Rates", style={'color': COLORS['accent']}),
        html.Div([
            html.Div([
                html.Span("🎰 Slots", style={'fontWeight': 'bold', 'color': COLORS['accent']}),
                html.P("E[wait] = 5 sec → ~12 bets/minute", style={'margin': '0.5rem 0', 'color': COLORS['text']}),
                html.P("~720 bets/hour", style={'margin': 0, 'color': COLORS['text'], 'opacity': '0.8'})
            ], style={'flex': '1', 'textAlign': 'center'}),
            html.Div([
                html.Span("🃏 Tables", style={'fontWeight': 'bold', 'color': COLORS['vip']}),
                html.P("E[wait] = 30 sec → ~2 bets/minute", style={'margin': '0.5rem 0', 'color': COLORS['text']}),
                html.P("~120 bets/hour", style={'margin': 0, 'color': COLORS['text'], 'opacity': '0.8'})
            ], style={'flex': '1', 'textAlign': 'center'})
        ], style={'display': 'flex', 'gap': '2rem'})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1rem', 'borderRadius': '8px', 'marginBottom': '1rem'})
    
    return html.Div([
        rate_info,
        dcc.Graph(figure=fig),
        dcc.Graph(figure=fig_theory)
    ])

def render_temporal_tab():
    """Temporal patterns - day of week, time of day"""
    np.random.seed(42)
    
    # Day of week weights
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weights = [0.10, 0.10, 0.12, 0.13, 0.20, 0.20, 0.15]
    
    fig_days = go.Figure(data=[go.Bar(
        x=days,
        y=weights,
        marker_color=[COLORS['regular'] if w < 0.15 else COLORS['accent'] if w < 0.20 else COLORS['whale'] 
                      for w in weights],
        text=[f'{w*100:.0f}%' for w in weights],
        textposition='outside'
    )])
    fig_days.update_layout(
        title={'text': 'Day of Week Activity Weights', 'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        yaxis_title='Weight',
        xaxis_title='Day of Week'
    )
    fig_days.update_xaxes(gridcolor='#4a5568')
    fig_days.update_yaxes(gridcolor='#4a5568')
    
    # Time of day distribution
    hours = []
    for _ in range(10000):
        if np.random.random() < 0.7:
            hour = int(np.random.normal(21, 2)) % 24
        else:
            hour = np.random.randint(0, 24)
        hours.append(hour)
    
    fig_hours = go.Figure(data=[go.Histogram(
        x=hours,
        nbinsx=24,
        marker_color=COLORS['accent']
    )])
    fig_hours.add_vline(x=21, line_dash="dash", line_color=COLORS['whale'],
                        annotation_text="Peak: 9 PM")
    fig_hours.update_layout(
        title={'text': 'Time of Day Distribution (Gaussian Mixture)', 
               'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=350,
        xaxis_title='Hour of Day (0-23)',
        yaxis_title='Session Frequency'
    )
    fig_hours.update_xaxes(gridcolor='#4a5568', tickmode='linear', tick0=0, dtick=2)
    fig_hours.update_yaxes(gridcolor='#4a5568')
    
    # Poisson session generation
    lambdas = [8, 5, 3, 2, 1]
    segment_names = ['Whale', 'VIP', 'High', 'Regular', 'Casual']
    
    fig_poisson = go.Figure()
    x = np.arange(0, 20)
    
    for i, (lam, name) in enumerate(zip(lambdas, segment_names)):
        y = stats.poisson.pmf(x, lam)
        fig_poisson.add_trace(go.Bar(x=x, y=y, name=f'{name} (λ={lam})',
                                     marker_color=SEGMENT_COLORS[i], opacity=0.7))
    
    fig_poisson.update_layout(
        title={'text': 'Weekly Session Count by Segment (Poisson)', 
               'font': {'size': 18, 'color': COLORS['text']}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400,
        barmode='overlay',
        xaxis_title='Number of Sessions per Week',
        yaxis_title='Probability'
    )
    fig_poisson.update_xaxes(gridcolor='#4a5568')
    fig_poisson.update_yaxes(gridcolor='#4a5568')
    
    # Code for temporal patterns
    code_block = html.Pre([
        html.Code("""# Time of day generation (Gaussian Mixture)
if np.random.random() < 0.7:  # 70% evening sessions
    hour = int(np.random.normal(21, 2)) % 24  # Peak at 9 PM
else:  # 30% uniform throughout day
    hour = np.random.randint(0, 24)

# Payday effect (+30% activity boost)
if 1 <= current_day <= 3 or 14 <= current_day <= 17:
    day_weights = [w * 1.3 for w in day_weights]""",
                 style={'color': '#68d391'})
    ], style={'backgroundColor': COLORS['card'], 'padding': '1rem', 'borderRadius': '8px',
              'overflow': 'auto', 'fontSize': '0.9rem', 'marginTop': '1rem'})
    
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=fig_days)], style={'flex': '1'}),
            html.Div([dcc.Graph(figure=fig_hours)], style={'flex': '1'})
        ], style={'display': 'flex', 'gap': '1rem', 'flexWrap': 'wrap'}),
        dcc.Graph(figure=fig_poisson),
        code_block
    ])

# ============================================
# RUN THE APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎰 Casino Generator - Mathematical Foundations Visualizer")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser to: http://127.0.0.1:8050")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8050)