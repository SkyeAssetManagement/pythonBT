"""
Plotly Dashboard MVP - Native Candlestick Rendering with Trade Navigation
This MVP demonstrates Plotly's native candlestick capabilities and trade jump functionality.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
from dash.exceptions import PreventUpdate
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Try to import data loader
try:
    from data.parquet_converter import ParquetConverter
except ImportError:
    print("Warning: ParquetConverter not available, will use synthetic data")
    ParquetConverter = None


class PlotlyDashboardMVP:
    """MVP Dashboard using Plotly's native candlestick charts."""
    
    def __init__(self):
        """Initialize the dashboard MVP."""
        self.app = dash.Dash(__name__)
        self.ohlcv_data = None
        self.trades_data = None
        self.current_viewport = None
        
        # Performance metrics
        self.load_time = 0
        self.render_time = 0
        
        # Setup the dashboard
        self._load_data()
        self._setup_layout()
        self._setup_callbacks()
        
    def _load_data(self):
        """Load OHLCV and trade data from 1/1/2020."""
        print("\n" + "="*60)
        print("LOADING DATA FOR PLOTLY MVP")
        print("="*60)
        
        start_time = time.time()
        
        # Try to load real data first
        if ParquetConverter:
            try:
                converter = ParquetConverter()
                # Load data from 2020-01-01
                self.ohlcv_data = converter.load_or_convert(
                    "AD", "1m", "diffAdjusted"
                )
                
                if self.ohlcv_data:
                    # Filter from 2020-01-01
                    self.ohlcv_data = converter.filter_data_by_date(
                        self.ohlcv_data, "2020-01-01", None
                    )
                    
                    # Convert to DataFrame for easier handling
                    self.df = pd.DataFrame({
                        'datetime': pd.to_datetime(self.ohlcv_data['datetime'], unit='ns'),
                        'open': self.ohlcv_data['open'],
                        'high': self.ohlcv_data['high'],
                        'low': self.ohlcv_data['low'],
                        'close': self.ohlcv_data['close'],
                        'volume': self.ohlcv_data.get('volume', np.ones(len(self.ohlcv_data['close'])))
                    })
                    
                    print(f"[OK] Loaded {len(self.df):,} bars from ParquetConverter")
                    
            except Exception as e:
                print(f"Failed to load real data: {e}")
                self.ohlcv_data = None
        
        # Fall back to synthetic data if needed
        if self.ohlcv_data is None:
            print("Generating synthetic data for testing...")
            self.df = self._generate_synthetic_data()
            print(f"[OK] Generated {len(self.df):,} synthetic bars")
        
        # Generate synthetic trades
        self.trades_data = self._generate_synthetic_trades()
        print(f"[OK] Generated {len(self.trades_data)} trades")
        
        # Calculate SMA
        self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['sma_50'] = self.df['close'].rolling(window=50).mean()
        
        self.load_time = time.time() - start_time
        print(f"\n[OK] Data loaded in {self.load_time:.2f} seconds")
        print(f"  Memory usage: ~{self.df.memory_usage().sum() / 1024**2:.1f} MB")
        
    def _generate_synthetic_data(self):
        """Generate synthetic OHLCV data from 2020-01-01."""
        # Generate 2 years of 1-minute data (simplified to daily for MVP)
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='1h')
        n = len(dates)
        
        # Generate realistic price movement
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, n)
        price = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        df = pd.DataFrame({
            'datetime': dates,
            'open': price * (1 + np.random.uniform(-0.002, 0.002, n)),
            'high': price * (1 + np.random.uniform(0, 0.005, n)),
            'low': price * (1 + np.random.uniform(-0.005, 0, n)),
            'close': price,
            'volume': np.random.uniform(1000, 10000, n)
        })
        
        # Ensure OHLC relationships are valid
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def _generate_synthetic_trades(self):
        """Generate synthetic trade data."""
        trades = []
        
        # Generate 50 trades throughout the data
        n_trades = 50
        indices = np.linspace(100, len(self.df) - 100, n_trades, dtype=int)
        
        for i, entry_idx in enumerate(indices):
            exit_idx = entry_idx + np.random.randint(10, 100)
            if exit_idx >= len(self.df):
                exit_idx = len(self.df) - 1
            
            entry_price = self.df.iloc[entry_idx]['close']
            exit_price = self.df.iloc[exit_idx]['close']
            pnl = exit_price - entry_price
            
            trades.append({
                'trade_id': f'T{i+1:03d}',
                'entry_time': self.df.iloc[entry_idx]['datetime'],
                'exit_time': self.df.iloc[exit_idx]['datetime'],
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'side': 'Long' if i % 2 == 0 else 'Short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': np.random.uniform(0.1, 2.0),
                'pnl': pnl,
                'pnl_pct': (pnl / entry_price) * 100
            })
        
        return pd.DataFrame(trades)
    
    def _setup_layout(self):
        """Setup the Dash layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Plotly Dashboard MVP - Native Candlestick Rendering", 
                       style={'color': 'white', 'margin': '10px'}),
                html.Div(id='performance-metrics', style={'color': '#00ff00', 'margin': '10px'})
            ], style={'backgroundColor': '#1e1e1e', 'padding': '10px'}),
            
            # Main container
            html.Div([
                # Left panel - Trade list and controls
                html.Div([
                    # Trade navigation input
                    html.Div([
                        html.Label("Jump to Trade:", style={'color': 'white'}),
                        dcc.Input(
                            id='trade-input',
                            type='text',
                            placeholder='Enter Trade ID (e.g., T001)',
                            style={'width': '150px', 'margin': '5px'}
                        ),
                        html.Button('Jump', id='jump-button', n_clicks=0,
                                  style={'margin': '5px'}),
                        html.Div(id='jump-status', style={'color': '#00ff00', 'margin': '5px'})
                    ], style={'padding': '10px', 'backgroundColor': '#2a2a2a', 'marginBottom': '10px'}),
                    
                    # Trade list table
                    html.Div([
                        html.H3("Trade List", style={'color': 'white', 'margin': '10px'}),
                        dash_table.DataTable(
                            id='trade-table',
                            columns=[
                                {'name': 'ID', 'id': 'trade_id'},
                                {'name': 'Side', 'id': 'side'},
                                {'name': 'Entry', 'id': 'entry_price', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                {'name': 'Exit', 'id': 'exit_price', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                {'name': 'PnL', 'id': 'pnl', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'PnL%', 'id': 'pnl_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}}
                            ],
                            data=self.trades_data.to_dict('records'),
                            style_cell={'textAlign': 'center', 'backgroundColor': '#2a2a2a', 'color': 'white'},
                            style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'pnl', 'filter_query': '{pnl} > 0'},
                                    'color': '#00ff00'
                                },
                                {
                                    'if': {'column_id': 'pnl', 'filter_query': '{pnl} < 0'},
                                    'color': '#ff0000'
                                },
                                {
                                    'if': {'column_id': 'side', 'filter_query': '{side} = "Long"'},
                                    'backgroundColor': '#004400'
                                },
                                {
                                    'if': {'column_id': 'side', 'filter_query': '{side} = "Short"'},
                                    'backgroundColor': '#440000'
                                }
                            ],
                            page_size=20,
                            style_table={'height': '500px', 'overflowY': 'auto'}
                        )
                    ], style={'padding': '10px', 'backgroundColor': '#2a2a2a'})
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Right panel - Chart
                html.Div([
                    dcc.Graph(
                        id='candlestick-chart',
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                        },
                        style={'height': '700px'}
                    ),
                    html.Div(id='chart-info', style={'color': 'white', 'padding': '10px'})
                ], style={'width': '70%', 'display': 'inline-block'})
            ], style={'backgroundColor': '#1e1e1e', 'minHeight': '800px'})
        ])
    
    def create_candlestick_figure(self, start_idx=None, end_idx=None):
        """Create the candlestick figure with SMA indicators."""
        start_time = time.time()
        
        # Default to showing last 1000 bars
        if start_idx is None:
            start_idx = max(0, len(self.df) - 1000)
        if end_idx is None:
            end_idx = len(self.df)
        
        # Slice data for viewport
        df_view = self.df.iloc[start_idx:end_idx]
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price Chart with SMA', 'Volume')
        )
        
        # Add candlestick chart - NATIVE PLOTLY CANDLESTICK!
        fig.add_trace(
            go.Candlestick(
                x=df_view['datetime'],
                open=df_view['open'],
                high=df_view['high'],
                low=df_view['low'],
                close=df_view['close'],
                name='OHLC',
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            ),
            row=1, col=1
        )
        
        # Add SMA indicators
        fig.add_trace(
            go.Scatter(
                x=df_view['datetime'],
                y=df_view['sma_20'],
                name='SMA 20',
                line=dict(color='#ffaa00', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_view['datetime'],
                y=df_view['sma_50'],
                name='SMA 50',
                line=dict(color='#00aaff', width=1)
            ),
            row=1, col=1
        )
        
        # Add trade markers
        trades_in_view = self.trades_data[
            (self.trades_data['entry_idx'] >= start_idx) & 
            (self.trades_data['entry_idx'] < end_idx)
        ]
        
        if not trades_in_view.empty:
            # Entry markers
            fig.add_trace(
                go.Scatter(
                    x=trades_in_view['entry_time'],
                    y=trades_in_view['entry_price'],
                    mode='markers',
                    name='Trade Entry',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='#00ff00',
                        line=dict(width=1, color='white')
                    ),
                    text=trades_in_view['trade_id'],
                    hovertemplate='Trade %{text}<br>Entry: %{y:.4f}<br>Time: %{x}'
                ),
                row=1, col=1
            )
            
            # Exit markers
            exits_in_view = self.trades_data[
                (self.trades_data['exit_idx'] >= start_idx) & 
                (self.trades_data['exit_idx'] < end_idx)
            ]
            
            if not exits_in_view.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exits_in_view['exit_time'],
                        y=exits_in_view['exit_price'],
                        mode='markers',
                        name='Trade Exit',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='#ff0000',
                            line=dict(width=1, color='white')
                        ),
                        text=exits_in_view['trade_id'],
                        hovertemplate='Trade %{text}<br>Exit: %{y:.4f}<br>Time: %{x}'
                    ),
                    row=1, col=1
                )
        
        # Add volume bars
        colors = ['#ff0000' if close < open else '#00ff00' 
                 for close, open in zip(df_view['close'], df_view['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df_view['datetime'],
                y=df_view['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Plotly Native Candlestick Chart - Showing {len(df_view)} bars',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=True,
            legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)'),
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridcolor='#333333')
        fig.update_yaxes(showgrid=True, gridcolor='#333333', title='Price', row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='#333333', title='Volume', row=2, col=1)
        
        self.render_time = time.time() - start_time
        self.current_viewport = (start_idx, end_idx)
        
        return fig
    
    def _setup_callbacks(self):
        """Setup Dash callbacks for interactivity."""
        
        @self.app.callback(
            [Output('candlestick-chart', 'figure'),
             Output('jump-status', 'children'),
             Output('performance-metrics', 'children'),
             Output('chart-info', 'children')],
            [Input('jump-button', 'n_clicks'),
             Input('trade-table', 'active_cell')],
            [State('trade-input', 'value'),
             State('trade-table', 'data')]
        )
        def update_chart(n_clicks, active_cell, trade_id_input, table_data):
            """Handle trade navigation from input or table click."""
            
            # Determine trigger
            triggered = ctx.triggered_id
            
            jump_status = ""
            start_idx = None
            end_idx = None
            
            if triggered == 'jump-button' and trade_id_input:
                # Jump to trade from input
                trade_id = trade_id_input.upper()
                trade = self.trades_data[self.trades_data['trade_id'] == trade_id]
                
                if not trade.empty:
                    entry_idx = trade.iloc[0]['entry_idx']
                    # Center viewport on trade
                    start_idx = max(0, entry_idx - 250)
                    end_idx = min(len(self.df), entry_idx + 250)
                    jump_status = f"[OK] Jumped to trade {trade_id}"
                else:
                    jump_status = f"[ERROR] Trade {trade_id} not found"
            
            elif triggered == 'trade-table' and active_cell:
                # Jump to trade from table click
                row = active_cell['row']
                if row < len(table_data):
                    trade_id = table_data[row]['trade_id']
                    trade = self.trades_data[self.trades_data['trade_id'] == trade_id]
                    
                    if not trade.empty:
                        entry_idx = trade.iloc[0]['entry_idx']
                        # Center viewport on trade
                        start_idx = max(0, entry_idx - 250)
                        end_idx = min(len(self.df), entry_idx + 250)
                        jump_status = f"[OK] Navigated to trade {trade_id}"
            
            # Create figure
            fig = self.create_candlestick_figure(start_idx, end_idx)
            
            # Performance metrics
            perf_metrics = f"""
            Data Points: {len(self.df):,} | 
            Load Time: {self.load_time:.2f}s | 
            Render Time: {self.render_time:.3f}s | 
            FPS: {1/self.render_time:.0f} (estimated)
            """
            
            # Chart info
            if self.current_viewport:
                viewport_size = self.current_viewport[1] - self.current_viewport[0]
                chart_info = f"""
                Viewport: Bars {self.current_viewport[0]:,} to {self.current_viewport[1]:,} 
                ({viewport_size:,} bars displayed) | 
                Native Plotly Candlestick Rendering
                """
            else:
                chart_info = "Use mouse wheel to zoom, drag to pan"
            
            return fig, jump_status, perf_metrics, chart_info
    
    def run(self, debug=True, port=8050):
        """Run the Dash application."""
        print(f"\n{'='*60}")
        print("STARTING PLOTLY DASHBOARD MVP")
        print(f"{'='*60}")
        print(f"\n[OK] Dashboard ready at http://localhost:{port}")
        print("\nFeatures demonstrated:")
        print("  - Native Plotly candlestick rendering (no manual drawing!)")
        print("  - SMA indicators overlaid on chart")
        print("  - Trade markers with hover information")
        print("  - Jump-to-trade functionality (input or click)")
        print("  - Smooth zoom and pan with mouse")
        print(f"\nData: {len(self.df):,} bars loaded")
        print(f"Trades: {len(self.trades_data)} trades with markers")
        
        self.app.run(debug=debug, port=port)


def main():
    """Run the MVP demonstration."""
    print("\n" + "="*80)
    print("PLOTLY DASHBOARD MVP - DEMONSTRATING NATIVE CANDLESTICK RENDERING")
    print("="*80)
    
    # Create and run dashboard
    dashboard = PlotlyDashboardMVP()
    
    # Print summary
    print("\n" + "="*60)
    print("MVP FEATURES COMPLETED:")
    print("="*60)
    print("[OK] Native Plotly candlestick chart (no manual vertex calculation!)")
    print("[OK] SMA 20 and SMA 50 indicators")
    print("[OK] Trade markers with entry/exit triangles")
    print("[OK] Trade list with clickable navigation")
    print("[OK] Jump-to-trade input box")
    print("[OK] Smooth zoom/pan with native Plotly controls")
    print("[OK] Dark theme professional appearance")
    print("[OK] Performance metrics display")
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON:")
    print("="*60)
    print("Old VisPy approach: Complex manual vertex calculation, rendering issues")
    print("New Plotly approach: Simple native candlestick, perfect rendering")
    print(f"Data handling: {len(dashboard.df):,} points loaded in {dashboard.load_time:.2f}s")
    
    # Run the server
    dashboard.run(debug=False, port=8050)


if __name__ == "__main__":
    main()