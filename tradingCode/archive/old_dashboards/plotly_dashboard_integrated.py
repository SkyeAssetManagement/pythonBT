"""
Plotly Dashboard Integrated - Works with main.py workflow
Run with: python main.py ES simpleSMA --useDefaults --start_date "2020-01-01" --plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
from pathlib import Path
import time
import sys
import threading
import webbrowser

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


class PlotlyDashboardIntegrated:
    """Integrated Plotly Dashboard for main.py workflow."""
    
    def __init__(self, ohlcv_data=None, trades_df=None, portfolio=None, symbol="ES", strategy_name="Strategy"):
        """Initialize integrated dashboard with trading results."""
        self.app = dash.Dash(__name__)
        self.symbol = symbol
        self.strategy_name = strategy_name
        
        # Store data
        self.ohlcv_data = ohlcv_data
        self.trades_df = trades_df
        self.portfolio = portfolio
        
        # Convert to DataFrame if we have data
        if ohlcv_data:
            self.df = self._prepare_dataframe(ohlcv_data)
        else:
            self.df = None
            
        # Performance metrics
        self.load_time = 0
        self.render_time = 0
        
        # Setup the dashboard
        if self.df is not None:
            self._calculate_indicators()
            self._setup_layout()
            self._setup_callbacks()
        
    def _prepare_dataframe(self, ohlcv_data):
        """Convert ohlcv_data dict to DataFrame."""
        # Handle datetime conversion
        if 'datetime' in ohlcv_data:
            datetime_vals = ohlcv_data['datetime']
            # Convert nanosecond timestamps to datetime
            if isinstance(datetime_vals[0], (int, np.integer)) and datetime_vals[0] > 1e15:
                datetime_index = pd.to_datetime(datetime_vals, unit='ns')
            else:
                datetime_index = pd.to_datetime(datetime_vals)
        else:
            datetime_index = pd.date_range(start='2020-01-01', periods=len(ohlcv_data['close']), freq='1min')
        
        df = pd.DataFrame({
            'datetime': datetime_index,
            'open': np.asarray(ohlcv_data['open'], dtype=np.float32),
            'high': np.asarray(ohlcv_data['high'], dtype=np.float32),
            'low': np.asarray(ohlcv_data['low'], dtype=np.float32),
            'close': np.asarray(ohlcv_data['close'], dtype=np.float32),
            'volume': np.asarray(ohlcv_data.get('volume', np.ones(len(ohlcv_data['close']))), dtype=np.float32)
        })
        
        return df
    
    def _calculate_indicators(self):
        """Calculate technical indicators."""
        # Simple Moving Averages
        self.df['sma_20'] = self.df['close'].rolling(window=20, min_periods=1).mean()
        self.df['sma_50'] = self.df['close'].rolling(window=50, min_periods=1).mean()
        self.df['sma_200'] = self.df['close'].rolling(window=200, min_periods=1).mean()
        
        # Add equity curve if portfolio available
        if self.portfolio is not None:
            try:
                # Get equity values
                if hasattr(self.portfolio, 'value'):
                    equity = self.portfolio.value
                    if hasattr(equity, 'values'):
                        equity_vals = equity.values
                        # Handle multi-column portfolios
                        if len(equity_vals.shape) > 1:
                            equity_vals = np.sum(equity_vals, axis=1)
                    else:
                        equity_vals = equity
                    
                    # Ensure length matches
                    if len(equity_vals) == len(self.df):
                        # Calculate cumulative net profit (equity minus initial capital)
                        initial_capital = equity_vals[0] if len(equity_vals) > 0 else 10000
                        self.df['equity'] = equity_vals - initial_capital  # Net profit since inception
                        print(f"DEBUG: Initial capital: {initial_capital:.2f}, Final equity: {equity_vals[-1]:.2f}, Net profit: {equity_vals[-1] - initial_capital:.2f}")
                    else:
                        print(f"Warning: Equity length {len(equity_vals)} doesn't match data length {len(self.df)}")
            except Exception as e:
                print(f"Warning: Could not add equity curve: {e}")
    
    def _prepare_trades_data(self):
        """Prepare trades data for display."""
        if self.trades_df is None or self.trades_df.empty:
            return pd.DataFrame()
        
        trades = self.trades_df.copy()
        
        # Ensure we have the required columns
        display_trades = pd.DataFrame()
        
        # Map column names flexibly
        if 'Exit Trade Id' in trades.columns:
            # VectorBT format - Exit Trade Id is the trade identifier
            display_trades['trade_id'] = trades['Exit Trade Id'].astype(str)
        elif 'Trade ID' in trades.columns:
            display_trades['trade_id'] = trades['Trade ID'].astype(str)
        elif 'trade_id' in trades.columns:
            display_trades['trade_id'] = trades['trade_id'].astype(str)
        else:
            display_trades['trade_id'] = [str(i) for i in range(len(trades))]
        
        # Side/Direction
        for col in ['Direction', 'direction', 'Side', 'side']:
            if col in trades.columns:
                display_trades['side'] = trades[col]
                break
        else:
            display_trades['side'] = 'Long'
        
        # Entry price
        for col in ['Avg Entry Price', 'entry_price', 'Entry Price']:
            if col in trades.columns:
                display_trades['entry_price'] = trades[col]
                break
        else:
            display_trades['entry_price'] = 0
        
        # Exit price
        for col in ['Avg Exit Price', 'exit_price', 'Exit Price']:
            if col in trades.columns:
                display_trades['exit_price'] = trades[col]
                break
        else:
            display_trades['exit_price'] = 0
        
        # PnL
        for col in ['PnL', 'pnl', 'Profit']:
            if col in trades.columns:
                display_trades['pnl'] = trades[col]
                break
        else:
            display_trades['pnl'] = 0
        
        # PnL %
        if 'pnl_pct' in trades.columns:
            display_trades['pnl_pct'] = trades['pnl_pct']
        else:
            display_trades['pnl_pct'] = (display_trades['pnl'] / (display_trades['entry_price'] + 0.0001)) * 100
        
        # Entry/Exit times or indices
        found_entry = False
        for col in ['Entry Index', 'EntryTime', 'entry_time', 'entry_index']:
            if col in trades.columns:
                display_trades['entry_idx'] = pd.to_numeric(trades[col], errors='coerce').fillna(0).astype(int)
                print(f"DEBUG: Using column '{col}' for entry_idx")
                print(f"DEBUG: Sample entry indices: {display_trades['entry_idx'].head(10).tolist()}")
                found_entry = True
                break
        
        if not found_entry:
            print("WARNING: No entry index column found, using default 0")
            display_trades['entry_idx'] = 0
            
        found_exit = False
        for col in ['Exit Index', 'ExitTime', 'exit_time', 'exit_index']:
            if col in trades.columns:
                display_trades['exit_idx'] = pd.to_numeric(trades[col], errors='coerce').fillna(display_trades['entry_idx'] + 10).astype(int)
                found_exit = True
                break
        
        if not found_exit:
            display_trades['exit_idx'] = display_trades['entry_idx'] + 10
        
        return display_trades
    
    def _setup_layout(self):
        """Setup the Dash layout."""
        # Prepare trades for display
        trades_display = self._prepare_trades_data()
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1(f"Plotly Dashboard - {self.symbol} - {self.strategy_name}", 
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
                            placeholder='Enter Trade ID',
                            style={'width': '150px', 'margin': '5px'}
                        ),
                        html.Button('Jump', id='jump-button', n_clicks=0,
                                  style={'margin': '5px'}),
                        html.Div(id='jump-status', style={'color': '#00ff00', 'margin': '5px'})
                    ], style={'padding': '10px', 'backgroundColor': '#2a2a2a', 'marginBottom': '10px'}),
                    
                    # Stats panel
                    html.Div([
                        html.H3("Performance Stats", style={'color': 'white'}),
                        html.Div(id='stats-panel', style={'color': 'white', 'padding': '10px'})
                    ], style={'backgroundColor': '#2a2a2a', 'marginBottom': '10px', 'padding': '10px'}),
                    
                    # Trade list table with infinite scroll
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
                            data=trades_display.to_dict('records') if not trades_display.empty else [],
                            style_cell={'textAlign': 'center', 'backgroundColor': '#2a2a2a', 'color': 'white', 'fontSize': '12px'},
                            style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'pnl', 'filter_query': '{pnl} > 0'},
                                    'color': '#00ff00'
                                },
                                {
                                    'if': {'column_id': 'pnl', 'filter_query': '{pnl} < 0'},
                                    'color': '#ff0000'
                                }
                            ],
                            # Virtualization for infinite scroll
                            virtualization=True,
                            fixed_rows={'headers': True},
                            style_table={'height': '400px', 'overflowY': 'auto'},
                            style_data={'height': 'auto', 'minWidth': '60px'}
                        )
                    ], style={'padding': '10px', 'backgroundColor': '#2a2a2a'})
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Right panel - Charts
                html.Div([
                    # Chart container with fixed data window
                    html.Div([
                        dcc.Graph(
                            id='main-chart',
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                'scrollZoom': True,  # Enable scroll to zoom
                                'doubleClick': 'autosize'  # Double-click to reset view
                            },
                            style={'height': '800px'}
                        ),
                        # Fixed data window in bottom left
                        html.Div(
                            id='data-window',
                            style={
                                'position': 'absolute',
                                'bottom': '10px',
                                'left': '10px',
                                'backgroundColor': 'rgba(30, 30, 30, 0.9)',
                                'color': 'white',
                                'padding': '10px',
                                'border': '1px solid #555',
                                'borderRadius': '5px',
                                'fontSize': '12px',
                                'fontFamily': 'monospace',
                                'minWidth': '200px',
                                'zIndex': '1000'
                            },
                            children=[
                                html.Div("Data Window", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                html.Div(id='data-content', children="Hover over chart for data")
                            ]
                        )
                    ], style={'position': 'relative'}),
                    html.Div(id='chart-info', style={'color': 'white', 'padding': '10px'})
                ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'backgroundColor': '#1e1e1e', 'minHeight': '900px'}),
            
            # Store trades data for callbacks
            dcc.Store(id='trades-store', data=trades_display.to_dict('records') if not trades_display.empty else [])
        ], style={'height': '100vh', 'overflow': 'hidden'})
    
    def create_main_figure(self, start_idx=None, end_idx=None):
        """Create the main chart figure with candlesticks, indicators, and equity curve."""
        start_time = time.time()
        
        # Default to showing last 2000 bars
        if start_idx is None:
            start_idx = max(0, len(self.df) - 2000)
        if end_idx is None:
            end_idx = len(self.df)
        
        # Slice data for viewport
        df_view = self.df.iloc[start_idx:end_idx]
        
        # Determine subplot configuration
        has_equity = 'equity' in self.df.columns
        n_rows = 3 if has_equity else 2
        # Adjusted sizes: Price chart 70%, Volume 10%, Equity 15%, spacing 5%
        row_heights = [0.70, 0.10, 0.15] if has_equity else [0.80, 0.15]
        subplot_titles = ('Price Chart with SMA', 'Volume', 'Cumulative Net Profit') if has_equity else ('Price Chart with SMA', 'Volume')
        
        # Create figure with subplots
        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
        
        # Add candlestick chart - NATIVE PLOTLY!
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
        
        # Add trade markers if we have trades
        if self.trades_df is not None and not self.trades_df.empty:
            trades_display = self._prepare_trades_data()
            
            # Get trades in viewport
            trades_in_view = trades_display[
                ((trades_display['entry_idx'] >= start_idx) & (trades_display['entry_idx'] < end_idx)) |
                ((trades_display['exit_idx'] >= start_idx) & (trades_display['exit_idx'] < end_idx))
            ]
            
            if not trades_in_view.empty:
                for _, trade in trades_in_view.iterrows():
                    # Get candle data for positioning
                    entry_idx = trade['entry_idx']
                    exit_idx = trade['exit_idx']
                    is_long = trade['side'].lower() in ['long', 'buy']
                    
                    # ENTRY MARKERS
                    if start_idx <= entry_idx < end_idx:
                        # Get candle low/high for this bar
                        candle_idx = entry_idx - start_idx
                        if 0 <= candle_idx < len(df_view):
                            candle_low = df_view.iloc[candle_idx]['low']
                            candle_high = df_view.iloc[candle_idx]['high']
                            entry_time = df_view.iloc[candle_idx]['datetime']
                            
                            if is_long:
                                # BUY: Filled green up arrow, 0.1% below candle low
                                y_position = candle_low * 0.999
                                fig.add_trace(
                                    go.Scatter(
                                        x=[entry_time],
                                        y=[y_position],
                                        mode='markers',
                                        name='Buy',
                                        marker=dict(
                                            symbol='triangle-up',
                                            size=14,
                                            color='#00ff00',  # Filled green
                                            line=dict(width=1, color='#00ff00')
                                        ),
                                        text=f"Trade {trade['trade_id']} BUY",
                                        hovertemplate='%{text}<br>Price: %{customdata:.4f}<br>Time: %{x}',
                                        customdata=[trade['entry_price']],
                                        showlegend=False
                                    ),
                                    row=1, col=1
                                )
                            else:
                                # SHORT: Filled red down arrow, 0.1% above candle high
                                y_position = candle_high * 1.001
                                fig.add_trace(
                                    go.Scatter(
                                        x=[entry_time],
                                        y=[y_position],
                                        mode='markers',
                                        name='Short',
                                        marker=dict(
                                            symbol='triangle-down',
                                            size=14,
                                            color='#ff0000',  # Filled red
                                            line=dict(width=1, color='#ff0000')
                                        ),
                                        text=f"Trade {trade['trade_id']} SHORT",
                                        hovertemplate='%{text}<br>Price: %{customdata:.4f}<br>Time: %{x}',
                                        customdata=[trade['entry_price']],
                                        showlegend=False
                                    ),
                                    row=1, col=1
                                )
                    
                    # EXIT MARKERS
                    if start_idx <= exit_idx < end_idx:
                        # Get candle low/high for exit bar
                        candle_idx = exit_idx - start_idx
                        if 0 <= candle_idx < len(df_view):
                            candle_low = df_view.iloc[candle_idx]['low']
                            candle_high = df_view.iloc[candle_idx]['high']
                            exit_time = df_view.iloc[candle_idx]['datetime']
                            
                            if is_long:
                                # SELL (exit long): Hollow red down arrow, 0.1% above candle high
                                y_position = candle_high * 1.001
                                fig.add_trace(
                                    go.Scatter(
                                        x=[exit_time],
                                        y=[y_position],
                                        mode='markers',
                                        name='Sell',
                                        marker=dict(
                                            symbol='triangle-down',
                                            size=14,
                                            color='rgba(255, 0, 0, 0)',  # Hollow (transparent fill)
                                            line=dict(width=2, color='#ff0000')  # Red outline
                                        ),
                                        text=f"Trade {trade['trade_id']} SELL",
                                        hovertemplate='%{text}<br>Price: %{customdata:.4f}<br>Time: %{x}',
                                        customdata=[trade['exit_price']],
                                        showlegend=False
                                    ),
                                    row=1, col=1
                                )
                            else:
                                # COVER (exit short): Hollow green up arrow, 0.1% below candle low
                                y_position = candle_low * 0.999
                                fig.add_trace(
                                    go.Scatter(
                                        x=[exit_time],
                                        y=[y_position],
                                        mode='markers',
                                        name='Cover',
                                        marker=dict(
                                            symbol='triangle-up',
                                            size=14,
                                            color='rgba(0, 255, 0, 0)',  # Hollow (transparent fill)
                                            line=dict(width=2, color='#00ff00')  # Green outline
                                        ),
                                        text=f"Trade {trade['trade_id']} COVER",
                                        hovertemplate='%{text}<br>Price: %{customdata:.4f}<br>Time: %{x}',
                                        customdata=[trade['exit_price']],
                                        showlegend=False
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
        
        # Add cumulative net profit curve if available
        if has_equity:
            # Color based on profit/loss
            profit_color = '#00ff00' if df_view['equity'].iloc[-1] >= 0 else '#ff0000'
            fill_color = 'rgba(0, 255, 0, 0.1)' if df_view['equity'].iloc[-1] >= 0 else 'rgba(255, 0, 0, 0.1)'
            
            fig.add_trace(
                go.Scatter(
                    x=df_view['datetime'],
                    y=df_view['equity'],
                    name='Net Profit',
                    line=dict(color=profit_color, width=2),
                    fill='tozeroy',
                    fillcolor=fill_color,
                    hovertemplate='Net Profit: $%{y:,.2f}<br>Time: %{x}'
                ),
                row=3, col=1
            )
            
            # Add zero line for reference
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1, opacity=0.5)
        
        # Update layout with Y-axis autoscaling
        fig.update_layout(
            title=f'{self.symbol} - {self.strategy_name} - Showing {len(df_view)} bars',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)'),
            hovermode='x unified',
            # Enable Y-axis autoscaling on zoom
            yaxis=dict(autorange=True, fixedrange=False),
            yaxis2=dict(autorange=True, fixedrange=False),
            yaxis3=dict(autorange=True, fixedrange=False) if has_equity else None
        )
        
        # Update axes with autorange enabled
        fig.update_xaxes(showgrid=True, gridcolor='#333333')
        fig.update_yaxes(showgrid=True, gridcolor='#333333', title='Price', autorange=True, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='#333333', title='Volume', autorange=True, row=2, col=1)
        if has_equity:
            fig.update_yaxes(showgrid=True, gridcolor='#333333', title='Net Profit ($)', autorange=True, row=3, col=1)
        
        self.render_time = time.time() - start_time
        return fig
    
    def _setup_callbacks(self):
        """Setup Dash callbacks for interactivity."""
        
        @self.app.callback(
            [Output('main-chart', 'figure'),
             Output('jump-status', 'children'),
             Output('performance-metrics', 'children'),
             Output('stats-panel', 'children'),
             Output('chart-info', 'children')],
            [Input('jump-button', 'n_clicks'),
             Input('trade-table', 'active_cell')],
            [State('trade-input', 'value'),
             State('trades-store', 'data')]
        )
        def update_chart(n_clicks, active_cell, trade_id_input, trades_data):
            """Handle trade navigation and update displays."""
            
            # Determine trigger
            triggered = ctx.triggered_id
            
            jump_status = ""
            start_idx = None
            end_idx = None
            
            if triggered == 'jump-button' and trade_id_input:
                # Jump to trade from input
                trade_id_search = str(trade_id_input).strip()
                print(f"DEBUG: Searching for trade ID: '{trade_id_search}'")
                
                # Try exact match first
                found = False
                for trade in trades_data:
                    trade_id_str = str(trade['trade_id'])
                    print(f"DEBUG: Comparing with trade ID: '{trade_id_str}', entry_idx: {trade.get('entry_idx', 'N/A')}")
                    
                    if trade_id_str == trade_id_search:
                        entry_idx = trade['entry_idx']
                        print(f"DEBUG: Found trade! Entry index: {entry_idx}")
                        start_idx = max(0, entry_idx - 500)
                        end_idx = min(len(self.df), entry_idx + 500)
                        jump_status = f"[OK] Jumped to trade {trade_id_search} (index {entry_idx})"
                        found = True
                        break
                
                if not found:
                    # Show available trade IDs for debugging
                    available_ids = [str(t['trade_id']) for t in trades_data[:10]]
                    jump_status = f"[ERROR] Trade '{trade_id_search}' not found. Try: {', '.join(available_ids)}..."
            
            elif triggered == 'trade-table' and active_cell:
                # Jump from table click
                row = active_cell['row']
                print(f"DEBUG: Table clicked - row {row}")
                if row < len(trades_data):
                    trade = trades_data[row]
                    entry_idx = trade['entry_idx']
                    print(f"DEBUG: Trade clicked - ID: {trade['trade_id']}, entry_idx: {entry_idx}")
                    start_idx = max(0, entry_idx - 500)
                    end_idx = min(len(self.df), entry_idx + 500)
                    jump_status = f"[OK] Navigated to trade {trade['trade_id']} (index {entry_idx})"
            
            # Create figure
            fig = self.create_main_figure(start_idx, end_idx)
            
            # Performance metrics
            total_trades = len(trades_data) if trades_data else 0
            profitable_trades = sum(1 for t in trades_data if t['pnl'] > 0) if trades_data else 0
            
            perf_metrics = f"""
            Data Points: {len(self.df):,} | 
            Trades: {total_trades} | 
            Profitable: {profitable_trades} | 
            Render Time: {self.render_time:.3f}s
            """
            
            # Stats panel
            if self.portfolio:
                try:
                    total_return = self.portfolio.total_return
                    if hasattr(total_return, 'iloc'):
                        total_return = total_return.iloc[0]
                    sharpe = self.portfolio.sharpe_ratio
                    if hasattr(sharpe, 'iloc'):
                        sharpe = sharpe.iloc[0]
                    
                    stats_html = f"""
                    Total Return: {total_return*100:.2f}%
                    Sharpe Ratio: {sharpe:.2f}
                    Win Rate: {(profitable_trades/total_trades*100) if total_trades > 0 else 0:.1f}%
                    """
                except:
                    stats_html = f"Win Rate: {(profitable_trades/total_trades*100) if total_trades > 0 else 0:.1f}%"
            else:
                stats_html = f"Win Rate: {(profitable_trades/total_trades*100) if total_trades > 0 else 0:.1f}%"
            
            # Chart info
            chart_info = "Native Plotly rendering | Use mouse wheel to zoom, drag to pan"
            
            return fig, jump_status, perf_metrics, stats_html, chart_info
    
    def run(self, port=8050, open_browser=True):
        """Run the dashboard."""
        print(f"\n{'='*60}")
        print(f"STARTING PLOTLY DASHBOARD - {self.symbol} - {self.strategy_name}")
        print(f"{'='*60}")
        print(f"[OK] Dashboard starting at http://localhost:{port}")
        print(f"[OK] Data: {len(self.df):,} bars loaded")
        print(f"[OK] Trades: {len(self.trades_df) if self.trades_df is not None else 0}")
        
        # Open browser automatically
        if open_browser:
            def open_browser_delayed():
                time.sleep(2)
                webbrowser.open(f'http://localhost:{port}')
            
            threading.Thread(target=open_browser_delayed).start()
        
        # Run the app
        self.app.run(debug=False, port=port)


def launch_plotly_dashboard(ohlcv_data, trades_csv_path=None, portfolio=None, symbol="ES", strategy_name="Strategy"):
    """Launch the Plotly dashboard with trading results.
    
    This function can be called from main.py after backtest completion.
    """
    print("\n" + "="*60)
    print("LAUNCHING PLOTLY DASHBOARD")
    print("="*60)
    
    # Load trades if CSV path provided
    trades_df = None
    if trades_csv_path and Path(trades_csv_path).exists():
        trades_df = pd.read_csv(trades_csv_path)
        print(f"[OK] Loaded {len(trades_df)} trades from {trades_csv_path}")
    
    # Create and run dashboard
    dashboard = PlotlyDashboardIntegrated(
        ohlcv_data=ohlcv_data,
        trades_df=trades_df,
        portfolio=portfolio,
        symbol=symbol,
        strategy_name=strategy_name
    )
    
    dashboard.run(port=8050, open_browser=True)
    
    return dashboard


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Plotly Dashboard Integration...")
    
    # Generate test data
    n_bars = 10000
    dates = pd.date_range(start='2020-01-01', periods=n_bars, freq='1min')
    
    # Generate price data
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, n_bars)
    price = 100 * np.exp(np.cumsum(returns))
    
    ohlcv_data = {
        'datetime': dates.values.astype(np.int64),
        'open': price * (1 + np.random.uniform(-0.002, 0.002, n_bars)),
        'high': price * (1 + np.random.uniform(0, 0.005, n_bars)),
        'low': price * (1 + np.random.uniform(-0.005, 0, n_bars)),
        'close': price,
        'volume': np.random.uniform(1000, 10000, n_bars)
    }
    
    # Launch dashboard
    dashboard = launch_plotly_dashboard(
        ohlcv_data=ohlcv_data,
        symbol="TEST",
        strategy_name="Test Strategy"
    )