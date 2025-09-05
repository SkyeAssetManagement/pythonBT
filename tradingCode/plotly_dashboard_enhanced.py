"""
Enhanced Plotly Dashboard with Intelligent Indicator Detection and Management
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx, ALL, MATCH
from dash.exceptions import PreventUpdate
from pathlib import Path
import time
import sys
import threading
import webbrowser
import re
import ast
import inspect
import importlib

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import VectorBT Pro
try:
    import vectorbtpro as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    print("Warning: VectorBT Pro not available. Some indicators may not work.")


class IndicatorDetector:
    """Detects indicators used in strategy code."""
    
    # Mapping of VBT indicator classes to their parameters
    VBT_INDICATORS = {
        'MA': {
            'name': 'Moving Average',
            'params': {'window': {'type': 'int', 'default': 20, 'min': 1, 'max': 500}},
            'function': lambda data, **params: vbt.MA.run(data['close'], **params).ma.values
        },
        'RSI': {
            'name': 'RSI',
            'params': {'window': {'type': 'int', 'default': 14, 'min': 2, 'max': 100}},
            'function': lambda data, **params: vbt.RSI.run(data['close'], **params).rsi.values
        },
        'MACD': {
            'name': 'MACD',
            'params': {
                'fast_window': {'type': 'int', 'default': 12, 'min': 2, 'max': 100},
                'slow_window': {'type': 'int', 'default': 26, 'min': 2, 'max': 200},
                'signal_window': {'type': 'int', 'default': 9, 'min': 1, 'max': 50}
            },
            'function': lambda data, **params: vbt.MACD.run(data['close'], **params).macd.values
        },
        'BB': {
            'name': 'Bollinger Bands',
            'params': {
                'window': {'type': 'int', 'default': 20, 'min': 2, 'max': 100},
                'alpha': {'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 4.0}
            },
            'function': lambda data, **params: vbt.BB.run(data['close'], **params).middle.values
        },
        'ATR': {
            'name': 'Average True Range',
            'params': {'window': {'type': 'int', 'default': 14, 'min': 1, 'max': 100}},
            'function': lambda data, **params: vbt.ATR.run(data['high'], data['low'], data['close'], **params).atr.values
        },
        'STOCH': {
            'name': 'Stochastic',
            'params': {
                'k_window': {'type': 'int', 'default': 14, 'min': 2, 'max': 100},
                'd_window': {'type': 'int', 'default': 3, 'min': 1, 'max': 50}
            },
            'function': lambda data, **params: vbt.STOCH.run(data['high'], data['low'], data['close'], **params).percent_k.values
        },
        'OBV': {
            'name': 'On Balance Volume',
            'params': {},
            'function': lambda data, **params: vbt.OBV.run(data['close'], data['volume']).obv.values
        },
        'VWAP': {
            'name': 'VWAP',
            'params': {},
            'function': lambda data, **params: vbt.VWAP.run(data['high'], data['low'], data['close'], data['volume']).vwap.values
        },
        'ADX': {
            'name': 'ADX',
            'params': {'window': {'type': 'int', 'default': 14, 'min': 2, 'max': 100}},
            'function': lambda data, **params: vbt.ADX.run(data['high'], data['low'], data['close'], **params).adx.values
        }
    }
    
    @classmethod
    def detect_indicators_from_strategy(cls, strategy_code: str) -> list:
        """
        Detect indicators used in strategy code.
        Returns list of detected indicators with their parameters.
        """
        detected = []
        
        # Parse for VectorBT indicators
        vbt_pattern = r'vbt\.(\w+)\.run\([^)]*\)'
        matches = re.findall(vbt_pattern, strategy_code)
        
        for indicator_name in matches:
            if indicator_name in cls.VBT_INDICATORS:
                indicator_info = cls.VBT_INDICATORS[indicator_name].copy()
                indicator_info['type'] = indicator_name
                detected.append(indicator_info)
        
        # Parse for MA with specific periods
        ma_pattern = r'vbt\.MA\.run\([^,]+,\s*(\d+)\)'
        ma_matches = re.findall(ma_pattern, strategy_code)
        
        for period in ma_matches:
            detected.append({
                'type': 'MA',
                'name': f'SMA {period}',
                'params': {'window': {'type': 'int', 'default': int(period), 'min': 1, 'max': 500}},
                'function': cls.VBT_INDICATORS['MA']['function']
            })
        
        # Parse for simple moving average calculations
        sma_pattern = r'(fast|slow)_period[\'"]?\s*[:\]]\s*(\d+)'
        sma_matches = re.findall(sma_pattern, strategy_code)
        
        for prefix, period in sma_matches:
            name = f'{prefix.capitalize()} SMA {period}'
            detected.append({
                'type': 'MA',
                'name': name,
                'params': {'window': {'type': 'int', 'default': int(period), 'min': 1, 'max': 500}},
                'function': cls.VBT_INDICATORS['MA']['function']
            })
        
        # Remove duplicates based on type and default parameters
        seen = set()
        unique_detected = []
        for ind in detected:
            # Create a unique key based on type and default params
            key = (ind['type'], tuple(p['default'] for p in ind.get('params', {}).values()))
            if key not in seen:
                seen.add(key)
                unique_detected.append(ind)
        
        return unique_detected


class EnhancedPlotlyDashboard:
    """Enhanced Plotly Dashboard with intelligent indicator management."""
    
    def __init__(self, ohlcv_data=None, trades_df=None, portfolio=None, 
                 symbol="ES", strategy_name="Strategy", strategy_code=None):
        """Initialize enhanced dashboard with trading results."""
        self.app = dash.Dash(__name__)
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.strategy_code = strategy_code
        
        # Store data
        self.ohlcv_data = ohlcv_data
        self.trades_df = trades_df
        self.portfolio = portfolio
        
        # Active indicators
        self.active_indicators = []
        
        # Convert to DataFrame if we have data
        if ohlcv_data:
            self.df = self._prepare_dataframe(ohlcv_data)
        else:
            self.df = None
        
        # Detect indicators from strategy
        if strategy_code:
            self.detected_indicators = IndicatorDetector.detect_indicators_from_strategy(strategy_code)
        else:
            self.detected_indicators = []
        
        # Performance metrics
        self.load_time = 0
        self.render_time = 0
        
        # Setup the dashboard
        if self.df is not None:
            self._apply_detected_indicators()
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
    
    def _apply_detected_indicators(self):
        """Apply detected indicators to the dataframe."""
        if not self.detected_indicators:
            # Default indicators if none detected
            self.detected_indicators = [
                {
                    'type': 'MA',
                    'name': 'SMA 20',
                    'params': {'window': {'type': 'int', 'default': 20, 'min': 1, 'max': 500}},
                    'function': IndicatorDetector.VBT_INDICATORS['MA']['function']
                },
                {
                    'type': 'MA',
                    'name': 'SMA 50',
                    'params': {'window': {'type': 'int', 'default': 50, 'min': 1, 'max': 500}},
                    'function': IndicatorDetector.VBT_INDICATORS['MA']['function']
                }
            ]
        
        # Apply each detected indicator
        for idx, indicator in enumerate(self.detected_indicators):
            indicator_id = f"indicator_{idx}"
            params = {k: v['default'] for k, v in indicator.get('params', {}).items()}
            
            try:
                if VBT_AVAILABLE and 'function' in indicator:
                    # Use VectorBT Pro to calculate indicator
                    values = indicator['function'](self.ohlcv_data, **params)
                    self.df[indicator_id] = values
                    self.active_indicators.append({
                        'id': indicator_id,
                        'name': indicator['name'],
                        'type': indicator['type'],
                        'params': params,
                        'visible': True
                    })
                else:
                    # Fallback to simple calculation for MA
                    if indicator['type'] == 'MA':
                        window = params.get('window', 20)
                        self.df[indicator_id] = self.df['close'].rolling(window=window, min_periods=1).mean()
                        self.active_indicators.append({
                            'id': indicator_id,
                            'name': indicator['name'],
                            'type': 'MA',
                            'params': params,
                            'visible': True
                        })
            except Exception as e:
                print(f"Warning: Could not calculate {indicator['name']}: {e}")
        
        # Add equity curve if portfolio available
        if self.portfolio is not None:
            try:
                if hasattr(self.portfolio, 'value'):
                    equity = self.portfolio.value
                    if hasattr(equity, 'values'):
                        equity_vals = equity.values
                        if len(equity_vals.shape) > 1:
                            equity_vals = np.sum(equity_vals, axis=1)
                    else:
                        equity_vals = equity
                    
                    if len(equity_vals) == len(self.df):
                        initial_capital = equity_vals[0] if len(equity_vals) > 0 else 10000
                        self.df['equity'] = equity_vals - initial_capital
            except Exception as e:
                print(f"Warning: Could not add equity curve: {e}")
    
    def _prepare_trades_data(self):
        """Prepare trades data for display."""
        if self.trades_df is None or self.trades_df.empty:
            return pd.DataFrame()
        
        trades = self.trades_df.copy()
        display_trades = pd.DataFrame()
        
        # Map columns flexibly
        if 'Exit Trade Id' in trades.columns:
            display_trades['trade_id'] = trades['Exit Trade Id'].astype(str)
        else:
            display_trades['trade_id'] = [str(i) for i in range(len(trades))]
        
        # Map other columns
        for col in ['Direction', 'direction', 'Side', 'side']:
            if col in trades.columns:
                display_trades['side'] = trades[col]
                break
        else:
            display_trades['side'] = 'Long'
        
        # Entry/Exit prices and indices
        for col in ['Avg Entry Price', 'entry_price', 'Entry Price']:
            if col in trades.columns:
                display_trades['entry_price'] = trades[col]
                break
        
        for col in ['Avg Exit Price', 'exit_price', 'Exit Price']:
            if col in trades.columns:
                display_trades['exit_price'] = trades[col]
                break
        
        for col in ['PnL', 'pnl', 'Profit']:
            if col in trades.columns:
                display_trades['pnl'] = trades[col]
                break
        
        # Entry/Exit indices
        for col in ['Entry Index', 'EntryTime', 'entry_time', 'entry_index']:
            if col in trades.columns:
                display_trades['entry_idx'] = pd.to_numeric(trades[col], errors='coerce').fillna(0).astype(int)
                break
        
        for col in ['Exit Index', 'ExitTime', 'exit_time', 'exit_index']:
            if col in trades.columns:
                display_trades['exit_idx'] = pd.to_numeric(trades[col], errors='coerce').fillna(display_trades['entry_idx'] + 10).astype(int)
                break
        
        return display_trades
    
    def _create_indicator_controls(self):
        """Create the indicator management panel."""
        available_indicators = list(IndicatorDetector.VBT_INDICATORS.keys())
        
        return html.Div([
            html.H3("Indicator Management", style={'color': 'white', 'marginBottom': '10px'}),
            
            # Add indicator section
            html.Div([
                html.Label("Add Indicator:", style={'color': 'white', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='indicator-dropdown',
                    options=[{'label': IndicatorDetector.VBT_INDICATORS[ind]['name'], 'value': ind} 
                            for ind in available_indicators],
                    value=None,
                    placeholder="Select indicator to add",
                    style={'width': '200px', 'display': 'inline-block', 'marginRight': '10px'}
                ),
                html.Button('Add', id='add-indicator-btn', n_clicks=0, 
                          style={'marginRight': '10px'}),
                html.Button('Clear All', id='clear-indicators-btn', n_clicks=0,
                          style={'backgroundColor': '#ff4444'})
            ], style={'marginBottom': '15px'}),
            
            # Dynamic parameter inputs section
            html.Div(id='indicator-params-section', style={'marginBottom': '15px'}),
            
            # Active indicators list
            html.Div([
                html.Label("Active Indicators:", style={'color': 'white', 'marginBottom': '5px'}),
                html.Div(id='active-indicators-list', children=[
                    html.Div([
                        html.Span(f"• {ind['name']}", style={'color': '#00ff00', 'marginRight': '10px'}),
                        html.Button('×', id={'type': 'remove-indicator', 'index': idx}, 
                                  style={'color': 'red', 'backgroundColor': 'transparent', 
                                        'border': 'none', 'cursor': 'pointer'})
                    ], style={'padding': '2px'})
                    for idx, ind in enumerate(self.active_indicators)
                ])
            ], style={'maxHeight': '150px', 'overflowY': 'auto', 'backgroundColor': '#3a3a3a', 
                     'padding': '10px', 'borderRadius': '5px'})
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px'})
    
    def _setup_layout(self):
        """Setup the enhanced Dash layout."""
        trades_display = self._prepare_trades_data()
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1(f"Enhanced Plotly Dashboard - {self.symbol} - {self.strategy_name}", 
                       style={'color': 'white', 'margin': '10px'}),
                html.Div(id='performance-metrics', style={'color': '#00ff00', 'margin': '10px'})
            ], style={'backgroundColor': '#1e1e1e', 'padding': '10px'}),
            
            # Store components
            dcc.Store(id='indicators-store', data=self.active_indicators),
            dcc.Store(id='trades-store', data=trades_display.to_dict('records') if not trades_display.empty else []),
            
            # Main container
            html.Div([
                # Left panel - Indicator controls and trade list
                html.Div([
                    # Indicator management panel (replaces trade input)
                    self._create_indicator_controls(),
                    
                    # Stats panel
                    html.Div([
                        html.H3("Performance Stats", style={'color': 'white'}),
                        html.Div(id='stats-panel', style={'color': 'white', 'padding': '10px'})
                    ], style={'backgroundColor': '#2a2a2a', 'marginBottom': '10px', 'padding': '10px'}),
                    
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
                                {'name': 'PnL', 'id': 'pnl', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                            ],
                            data=trades_display.to_dict('records') if not trades_display.empty else [],
                            style_cell={'textAlign': 'center', 'backgroundColor': '#2a2a2a', 'color': 'white'},
                            style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold'},
                            virtualization=True,
                            fixed_rows={'headers': True},
                            style_table={'height': '300px', 'overflowY': 'auto'}
                        )
                    ], style={'padding': '10px', 'backgroundColor': '#2a2a2a'})
                ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Right panel - Chart
                html.Div([
                    dcc.Graph(
                        id='main-chart',
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'scrollZoom': True,
                            'doubleClick': 'autosize'
                        },
                        style={'height': '800px'}
                    ),
                    html.Div(id='chart-info', style={'color': 'white', 'padding': '10px'})
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'backgroundColor': '#1e1e1e', 'minHeight': '900px'})
        ], style={'height': '100vh', 'overflow': 'hidden'})
    
    def create_main_figure(self, indicators_data=None):
        """Create the main chart figure with dynamic indicators."""
        start_time = time.time()
        
        # Default to showing last 2000 bars
        start_idx = max(0, len(self.df) - 2000)
        end_idx = len(self.df)
        df_view = self.df.iloc[start_idx:end_idx]
        
        # Determine subplot configuration
        has_equity = 'equity' in self.df.columns
        n_rows = 3 if has_equity else 2
        row_heights = [0.70, 0.10, 0.15] if has_equity else [0.80, 0.15]
        subplot_titles = ('Price Chart with Indicators', 'Volume', 'Net Profit') if has_equity else ('Price Chart', 'Volume')
        
        # Create figure
        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
        
        # Add candlestick chart
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
        
        # Add active indicators
        colors = ['#ffaa00', '#00aaff', '#ff00ff', '#00ffff', '#ffff00', '#ff6600']
        if indicators_data:
            for idx, indicator in enumerate(indicators_data):
                if indicator['visible'] and indicator['id'] in df_view.columns:
                    color = colors[idx % len(colors)]
                    fig.add_trace(
                        go.Scatter(
                            x=df_view['datetime'],
                            y=df_view[indicator['id']],
                            name=indicator['name'],
                            line=dict(color=color, width=1)
                        ),
                        row=1, col=1
                    )
        
        # Add volume bars
        colors_vol = ['#ff0000' if close < open else '#00ff00' 
                     for close, open in zip(df_view['close'], df_view['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df_view['datetime'],
                y=df_view['volume'],
                name='Volume',
                marker_color=colors_vol,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # Add equity curve if available
        if has_equity:
            profit_color = '#00ff00' if df_view['equity'].iloc[-1] >= 0 else '#ff0000'
            fill_color = 'rgba(0, 255, 0, 0.1)' if df_view['equity'].iloc[-1] >= 0 else 'rgba(255, 0, 0, 0.1)'
            
            fig.add_trace(
                go.Scatter(
                    x=df_view['datetime'],
                    y=df_view['equity'],
                    name='Net Profit',
                    line=dict(color=profit_color, width=2),
                    fill='tozeroy',
                    fillcolor=fill_color
                ),
                row=3, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1, opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title=f'{self.symbol} - {self.strategy_name} - Dynamic Indicators',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)'),
            hovermode='x unified',
            yaxis=dict(autorange=True, fixedrange=False),
            yaxis2=dict(autorange=True, fixedrange=False),
            yaxis3=dict(autorange=True, fixedrange=False) if has_equity else None
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='#333333')
        fig.update_yaxes(showgrid=True, gridcolor='#333333', autorange=True)
        
        self.render_time = time.time() - start_time
        return fig
    
    def _setup_callbacks(self):
        """Setup enhanced Dash callbacks."""
        
        @self.app.callback(
            Output('indicator-params-section', 'children'),
            Input('indicator-dropdown', 'value')
        )
        def update_parameter_inputs(selected_indicator):
            """Dynamically create parameter inputs for selected indicator."""
            if not selected_indicator:
                return []
            
            indicator_info = IndicatorDetector.VBT_INDICATORS.get(selected_indicator, {})
            params = indicator_info.get('params', {})
            
            if not params:
                return html.Div("No parameters for this indicator", style={'color': 'white'})
            
            param_inputs = []
            for param_name, param_info in params.items():
                param_inputs.append(html.Div([
                    html.Label(f"{param_name.replace('_', ' ').title()}:", 
                             style={'color': 'white', 'marginRight': '10px', 'width': '120px', 
                                   'display': 'inline-block'}),
                    dcc.Input(
                        id={'type': 'param-input', 'param': param_name},
                        type='number' if param_info['type'] == 'int' else 'text',
                        value=param_info['default'],
                        min=param_info.get('min'),
                        max=param_info.get('max'),
                        style={'width': '100px', 'marginRight': '10px'}
                    ),
                    html.Span(f"({param_info.get('min', 'N/A')} - {param_info.get('max', 'N/A')})", 
                            style={'color': '#888', 'fontSize': '12px'})
                ], style={'marginBottom': '5px'}))
            
            return html.Div(param_inputs)
        
        @self.app.callback(
            [Output('indicators-store', 'data'),
             Output('active-indicators-list', 'children'),
             Output('main-chart', 'figure')],
            [Input('add-indicator-btn', 'n_clicks'),
             Input('clear-indicators-btn', 'n_clicks'),
             Input({'type': 'remove-indicator', 'index': ALL}, 'n_clicks')],
            [State('indicator-dropdown', 'value'),
             State({'type': 'param-input', 'param': ALL}, 'value'),
             State({'type': 'param-input', 'param': ALL}, 'id'),
             State('indicators-store', 'data')]
        )
        def manage_indicators(add_clicks, clear_clicks, remove_clicks, 
                             selected_indicator, param_values, param_ids, current_indicators):
            """Handle adding, removing, and clearing indicators."""
            
            if not current_indicators:
                current_indicators = self.active_indicators.copy()
            
            triggered = ctx.triggered_id
            
            if triggered == 'clear-indicators-btn':
                # Clear all indicators
                current_indicators = []
            
            elif triggered == 'add-indicator-btn' and selected_indicator:
                # Add new indicator
                indicator_info = IndicatorDetector.VBT_INDICATORS.get(selected_indicator, {})
                
                # Collect parameters
                params = {}
                for param_id, value in zip(param_ids, param_values):
                    if value is not None:
                        params[param_id['param']] = value
                
                # Calculate indicator values
                new_id = f"indicator_{len(current_indicators)}"
                
                try:
                    if VBT_AVAILABLE and 'function' in indicator_info:
                        values = indicator_info['function'](self.ohlcv_data, **params)
                        self.df[new_id] = values
                    else:
                        # Fallback for MA
                        if selected_indicator == 'MA':
                            window = params.get('window', 20)
                            self.df[new_id] = self.df['close'].rolling(window=window, min_periods=1).mean()
                    
                    # Add to active indicators
                    current_indicators.append({
                        'id': new_id,
                        'name': f"{indicator_info['name']} ({', '.join(f'{k}={v}' for k, v in params.items())})",
                        'type': selected_indicator,
                        'params': params,
                        'visible': True
                    })
                except Exception as e:
                    print(f"Error adding indicator: {e}")
            
            elif isinstance(triggered, dict) and triggered.get('type') == 'remove-indicator':
                # Remove specific indicator
                idx_to_remove = triggered['index']
                if 0 <= idx_to_remove < len(current_indicators):
                    current_indicators.pop(idx_to_remove)
            
            # Update indicator list display
            indicators_display = [
                html.Div([
                    html.Span(f"• {ind['name']}", style={'color': '#00ff00', 'marginRight': '10px'}),
                    html.Button('×', id={'type': 'remove-indicator', 'index': idx}, 
                              style={'color': 'red', 'backgroundColor': 'transparent', 
                                    'border': 'none', 'cursor': 'pointer'})
                ], style={'padding': '2px'})
                for idx, ind in enumerate(current_indicators)
            ]
            
            # Update chart
            fig = self.create_main_figure(current_indicators)
            
            return current_indicators, indicators_display, fig
        
        @self.app.callback(
            [Output('performance-metrics', 'children'),
             Output('stats-panel', 'children')],
            Input('main-chart', 'figure')
        )
        def update_metrics(figure):
            """Update performance metrics display."""
            
            trades_data = self._prepare_trades_data()
            total_trades = len(trades_data) if not trades_data.empty else 0
            profitable_trades = sum(trades_data['pnl'] > 0) if not trades_data.empty else 0
            
            perf_metrics = f"""
            Data Points: {len(self.df):,} | 
            Trades: {total_trades} | 
            Indicators: {len(self.active_indicators)} | 
            Render Time: {self.render_time:.3f}s
            """
            
            if self.portfolio and total_trades > 0:
                win_rate = (profitable_trades/total_trades*100)
                stats_html = f"Win Rate: {win_rate:.1f}%"
            else:
                stats_html = "No trades to analyze"
            
            return perf_metrics, stats_html
    
    def run(self, port=8050, open_browser=True):
        """Run the enhanced dashboard."""
        print(f"\n{'='*60}")
        print(f"ENHANCED PLOTLY DASHBOARD - {self.symbol} - {self.strategy_name}")
        print(f"{'='*60}")
        print(f"[OK] Dashboard starting at http://localhost:{port}")
        print(f"[OK] Data: {len(self.df):,} bars loaded")
        print(f"[OK] Detected {len(self.detected_indicators)} indicators from strategy")
        print(f"[OK] Active indicators: {', '.join(ind['name'] for ind in self.active_indicators)}")
        
        if open_browser:
            def open_browser_delayed():
                time.sleep(2)
                webbrowser.open(f'http://localhost:{port}')
            
            threading.Thread(target=open_browser_delayed).start()
        
        self.app.run(debug=False, port=port)


def launch_enhanced_dashboard(ohlcv_data, trades_csv_path=None, portfolio=None, 
                             symbol="ES", strategy_name="Strategy", strategy_path=None):
    """Launch the enhanced Plotly dashboard."""
    
    print("\n" + "="*60)
    print("LAUNCHING ENHANCED PLOTLY DASHBOARD")
    print("="*60)
    
    # Load strategy code for indicator detection
    strategy_code = None
    if strategy_path and Path(strategy_path).exists():
        with open(strategy_path, 'r') as f:
            strategy_code = f.read()
        print(f"[OK] Loaded strategy code from {strategy_path}")
    
    # Load trades if CSV path provided
    trades_df = None
    if trades_csv_path and Path(trades_csv_path).exists():
        trades_df = pd.read_csv(trades_csv_path)
        print(f"[OK] Loaded {len(trades_df)} trades")
    
    # Create and run dashboard
    dashboard = EnhancedPlotlyDashboard(
        ohlcv_data=ohlcv_data,
        trades_df=trades_df,
        portfolio=portfolio,
        symbol=symbol,
        strategy_name=strategy_name,
        strategy_code=strategy_code
    )
    
    dashboard.run(port=8050, open_browser=True)
    
    return dashboard


if __name__ == "__main__":
    # Test with sample data
    print("Testing Enhanced Plotly Dashboard...")
    
    n_bars = 10000
    dates = pd.date_range(start='2020-01-01', periods=n_bars, freq='1min')
    
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
    
    # Sample strategy code for testing
    test_strategy_code = """
    fast_ma = vbt.MA.run(close_prices, 20).ma.values
    slow_ma = vbt.MA.run(close_prices, 50).ma.values
    rsi = vbt.RSI.run(close_prices, 14).rsi.values
    """
    
    dashboard = EnhancedPlotlyDashboard(
        ohlcv_data=ohlcv_data,
        symbol="TEST",
        strategy_name="Test Strategy",
        strategy_code=test_strategy_code
    )
    
    dashboard.run()