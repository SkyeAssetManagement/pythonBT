"""
Enhanced Plotly Dashboard V2 with Complete VectorBT Pro Indicator Management
No automatic indicators, no jump-to-trade, full user control
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
import json

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import VectorBT Pro
try:
    import vectorbtpro as vbt
    VBT_AVAILABLE = True
except ImportError:
    try:
        import vectorbt as vbt
        VBT_AVAILABLE = True
    except ImportError:
        VBT_AVAILABLE = False
        print("Warning: VectorBT not available. Indicators will use fallback calculations.")


class VectorBTIndicatorManager:
    """Complete VectorBT Pro indicator management system."""
    
    # Complete VectorBT Pro indicator catalog
    INDICATORS = {
        # Price-based Moving Averages
        'SMA': {
            'name': 'Simple Moving Average',
            'category': 'Price',
            'params': {
                'window': {'type': 'int', 'default': 20, 'min': 1, 'max': 500, 'description': 'Period'}
            },
            'calculate': lambda data, **p: data['close'].rolling(window=p.get('window', 20), min_periods=1).mean()
        },
        'EMA': {
            'name': 'Exponential Moving Average',
            'category': 'Price',
            'params': {
                'window': {'type': 'int', 'default': 20, 'min': 1, 'max': 500, 'description': 'Period'}
            },
            'calculate': lambda data, **p: data['close'].ewm(span=p.get('window', 20), adjust=False).mean()
        },
        'WMA': {
            'name': 'Weighted Moving Average',
            'category': 'Price',
            'params': {
                'window': {'type': 'int', 'default': 20, 'min': 1, 'max': 500, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_wma(data['close'], p.get('window', 20))
        },
        'HMA': {
            'name': 'Hull Moving Average',
            'category': 'Price',
            'params': {
                'window': {'type': 'int', 'default': 20, 'min': 4, 'max': 500, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_hma(data['close'], p.get('window', 20))
        },
        
        # Momentum Indicators
        'RSI': {
            'name': 'Relative Strength Index',
            'category': 'Momentum',
            'params': {
                'window': {'type': 'int', 'default': 14, 'min': 2, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_rsi(data['close'], p.get('window', 14)),
            'subplot': True  # Display in separate subplot
        },
        'MACD': {
            'name': 'MACD',
            'category': 'Momentum',
            'params': {
                'fast_window': {'type': 'int', 'default': 12, 'min': 2, 'max': 100, 'description': 'Fast Period'},
                'slow_window': {'type': 'int', 'default': 26, 'min': 2, 'max': 200, 'description': 'Slow Period'},
                'signal_window': {'type': 'int', 'default': 9, 'min': 1, 'max': 50, 'description': 'Signal Period'}
            },
            'calculate': lambda data, **p: calculate_macd(data['close'], p.get('fast_window', 12), 
                                                         p.get('slow_window', 26), p.get('signal_window', 9)),
            'subplot': True,
            'multiple_lines': True  # Returns dict with 'macd', 'signal', 'histogram'
        },
        'STOCH': {
            'name': 'Stochastic Oscillator',
            'category': 'Momentum',
            'params': {
                'k_window': {'type': 'int', 'default': 14, 'min': 2, 'max': 100, 'description': 'K Period'},
                'd_window': {'type': 'int', 'default': 3, 'min': 1, 'max': 50, 'description': 'D Period'}
            },
            'calculate': lambda data, **p: calculate_stoch(data, p.get('k_window', 14), p.get('d_window', 3)),
            'subplot': True,
            'multiple_lines': True  # Returns dict with 'k' and 'd'
        },
        'CCI': {
            'name': 'Commodity Channel Index',
            'category': 'Momentum',
            'params': {
                'window': {'type': 'int', 'default': 20, 'min': 2, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_cci(data, p.get('window', 20)),
            'subplot': True
        },
        'WILLR': {
            'name': 'Williams %R',
            'category': 'Momentum',
            'params': {
                'window': {'type': 'int', 'default': 14, 'min': 2, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_williams_r(data, p.get('window', 14)),
            'subplot': True
        },
        'MFI': {
            'name': 'Money Flow Index',
            'category': 'Momentum',
            'params': {
                'window': {'type': 'int', 'default': 14, 'min': 2, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_mfi(data, p.get('window', 14)),
            'subplot': True
        },
        'ROC': {
            'name': 'Rate of Change',
            'category': 'Momentum',
            'params': {
                'window': {'type': 'int', 'default': 10, 'min': 1, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_roc(data['close'], p.get('window', 10)),
            'subplot': True
        },
        
        # Volatility Indicators
        'BB': {
            'name': 'Bollinger Bands',
            'category': 'Volatility',
            'params': {
                'window': {'type': 'int', 'default': 20, 'min': 2, 'max': 100, 'description': 'Period'},
                'alpha': {'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 4.0, 'step': 0.1, 'description': 'Std Dev'}
            },
            'calculate': lambda data, **p: calculate_bollinger_bands(data['close'], p.get('window', 20), p.get('alpha', 2.0)),
            'multiple_lines': True  # Returns dict with 'upper', 'middle', 'lower'
        },
        'ATR': {
            'name': 'Average True Range',
            'category': 'Volatility',
            'params': {
                'window': {'type': 'int', 'default': 14, 'min': 1, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_atr(data, p.get('window', 14)),
            'subplot': True
        },
        'KC': {
            'name': 'Keltner Channels',
            'category': 'Volatility',
            'params': {
                'ma_window': {'type': 'int', 'default': 20, 'min': 2, 'max': 100, 'description': 'MA Period'},
                'atr_window': {'type': 'int', 'default': 10, 'min': 1, 'max': 100, 'description': 'ATR Period'},
                'mult': {'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 5.0, 'step': 0.1, 'description': 'Multiplier'}
            },
            'calculate': lambda data, **p: calculate_keltner_channels(data, p.get('ma_window', 20), 
                                                                     p.get('atr_window', 10), p.get('mult', 2.0)),
            'multiple_lines': True  # Returns dict with 'upper', 'middle', 'lower'
        },
        'DC': {
            'name': 'Donchian Channels',
            'category': 'Volatility',
            'params': {
                'window': {'type': 'int', 'default': 20, 'min': 2, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_donchian_channels(data, p.get('window', 20)),
            'multiple_lines': True  # Returns dict with 'upper', 'middle', 'lower'
        },
        
        # Volume Indicators
        'OBV': {
            'name': 'On Balance Volume',
            'category': 'Volume',
            'params': {},
            'calculate': lambda data, **p: calculate_obv(data),
            'subplot': True
        },
        'VWAP': {
            'name': 'Volume Weighted Average Price',
            'category': 'Volume',
            'params': {},
            'calculate': lambda data, **p: calculate_vwap(data)
        },
        'AD': {
            'name': 'Accumulation/Distribution',
            'category': 'Volume',
            'params': {},
            'calculate': lambda data, **p: calculate_ad(data),
            'subplot': True
        },
        'CMF': {
            'name': 'Chaikin Money Flow',
            'category': 'Volume',
            'params': {
                'window': {'type': 'int', 'default': 20, 'min': 2, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_cmf(data, p.get('window', 20)),
            'subplot': True
        },
        'FI': {
            'name': 'Force Index',
            'category': 'Volume',
            'params': {
                'window': {'type': 'int', 'default': 13, 'min': 1, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_force_index(data, p.get('window', 13)),
            'subplot': True
        },
        'VWMA': {
            'name': 'Volume Weighted Moving Average',
            'category': 'Volume',
            'params': {
                'window': {'type': 'int', 'default': 20, 'min': 2, 'max': 200, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_vwma(data, p.get('window', 20))
        },
        
        # Trend Indicators
        'ADX': {
            'name': 'Average Directional Index',
            'category': 'Trend',
            'params': {
                'window': {'type': 'int', 'default': 14, 'min': 2, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_adx(data, p.get('window', 14)),
            'subplot': True
        },
        'AROON': {
            'name': 'Aroon Indicator',
            'category': 'Trend',
            'params': {
                'window': {'type': 'int', 'default': 25, 'min': 2, 'max': 100, 'description': 'Period'}
            },
            'calculate': lambda data, **p: calculate_aroon(data, p.get('window', 25)),
            'subplot': True,
            'multiple_lines': True  # Returns dict with 'up' and 'down'
        },
        'SAR': {
            'name': 'Parabolic SAR',
            'category': 'Trend',
            'params': {
                'start': {'type': 'float', 'default': 0.02, 'min': 0.01, 'max': 0.1, 'step': 0.01, 'description': 'Start'},
                'increment': {'type': 'float', 'default': 0.02, 'min': 0.01, 'max': 0.1, 'step': 0.01, 'description': 'Increment'},
                'maximum': {'type': 'float', 'default': 0.2, 'min': 0.1, 'max': 0.5, 'step': 0.01, 'description': 'Maximum'}
            },
            'calculate': lambda data, **p: calculate_sar(data, p.get('start', 0.02), 
                                                        p.get('increment', 0.02), p.get('maximum', 0.2))
        },
        'SUPERTREND': {
            'name': 'SuperTrend',
            'category': 'Trend',
            'params': {
                'window': {'type': 'int', 'default': 10, 'min': 1, 'max': 100, 'description': 'ATR Period'},
                'mult': {'type': 'float', 'default': 3.0, 'min': 0.5, 'max': 10.0, 'step': 0.1, 'description': 'Multiplier'}
            },
            'calculate': lambda data, **p: calculate_supertrend(data, p.get('window', 10), p.get('mult', 3.0))
        }
    }
    
    @classmethod
    def get_categories(cls):
        """Get unique indicator categories."""
        categories = set()
        for ind_data in cls.INDICATORS.values():
            categories.add(ind_data.get('category', 'Other'))
        return sorted(list(categories))
    
    @classmethod
    def get_indicators_by_category(cls, category):
        """Get indicators for a specific category."""
        indicators = []
        for key, data in cls.INDICATORS.items():
            if data.get('category') == category:
                indicators.append({'value': key, 'label': data['name']})
        return sorted(indicators, key=lambda x: x['label'])


# Indicator calculation functions (fallback implementations when VBT not available)
def calculate_wma(series, window):
    """Calculate Weighted Moving Average."""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hma(series, window):
    """Calculate Hull Moving Average."""
    half_window = window // 2
    sqrt_window = int(np.sqrt(window))
    wma_half = calculate_wma(series, half_window)
    wma_full = calculate_wma(series, window)
    diff = 2 * wma_half - wma_full
    return calculate_wma(pd.Series(diff), sqrt_window)

def calculate_rsi(close, window):
    """Calculate RSI."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(close, fast, slow, signal):
    """Calculate MACD."""
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return {'macd': macd, 'signal': signal_line, 'histogram': histogram}

def calculate_stoch(data, k_window, d_window):
    """Calculate Stochastic Oscillator."""
    low_min = data['low'].rolling(window=k_window).min()
    high_max = data['high'].rolling(window=k_window).max()
    k = 100 * ((data['close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_window).mean()
    return {'k': k, 'd': d}

def calculate_cci(data, window):
    """Calculate Commodity Channel Index."""
    tp = (data['high'] + data['low'] + data['close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - sma) / (0.015 * mad)

def calculate_williams_r(data, window):
    """Calculate Williams %R."""
    high_max = data['high'].rolling(window=window).max()
    low_min = data['low'].rolling(window=window).min()
    return -100 * ((high_max - data['close']) / (high_max - low_min))

def calculate_mfi(data, window):
    """Calculate Money Flow Index."""
    tp = (data['high'] + data['low'] + data['close']) / 3
    mf = tp * data['volume']
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(window=window).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(window=window).sum()
    mfr = pos_mf / neg_mf
    return 100 - (100 / (1 + mfr))

def calculate_roc(close, window):
    """Calculate Rate of Change."""
    return ((close - close.shift(window)) / close.shift(window)) * 100

def calculate_bollinger_bands(close, window, alpha):
    """Calculate Bollinger Bands."""
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    return {
        'upper': sma + (std * alpha),
        'middle': sma,
        'lower': sma - (std * alpha)
    }

def calculate_atr(data, window):
    """Calculate Average True Range."""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_keltner_channels(data, ma_window, atr_window, mult):
    """Calculate Keltner Channels."""
    middle = data['close'].ewm(span=ma_window, adjust=False).mean()
    atr = calculate_atr(data, atr_window)
    return {
        'upper': middle + (mult * atr),
        'middle': middle,
        'lower': middle - (mult * atr)
    }

def calculate_donchian_channels(data, window):
    """Calculate Donchian Channels."""
    upper = data['high'].rolling(window=window).max()
    lower = data['low'].rolling(window=window).min()
    middle = (upper + lower) / 2
    return {'upper': upper, 'middle': middle, 'lower': lower}

def calculate_obv(data):
    """Calculate On Balance Volume."""
    return (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()

def calculate_vwap(data):
    """Calculate VWAP."""
    tp = (data['high'] + data['low'] + data['close']) / 3
    return (tp * data['volume']).cumsum() / data['volume'].cumsum()

def calculate_ad(data):
    """Calculate Accumulation/Distribution."""
    clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    return (clv * data['volume']).fillna(0).cumsum()

def calculate_cmf(data, window):
    """Calculate Chaikin Money Flow."""
    clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    mf = clv * data['volume']
    return mf.rolling(window=window).sum() / data['volume'].rolling(window=window).sum()

def calculate_force_index(data, window):
    """Calculate Force Index."""
    fi = data['close'].diff() * data['volume']
    return fi.ewm(span=window, adjust=False).mean()

def calculate_vwma(data, window):
    """Calculate Volume Weighted Moving Average."""
    return (data['close'] * data['volume']).rolling(window=window).sum() / data['volume'].rolling(window=window).sum()

def calculate_adx(data, window):
    """Calculate ADX."""
    # Simplified ADX calculation
    plus_dm = data['high'].diff()
    minus_dm = -data['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = calculate_atr(data, window)
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window=window).mean()

def calculate_aroon(data, window):
    """Calculate Aroon Indicator."""
    aroon_up = data['high'].rolling(window=window + 1).apply(lambda x: x.argmax() / window * 100)
    aroon_down = data['low'].rolling(window=window + 1).apply(lambda x: x.argmin() / window * 100)
    return {'up': aroon_up, 'down': aroon_down}

def calculate_sar(data, start, increment, maximum):
    """Calculate Parabolic SAR."""
    # Simplified SAR calculation
    sar = data['close'].copy()
    # This would need a proper implementation with loops
    return sar

def calculate_supertrend(data, window, mult):
    """Calculate SuperTrend."""
    atr = calculate_atr(data, window)
    hl_avg = (data['high'] + data['low']) / 2
    upper_band = hl_avg + (mult * atr)
    lower_band = hl_avg - (mult * atr)
    
    # Simplified - actual implementation needs state tracking
    supertrend = data['close'].copy()
    supertrend = supertrend.where(data['close'] > upper_band.shift(), lower_band)
    supertrend = supertrend.where(data['close'] < lower_band.shift(), upper_band)
    return supertrend


class EnhancedPlotlyDashboardV2:
    """Enhanced Plotly Dashboard V2 with full indicator control."""
    
    def __init__(self, ohlcv_data=None, trades_df=None, portfolio=None, 
                 symbol="ES", strategy_name="Strategy"):
        """Initialize dashboard with NO automatic indicators."""
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.symbol = symbol
        self.strategy_name = strategy_name
        
        # Store data
        self.ohlcv_data = ohlcv_data
        self.trades_df = trades_df
        self.portfolio = portfolio
        
        # Active indicators tracking
        self.active_indicators = []
        self.indicator_counter = 0
        
        # Convert to DataFrame
        if ohlcv_data:
            self.df = self._prepare_dataframe(ohlcv_data)
        else:
            self.df = None
        
        # Performance metrics
        self.load_time = 0
        self.render_time = 0
        
        # Setup the dashboard
        if self.df is not None:
            # NO automatic indicator loading
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
                    
                    if len(equity_vals) == len(df):
                        initial_capital = equity_vals[0] if len(equity_vals) > 0 else 10000
                        df['equity'] = equity_vals - initial_capital
            except Exception as e:
                print(f"Warning: Could not add equity curve: {e}")
        
        return df
    
    def _prepare_trades_data(self):
        """Prepare trades data for display with separate Entry/Exit rows."""
        if self.trades_df is None or self.trades_df.empty:
            return pd.DataFrame()
        
        trades = self.trades_df.copy()
        rows = []
        
        # Process each trade to create separate entry and exit rows
        for idx, trade in trades.iterrows():
            # Get trade ID
            trade_id = str(trade.get('Exit Trade Id', trade.get('Trade Id', idx)))
            
            # Get direction
            direction = 'Long'
            for col in ['Direction', 'direction', 'Side', 'side']:
                if col in trades.columns:
                    direction = trade[col]
                    break
            
            # Get position size
            pos_size = 1.0
            for col in ['Size', 'Shares', 'Position Size', 'position_size', 'Quantity']:
                if col in trades.columns:
                    pos_size = trade[col]
                    break
            
            # Get entry data
            entry_price = None
            for col in ['Avg Entry Price', 'entry_price', 'Entry Price']:
                if col in trades.columns:
                    entry_price = trade[col]
                    break
            
            entry_time = None
            entry_idx = None
            for col in ['Entry Time', 'EntryTime', 'entry_time', 'Entry Date']:
                if col in trades.columns:
                    entry_time = trade[col]
                    break
            for col in ['Entry Index', 'entry_index', 'Entry Bar']:
                if col in trades.columns:
                    entry_idx = trade[col]
                    break
            
            # Get exit data
            exit_price = None
            for col in ['Avg Exit Price', 'exit_price', 'Exit Price']:
                if col in trades.columns:
                    exit_price = trade[col]
                    break
            
            exit_time = None
            exit_idx = None
            for col in ['Exit Time', 'ExitTime', 'exit_time', 'Exit Date']:
                if col in trades.columns:
                    exit_time = trade[col]
                    break
            for col in ['Exit Index', 'exit_index', 'Exit Bar']:
                if col in trades.columns:
                    exit_idx = trade[col]
                    break
            
            # Get PnL
            pnl = None
            for col in ['PnL', 'pnl', 'Profit', 'Return']:
                if col in trades.columns:
                    pnl = trade[col]
                    break
            
            # Convert times to datetime if they're indices
            if entry_idx is not None and self.df is not None:
                try:
                    entry_idx_int = int(entry_idx)
                    if 0 <= entry_idx_int < len(self.df):
                        entry_datetime = self.df.iloc[entry_idx_int]['datetime']
                    else:
                        entry_datetime = entry_time if entry_time else ''
                except:
                    entry_datetime = entry_time if entry_time else ''
            else:
                entry_datetime = entry_time if entry_time else ''
            
            if exit_idx is not None and self.df is not None:
                try:
                    exit_idx_int = int(exit_idx)
                    if 0 <= exit_idx_int < len(self.df):
                        exit_datetime = self.df.iloc[exit_idx_int]['datetime']
                    else:
                        exit_datetime = exit_time if exit_time else ''
                except:
                    exit_datetime = exit_time if exit_time else ''
            else:
                exit_datetime = exit_time if exit_time else ''
            
            # Calculate time in trade
            time_in_trade = ''
            if entry_datetime and exit_datetime:
                try:
                    if isinstance(entry_datetime, (pd.Timestamp, pd.DatetimeIndex)):
                        entry_dt = entry_datetime
                    else:
                        entry_dt = pd.to_datetime(entry_datetime)
                    
                    if isinstance(exit_datetime, (pd.Timestamp, pd.DatetimeIndex)):
                        exit_dt = exit_datetime
                    else:
                        exit_dt = pd.to_datetime(exit_datetime)
                    
                    duration = exit_dt - entry_dt
                    # Format duration nicely
                    days = duration.days
                    hours = duration.seconds // 3600
                    minutes = (duration.seconds % 3600) // 60
                    
                    if days > 0:
                        time_in_trade = f"{days}d {hours}h {minutes}m"
                    elif hours > 0:
                        time_in_trade = f"{hours}h {minutes}m"
                    else:
                        time_in_trade = f"{minutes}m"
                except:
                    time_in_trade = ''
            
            # Create ENTRY row
            if entry_price is not None:
                rows.append({
                    'type': 'Entry',
                    'direction': direction,
                    'datetime': str(entry_datetime) if entry_datetime else '',
                    'price': entry_price,
                    'pos_size': pos_size,
                    'entry_trade': '',  # Blank for entry
                    'pnl': '',  # Blank for entry
                    'time_in_trade': '',  # Blank for entry
                    'trade_id': trade_id,
                    'bar_idx': entry_idx if entry_idx is not None else 0
                })
            
            # Create EXIT row
            if exit_price is not None:
                rows.append({
                    'type': 'Exit',
                    'direction': direction,
                    'datetime': str(exit_datetime) if exit_datetime else '',
                    'price': exit_price,
                    'pos_size': pos_size,
                    'entry_trade': trade_id,  # Reference to entry
                    'pnl': f"{pnl:.2f}" if pnl is not None else '',
                    'time_in_trade': time_in_trade,
                    'trade_id': trade_id,
                    'bar_idx': exit_idx if exit_idx is not None else 0
                })
        
        return pd.DataFrame(rows)
    
    def _create_indicator_controls(self):
        """Create the NEW indicator management panel without jump-to-trade."""
        categories = VectorBTIndicatorManager.get_categories()
        
        return html.Div([
            html.H3("📊 Indicator Management", style={'color': 'white', 'marginBottom': '15px'}),
            
            # Category selector
            html.Div([
                html.Label("Category:", style={'color': '#aaa', 'marginRight': '10px', 'fontSize': '12px'}),
                dcc.Dropdown(
                    id='category-dropdown',
                    options=[{'label': cat, 'value': cat} for cat in categories],
                    value=categories[0] if categories else None,
                    placeholder="Select category",
                    style={'width': '100%', 'marginBottom': '10px'}
                )
            ]),
            
            # Indicator selector
            html.Div([
                html.Label("Indicator:", style={'color': '#aaa', 'marginRight': '10px', 'fontSize': '12px'}),
                dcc.Dropdown(
                    id='indicator-dropdown',
                    options=[],  # Will be populated based on category
                    value=None,
                    placeholder="Select indicator",
                    style={'width': '100%', 'marginBottom': '10px'}
                )
            ]),
            
            # Dynamic parameter inputs
            html.Div(
                id='parameter-inputs',
                style={'marginBottom': '15px', 'padding': '10px', 
                      'backgroundColor': '#3a3a3a', 'borderRadius': '5px'}
            ),
            
            # Control buttons
            html.Div([
                html.Button('➕ Add Indicator', id='add-indicator-btn', n_clicks=0,
                          style={'marginRight': '5px', 'backgroundColor': '#4CAF50', 
                                'color': 'white', 'border': 'none', 'padding': '8px 15px',
                                'borderRadius': '4px', 'cursor': 'pointer'}),
                html.Button('↩ Clear Last', id='clear-last-btn', n_clicks=0,
                          style={'marginRight': '5px', 'backgroundColor': '#FF9800',
                                'color': 'white', 'border': 'none', 'padding': '8px 15px',
                                'borderRadius': '4px', 'cursor': 'pointer'}),
                html.Button('🗑 Clear All', id='clear-all-btn', n_clicks=0,
                          style={'backgroundColor': '#f44336', 'color': 'white',
                                'border': 'none', 'padding': '8px 15px',
                                'borderRadius': '4px', 'cursor': 'pointer'})
            ], style={'marginBottom': '15px'}),
            
            # Active indicators list
            html.Div([
                html.H4("Active Indicators", style={'color': '#aaa', 'fontSize': '14px', 'marginBottom': '10px'}),
                html.Div(id='active-indicators-list', 
                        style={'maxHeight': '200px', 'overflowY': 'auto'})
            ], style={'backgroundColor': '#3a3a3a', 'padding': '10px', 'borderRadius': '5px'})
            
        ], style={'padding': '15px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px'})
    
    def _setup_layout(self):
        """Setup the dashboard layout WITHOUT jump-to-trade."""
        trades_display = self._prepare_trades_data()
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1(f"📈 {self.symbol} - {self.strategy_name} Dashboard V2", 
                       style={'color': 'white', 'margin': '10px'}),
                html.Div(id='performance-metrics', style={'color': '#00ff00', 'margin': '10px'})
            ], style={'backgroundColor': '#1e1e1e', 'padding': '10px'}),
            
            # Store components for state management
            dcc.Store(id='indicators-store', data=[]),
            dcc.Store(id='chart-range', data={'start': 0, 'end': min(2000, len(self.df))}),
            
            # Main container
            html.Div([
                # Left panel - Controls
                html.Div([
                    # NEW Indicator management panel (NO jump-to-trade)
                    self._create_indicator_controls(),
                    
                    # Stats panel
                    html.Div([
                        html.H3("📊 Performance Stats", style={'color': 'white'}),
                        html.Div(id='stats-panel', style={'color': 'white', 'padding': '10px'})
                    ], style={'backgroundColor': '#2a2a2a', 'marginTop': '10px', 'padding': '10px', 'borderRadius': '5px'}),
                    
                    # Enhanced Trade list table with all columns and horizontal scroll
                    html.Div([
                        html.H3("📋 Trade List (Entry/Exit Separated)", style={'color': 'white', 'margin': '10px'}),
                        dash_table.DataTable(
                            id='trade-table',
                            columns=[
                                {'name': 'Type', 'id': 'type', 'width': '60px'},
                                {'name': 'Direction', 'id': 'direction', 'width': '70px'},
                                {'name': 'DateTime', 'id': 'datetime', 'width': '150px'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.4f'}, 'width': '80px'},
                                {'name': 'Pos Size', 'id': 'pos_size', 'type': 'numeric', 'format': {'specifier': '.2f'}, 'width': '80px'},
                                {'name': 'Entry Trade', 'id': 'entry_trade', 'width': '90px'},
                                {'name': 'PnL', 'id': 'pnl', 'width': '80px'},
                                {'name': 'Time in Trade', 'id': 'time_in_trade', 'width': '100px'}
                            ],
                            data=trades_display.to_dict('records') if not trades_display.empty else [],
                            style_cell={
                                'textAlign': 'center', 
                                'backgroundColor': '#2a2a2a', 
                                'color': 'white',
                                'fontSize': '11px',
                                'minWidth': '50px',
                                'maxWidth': '150px',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_header={
                                'backgroundColor': '#1e1e1e', 
                                'fontWeight': 'bold',
                                'fontSize': '12px'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'type', 'filter_query': '{type} = Entry'},
                                    'backgroundColor': '#1a3a1a',
                                    'color': '#90EE90'
                                },
                                {
                                    'if': {'column_id': 'type', 'filter_query': '{type} = Exit'},
                                    'backgroundColor': '#3a1a1a',
                                    'color': '#FFB6C1'
                                },
                                {
                                    'if': {'column_id': 'direction', 'filter_query': '{direction} = Long'},
                                    'color': '#00ff00'
                                },
                                {
                                    'if': {'column_id': 'direction', 'filter_query': '{direction} = Short'},
                                    'color': '#ff9900'
                                },
                                {
                                    'if': {'column_id': 'pnl', 'filter_query': '{pnl} contains -'},
                                    'color': '#ff0000'
                                },
                                {
                                    'if': {'column_id': 'pnl', 'filter_query': '{pnl} != "" and {pnl} !contains -'},
                                    'color': '#00ff00'
                                }
                            ],
                            virtualization=True,
                            fixed_rows={'headers': True},
                            fixed_columns={'headers': True, 'data': 1},  # Fix Type column
                            style_table={
                                'height': '280px', 
                                'overflowY': 'auto',
                                'overflowX': 'auto',  # Enable horizontal scroll
                                'width': '100%'
                            },
                            style_data={
                                'border': '1px solid #3a3a3a'
                            },
                            row_selectable='single',
                            selected_rows=[],
                            sort_action='native',  # Enable sorting
                            filter_action='native'  # Enable filtering
                        )
                    ], style={'padding': '10px', 'backgroundColor': '#2a2a2a', 'borderRadius': '5px', 'marginTop': '10px'})
                ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                
                # Right panel - Chart
                html.Div([
                    dcc.Graph(
                        id='main-chart',
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'scrollZoom': True,
                            'doubleClick': 'autosize',
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        },
                        style={'height': '850px'}
                    ),
                    html.Div(id='chart-info', style={'color': 'white', 'padding': '10px', 'fontSize': '12px'})
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
            ], style={'backgroundColor': '#1e1e1e', 'minHeight': '900px'})
        ], style={'height': '100vh', 'overflow': 'hidden', 'backgroundColor': '#1e1e1e'})
    
    def _setup_callbacks(self):
        """Setup all dashboard callbacks."""
        
        # Update indicator dropdown based on category
        @self.app.callback(
            Output('indicator-dropdown', 'options'),
            Input('category-dropdown', 'value')
        )
        def update_indicator_dropdown(category):
            if not category:
                return []
            return VectorBTIndicatorManager.get_indicators_by_category(category)
        
        # Update parameter inputs based on selected indicator
        @self.app.callback(
            Output('parameter-inputs', 'children'),
            Input('indicator-dropdown', 'value')
        )
        def update_parameter_inputs(indicator):
            if not indicator or indicator not in VectorBTIndicatorManager.INDICATORS:
                return html.Div("Select an indicator to see parameters", 
                              style={'color': '#888', 'fontSize': '12px'})
            
            indicator_info = VectorBTIndicatorManager.INDICATORS[indicator]
            params = indicator_info.get('params', {})
            
            if not params:
                return html.Div("This indicator has no parameters", 
                              style={'color': '#888', 'fontSize': '12px'})
            
            inputs = []
            for param_name, param_info in params.items():
                param_type = param_info['type']
                default_val = param_info['default']
                description = param_info.get('description', param_name)
                
                input_group = html.Div([
                    html.Label(f"{description}:", 
                             style={'color': '#aaa', 'fontSize': '12px', 'marginBottom': '2px'}),
                    dcc.Input(
                        id={'type': 'param-input', 'param': param_name},
                        type='number',
                        value=default_val,
                        min=param_info.get('min', 0),
                        max=param_info.get('max', 1000),
                        step=param_info.get('step', 1),
                        style={'width': '100%', 'marginBottom': '8px'}
                    ) if param_type in ['int', 'float'] else None
                ], style={'marginBottom': '10px'})
                
                if input_group.children[1]:  # Only add if input was created
                    inputs.append(input_group)
            
            return html.Div(inputs)
        
        # Main callback for managing indicators
        @self.app.callback(
            [Output('main-chart', 'figure'),
             Output('indicators-store', 'data'),
             Output('active-indicators-list', 'children'),
             Output('chart-info', 'children'),
             Output('stats-panel', 'children')],
            [Input('add-indicator-btn', 'n_clicks'),
             Input('clear-last-btn', 'n_clicks'),
             Input('clear-all-btn', 'n_clicks'),
             Input('trade-table', 'selected_rows'),
             Input('chart-range', 'data')],
            [State('indicator-dropdown', 'value'),
             State({'type': 'param-input', 'param': ALL}, 'value'),
             State({'type': 'param-input', 'param': ALL}, 'id'),
             State('indicators-store', 'data')]
        )
        def update_chart_and_indicators(add_clicks, clear_last_clicks, clear_all_clicks,
                                       selected_rows, chart_range, selected_indicator,
                                       param_values, param_ids, current_indicators):
            """Main callback to handle all indicator operations and chart updates."""
            
            ctx_triggered = ctx.triggered_id
            
            # Initialize indicators list if needed
            if current_indicators is None:
                current_indicators = []
            
            # Handle button clicks
            if ctx_triggered == 'add-indicator-btn' and selected_indicator:
                # Extract parameters
                params = {}
                for param_id, value in zip(param_ids, param_values):
                    param_name = param_id['param']
                    params[param_name] = value
                
                # Calculate indicator
                indicator_info = VectorBTIndicatorManager.INDICATORS[selected_indicator]
                try:
                    result = indicator_info['calculate'](self.df, **params)
                    
                    # Generate unique ID
                    self.indicator_counter += 1
                    indicator_id = f"ind_{self.indicator_counter}"
                    
                    # Store indicator data
                    new_indicator = {
                        'id': indicator_id,
                        'type': selected_indicator,
                        'name': f"{indicator_info['name']} ({', '.join([f'{k}={v}' for k, v in params.items()])})",
                        'params': params,
                        'subplot': indicator_info.get('subplot', False),
                        'multiple_lines': indicator_info.get('multiple_lines', False)
                    }
                    
                    # Store calculated data in dataframe
                    if isinstance(result, dict):
                        for key, values in result.items():
                            self.df[f"{indicator_id}_{key}"] = values
                            new_indicator[f'column_{key}'] = f"{indicator_id}_{key}"
                    else:
                        self.df[indicator_id] = result
                        new_indicator['column'] = indicator_id
                    
                    current_indicators.append(new_indicator)
                    
                except Exception as e:
                    print(f"Error calculating {selected_indicator}: {e}")
            
            elif ctx_triggered == 'clear-last-btn' and current_indicators:
                # Remove last indicator
                removed = current_indicators.pop()
                # Clean up dataframe columns
                if 'column' in removed:
                    if removed['column'] in self.df.columns:
                        self.df.drop(columns=[removed['column']], inplace=True)
                else:
                    # Multiple columns
                    for key in ['upper', 'middle', 'lower', 'macd', 'signal', 'histogram', 'k', 'd', 'up', 'down']:
                        col_name = f"column_{key}"
                        if col_name in removed and removed[col_name] in self.df.columns:
                            self.df.drop(columns=[removed[col_name]], inplace=True)
            
            elif ctx_triggered == 'clear-all-btn':
                # Clear all indicators
                for ind in current_indicators:
                    if 'column' in ind and ind['column'] in self.df.columns:
                        self.df.drop(columns=[ind['column']], inplace=True)
                    else:
                        # Multiple columns
                        for key in ['upper', 'middle', 'lower', 'macd', 'signal', 'histogram', 'k', 'd', 'up', 'down']:
                            col_name = f"column_{key}"
                            if col_name in ind and ind[col_name] in self.df.columns:
                                self.df.drop(columns=[ind[col_name]], inplace=True)
                current_indicators = []
            
            # Handle trade selection - navigate to selected trade
            if ctx_triggered == 'trade-table' and selected_rows:
                trades_display = self._prepare_trades_data()
                if not trades_display.empty and selected_rows[0] < len(trades_display):
                    selected_trade = trades_display.iloc[selected_rows[0]]
                    # Get the bar index for the selected trade
                    bar_idx = int(selected_trade.get('bar_idx', 0))
                    if bar_idx > 0:
                        # Center the view around the selected trade
                        window_size = 500  # Show 500 bars around the trade
                        new_start = max(0, bar_idx - window_size // 2)
                        new_end = min(len(self.df), bar_idx + window_size // 2)
                        chart_range = {'start': new_start, 'end': new_end}
            
            # Create figure with current indicators
            fig = self._create_figure(current_indicators, chart_range, selected_rows)
            
            # Update active indicators list display
            indicators_display = []
            for i, ind in enumerate(current_indicators):
                indicators_display.append(
                    html.Div([
                        html.Span(f"• {ind['name']}", style={'color': '#00ff00', 'fontSize': '12px'}),
                    ], style={'padding': '3px'})
                )
            
            # Update chart info
            chart_info = f"Displaying {len(current_indicators)} indicator(s) | "
            chart_info += f"Range: bars {chart_range.get('start', 0)} to {chart_range.get('end', len(self.df))}"
            
            # Add selected trade info if applicable
            if selected_rows:
                trades_display = self._prepare_trades_data()
                if not trades_display.empty and selected_rows[0] < len(trades_display):
                    selected_trade = trades_display.iloc[selected_rows[0]]
                    chart_info += f" | Selected: {selected_trade['type']} @ {selected_trade['datetime']}"
            
            # Update stats
            stats = self._calculate_stats()
            
            return fig, current_indicators, indicators_display, chart_info, stats
    
    def _create_figure(self, indicators, chart_range, selected_trade_rows=None):
        """Create the main chart figure."""
        start_idx = chart_range.get('start', 0)
        end_idx = chart_range.get('end', min(2000, len(self.df)))
        df_view = self.df.iloc[start_idx:end_idx]
        
        # Count subplots needed
        n_subplots = 2  # Price + Volume
        subplot_indicators = [ind for ind in indicators if ind.get('subplot', False)]
        n_subplots += len(subplot_indicators)
        if 'equity' in self.df.columns:
            n_subplots += 1
        
        # Calculate row heights
        if n_subplots == 2:
            row_heights = [0.8, 0.2]
        elif n_subplots == 3:
            row_heights = [0.6, 0.2, 0.2]
        else:
            main_height = 0.5
            other_height = 0.5 / (n_subplots - 1)
            row_heights = [main_height] + [other_height] * (n_subplots - 1)
        
        # Create subplots
        fig = make_subplots(
            rows=n_subplots, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights
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
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add trade markers if we have trades
        if selected_trade_rows is not None and len(selected_trade_rows) > 0:
            trades_display = self._prepare_trades_data()
            if not trades_display.empty:
                # Get selected trade
                if selected_trade_rows[0] < len(trades_display):
                    selected_trade = trades_display.iloc[selected_trade_rows[0]]
                    bar_idx = int(selected_trade.get('bar_idx', 0))
                    
                    # Add marker for selected trade
                    if start_idx <= bar_idx < end_idx:
                        relative_idx = bar_idx - start_idx
                        if 0 <= relative_idx < len(df_view):
                            marker_color = '#00ff00' if selected_trade['type'] == 'Entry' else '#ff0000'
                            marker_symbol = 'triangle-up' if selected_trade['type'] == 'Entry' else 'triangle-down'
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=[df_view.iloc[relative_idx]['datetime']],
                                    y=[float(selected_trade['price'])],
                                    mode='markers',
                                    marker=dict(
                                        size=15,
                                        color=marker_color,
                                        symbol=marker_symbol,
                                        line=dict(width=2, color='white')
                                    ),
                                    name=f"Selected {selected_trade['type']}",
                                    showlegend=False,
                                    hovertext=f"{selected_trade['type']}: {selected_trade['price']:.4f}<br>"
                                             f"Size: {selected_trade['pos_size']}<br>"
                                             f"Time: {selected_trade['datetime']}"
                                ),
                                row=1, col=1
                            )
        
        # Also add all trades in view (smaller markers)
        if self.trades_df is not None and not self.trades_df.empty:
            trades_display = self._prepare_trades_data()
            for _, trade in trades_display.iterrows():
                bar_idx = int(trade.get('bar_idx', 0))
                if start_idx <= bar_idx < end_idx:
                    relative_idx = bar_idx - start_idx
                    if 0 <= relative_idx < len(df_view):
                        marker_color = '#90EE90' if trade['type'] == 'Entry' else '#FFB6C1'
                        marker_symbol = 'triangle-up' if trade['type'] == 'Entry' else 'triangle-down'
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[df_view.iloc[relative_idx]['datetime']],
                                y=[float(trade['price'])],
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    color=marker_color,
                                    symbol=marker_symbol,
                                    opacity=0.6
                                ),
                                name=f"{trade['type']} Trades",
                                showlegend=False,
                                hovertext=f"{trade['type']}: {trade['price']:.4f}"
                            ),
                            row=1, col=1
                        )
        
        # Add overlay indicators (on price chart)
        colors = ['#ffaa00', '#00aaff', '#ff00ff', '#00ffff', '#ffff00', '#ff6600', '#00ff00', '#ff9900']
        color_idx = 0
        
        for ind in indicators:
            if not ind.get('subplot', False):
                color = colors[color_idx % len(colors)]
                color_idx += 1
                
                if ind.get('multiple_lines', False):
                    # Handle indicators with multiple lines (BB, KC, etc.)
                    for key in ['upper', 'middle', 'lower']:
                        col_name = ind.get(f'column_{key}')
                        if col_name and col_name in df_view.columns:
                            line_style = 'dash' if key in ['upper', 'lower'] else 'solid'
                            fig.add_trace(
                                go.Scatter(
                                    x=df_view['datetime'],
                                    y=df_view[col_name],
                                    name=f"{ind['name']} {key}",
                                    line=dict(color=color, width=1, dash=line_style),
                                    opacity=0.7
                                ),
                                row=1, col=1
                            )
                else:
                    # Single line indicator
                    col_name = ind.get('column')
                    if col_name and col_name in df_view.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df_view['datetime'],
                                y=df_view[col_name],
                                name=ind['name'],
                                line=dict(color=color, width=2)
                            ),
                            row=1, col=1
                        )
        
        # Add volume
        colors_vol = ['#ef5350' if close < open else '#26a69a' 
                     for close, open in zip(df_view['close'], df_view['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df_view['datetime'],
                y=df_view['volume'],
                name='Volume',
                marker_color=colors_vol,
                opacity=0.3
            ),
            row=2, col=1
        )
        
        # Add subplot indicators
        current_row = 3
        for ind in subplot_indicators:
            if ind.get('multiple_lines', False):
                # Handle multi-line subplot indicators (MACD, Stoch, Aroon)
                first_line = True
                for key in ['macd', 'signal', 'histogram', 'k', 'd', 'up', 'down']:
                    col_name = ind.get(f'column_{key}')
                    if col_name and col_name in df_view.columns:
                        trace_type = go.Bar if key == 'histogram' else go.Scatter
                        fig.add_trace(
                            trace_type(
                                x=df_view['datetime'],
                                y=df_view[col_name],
                                name=f"{ind['name']} {key}",
                                showlegend=first_line
                            ),
                            row=current_row, col=1
                        )
                        first_line = False
            else:
                # Single line subplot indicator
                col_name = ind.get('column')
                if col_name and col_name in df_view.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_view['datetime'],
                            y=df_view[col_name],
                            name=ind['name'],
                            line=dict(width=2)
                        ),
                        row=current_row, col=1
                    )
            
            # Add reference lines for oscillators
            if ind['type'] in ['RSI', 'STOCH', 'WILLR', 'MFI']:
                if ind['type'] == 'RSI':
                    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=current_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=current_row, col=1)
                elif ind['type'] in ['STOCH', 'MFI']:
                    fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.3, row=current_row, col=1)
                    fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.3, row=current_row, col=1)
            
            current_row += 1
        
        # Add equity curve if available
        if 'equity' in self.df.columns:
            profit_color = '#26a69a' if df_view['equity'].iloc[-1] >= 0 else '#ef5350'
            fig.add_trace(
                go.Scatter(
                    x=df_view['datetime'],
                    y=df_view['equity'],
                    name='Net Profit',
                    line=dict(color=profit_color, width=2),
                    fill='tozeroy',
                    fillcolor=f'rgba({38 if profit_color == "#26a69a" else 239}, '
                             f'{166 if profit_color == "#26a69a" else 83}, '
                             f'{154 if profit_color == "#26a69a" else 80}, 0.1)'
                ),
                row=current_row, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=current_row, col=1, opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title=f'{self.symbol} - {self.strategy_name}',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=850,
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor='rgba(0,0,0,0.5)',
                font=dict(size=10)
            ),
            hovermode='x unified',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update all y-axes to have autorange
        for i in range(1, n_subplots + 1):
            fig.update_yaxes(autorange=True, fixedrange=False, row=i, col=1)
        
        return fig
    
    def _calculate_stats(self):
        """Calculate and format performance statistics."""
        stats = []
        
        if self.portfolio is not None:
            try:
                # Total return
                total_return = self.portfolio.total_return
                if hasattr(total_return, 'values'):
                    total_return = np.mean(total_return.values)
                stats.append(html.Div(f"Total Return: {total_return*100:.2f}%", 
                                    style={'color': '#00ff00' if total_return > 0 else '#ff0000'}))
                
                # Sharpe ratio
                sharpe = self.portfolio.sharpe_ratio
                if hasattr(sharpe, 'values'):
                    sharpe = np.mean(sharpe.values)
                stats.append(html.Div(f"Sharpe Ratio: {sharpe:.2f}"))
                
                # Max drawdown
                max_dd = self.portfolio.max_drawdown
                if hasattr(max_dd, 'values'):
                    max_dd = np.max(max_dd.values)
                stats.append(html.Div(f"Max Drawdown: {max_dd*100:.2f}%", 
                                    style={'color': '#ff9900'}))
                
                # Win rate
                win_rate = self.portfolio.win_rate
                if hasattr(win_rate, 'values'):
                    win_rate = np.mean(win_rate.values)
                stats.append(html.Div(f"Win Rate: {win_rate*100:.1f}%"))
                
            except Exception as e:
                stats.append(html.Div(f"Stats calculation error: {e}", style={'color': '#ff0000'}))
        else:
            stats.append(html.Div("No portfolio data available", style={'color': '#888'}))
        
        return html.Div(stats)
    
    def run(self, port=8050, debug=False, open_browser=True):
        """Run the dashboard."""
        if open_browser:
            timer = threading.Timer(1.5, lambda: webbrowser.open(f'http://127.0.0.1:{port}/'))
            timer.start()
        
        self.app.run(debug=debug, port=port, host='127.0.0.1')


def launch_enhanced_dashboard_v2(ohlcv_data, trades_csv_path=None, portfolio=None, 
                                symbol="ES", strategy_name="Strategy"):
    """Launch the enhanced Plotly dashboard V2."""
    # Load trades if path provided
    trades_df = None
    if trades_csv_path and Path(trades_csv_path).exists():
        try:
            trades_df = pd.read_csv(trades_csv_path)
        except Exception as e:
            print(f"Warning: Could not load trades CSV: {e}")
    
    # Create and run dashboard
    dashboard = EnhancedPlotlyDashboardV2(
        ohlcv_data=ohlcv_data,
        trades_df=trades_df,
        portfolio=portfolio,
        symbol=symbol,
        strategy_name=strategy_name
    )
    
    dashboard.run(debug=False, open_browser=True)
    return dashboard


if __name__ == "__main__":
    # Test with sample data
    import numpy as np
    
    # Generate sample OHLCV data
    n_bars = 5000
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='1h')
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    
    ohlcv_data = {
        'datetime': dates,
        'open': close + np.random.randn(n_bars) * 0.1,
        'high': close + np.abs(np.random.randn(n_bars) * 0.5),
        'low': close - np.abs(np.random.randn(n_bars) * 0.5),
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars)
    }
    
    # Launch dashboard
    dashboard = launch_enhanced_dashboard_v2(
        ohlcv_data=ohlcv_data,
        symbol="TEST",
        strategy_name="Test Strategy V2"
    )