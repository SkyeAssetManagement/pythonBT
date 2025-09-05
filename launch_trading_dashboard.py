#!/usr/bin/env python3
"""
Direct Trading Dashboard Launcher
==================================
Launches the trading dashboard with candlestick charts and trade visualization
Without heavy backtesting or optimization
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add paths for imports
sys.path.insert(0, 'tradingCode')
sys.path.insert(0, 'tradingCode/src')
sys.path.insert(0, 'src')

# Import the dashboard
os.chdir('tradingCode')  # Change to tradingCode directory for proper imports
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data
from PyQt5.QtWidgets import QApplication

def load_sample_data():
    """Load sample data from available files"""
    # Try to find data files
    data_files = [
        '../parquetData/ES-DIFF-daily-with-atr.csv',
        '../dataRaw/range-ATR30x0.05/ES/diffAdjusted/ES-DIFF-range-ATR30x0.05-dailyATR.csv',
        '../data/ES_continuous.csv',
        'data/ES_sample.parquet'
    ]
    
    for file in data_files:
        if os.path.exists(file):
            print(f"Loading data from: {file}")
            try:
                if file.endswith('.parquet'):
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_csv(file)
                return df
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
    
    # If no file found, create sample data
    print("No data files found, creating sample data...")
    return create_sample_ohlcv_data()

def create_sample_ohlcv_data():
    """Create sample OHLCV data for demonstration"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create 500 bars of sample data
    n_bars = 500
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1H')
    
    # Generate realistic price data
    np.random.seed(42)
    close = 4000 + np.cumsum(np.random.randn(n_bars) * 5)
    
    data = pd.DataFrame({
        'datetime': dates,
        'open': close + np.random.randn(n_bars) * 2,
        'high': close + np.abs(np.random.randn(n_bars) * 3),
        'low': close - np.abs(np.random.randn(n_bars) * 3),
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars)
    })
    
    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

def create_sample_trades(data_length):
    """Create sample trade data for visualization"""
    n_trades = 20
    trade_indices = np.random.choice(range(50, data_length - 50), n_trades, replace=False)
    trade_indices.sort()
    
    trades = []
    for i, idx in enumerate(trade_indices):
        trades.append({
            'index': idx,
            'type': 'Buy' if i % 2 == 0 else 'Sell',
            'price': None,  # Will be set from OHLC data
            'size': np.random.randint(1, 5),
            'timestamp': None  # Will be set from datetime
        })
    
    return trades

def main():
    """Main entry point for dashboard"""
    print("="*60)
    print("SAM - Trading Dashboard Launcher")
    print("Candlestick Charts with Trade Visualization")
    print("="*60)
    
    # Create Qt Application
    app = QApplication(sys.argv)
    
    # Load or create data
    ohlcv_data = load_sample_data()
    
    # Ensure proper column names
    if 'datetime' not in ohlcv_data.columns and ohlcv_data.index.name:
        ohlcv_data = ohlcv_data.reset_index()
        if 'index' in ohlcv_data.columns:
            ohlcv_data = ohlcv_data.rename(columns={'index': 'datetime'})
    
    # Standardize column names
    column_mapping = {
        'Date': 'datetime',
        'date': 'datetime', 
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'vol': 'volume'
    }
    
    ohlcv_data = ohlcv_data.rename(columns=column_mapping)
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in ohlcv_data.columns:
            # Try to find similar column
            for orig_col in ohlcv_data.columns:
                if col in orig_col.lower():
                    ohlcv_data[col] = ohlcv_data[orig_col]
                    break
    
    # Add volume if missing
    if 'volume' not in ohlcv_data.columns:
        ohlcv_data['volume'] = np.random.randint(1000, 10000, len(ohlcv_data))
    
    print(f"Loaded {len(ohlcv_data)} bars of data")
    print(f"Columns: {list(ohlcv_data.columns)}")
    
    # Create sample trades
    trades_data = create_sample_trades(len(ohlcv_data))
    
    # Update trade prices from OHLC data
    for trade in trades_data:
        idx = trade['index']
        if idx < len(ohlcv_data):
            trade['price'] = float(ohlcv_data.iloc[idx]['close'])
            if 'datetime' in ohlcv_data.columns:
                trade['timestamp'] = ohlcv_data.iloc[idx]['datetime']
    
    print(f"Generated {len(trades_data)} sample trades")
    
    # Create test data dictionary
    test_data = {
        'ohlcv': ohlcv_data,
        'trades': trades_data,
        'equity_curve': pd.DataFrame({
            'datetime': ohlcv_data['datetime'] if 'datetime' in ohlcv_data.columns else ohlcv_data.index,
            'equity': 100000 + np.cumsum(np.random.randn(len(ohlcv_data)) * 100)
        }),
        'metrics': {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.08,
            'win_rate': 0.65,
            'total_trades': len(trades_data)
        }
    }
    
    # Create and show dashboard
    print("\nLaunching Trading Dashboard...")
    dashboard = FinalTradingDashboard(test_data)
    dashboard.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()