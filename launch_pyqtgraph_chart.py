#!/usr/bin/env python3
"""
PyQtGraph Range Bars Chart Launcher
====================================
Launches the pyqtgraph range bars visualization with candlesticks and trade X markers
"""

import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from datetime import datetime

# Import the main chart
from pyqtgraph_range_bars_final import RangeBarChartFinal
from trade_data import TradeCollection, TradeData

def load_data():
    """Load data from available files"""
    # Try to find data files in order of preference
    data_files = [
        'parquetData/ES-DIFF-daily-with-atr.csv',
        'dataRaw/range-ATR30x0.05/ES/diffAdjusted/ES-DIFF-range-ATR30x0.05-dailyATR.csv',
        'dataRaw/range-ATR30x0.1/ES/diffAdjusted/ES-DIFF-range-ATR30x0.1-dailyATR.csv',
        'dataRaw/range-ATR30x0.2/ES/diffAdjusted/ES-DIFF-range-ATR30x0.2-dailyATR.csv',
        'data/ES_continuous.csv'
    ]
    
    for file in data_files:
        if os.path.exists(file):
            print(f"Loading data from: {file}")
            try:
                if file.endswith('.parquet'):
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_csv(file)
                    
                print(f"Loaded {len(df)} bars from {file}")
                print(f"Columns: {list(df.columns)}")
                return df, file
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
    
    # If no file found, create sample data
    print("No data files found, creating sample data...")
    return create_sample_data(), "Sample Data"

def create_sample_data():
    """Create sample OHLCV data for demonstration"""
    n_bars = 1000
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='15min')
    
    # Generate realistic price data
    np.random.seed(42)
    close = 4000 + np.cumsum(np.random.randn(n_bars) * 2)
    
    data = pd.DataFrame({
        'DateTime': dates,
        'Open': close + np.random.randn(n_bars) * 1,
        'High': close + np.abs(np.random.randn(n_bars) * 2),
        'Low': close - np.abs(np.random.randn(n_bars) * 2),
        'Close': close,
        'Volume': np.random.randint(100, 1000, n_bars),
        'AUX1': np.random.uniform(10, 50, n_bars),  # ATR values
        'AUX2': np.random.uniform(5, 20, n_bars)    # Range values
    })
    
    # Ensure OHLC relationships are valid
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data

def create_sample_trades(data_length):
    """Create sample trades for visualization"""
    num_trades = min(30, data_length // 20)  # About 30 trades or 1 per 20 bars
    
    trades = []
    trade_indices = np.random.choice(range(50, min(data_length - 50, 950)), num_trades, replace=False)
    trade_indices.sort()
    
    for i, idx in enumerate(trade_indices):
        trades.append(TradeData(
            bar_index=idx,
            price=4000 + np.random.randn() * 10,  # Price around 4000
            trade_type='Buy' if i % 2 == 0 else 'Sell',
            timestamp=None
        ))
    
    return TradeCollection(trades)

def main():
    """Main entry point"""
    print("="*60)
    print("SAM - PyQtGraph Range Bars Visualization")
    print("Candlestick Charts with Trade X Markers")
    print("="*60)
    
    # Create Qt Application
    app = QtWidgets.QApplication(sys.argv)
    
    # Load data
    data, filename = load_data()
    
    # Standardize column names
    column_mapping = {
        'datetime': 'DateTime',
        'date': 'DateTime',
        'Date': 'DateTime',
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'vol': 'Volume'
    }
    
    # Rename columns to standard format
    for old_name, new_name in column_mapping.items():
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})
    
    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, trying to find alternatives...")
        # Try to find columns with these words in them
        for req_col in missing_cols:
            for col in data.columns:
                if req_col.lower() in col.lower():
                    data[req_col] = data[col]
                    print(f"Using {col} as {req_col}")
                    break
    
    # Add default columns if missing
    if 'Volume' not in data.columns:
        data['Volume'] = np.random.randint(100, 1000, len(data))
        print("Added synthetic Volume data")
    
    if 'DateTime' not in data.columns:
        if data.index.name and 'date' in data.index.name.lower():
            data['DateTime'] = data.index
        else:
            data['DateTime'] = pd.date_range(end=datetime.now(), periods=len(data), freq='15min')
        print("Added synthetic DateTime data")
    
    # Add AUX fields if missing (for ATR and Range display)
    if 'AUX1' not in data.columns:
        data['AUX1'] = np.random.uniform(10, 50, len(data))
        print("Added synthetic AUX1 (ATR) data")
    
    if 'AUX2' not in data.columns:
        data['AUX2'] = np.random.uniform(5, 20, len(data))
        print("Added synthetic AUX2 (Range) data")
    
    print(f"\nData prepared: {len(data)} bars")
    print(f"Columns: {list(data.columns)}")
    print(f"Date range: {data['DateTime'].iloc[0]} to {data['DateTime'].iloc[-1]}")
    
    # Create sample trades
    trades = create_sample_trades(len(data))
    print(f"Created {len(trades)} sample trades")
    
    # Create and show the chart
    chart = RangeBarChartFinal()
    
    # The chart loads its own data in load_data() method
    # But we need to override it with our data
    # Set the data directly
    chart.full_data = {
        'timestamp': pd.to_datetime(data['DateTime']),
        'open': data['Open'].values.astype(np.float32),
        'high': data['High'].values.astype(np.float32),
        'low': data['Low'].values.astype(np.float32),
        'close': data['Close'].values.astype(np.float32),
        'volume': data['Volume'].values.astype(np.float32),
        'aux1': data['AUX1'].values.astype(np.float32),  # ATR
        'aux2': data['AUX2'].values.astype(np.float32)   # Range
    }
    chart.total_bars = len(data)
    
    # Initialize the chart if it has this method
    if hasattr(chart, 'init_chart'):
        chart.init_chart()
    elif hasattr(chart, 'init_ui'):
        chart.init_ui()
    
    # Add trades if the chart supports it
    if hasattr(chart, 'set_trades'):
        chart.set_trades(trades)
    elif hasattr(chart, 'trade_collection'):
        chart.trade_collection = trades
    
    # Show the window
    chart.show()
    
    print("\nChart window opened. Use:")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Click and drag: Pan")
    print("  - Hover: See price/time/ATR/Range info")
    print("  - White X marks: Trade locations")
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()