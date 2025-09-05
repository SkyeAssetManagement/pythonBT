#!/usr/bin/env python3
"""
Simple PyQtGraph Chart Launcher
================================
Directly modifies the pyqtgraph_range_bars_final to use our data
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
import time

# Import required components
from trade_data import TradeCollection, TradeData

def create_sample_trades(num_bars):
    """Create sample trades"""
    num_trades = min(30, num_bars // 20)
    trades = []
    
    np.random.seed(42)
    indices = np.random.choice(range(50, min(num_bars - 50, num_bars)), 
                              min(num_trades, num_bars - 100), replace=False)
    indices.sort()
    
    for i, idx in enumerate(indices):
        trades.append(TradeData(
            bar_index=idx,
            price=4000 + np.random.randn() * 10,
            trade_type='Buy' if i % 2 == 0 else 'Sell'
        ))
    
    return TradeCollection(trades)

def main():
    """Main entry point"""
    print("="*60)
    print("SAM - PyQtGraph Range Bars Visualization")
    print("="*60)
    
    # Create Qt Application
    app = QtWidgets.QApplication(sys.argv)
    
    # Import and modify the chart class to use our data
    from pyqtgraph_range_bars_final import RangeBarChartFinal
    
    # Create a modified version that loads our data
    class ModifiedChart(RangeBarChartFinal):
        def load_data(self):
            """Override to load our data"""
            print("Loading data from workspace...")
            start_time = time.time()
            
            # Try multiple file locations
            file_paths = [
                Path("parquetData/ES-DIFF-daily-with-atr.csv"),
                Path("dataRaw/range-ATR30x0.05/ES/diffAdjusted/ES-DIFF-range-ATR30x0.05-dailyATR.csv"),
                Path("dataRaw/range-ATR30x0.1/ES/diffAdjusted/ES-DIFF-range-ATR30x0.1-dailyATR.csv"),
                Path("dataRaw/range-ATR30x0.2/ES/diffAdjusted/ES-DIFF-range-ATR30x0.2-dailyATR.csv"),
            ]
            
            df = None
            loaded_file = None
            
            # Try to load a file
            for file_path in file_paths:
                if file_path.exists():
                    try:
                        if str(file_path).endswith('.parquet'):
                            df = pd.read_parquet(file_path)
                        else:
                            df = pd.read_csv(file_path)
                        loaded_file = file_path
                        print(f"Loaded {len(df)} bars from {file_path.name}")
                        break
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue
            
            if df is None:
                # Create sample data
                print("Creating sample data...")
                n = 1000
                base = 4000
                prices = base + np.cumsum(np.random.randn(n) * 2)
                
                df = pd.DataFrame({
                    'DateTime': pd.date_range('2024-01-01', periods=n, freq='15min'),
                    'Open': np.roll(prices, 1),
                    'High': prices + np.abs(np.random.randn(n) * 3),
                    'Low': prices - np.abs(np.random.randn(n) * 3),
                    'Close': prices,
                    'Volume': np.random.uniform(1000, 10000, n)
                })
                df['Open'][0] = base
            
            # Standardize column names
            column_mapping = {
                'datetime': 'DateTime', 'date': 'DateTime', 'Date': 'DateTime',
                'timestamp': 'DateTime', 'Timestamp': 'DateTime',
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'volume': 'Volume', 'vol': 'Volume'
            }
            
            for old, new in column_mapping.items():
                if old in df.columns and new not in df.columns:
                    df[new] = df[old]
            
            # Ensure DateTime column
            if 'DateTime' not in df.columns:
                if 'Time' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['Time'])
                elif df.index.name and 'date' in df.index.name.lower():
                    df['DateTime'] = df.index
                else:
                    df['DateTime'] = pd.date_range('2024-01-01', periods=len(df), freq='15min')
            else:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
            
            # Ensure required columns
            required = ['Open', 'High', 'Low', 'Close']
            for col in required:
                if col not in df.columns:
                    # Try to find it
                    for c in df.columns:
                        if col.lower() in c.lower():
                            df[col] = df[c]
                            break
            
            # Add volume if missing
            if 'Volume' not in df.columns:
                df['Volume'] = np.random.uniform(1000, 10000, len(df))
            
            # Add AUX fields
            if 'AUX1' not in df.columns:
                df['AUX1'] = np.random.uniform(10, 30, len(df))  # Simulated ATR
            if 'AUX2' not in df.columns:
                df['AUX2'] = np.random.uniform(0.05, 0.2, len(df))  # Range multiplier
            
            # Convert to the format the chart expects
            # Ensure DateTime is properly converted to array
            timestamps = pd.to_datetime(df['DateTime'])
            if isinstance(timestamps, pd.Series):
                timestamps = timestamps.values
            
            self.full_data = {
                'timestamp': timestamps,
                'open': df['Open'].values.astype(np.float32),
                'high': df['High'].values.astype(np.float32),
                'low': df['Low'].values.astype(np.float32),
                'close': df['Close'].values.astype(np.float32),
                'volume': df['Volume'].values.astype(np.float32) if 'Volume' in df else None,
                'aux1': df['AUX1'].values.astype(np.float32) if 'AUX1' in df else None,
                'aux2': df['AUX2'].values.astype(np.float32) if 'AUX2' in df else None
            }
            self.total_bars = len(self.full_data['open'])
            
            print(f"Data loaded: {self.total_bars:,} bars in {time.time()-start_time:.2f}s")
            
            # Update trade panel if it exists
            if self.trade_panel and self.full_data['timestamp'] is not None:
                self.trade_panel.set_chart_timestamps(self.full_data['timestamp'])
                
                bar_data = {
                    'open': self.full_data['open'],
                    'high': self.full_data['high'],
                    'low': self.full_data['low'],
                    'close': self.full_data['close']
                }
                self.trade_panel.set_bar_data(bar_data)
                
                print(f"Set chart timestamps and OHLC data in trade panel: "
                     f"{self.full_data['timestamp'][0]} to {self.full_data['timestamp'][-1]}")
    
    # Create the modified chart
    chart = ModifiedChart()
    
    # Create and load sample trades
    trades = create_sample_trades(chart.total_bars)
    if hasattr(chart, 'load_trades'):
        chart.load_trades(trades)
        print(f"Loaded {len(trades)} sample trades")
    
    # Show the window
    chart.show()
    
    print("\nChart window opened. Controls:")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Click and drag: Pan")
    print("  - Hover: See price/time/ATR/Range info")
    print("  - White X marks: Trade locations")
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()