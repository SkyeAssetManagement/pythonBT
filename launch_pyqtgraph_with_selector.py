#!/usr/bin/env python3
"""
PyQtGraph Chart Launcher with Data Selector
============================================
Shows data selector dialog first, then launches the chart with selected configuration
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
from pyqtgraph_data_selector import DataSelectorDialog
from pyqtgraph_range_bars_final import RangeBarChartFinal
from trade_data import TradeCollection, TradeData
from csv_trade_loader import CSVTradeLoader

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

def generate_system_trades(system_name, data):
    """Generate trades from a simple trading system"""
    trades = []
    
    if 'Close' not in data.columns:
        return TradeCollection(trades)
    
    prices = data['Close'].values
    
    if system_name == "Simple Moving Average":
        # Simple SMA crossover
        sma20 = pd.Series(prices).rolling(20).mean()
        sma50 = pd.Series(prices).rolling(50).mean()
        
        for i in range(51, len(prices)):
            if sma20.iloc[i] > sma50.iloc[i] and sma20.iloc[i-1] <= sma50.iloc[i-1]:
                trades.append(TradeData(
                    bar_index=i,
                    price=prices[i],
                    trade_type='Buy'
                ))
            elif sma20.iloc[i] < sma50.iloc[i] and sma20.iloc[i-1] >= sma50.iloc[i-1]:
                trades.append(TradeData(
                    bar_index=i,
                    price=prices[i],
                    trade_type='Sell'
                ))
    
    elif system_name == "RSI Momentum":
        # RSI based trades
        rsi = calculate_rsi(prices, 14)
        for i in range(15, len(prices)):
            if rsi[i] < 30 and rsi[i-1] >= 30:
                trades.append(TradeData(
                    bar_index=i,
                    price=prices[i],
                    trade_type='Buy'
                ))
            elif rsi[i] > 70 and rsi[i-1] <= 70:
                trades.append(TradeData(
                    bar_index=i,
                    price=prices[i],
                    trade_type='Sell'
                ))
    
    else:
        # Default to sample trades
        return create_sample_trades(len(data))
    
    return TradeCollection(trades)

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi

class ConfiguredChart(RangeBarChartFinal):
    """Modified chart that uses configuration from selector"""

    def __init__(self, config=None):
        # IMPORTANT: Set config BEFORE calling parent __init__
        # Parent will skip load_data() because we have config attribute
        self.config = config or {}
        super().__init__()
        # Now load our configured data
        self.load_data()
    
    def load_data(self):
        """Override to load configured data (real data only)."""
        print("Loading configured data...")
        start_time = time.time()
        
        if self.config.get('data_file') and os.path.exists(self.config['data_file']):
            file_path = self.config['data_file']
            print(f"Loading from: {file_path}")
            
            try:
                if str(file_path).endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                print(f"Loaded {len(df)} bars from {Path(file_path).name}")
            except Exception as e:
                raise RuntimeError(f"Error loading file '{file_path}': {e}")
        else:
            raise FileNotFoundError(f"Configured data_file not found: {self.config.get('data_file')}")
        
        # Store dataframe for trade generation
        self.dataframe = df
        

        # Standardize column names
        # CRITICAL: Don't map 'Date' to 'DateTime' - we need to combine Date+Time first!
        column_mapping = {
            'datetime': 'DateTime', 'date': 'DateTime',  # lowercase date is ok
            # 'Date': 'DateTime',  # REMOVED - uppercase Date needs to be combined with Time first
            'timestamp': 'DateTime', 'Timestamp': 'DateTime',
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'volume': 'Volume', 'vol': 'Volume'
        }
        
        for old, new in column_mapping.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        
        # Ensure DateTime column - handle separate Date and Time columns
        if 'DateTime' not in df.columns:
            if 'Date' in df.columns and 'Time' in df.columns:
                # Combine Date and Time columns properly
                # Combine Date and Time columns properly
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            elif 'Time' in df.columns:
                df['DateTime'] = pd.to_datetime(df['Time'])
            elif df.index.name and 'date' in df.index.name.lower():
                df['DateTime'] = df.index
            else:
                raise ValueError("No DateTime/Time column found in input data")
        else:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Ensure required columns
        required = ['Open', 'High', 'Low', 'Close']
        missing_required = [c for c in required if c not in df.columns]
        if missing_required:
            # Try soft mapping by case-insensitive contains
            for col in list(missing_required):
                for c in df.columns:
                    if col.lower() in c.lower():
                        df[col] = df[c]
                        if col in missing_required:
                            missing_required.remove(col)
                        break
        if missing_required:
            raise ValueError(f"Missing required columns after normalization: {missing_required}")
        
        # Add volume if missing (optional)
        if 'Volume' not in df.columns:
            df['Volume'] = np.nan
        
        # AUX fields optional; if not present, set NaN
        if 'AUX1' not in df.columns:
            df['AUX1'] = np.nan
        if 'AUX2' not in df.columns:
            df['AUX2'] = np.nan
        
        # Convert to the format the chart expects
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

        # Set ViewBox limits based on actual data size
        if hasattr(self, 'viewBox'):
            self.viewBox.setLimits(
                xMin=-100,  # Allow some padding
                xMax=self.total_bars + 100,  # Dynamic based on actual data
                yMin=0,  # Price shouldn't be negative
                yMax=100000,  # Large but finite limit
                minXRange=10,  # Minimum 10 bars visible
                maxXRange=self.total_bars + 1000,  # Allow viewing all bars
                minYRange=1,  # Minimum price range
                maxYRange=50000  # Maximum price range
            )
            print(f"ViewBox limits set dynamically: xMax={self.total_bars + 100}")

        # Debug: Verify we have all data including June 2023
        if self.total_bars > 199628:
            june_idx = 199628
            print(f"DEBUG: Checking bar at June 2023 (index {june_idx})...")
            if june_idx < len(self.full_data['timestamp']):
                june_ts = self.full_data['timestamp'][june_idx]
                june_open = self.full_data['open'][june_idx]
                print(f"  Bar {june_idx}: timestamp={june_ts}, open={june_open}")
            print(f"  Total bars available: {self.total_bars}")

        # Load trades based on configuration
        self.load_configured_trades()

        # Update trade panel if it exists
        if self.trade_panel and self.full_data['timestamp'] is not None:
            self.trade_panel.set_chart_timestamps(self.full_data['timestamp'])

            bar_data = {
                'timestamp': self.full_data['timestamp'],  # CRITICAL: Include timestamp!
                'open': self.full_data['open'],
                'high': self.full_data['high'],
                'low': self.full_data['low'],
                'close': self.full_data['close']
            }
            self.trade_panel.set_bar_data(bar_data)

        # Initial render - show recent data instead of just first 500 bars
        if getattr(self, 'total_bars', 0) > 0:
            # Show last 500 bars instead of first 500
            if self.total_bars > 500:
                start_idx = self.total_bars - 500
                end_idx = self.total_bars
            else:
                start_idx = 0
                end_idx = self.total_bars
            # Initialize current_x_range before render
            self.current_x_range = (start_idx, end_idx)
            self.render_range(start_idx, end_idx)
    
    def create_sample_data(self):
        """Disabled: sample data not permitted."""
        raise RuntimeError("Sample data is disabled. Provide a valid data_file.")
    
    def load_configured_trades(self):
        """Load trades based on configuration (no sample mode)."""
        trade_source = self.config.get('trade_source', 'none')

        if trade_source == 'none':
            print("No trades loaded")
            return

        if trade_source == 'csv' and self.config.get('trade_file'):
            loader = CSVTradeLoader()
            trades = loader.load_csv_trades(self.config['trade_file'])
            if trades and hasattr(self, 'load_trades'):
                self.load_trades(trades)
                print(f"Loaded {len(trades)} trades from CSV")
            else:
                raise RuntimeError("Failed to load trades from CSV")

        elif trade_source == 'system' and self.config.get('system'):
            if hasattr(self, 'dataframe'):
                print(f"Generating system trades for {self.config['system']}")
                print(f"DataFrame columns available: {self.dataframe.columns.tolist()}")
                trades = generate_system_trades(self.config['system'], self.dataframe)
                if hasattr(self, 'load_trades'):
                    self.load_trades(trades)
                    print(f"Generated {len(trades)} trades from {self.config['system']}")
            else:
                print("Warning: dataframe not available for system trade generation")

        elif trade_source == 'sample':
            print("Sample trade source is disabled; no trades loaded")

def main():
    """Main entry point"""
    print("="*60)
    print("SAM - PyQtGraph Range Bars Visualization")
    print("With Data Selection Dialog")
    print("="*60)
    
    # Create Qt Application
    app = QtWidgets.QApplication(sys.argv)
    
    # Show data selector dialog
    selector = DataSelectorDialog()
    
    if selector.exec_() == QtWidgets.QDialog.Accepted:
        # Get configuration
        config = selector.get_configuration()
        
        print("\nConfiguration selected:")
        print(f"  Data file: {config['data_file']}")
        print(f"  Trade source: {config['trade_source']}")
        if config['trade_file']:
            print(f"  Trade file: {config['trade_file']}")
        if config['system']:
            print(f"  System: {config['system']}")
        print(f"  Indicators: {config['indicators']}")
        
        # Create the configured chart
        chart = ConfiguredChart(config)
        
        # Show the window
        chart.show()
        
        print("\nChart window opened. Controls:")
        print("  - Mouse wheel: Zoom in/out")
        print("  - Click and drag: Pan")
        print("  - Hover: See price/time/ATR/Range info")
        if config['trade_source'] != 'none':
            print("  - White X marks: Trade locations")
        
        # Run the application
        sys.exit(app.exec_())
    else:
        print("Data selection cancelled")
        sys.exit(0)

if __name__ == "__main__":
    main()