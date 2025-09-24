#!/usr/bin/env python3
"""
Unified System Launcher - Properly Integrated Version
======================================================
Uses the unified execution engine with all new components
"""

import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')
sys.path.insert(0, 'src/trading')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from datetime import datetime
import time
import yaml

# Import chart components
from pyqtgraph_data_selector import DataSelectorDialog
from pyqtgraph_range_bars_final import RangeBarChartFinal

# Import NEW unified components
from core.standalone_execution import ExecutionConfig, StandaloneExecutionEngine
from strategies.strategy_wrapper import StrategyFactory
from core.strategy_runner_adapter import StrategyRunnerAdapter
from core.trade_types import TradeRecord, TradeRecordCollection
from visualization.enhanced_trade_panel import EnhancedTradeListPanel

# Import legacy components for compatibility
from trade_data import TradeCollection, TradeData
from csv_trade_loader import CSVTradeLoader


def load_config():
    """Load configuration from config.yaml"""
    config_path = "C:\\code\\PythonBT\\tradingCode\\config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def generate_unified_trades(system_name, data):
    """Generate trades using the unified execution engine"""
    print(f"\n[UNIFIED] Generating trades with unified engine for: {system_name}")

    # Load config
    config = load_config()
    use_unified = config.get('use_unified_engine', False)

    if not use_unified:
        print("[UNIFIED] Warning: use_unified_engine is false in config.yaml")

    # Create execution config
    exec_config = ExecutionConfig.from_yaml()
    print(f"[UNIFIED] Signal lag: {exec_config.signal_lag} bars")
    print(f"[UNIFIED] Buy formula: {exec_config.buy_execution_formula}")
    print(f"[UNIFIED] Sell formula: {exec_config.sell_execution_formula}")

    # Map system names to strategy names
    strategy_map = {
        "Simple Moving Average": "sma_crossover",
        "RSI Momentum": "rsi_momentum"
    }

    strategy_name = strategy_map.get(system_name, "sma_crossover")

    # Create wrapped strategy with unified execution
    if strategy_name == "sma_crossover":
        wrapped_strategy = StrategyFactory.create_sma_crossover(
            fast_period=20,
            slow_period=50,
            long_only=True,
            execution_config=exec_config
        )
    elif strategy_name == "rsi_momentum":
        wrapped_strategy = StrategyFactory.create_rsi_momentum(
            rsi_period=14,
            oversold=30,
            overbought=70,
            long_only=True,
            execution_config=exec_config
        )
    else:
        print(f"[UNIFIED] Unknown strategy: {system_name}")
        return TradeCollection([])

    # Execute trades with unified engine
    trade_records = wrapped_strategy.execute_trades(data)

    print(f"[UNIFIED] Generated {len(trade_records)} trades")

    # Show first few trades to verify lag
    if len(trade_records) > 0:
        print("\n[UNIFIED] First 3 trades with execution details:")
        for trade in trade_records[:3]:
            print(f"  Trade {trade.trade_id}: {trade.trade_type}")
            print(f"    Signal bar: {trade.signal_bar}, Execution bar: {trade.execution_bar}")
            print(f"    Lag: {trade.lag} bars")
            print(f"    Price: ${trade.price:.2f}")
            if trade.pnl_percent is not None:
                print(f"    P&L: {trade.pnl_percent:.2f}%")

    # Convert to legacy format for compatibility with chart
    # But keep P&L percentage info
    legacy_trades = []
    for trade in trade_records:
        legacy_trade = TradeData(
            bar_index=trade.bar_index,
            trade_type=trade.trade_type,
            price=trade.price,
            trade_id=trade.trade_id,
            timestamp=trade.timestamp,
            pnl=trade.pnl_points,
            strategy=trade.strategy
        )
        # Add percentage P&L as custom attribute
        if trade.pnl_percent is not None:
            legacy_trade.pnl_percent = trade.pnl_percent
        legacy_trades.append(legacy_trade)

    return TradeCollection(legacy_trades)


class UnifiedConfiguredChart(RangeBarChartFinal):
    """Chart that uses unified execution engine for trades"""

    def __init__(self, config=None):
        # Set config BEFORE parent init so parent skips load_data()
        self.config = config or {}
        # Initialize required attributes
        self.full_data = None
        self.current_x_range = None
        self.is_rendering = False
        # Call parent init - this sets up UI and hover mechanism
        super().__init__()
        # Now load our data with unified execution
        self.load_data()

    def load_data(self):
        """Load data with unified execution"""
        print("[UNIFIED CHART] Loading configured data...")
        start_time = time.time()

        if self.config.get('data_file') and os.path.exists(self.config['data_file']):
            file_path = self.config['data_file']
            print(f"[UNIFIED CHART] Loading from: {file_path}")

            try:
                if str(file_path).endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)

                print(f"[UNIFIED CHART] Raw data shape: {df.shape}")
                print(f"[UNIFIED CHART] Columns: {df.columns.tolist()}")

                # Handle Date and Time columns
                if 'Date' in df.columns and 'Time' in df.columns:
                    print("[UNIFIED CHART] Combining Date and Time columns...")
                    # Convert Date to datetime if needed
                    if df['Date'].dtype == 'object':
                        df['Date'] = pd.to_datetime(df['Date'])

                    # Handle Time column - could be timedelta or string
                    if pd.api.types.is_timedelta64_dtype(df['Time']):
                        # Time is already timedelta, combine properly
                        df['DateTime'] = df['Date'] + df['Time']
                    else:
                        # Time is string, combine as strings
                        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

                # Ensure we have DateTime column
                if 'DateTime' not in df.columns and 'timestamp' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['timestamp'])
                elif 'DateTime' not in df.columns:
                    df['DateTime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='5min')

                # Rename columns to standard names (but preserve Date separately)
                column_mapping = {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                df.rename(columns=column_mapping, inplace=True)

                # Convert to numpy arrays with proper dtypes
                # IMPORTANT: Chart expects both 'timestamp' and lowercase OHLC keys
                self.full_data = {
                    'timestamp': df['DateTime'].values,  # Chart expects 'timestamp'
                    'DateTime': df['DateTime'].values,   # Keep for compatibility
                    'open': df['Open'].values.astype(np.float64),
                    'high': df['High'].values.astype(np.float64),
                    'low': df['Low'].values.astype(np.float64),
                    'close': df['Close'].values.astype(np.float64),
                    'volume': df['Volume'].values.astype(np.float64) if 'Volume' in df.columns else np.zeros(len(df)),
                    # Load ATR if available - check multiple column names
                    'aux1': (df['AUX1'].values.astype(np.float64) if 'AUX1' in df.columns else
                            df['ATR'].values.astype(np.float64) if 'ATR' in df.columns else
                            df['atr'].values.astype(np.float64) if 'atr' in df.columns else None),
                    # Load multiplier or use default
                    'aux2': (df['AUX2'].values.astype(np.float64) if 'AUX2' in df.columns else
                            np.full(len(df), 0.1, dtype=np.float64) if any(col in df.columns for col in ['AUX1', 'ATR', 'atr']) else None)
                }

                # Calculate returns if not present
                if 'Returns' not in df.columns:
                    self.full_data['Returns'] = np.concatenate([[0], np.diff(self.full_data['close']) / self.full_data['close'][:-1]])

                print(f"[UNIFIED CHART] Loaded {len(self.full_data['timestamp'])} bars")
                print(f"[UNIFIED CHART] Load time: {time.time() - start_time:.2f} seconds")

                # Debug ATR loading
                if self.full_data.get('aux1') is not None:
                    atr_values = self.full_data['aux1']
                    non_zero = atr_values[atr_values != 0]
                    print(f"[UNIFIED CHART] ATR data loaded: {len(non_zero)} non-zero values out of {len(atr_values)}")
                    if len(non_zero) > 0:
                        print(f"[UNIFIED CHART] ATR range: {non_zero.min():.2f} - {non_zero.max():.2f}, mean: {non_zero.mean():.2f}")
                else:
                    print("[UNIFIED CHART] No ATR data found in file")

                # Set required attributes for chart
                self.total_bars = len(self.full_data['timestamp'])
                self.current_x_range = (0, min(1000, self.total_bars))  # Initialize x range
                self.is_rendering = False

                # CRITICAL: Pass data to trade panel and strategy runner
                if self.trade_panel and self.full_data['timestamp'] is not None:
                    print("[UNIFIED CHART] Passing data to trade panel and strategy runner...")
                    self.trade_panel.set_chart_timestamps(self.full_data['timestamp'])

                    # Pass bar data for strategy runner
                    bar_data = {
                        'timestamp': self.full_data['timestamp'],
                        'open': self.full_data['open'],
                        'high': self.full_data['high'],
                        'low': self.full_data['low'],
                        'close': self.full_data['close'],
                        'volume': self.full_data.get('volume', np.zeros(len(self.full_data['close'])))
                    }
                    self.trade_panel.set_bar_data(bar_data)
                    print("[UNIFIED CHART] Data passed to strategy runner successfully")

                # Trigger initial render
                self.render_range(0, min(1000, self.total_bars))

            except Exception as e:
                print(f"[UNIFIED CHART] Error loading file: {e}")
                self.generate_sample_data()
                # Set required attributes for chart
                self.total_bars = len(self.full_data['timestamp'])
                self.current_x_range = (0, min(1000, self.total_bars))
                self.is_rendering = False
                self.pass_data_to_trade_panel()
                self.render_range(0, min(1000, self.total_bars))
        else:
            print("[UNIFIED CHART] No data file configured, using sample data")
            self.generate_sample_data()
            # Set required attributes for chart
            self.total_bars = len(self.full_data['timestamp'])
            self.current_x_range = (0, min(1000, self.total_bars))
            self.is_rendering = False
            self.pass_data_to_trade_panel()
            self.render_range(0, min(1000, self.total_bars))

    def pass_data_to_trade_panel(self):
        """Helper method to pass data to trade panel and strategy runner"""
        if self.trade_panel and self.full_data and self.full_data.get('timestamp') is not None:
            print("[UNIFIED CHART] Passing data to trade panel and strategy runner...")
            self.trade_panel.set_chart_timestamps(self.full_data['timestamp'])

            # Pass bar data for strategy runner
            bar_data = {
                'timestamp': self.full_data['timestamp'],
                'open': self.full_data['open'],
                'high': self.full_data['high'],
                'low': self.full_data['low'],
                'close': self.full_data['close'],
                'volume': self.full_data.get('volume', np.zeros(len(self.full_data['close'])))
            }
            self.trade_panel.set_bar_data(bar_data)
            print("[UNIFIED CHART] Data passed to strategy runner successfully")

    def generate_sample_data(self):
        """Generate sample data as fallback"""
        print("[UNIFIED CHART] Generating sample data...")
        num_bars = 500
        dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='5min')
        prices = 4000 + np.cumsum(np.random.randn(num_bars) * 2)

        self.full_data = {
            'timestamp': dates.to_numpy(),  # Chart expects 'timestamp'
            'DateTime': dates.to_numpy(),   # Keep for compatibility
            'open': prices + np.random.randn(num_bars) * 1,
            'high': prices + np.abs(np.random.randn(num_bars) * 2),
            'low': prices - np.abs(np.random.randn(num_bars) * 2),
            'close': prices,
            'volume': np.random.randint(1000, 10000, num_bars).astype(np.float64),
            'Returns': np.concatenate([[0], np.diff(prices) / prices[:-1]])
        }

        # Ensure High is highest and Low is lowest
        self.full_data['high'] = np.maximum.reduce([
            self.full_data['open'],
            self.full_data['high'],
            self.full_data['low'],
            self.full_data['close']
        ])
        self.full_data['low'] = np.minimum.reduce([
            self.full_data['open'],
            self.full_data['high'],
            self.full_data['low'],
            self.full_data['close']
        ])

    def load_configured_trades(self):
        """Load trades using unified execution engine"""
        trade_source = self.config.get('trade_source', 'None')

        print(f"[UNIFIED CHART] Trade source selected: {trade_source}")

        if trade_source == 'CSV':
            csv_file = self.config.get('csv_file')
            if csv_file and os.path.exists(csv_file):
                loader = CSVTradeLoader()
                self.trades = loader.load_csv_trades(csv_file)
                print(f"[UNIFIED CHART] Loaded {len(self.trades)} trades from CSV")

        elif trade_source == 'System':
            # Use UNIFIED execution engine
            system_name = self.config.get('trading_system', 'Simple Moving Average')
            print(f"[UNIFIED CHART] Generating trades for system: {system_name}")

            # Convert full_data dict to DataFrame for strategies
            # Note: strategies expect uppercase column names
            df = pd.DataFrame({
                'DateTime': self.full_data['timestamp'],  # Use timestamp array
                'Open': self.full_data['open'],
                'High': self.full_data['high'],
                'Low': self.full_data['low'],
                'Close': self.full_data['close'],
                'Volume': self.full_data['volume']
            })

            # Generate trades with unified engine
            self.trades = generate_unified_trades(system_name, df)
            print(f"[UNIFIED CHART] Generated {len(self.trades)} trades with unified engine")

            # Verify lag in displayed trades
            if len(self.trades) > 0:
                print("\n[UNIFIED CHART] Verifying execution lag in chart trades:")
                for i, trade in enumerate(self.trades[:3]):
                    print(f"  Trade {i}: {trade.trade_type} at bar {trade.bar_index}")
                    if hasattr(trade, 'pnl_percent'):
                        print(f"    P&L: {trade.pnl_percent:.2f}%")

        elif trade_source == 'Sample':
            print("[UNIFIED CHART] Sample trades not supported in unified system")
            self.trades = TradeCollection([])

        else:
            self.trades = TradeCollection([])

        # Load trades into chart
        if self.trades and len(self.trades) > 0:
            self.load_trades(self.trades)


def main():
    """Main entry point for unified system launcher"""
    print("\n" + "=" * 80)
    print("UNIFIED TRADING SYSTEM LAUNCHER")
    print("Using unified execution engine with lag and formulas")
    print("=" * 80)

    # Check config
    config = load_config()
    use_unified = config.get('use_unified_engine', False)

    if use_unified:
        print("[OK] Unified engine ENABLED in config.yaml")
    else:
        print("[WARNING] Unified engine DISABLED in config.yaml - set 'use_unified_engine: true' to enable")

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)

    # Show data selector dialog
    dialog = DataSelectorDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        config = dialog.get_configuration()

        # Create main window
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle('PyQtGraph Range Bars - UNIFIED SYSTEM')
        main_window.resize(1600, 900)

        # Create chart with unified execution
        chart = UnifiedConfiguredChart(config)

        # Load trades with unified engine - AFTER chart is initialized
        QtCore.QTimer.singleShot(100, lambda: chart.load_configured_trades())

        # TODO: Replace trade panel with enhanced version
        # For now, using existing panel but trades have P&L percentage data

        # Set as central widget
        main_window.setCentralWidget(chart)

        # Show window
        main_window.show()

        # Run application
        sys.exit(app.exec_())
    else:
        print("Data selection cancelled")
        sys.exit(0)


if __name__ == '__main__':
    main()