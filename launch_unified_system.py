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

# Import headless backtesting components
# (old unified components removed - now using CSV-based headless system)

# Import legacy components for compatibility
from src.trading.visualization.trade_data import TradeCollection, TradeData
from csv_trade_loader import CSVTradeLoader


def load_config():
    """Load configuration from config.yaml"""
    config_path = "C:\\code\\PythonBT\\tradingCode\\config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


# Removed old generate_unified_trades function - now using headless CSV system


class UnifiedConfiguredChart(RangeBarChartFinal):
    """Chart that uses headless CSV system for trades"""

    def __init__(self, config=None):
        # Set config BEFORE parent init so parent skips load_data()
        self.config = config or {}
        # Initialize required attributes
        self.full_data = None
        self.current_x_range = None
        self.is_rendering = False
        # Call parent init - this sets up UI and hover mechanism
        super().__init__()

        # Add new button controls AFTER parent init
        self.add_trading_controls()

        # Now load our data (but NOT trades - wait for button clicks)
        self.load_data()

    def add_trading_controls(self):
        """Add trading control buttons to the chart"""

        # Create controls widget
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout()
        controls_widget.setMaximumHeight(60)
        controls_widget.setStyleSheet("QWidget { background-color: #f0f0f0; border: 1px solid #ccc; }")

        # Button 1: Load Previous Backtests
        load_previous_btn = QtWidgets.QPushButton("📁 Load Previous Backtests")
        load_previous_btn.clicked.connect(self.load_previous_backtests)
        load_previous_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        controls_layout.addWidget(load_previous_btn)

        # Button 2: Run Strategy + Auto Load
        run_strategy_btn = QtWidgets.QPushButton("🔄 Run Strategy + Load")
        run_strategy_btn.clicked.connect(self.run_strategy_and_load)
        run_strategy_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        controls_layout.addWidget(run_strategy_btn)

        # Clear trades button
        clear_btn = QtWidgets.QPushButton("🧹 Clear")
        clear_btn.clicked.connect(self.clear_trades)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px 15px;
                border-radius: 5px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #da190b; }
        """)
        controls_layout.addWidget(clear_btn)

        # Status label
        self.status_label = QtWidgets.QLabel("Chart loaded - Ready for trading")
        self.status_label.setStyleSheet("QLabel { color: #666; font-size: 12px; padding: 10px; font-weight: bold; }")
        controls_layout.addWidget(self.status_label)

        controls_layout.addStretch()

        # System info
        info_label = QtWidgets.QLabel("INTEGRATED HEADLESS SYSTEM")
        info_label.setStyleSheet("QLabel { color: #ff9800; font-weight: bold; font-size: 11px; }")
        controls_layout.addWidget(info_label)

        controls_widget.setLayout(controls_layout)

        # Insert at top of main widget
        if hasattr(self, 'layout') and self.layout():
            self.layout().insertWidget(0, controls_widget)
        else:
            # If no layout exists, create one
            main_widget = QtWidgets.QWidget()
            main_layout = QtWidgets.QVBoxLayout()
            main_layout.addWidget(controls_widget)
            main_layout.addWidget(self)  # Add the chart itself
            main_widget.setLayout(main_layout)

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
        """Load trades using headless CSV system"""
        trade_source = self.config.get('trade_source', 'None')

        print(f"[UNIFIED CHART] Trade source selected: {trade_source}")

        if trade_source == 'CSV':
            csv_file = self.config.get('csv_file')
            if csv_file and os.path.exists(csv_file):
                loader = CSVTradeLoader()
                self.trades = loader.load_csv_trades(csv_file)
                print(f"[UNIFIED CHART] Loaded {len(self.trades)} trades from CSV")

        elif trade_source == 'System':
            # Use HEADLESS BACKTESTING instead of old unified engine
            system_name = self.config.get('trading_system', 'Simple Moving Average')
            print(f"[UNIFIED CHART] Running headless backtest for system: {system_name}")

            try:
                from src.trading.backtesting.headless_backtester import HeadlessBacktester
                from src.trading.visualization.backtest_result_loader import BacktestResultLoader

                # Map system names to strategy names
                strategy_map = {
                    "Simple Moving Average": "sma_crossover",
                    "RSI Momentum": "rsi_momentum"
                }
                strategy_name = strategy_map.get(system_name, "sma_crossover")

                # Set up parameters based on system
                if strategy_name == "sma_crossover":
                    params = {
                        'fast_period': 20,
                        'slow_period': 50,
                        'long_only': True,
                        'signal_lag': 2,
                        'position_size': 1.0,
                        'min_execution_time': 5.0
                    }
                else:
                    params = {
                        'rsi_period': 14,
                        'oversold': 30,
                        'overbought': 70,
                        'signal_lag': 2,
                        'position_size': 1.0
                    }

                # Check if we have the data file configured
                data_file = self.config.get('data_file', 'data/sample_trading_data_small.csv')
                if not os.path.exists(data_file):
                    data_file = 'data/sample_trading_data_small.csv'

                print(f"[UNIFIED CHART] Running headless backtest with {data_file}")

                # Run headless backtest
                backtester = HeadlessBacktester()
                run_id = backtester.run_backtest(
                    strategy_name=strategy_name,
                    parameters=params,
                    data_file=data_file,
                    execution_mode='standard'
                )

                # Load results from CSV
                loader = BacktestResultLoader()
                csv_trades = loader.load_trade_list(run_id)

                if csv_trades and len(csv_trades) > 0:
                    # Convert to legacy trade format for chart compatibility
                    legacy_trades = []
                    for trade in csv_trades:
                        legacy_trade = TradeData(
                            bar_index=trade.bar_index,
                            trade_type=trade.trade_type,
                            price=trade.price,
                            trade_id=getattr(trade, 'trade_id', len(legacy_trades)),
                            timestamp=trade.timestamp,
                            pnl=getattr(trade, 'pnl', 0),
                            strategy=trade.strategy
                        )
                        legacy_trades.append(legacy_trade)

                    self.trades = TradeCollection(legacy_trades)
                    print(f"[UNIFIED CHART] Loaded {len(self.trades)} trades from headless CSV")

                    # Show P&L info if available
                    if len(self.trades) > 0:
                        print("\n[UNIFIED CHART] First 3 trades from CSV:")
                        for i, trade in enumerate(self.trades[:3]):
                            print(f"  Trade {i}: {trade.trade_type} at bar {trade.bar_index}, price ${trade.price:.2f}")
                            if hasattr(trade, 'pnl'):
                                print(f"    P&L: {trade.pnl:.2f}")

                else:
                    print("[UNIFIED CHART] No trades generated from headless backtest")
                    self.trades = TradeCollection([])

            except Exception as e:
                print(f"[UNIFIED CHART] Error running headless backtest: {e}")
                print("[UNIFIED CHART] Falling back to no trades")
                self.trades = TradeCollection([])

        elif trade_source == 'Sample':
            print("[UNIFIED CHART] Sample trades not supported - use System instead")
            self.trades = TradeCollection([])

        else:
            self.trades = TradeCollection([])

        # Load trades into chart
        if self.trades and len(self.trades) > 0:
            self.load_trades(self.trades)

    def load_previous_backtests(self):
        """Button 1: Load previous backtests from folder structure"""
        print("[UNIFIED CHART] Loading previous backtests...")

        try:
            from src.trading.visualization.backtest_result_loader import BacktestResultLoader

            # Get available backtest runs
            loader = BacktestResultLoader()
            runs = loader.list_available_runs()

            if not runs:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Previous Backtests",
                    "No backtest results found in backtest_results/ folder.\n\nTo create backtests:\n1. Click 'Run Strategy + Load'\n2. Or run headless backtests separately"
                )
                return

            # Create selection dialog with detailed info
            items = []
            for run in runs:
                timestamp = run['timestamp']
                readable_time = f"{timestamp[6:8]}/{timestamp[4:6]}/{timestamp[0:4]} {timestamp[9:11]}:{timestamp[11:13]}"
                strategy = run['strategy_name'].replace('_', ' ').title()
                execution = run['execution_mode'].upper()
                items.append(f"{strategy} - {readable_time} ({execution})")

            item, ok = QtWidgets.QInputDialog.getItem(
                self, "Load Previous Backtest",
                f"Select from {len(runs)} available backtest runs:",
                items, 0, False
            )

            if ok and item:
                selected_run = runs[items.index(item)]
                print(f"[UNIFIED CHART] Loading run: {selected_run['run_id']}")

                # Load trades from CSV
                csv_trades = loader.load_trade_list(selected_run['run_id'])

                if csv_trades and len(csv_trades) > 0:
                    # Convert to legacy trade format for chart compatibility
                    legacy_trades = []
                    for trade in csv_trades:
                        legacy_trade = TradeData(
                            bar_index=trade.bar_index,
                            trade_type=trade.trade_type,
                            price=trade.price,
                            trade_id=len(legacy_trades),
                            timestamp=trade.timestamp,
                            pnl=getattr(trade, 'pnl', 0),
                            strategy=trade.strategy
                        )
                        # Preserve metadata if available
                        if hasattr(trade, 'metadata') and trade.metadata:
                            legacy_trade.metadata = trade.metadata
                        legacy_trades.append(legacy_trade)

                    trade_collection = TradeCollection(legacy_trades)

                    # Load into chart
                    self.load_trades(trade_collection)

                    # Update status
                    strategy_name = selected_run['strategy_name'].replace('_', ' ').title()
                    self.status_label.setText(f"✅ Loaded {len(legacy_trades)} trades from {strategy_name} backtest")

                    print(f"[UNIFIED CHART] Successfully loaded {len(legacy_trades)} trades from CSV")

                    # Show summary
                    buy_count = sum(1 for t in legacy_trades if t.trade_type == 'BUY')
                    sell_count = sum(1 for t in legacy_trades if t.trade_type == 'SELL')
                    print(f"[UNIFIED CHART] Trade summary: {buy_count} BUY, {sell_count} SELL")

                else:
                    QtWidgets.QMessageBox.warning(
                        self, "No Trades Found",
                        f"No trades found in backtest run:\n{selected_run['run_id']}"
                    )

        except Exception as e:
            print(f"[UNIFIED CHART] Error loading previous backtests: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Error Loading Backtests",
                f"Failed to load previous backtests:\n\n{e}\n\nCheck that backtest_results/ folder exists with valid CSV files."
            )

    def run_strategy_and_load(self):
        """Button 2: Run headless backtest + auto-load results"""
        print("[UNIFIED CHART] Running strategy and auto-loading results...")

        try:
            from src.trading.backtesting.headless_backtester import HeadlessBacktester
            from src.trading.visualization.backtest_result_loader import BacktestResultLoader

            # Get strategy parameters from current config
            system_name = self.config.get('trading_system', 'Simple Moving Average')
            data_file = self.config.get('data_file', 'data/sample_trading_data_small.csv')

            # Map system names to strategy names
            strategy_map = {
                "Simple Moving Average": "sma_crossover",
                "RSI Momentum": "rsi_momentum"
            }
            strategy_name = strategy_map.get(system_name, "sma_crossover")

            # Set up parameters based on system
            if strategy_name == "sma_crossover":
                params = {
                    'fast_period': 20,
                    'slow_period': 50,
                    'long_only': True,
                    'signal_lag': 2,
                    'position_size': 1.0,
                    'min_execution_time': 5.0
                }
            else:
                params = {
                    'rsi_period': 14,
                    'oversold': 30,
                    'overbought': 70,
                    'signal_lag': 2,
                    'position_size': 1.0
                }

            # Check data file exists
            if not os.path.exists(data_file):
                data_file = 'data/sample_trading_data_small.csv'
                if not os.path.exists(data_file):
                    QtWidgets.QMessageBox.warning(
                        self, "Data File Missing",
                        "No data file found. Please select a data file with range bars or price data."
                    )
                    return

            # Show progress dialog
            progress = QtWidgets.QProgressDialog("Running headless backtest...", "Cancel", 0, 0, self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setAutoClose(False)
            progress.show()
            QtWidgets.QApplication.processEvents()

            # Update status
            self.status_label.setText(f"🔄 Running {system_name} backtest...")
            QtWidgets.QApplication.processEvents()

            print(f"[UNIFIED CHART] Running {strategy_name} with {data_file}")
            print(f"[UNIFIED CHART] Parameters: {params}")

            # Run headless backtest
            backtester = HeadlessBacktester()
            run_id = backtester.run_backtest(
                strategy_name=strategy_name,
                parameters=params,
                data_file=data_file,
                execution_mode='standard'
            )

            progress.setLabelText("Loading results...")
            QtWidgets.QApplication.processEvents()

            # Load results automatically
            loader = BacktestResultLoader()
            csv_trades = loader.load_trade_list(run_id)

            progress.close()

            if csv_trades and len(csv_trades) > 0:
                # Convert to legacy trade format
                legacy_trades = []
                for trade in csv_trades:
                    legacy_trade = TradeData(
                        bar_index=trade.bar_index,
                        trade_type=trade.trade_type,
                        price=trade.price,
                        trade_id=len(legacy_trades),
                        timestamp=trade.timestamp,
                        pnl=getattr(trade, 'pnl', 0),
                        strategy=trade.strategy
                    )
                    # Preserve metadata
                    if hasattr(trade, 'metadata') and trade.metadata:
                        legacy_trade.metadata = trade.metadata
                    legacy_trades.append(legacy_trade)

                trade_collection = TradeCollection(legacy_trades)

                # Load into chart automatically
                self.load_trades(trade_collection)

                # Update status with success message
                strategy_display = system_name
                self.status_label.setText(f"✅ Generated & loaded {len(legacy_trades)} trades from {strategy_display}")

                print(f"[UNIFIED CHART] Successfully generated and loaded {len(legacy_trades)} trades")

                # Show success message
                buy_count = sum(1 for t in legacy_trades if t.trade_type == 'BUY')
                sell_count = sum(1 for t in legacy_trades if t.trade_type == 'SELL')

                QtWidgets.QMessageBox.information(
                    self, "Strategy Complete",
                    f"Successfully generated and loaded trades!\n\n"
                    f"Strategy: {strategy_display}\n"
                    f"Total trades: {len(legacy_trades)}\n"
                    f"BUY trades: {buy_count}\n"
                    f"SELL trades: {sell_count}\n\n"
                    f"Results saved to: {run_id}"
                )

            else:
                progress.close()
                self.status_label.setText("⚠️ Backtest completed but no trades generated")
                QtWidgets.QMessageBox.warning(
                    self, "No Trades Generated",
                    f"Backtest completed but no trades were generated.\n\nThis might be due to:\n- No valid signals in the data\n- Strategy parameters too restrictive\n\nBacktest saved as: {run_id}"
                )

        except Exception as e:
            if 'progress' in locals():
                progress.close()
            print(f"[UNIFIED CHART] Error running strategy: {e}")
            self.status_label.setText("❌ Strategy execution failed")
            QtWidgets.QMessageBox.critical(
                self, "Strategy Execution Failed",
                f"Failed to run strategy:\n\n{e}\n\nCheck console for detailed error information."
            )

    def clear_trades(self):
        """Clear all trades from chart"""
        print("[UNIFIED CHART] Clearing trades...")
        empty_collection = TradeCollection([])
        self.load_trades(empty_collection)
        self.status_label.setText("🧹 Trades cleared - Chart ready")


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

        # Create chart with headless CSV system
        chart = UnifiedConfiguredChart(config)

        # NO automatic trade loading - wait for button clicks
        # This prevents the hanging issue from old load_configured_trades()

        # Using enhanced trade panel with P&L percentage data and sorting capabilities

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