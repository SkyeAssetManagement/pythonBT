#!/usr/bin/env python3
"""
Chart Only Launcher - NO Strategy Execution
===========================================
Pure chart display that ONLY loads existing CSV trades
NO strategy execution, NO hanging, NO backtesting
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

# Import chart components
from pyqtgraph_data_selector import DataSelectorDialog
from pyqtgraph_range_bars_final import RangeBarChartFinal

# Import CSV loader ONLY
from src.trading.visualization.backtest_result_loader import BacktestResultLoader
from src.trading.visualization.trade_data import TradeCollection, TradeData


class ChartOnlySystem(RangeBarChartFinal):
    """Chart system that ONLY displays data and loads existing CSV trades"""

    def __init__(self, config=None):
        # Set config BEFORE parent init
        self.config = config or {}
        # Initialize required attributes
        self.full_data = None
        self.current_x_range = None
        self.is_rendering = False

        # Call parent init - sets up UI
        super().__init__()

        # Add CSV trade loading controls
        self.add_csv_trade_controls()

        # Load chart data
        self.load_data()

    def add_csv_trade_controls(self):
        """Add controls for CSV trade loading ONLY"""

        # Create controls widget
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout()
        controls_widget.setMaximumHeight(60)

        # Load CSV trades button
        load_csv_btn = QtWidgets.QPushButton("📊 Load CSV Trades")
        load_csv_btn.clicked.connect(self.load_csv_trades)
        load_csv_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        controls_layout.addWidget(load_csv_btn)

        # Clear trades button
        clear_btn = QtWidgets.QPushButton("🧹 Clear Trades")
        clear_btn.clicked.connect(self.clear_trades)
        clear_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 8px; }")
        controls_layout.addWidget(clear_btn)

        # Status label
        self.status_label = QtWidgets.QLabel("Chart loaded - No trades")
        self.status_label.setStyleSheet("QLabel { color: #666; font-size: 12px; padding: 8px; }")
        controls_layout.addWidget(self.status_label)

        controls_layout.addStretch()

        # Info label
        info_label = QtWidgets.QLabel("CHART ONLY MODE - No strategy execution")
        info_label.setStyleSheet("QLabel { color: #ff9800; font-weight: bold; font-size: 11px; }")
        controls_layout.addWidget(info_label)

        controls_widget.setLayout(controls_layout)

        # Insert at top of main widget
        if hasattr(self, 'layout') and self.layout():
            self.layout().insertWidget(0, controls_widget)

    def load_csv_trades(self):
        """Load trades from existing CSV files ONLY"""
        print("[CHART] Loading CSV trades...")

        try:
            # Get available backtest runs
            loader = BacktestResultLoader()
            runs = loader.list_available_runs()

            if not runs:
                QtWidgets.QMessageBox.information(
                    self,
                    "No CSV Trades",
                    "No backtest CSV files found.\n\nTo create trades:\n1. Run headless backtests separately\n2. Use: python test_headless_system.py"
                )
                return

            # Show selection dialog
            items = []
            for run in runs:
                timestamp = run['timestamp']
                readable_time = f"{timestamp[6:8]}/{timestamp[4:6]} {timestamp[9:11]}:{timestamp[11:13]}"
                strategy = run['strategy_name']
                execution = run['execution_mode']
                items.append(f"{strategy} - {readable_time} ({execution})")

            item, ok = QtWidgets.QInputDialog.getItem(
                self, "Load CSV Trades",
                "Select backtest results to load:",
                items, 0, False
            )

            if ok and item:
                selected_run = runs[items.index(item)]
                print(f"[CHART] Loading run: {selected_run['run_id']}")

                # Load trades from CSV
                csv_trades = loader.load_trade_list(selected_run['run_id'])

                if csv_trades and len(csv_trades) > 0:
                    print(f"[CHART] Loaded {len(csv_trades)} trades from CSV")

                    # Convert to chart format
                    chart_trades = []
                    for trade in csv_trades:
                        chart_trade = TradeData(
                            bar_index=trade.bar_index,
                            trade_type=trade.trade_type,
                            price=trade.price,
                            trade_id=len(chart_trades),
                            timestamp=trade.timestamp,
                            pnl=getattr(trade, 'pnl', 0),
                            strategy=trade.strategy
                        )
                        # Preserve metadata if available
                        if hasattr(trade, 'metadata') and trade.metadata:
                            chart_trade.metadata = trade.metadata
                        chart_trades.append(chart_trade)

                    trade_collection = TradeCollection(chart_trades)

                    # Load into chart
                    self.load_trades(trade_collection)

                    # Update status
                    self.status_label.setText(f"✅ Loaded {len(chart_trades)} trades from {selected_run['strategy_name']}")

                    print(f"[CHART] Successfully loaded trades into chart display")

                    # Show first few trades
                    for i, trade in enumerate(chart_trades[:3]):
                        pnl_info = f", P&L: {trade.pnl:.2f}" if hasattr(trade, 'pnl') else ""
                        print(f"  Trade {i+1}: {trade.trade_type} at bar {trade.bar_index}{pnl_info}")

                else:
                    QtWidgets.QMessageBox.warning(self, "No Trades", f"No trades found in {selected_run['run_id']}")

        except Exception as e:
            print(f"[CHART] Error loading CSV trades: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load CSV trades:\n{e}")

    def clear_trades(self):
        """Clear all trades from chart"""
        print("[CHART] Clearing trades...")
        empty_collection = TradeCollection([])
        self.load_trades(empty_collection)
        self.status_label.setText("Chart cleared - No trades")

    def load_data(self):
        """Load chart data - same as original system"""
        print("[CHART] Loading chart data...")
        start_time = datetime.now()

        data_file = self.config.get('data_file')
        if data_file and os.path.exists(data_file):
            print(f"[CHART] Loading from: {data_file}")

            try:
                if str(data_file).endswith('.parquet'):
                    df = pd.read_parquet(data_file)
                else:
                    df = pd.read_csv(data_file)

                print(f"[CHART] Raw data shape: {df.shape}")

                # Handle Date and Time columns - EXACT same logic as original
                if 'Date' in df.columns and 'Time' in df.columns:
                    print("[CHART] Combining Date and Time columns...")
                    if df['Date'].dtype == 'object':
                        df['Date'] = pd.to_datetime(df['Date'])

                    if pd.api.types.is_timedelta64_dtype(df['Time']):
                        df['DateTime'] = df['Date'] + df['Time']
                    else:
                        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

                # Ensure DateTime column
                if 'DateTime' not in df.columns and 'timestamp' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['timestamp'])
                elif 'DateTime' not in df.columns:
                    df['DateTime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='5min')

                # Column mapping - EXACT same as original
                column_mapping = {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                df.rename(columns=column_mapping, inplace=True)

                # Convert to chart data format - EXACT same as original
                self.full_data = {
                    'timestamp': df['DateTime'].values,
                    'DateTime': df['DateTime'].values,
                    'open': df['Open'].values.astype(np.float64),
                    'high': df['High'].values.astype(np.float64),
                    'low': df['Low'].values.astype(np.float64),
                    'close': df['Close'].values.astype(np.float64),
                    'volume': df['Volume'].values.astype(np.float64) if 'Volume' in df.columns else np.zeros(len(df)),
                    # Load ATR if available
                    'aux1': (df['AUX1'].values.astype(np.float64) if 'AUX1' in df.columns else
                            df['ATR'].values.astype(np.float64) if 'ATR' in df.columns else
                            df['atr'].values.astype(np.float64) if 'atr' in df.columns else None),
                    'aux2': (df['AUX2'].values.astype(np.float64) if 'AUX2' in df.columns else
                            np.full(len(df), 0.1, dtype=np.float64) if any(col in df.columns for col in ['AUX1', 'ATR', 'atr']) else None)
                }

                # Calculate returns if not present
                if 'Returns' not in df.columns:
                    self.full_data['Returns'] = np.concatenate([[0], np.diff(self.full_data['close']) / self.full_data['close'][:-1]])

                print(f"[CHART] Loaded {len(self.full_data['timestamp'])} bars")
                load_time = (datetime.now() - start_time).total_seconds()
                print(f"[CHART] Load time: {load_time:.2f} seconds")

                # Set required attributes for chart - EXACT same as original
                self.total_bars = len(self.full_data['timestamp'])
                self.current_x_range = (0, min(1000, self.total_bars))
                self.is_rendering = False

                # Initial render - EXACT same as original
                self.render_range(0, min(1000, self.total_bars))

                self.status_label.setText(f"📈 Chart loaded: {self.total_bars} bars")

            except Exception as e:
                print(f"[CHART] Error loading data: {e}")
                self.generate_sample_data()

        else:
            print(f"[CHART] Data file not found: {data_file}")
            self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample data - EXACT same as original"""
        print("[CHART] Generating sample data...")
        num_bars = 1000
        dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='5min')
        prices = 4000 + np.cumsum(np.random.randn(num_bars) * 2)

        self.full_data = {
            'timestamp': dates.to_numpy(),
            'DateTime': dates.to_numpy(),
            'open': prices + np.random.randn(num_bars) * 1,
            'high': prices + np.abs(np.random.randn(num_bars) * 2),
            'low': prices - np.abs(np.random.randn(num_bars) * 2),
            'close': prices,
            'volume': np.random.randint(1000, 10000, num_bars).astype(np.float64),
            'Returns': np.concatenate([[0], np.diff(prices) / prices[:-1]])
        }

        # Fix OHLC relationships - EXACT same as original
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

        # Set chart attributes
        self.total_bars = num_bars
        self.current_x_range = (0, min(1000, self.total_bars))
        self.is_rendering = False

        # Initial render
        self.render_range(0, min(1000, self.total_bars))

        self.status_label.setText(f"📈 Sample data: {num_bars} bars")


def main():
    """Chart only launcher - NO strategy execution"""
    print("\n" + "=" * 70)
    print("CHART ONLY SYSTEM - NO STRATEGY EXECUTION")
    print("Displays chart data and loads existing CSV trades ONLY")
    print("=" * 70)

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)

    # Show data selector dialog
    dialog = DataSelectorDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        config = dialog.get_configuration()

        # Force trade source to None to prevent any strategy execution
        config['trade_source'] = 'None'

        # Create main window
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle('Chart Only System - Load CSV Trades (NO Strategy Execution)')
        main_window.resize(1600, 900)

        # Create chart with NO strategy execution
        chart = ChartOnlySystem(config)

        # Set as central widget
        main_window.setCentralWidget(chart)

        # Show window
        main_window.show()

        print("[CHART] System ready!")
        print("  - Chart data loaded with 1000 bar rendering")
        print("  - Use 'Load CSV Trades' to display existing trades")
        print("  - NO strategy execution - NO hanging")
        print("  - Run headless backtests separately if needed")

        # Run application
        sys.exit(app.exec_())
    else:
        print("Data selection cancelled")
        sys.exit(0)


if __name__ == '__main__':
    main()