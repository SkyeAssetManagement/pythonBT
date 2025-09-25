#!/usr/bin/env python3
"""
Simple Headless System Launcher
===============================
Direct launch without dialog - loads sample data and headless backtesting
"""

import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')
sys.path.insert(0, 'src/trading')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5 import QtWidgets, QtCore
import pandas as pd

# Import NEW headless components
from src.trading.visualization.integrated_chart_runner import IntegratedChartRunner
from src.trading.visualization.backtest_result_loader import BacktestResultLoader

# Import chart components for display
from pyqtgraph_range_bars_final import RangeBarChartFinal


class SimpleHeadlessChart(RangeBarChartFinal):
    """Simplified chart with automatic data loading and headless integration"""

    def __init__(self):
        # Set up config with sample data
        self.config = {
            'data_file': 'data/sample_trading_data_small.csv',
            'trade_source': 'System',
            'trading_system': 'Simple Moving Average'
        }

        self.runner = IntegratedChartRunner()
        super().__init__()

        print(f"[SIMPLE] Initialized with data file: {self.config['data_file']}")

        # Add headless backtest buttons
        self.add_headless_controls()

        # Automatically load data
        self.load_data()

    def add_headless_controls(self):
        """Add simplified headless controls"""

        # Create a toolbar-like widget
        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout()
        toolbar.setMaximumHeight(50)

        # Headless backtest button
        headless_btn = QtWidgets.QPushButton("🔄 New Headless Backtest")
        headless_btn.clicked.connect(self.run_new_backtest)
        headless_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        toolbar_layout.addWidget(headless_btn)

        # Load results button
        load_btn = QtWidgets.QPushButton("📁 Load Results")
        load_btn.clicked.connect(self.load_existing_results)
        load_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        toolbar_layout.addWidget(load_btn)

        # Results info label
        self.info_label = QtWidgets.QLabel("Ready - No trades loaded")
        self.info_label.setStyleSheet("QLabel { color: #666; font-size: 12px; }")
        toolbar_layout.addWidget(self.info_label)

        toolbar_layout.addStretch()

        # Quick test button
        test_btn = QtWidgets.QPushButton("⚡ Quick Test")
        test_btn.clicked.connect(self.run_quick_test)
        test_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        toolbar_layout.addWidget(test_btn)

        toolbar.setLayout(toolbar_layout)

        # Insert at top of main widget
        if hasattr(self, 'layout') and self.layout():
            self.layout().insertWidget(0, toolbar)

    def run_new_backtest(self):
        """Run new headless backtest using the integrated workflow"""
        print("[SIMPLE] Running new headless backtest...")

        try:
            result = self.runner.execute_backtest_workflow(parent=self)

            if result:
                trades, metadata = result
                print(f"[SIMPLE] Got {len(trades)} trades from {metadata['source']}")

                self.info_label.setText(f"✅ {len(trades)} trades from {metadata['strategy_name']} ({metadata['execution_mode']})")

                # Load trades into chart
                self.load_headless_trades(trades)

            else:
                print("[SIMPLE] No results - cancelled or error")
                self.info_label.setText("❌ No results - cancelled")

        except Exception as e:
            print(f"[SIMPLE] Error: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Backtest failed: {e}")

    def load_existing_results(self):
        """Load existing results with simple selection"""
        print("[SIMPLE] Loading existing results...")

        try:
            loader = BacktestResultLoader()
            runs = loader.list_available_runs()

            if not runs:
                QtWidgets.QMessageBox.information(self, "No Results",
                    "No backtest results found.\n\nTip: Run 'Quick Test' to generate sample results.")
                return

            # Simple list dialog
            items = []
            for run in runs:
                timestamp = run['timestamp']
                readable_time = f"{timestamp[6:8]}/{timestamp[4:6]} {timestamp[9:11]}:{timestamp[11:13]}"
                items.append(f"{run['strategy_name']} - {readable_time} ({run['execution_mode']})")

            item, ok = QtWidgets.QInputDialog.getItem(
                self, "Load Backtest Results",
                "Select a backtest run to load:", items, 0, False
            )

            if ok and item:
                selected_run = runs[items.index(item)]
                trades = loader.load_trade_list(selected_run['run_id'])

                if trades:
                    print(f"[SIMPLE] Loaded {len(trades)} trades")
                    self.info_label.setText(f"📊 Loaded {len(trades)} trades from {selected_run['strategy_name']}")
                    self.load_headless_trades(trades)
                else:
                    QtWidgets.QMessageBox.warning(self, "Load Error", "No trades found in selected run")

        except Exception as e:
            print(f"[SIMPLE] Error loading: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load results: {e}")

    def run_quick_test(self):
        """Run a quick headless backtest for testing"""
        print("[SIMPLE] Running quick test...")

        try:
            from src.trading.backtesting.headless_backtester import HeadlessBacktester

            backtester = HeadlessBacktester()
            params = {
                'fast_period': 10,
                'slow_period': 30,
                'long_only': False,
                'signal_lag': 2,
                'position_size': 1.0
            }

            # Show progress
            progress = QtWidgets.QProgressDialog("Running quick test...", "Cancel", 0, 0, self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.show()
            QtWidgets.QApplication.processEvents()

            # Run backtest
            run_id = backtester.run_backtest(
                'sma_crossover',
                params,
                'data/sample_trading_data_small.csv',
                'standard'
            )

            progress.close()

            # Load results
            loader = BacktestResultLoader()
            trades = loader.load_trade_list(run_id)

            if trades:
                print(f"[SIMPLE] Quick test generated {len(trades)} trades")
                self.info_label.setText(f"⚡ Quick test: {len(trades)} trades generated")
                self.load_headless_trades(trades)
            else:
                self.info_label.setText("⚠️ Quick test completed but no trades found")

        except Exception as e:
            print(f"[SIMPLE] Quick test error: {e}")
            self.info_label.setText(f"❌ Quick test failed: {str(e)[:50]}...")

    def load_headless_trades(self, trades):
        """Load trades into chart display"""
        print(f"[SIMPLE] Loading {len(trades)} trades into chart...")

        # Convert to chart format
        from trade_data import TradeCollection, TradeData

        chart_trades = []
        for trade in trades:
            chart_trade = TradeData(
                bar_index=trade.bar_index,
                trade_type=trade.trade_type,
                price=trade.price,
                timestamp=trade.timestamp,
                size=trade.size,
                strategy=trade.strategy
            )

            # Preserve metadata if available
            if hasattr(trade, 'metadata') and trade.metadata:
                chart_trade.metadata = trade.metadata

            chart_trades.append(chart_trade)

        trade_collection = TradeCollection(chart_trades)
        self.load_trades(trade_collection)

        print(f"[SIMPLE] Successfully loaded trades into chart")

    def load_data(self):
        """Load chart data automatically"""
        data_file = self.config.get('data_file')

        if not data_file or not os.path.exists(data_file):
            print(f"[SIMPLE] Data file not found: {data_file}, using sample data")
            self.generate_sample_data()
            return

        print(f"[SIMPLE] Loading data from: {data_file}")

        try:
            df = pd.read_csv(data_file)

            # Process datetime
            if 'Date' in df.columns and 'Time' in df.columns:
                df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
            else:
                df['DateTime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='5min')

            # Convert to chart format
            self.full_data = {
                'timestamp': df['DateTime'].values,
                'open': df['Open'].values.astype(float),
                'high': df['High'].values.astype(float),
                'low': df['Low'].values.astype(float),
                'close': df['Close'].values.astype(float),
                'volume': df['Volume'].values.astype(float) if 'Volume' in df.columns else None
            }

            self.total_bars = len(df)
            self.current_x_range = (0, min(1000, self.total_bars))

            print(f"[SIMPLE] Loaded {len(df)} bars, price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

            # Initial render
            self.render_range(0, min(1000, self.total_bars))

        except Exception as e:
            print(f"[SIMPLE] Error loading data: {e}")
            self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample data as fallback"""
        import numpy as np

        print("[SIMPLE] Generating sample data...")
        num_bars = 1000
        dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='5min')
        prices = 4000 + np.cumsum(np.random.randn(num_bars) * 2)

        self.full_data = {
            'timestamp': dates.to_numpy(),
            'open': prices + np.random.randn(num_bars) * 1,
            'high': prices + np.abs(np.random.randn(num_bars) * 2),
            'low': prices - np.abs(np.random.randn(num_bars) * 2),
            'close': prices,
            'volume': np.random.randint(1000, 10000, num_bars).astype(float)
        }

        # Fix OHLC relationships
        import numpy as np
        self.full_data['high'] = np.maximum.reduce([
            self.full_data['open'], self.full_data['high'],
            self.full_data['low'], self.full_data['close']
        ])
        self.full_data['low'] = np.minimum.reduce([
            self.full_data['open'], self.full_data['high'],
            self.full_data['low'], self.full_data['close']
        ])

        self.total_bars = num_bars
        self.current_x_range = (0, min(1000, self.total_bars))

        print(f"[SIMPLE] Generated {num_bars} bars of sample data")
        self.render_range(0, min(1000, self.total_bars))


def main():
    """Simple launcher - no dialogs, direct data loading"""
    print("\n" + "=" * 70)
    print("SIMPLE HEADLESS BACKTESTING SYSTEM")
    print("Auto-loads sample data with headless backtest integration")
    print("=" * 70)

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)

    # Create main window
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle('Simple Headless System - Chart Data + P&L Backtests')
    main_window.resize(1600, 900)

    # Create simplified chart
    chart = SimpleHeadlessChart()
    main_window.setCentralWidget(chart)

    # Show window
    main_window.show()

    print("[SIMPLE] System ready!")
    print("  - Chart data should be visible")
    print("  - Use 'Quick Test' for immediate backtest")
    print("  - Use 'New Headless Backtest' for full workflow")
    print("  - Use 'Load Results' for existing backtests")

    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()