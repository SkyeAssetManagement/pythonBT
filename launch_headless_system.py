#!/usr/bin/env python3
"""
Headless Backtesting System Launcher
====================================
Uses the new headless backtesting architecture with P&L calculations
and integrated chart runner (load existing results OR run new backtests)
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
from src.trading.backtesting.headless_backtester import HeadlessBacktester

# Import chart components for display
from pyqtgraph_data_selector import DataSelectorDialog
from pyqtgraph_range_bars_final import RangeBarChartFinal


class HeadlessSystemChart(RangeBarChartFinal):
    """Chart that integrates with headless backtesting system"""

    def __init__(self, config=None):
        self.config = config or {}
        self.runner = IntegratedChartRunner()
        super().__init__()

        # Add headless backtest button to UI
        self.add_headless_controls()

    def add_headless_controls(self):
        """Add controls for headless backtesting"""

        # Create controls widget
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout()

        # Headless backtest button
        self.headless_btn = QtWidgets.QPushButton("Headless Backtest")
        self.headless_btn.clicked.connect(self.run_headless_workflow)
        controls_layout.addWidget(self.headless_btn)

        # Load results button
        self.load_results_btn = QtWidgets.QPushButton("Load Results")
        self.load_results_btn.clicked.connect(self.load_existing_results)
        controls_layout.addWidget(self.load_results_btn)

        # Results info label
        self.results_label = QtWidgets.QLabel("No results loaded")
        controls_layout.addWidget(self.results_label)

        controls_layout.addStretch()
        controls_widget.setLayout(controls_layout)

        # Add to main layout (assuming parent has layout)
        if hasattr(self, 'main_layout'):
            self.main_layout.insertWidget(0, controls_widget)

    def run_headless_workflow(self):
        """Run the headless backtest workflow"""
        print("[HEADLESS] Starting headless backtest workflow...")

        try:
            # Execute backtest workflow (load existing OR run new)
            result = self.runner.execute_backtest_workflow(parent=self)

            if result:
                trades, metadata = result
                print(f"[HEADLESS] Got {len(trades)} trades from {metadata['source']}")

                # Update results label
                self.results_label.setText(f"Loaded {len(trades)} trades from {metadata['strategy_name']} ({metadata['execution_mode']})")

                # Load trades into chart
                self.load_headless_trades(trades, metadata)

            else:
                print("[HEADLESS] No results - user cancelled or error occurred")
                self.results_label.setText("No results - cancelled or error")

        except Exception as e:
            print(f"[HEADLESS] Error in workflow: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Headless backtest failed: {e}")

    def load_existing_results(self):
        """Load existing backtest results directly"""
        print("[HEADLESS] Loading existing results...")

        try:
            loader = BacktestResultLoader()
            runs = loader.list_available_runs()

            if not runs:
                QtWidgets.QMessageBox.information(self, "No Results", "No backtest results found in backtest_results folder.")
                return

            # Show selection dialog
            items = [f"{run['strategy_name']} ({run['timestamp']}) - {run['execution_mode']}" for run in runs]
            item, ok = QtWidgets.QInputDialog.getItem(
                self, "Select Backtest Run", "Choose run to load:", items, 0, False
            )

            if ok and item:
                selected_run = runs[items.index(item)]

                # Load trades
                trades = loader.load_trade_list(selected_run['run_id'])

                if trades:
                    print(f"[HEADLESS] Loaded {len(trades)} trades from {selected_run['run_id']}")

                    # Update results label
                    self.results_label.setText(f"Loaded {len(trades)} trades from {selected_run['strategy_name']} ({selected_run['execution_mode']})")

                    # Prepare metadata
                    metadata = {
                        'run_id': selected_run['run_id'],
                        'strategy_name': selected_run['strategy_name'],
                        'execution_mode': selected_run['execution_mode'],
                        'source': 'loaded_from_csv'
                    }

                    # Load trades into chart
                    self.load_headless_trades(trades, metadata)

                else:
                    QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load trades from {selected_run['run_id']}")

        except Exception as e:
            print(f"[HEADLESS] Error loading results: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load results: {e}")

    def load_headless_trades(self, trades, metadata):
        """Load trades from headless backtest into chart display"""

        print(f"[HEADLESS] Loading {len(trades)} trades into chart...")
        print(f"[HEADLESS] Strategy: {metadata['strategy_name']}")
        print(f"[HEADLESS] Execution mode: {metadata['execution_mode']}")

        # Show first few trades with P&L info
        for i, trade in enumerate(trades[:3]):
            print(f"  Trade {i+1}: {trade.trade_type} at {trade.price:.2f}")
            if hasattr(trade, 'metadata') and trade.metadata:
                exec_bars = trade.metadata.get('exec_bars', 'N/A')
                print(f"    execBars: {exec_bars}")

        # Convert to format expected by chart
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

            # Preserve TWAP metadata if available
            if hasattr(trade, 'metadata') and trade.metadata:
                chart_trade.metadata = trade.metadata

            chart_trades.append(chart_trade)

        trade_collection = TradeCollection(chart_trades)

        # Load into chart
        self.load_trades(trade_collection)

        print(f"[HEADLESS] Successfully loaded {len(chart_trades)} trades into chart")

    def load_data(self):
        """Load data - use configured file or sample data"""

        data_file = self.config.get('data_file')
        if data_file and os.path.exists(data_file):
            print(f"[HEADLESS] Loading data from: {data_file}")

            try:
                if data_file.endswith('.parquet'):
                    df = pd.read_parquet(data_file)
                else:
                    df = pd.read_csv(data_file)

                # Standard data processing (similar to original)
                if 'Date' in df.columns and 'Time' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
                elif 'timestamp' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['timestamp'])
                elif 'DateTime' not in df.columns:
                    df['DateTime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='5min')

                # Ensure standard OHLCV columns
                column_mapping = {
                    'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                }
                df.rename(columns=column_mapping, inplace=True)

                # Convert to format expected by chart
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

                print(f"[HEADLESS] Loaded {len(df)} bars of data")

                # Initial render
                self.render_range(0, min(1000, self.total_bars))

            except Exception as e:
                print(f"[HEADLESS] Error loading data: {e}")
                self.generate_sample_data()
        else:
            print("[HEADLESS] No data file specified, using sample data")
            self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample data as fallback"""
        import numpy as np

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

        # Ensure High is highest and Low is lowest
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

        print(f"[HEADLESS] Generated {num_bars} bars of sample data")

        # Initial render
        self.render_range(0, min(1000, self.total_bars))


def main():
    """Main entry point for headless system launcher"""
    print("\n" + "=" * 80)
    print("HEADLESS BACKTESTING SYSTEM LAUNCHER")
    print("Load existing results OR run new headless backtests with P&L")
    print("=" * 80)

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)

    # Show data selector dialog
    dialog = DataSelectorDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        config = dialog.get_configuration()

        # Create main window
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle('Headless Backtesting System - Load Results OR Run New')
        main_window.resize(1600, 900)

        # Create chart with headless integration
        chart = HeadlessSystemChart(config)

        # Set as central widget
        main_window.setCentralWidget(chart)

        # Show window
        main_window.show()

        print("[HEADLESS] System ready - use 'Headless Backtest' to run workflow or 'Load Results' to load existing")

        # Run application
        sys.exit(app.exec_())
    else:
        print("Data selection cancelled")
        sys.exit(0)


if __name__ == '__main__':
    main()