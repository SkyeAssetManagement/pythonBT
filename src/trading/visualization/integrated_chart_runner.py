#!/usr/bin/env python3
"""
Integrated Chart Runner
=======================
Integrates headless backtesting with chart visualizer.
Provides options to either:
1. Load existing backtest results from CSV files
2. Run new headless backtest and display results
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, Any, Optional, List, Tuple
from PyQt5 import QtWidgets, QtCore
import pandas as pd

from .backtest_result_loader import BacktestResultLoader, ChartVisualizerIntegration
from src.trading.data.trade_data import TradeData, TradeCollection


class BacktestModeSelector(QtWidgets.QDialog):
    """
    Dialog to choose between loading existing results or running new backtest
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Backtest Mode Selection")
        self.setModal(True)
        self.resize(500, 300)

        self.selected_mode = None
        self.selected_run = None
        self.new_backtest_params = None

        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()

        # Title
        title = QtWidgets.QLabel("Choose Backtest Mode")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Mode selection
        mode_group = QtWidgets.QGroupBox("Select Mode:")
        mode_layout = QtWidgets.QVBoxLayout()

        self.load_mode_radio = QtWidgets.QRadioButton("Load Existing Backtest Results")
        self.run_mode_radio = QtWidgets.QRadioButton("Run New Headless Backtest")
        self.load_mode_radio.setChecked(True)  # Default selection

        mode_layout.addWidget(self.load_mode_radio)
        mode_layout.addWidget(self.run_mode_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Stacked widget for mode-specific options
        self.stacked_widget = QtWidgets.QStackedWidget()

        # Load mode widget
        load_widget = self.create_load_widget()
        self.stacked_widget.addWidget(load_widget)

        # Run mode widget
        run_widget = self.create_run_widget()
        self.stacked_widget.addWidget(run_widget)

        layout.addWidget(self.stacked_widget)

        # Connect radio buttons
        self.load_mode_radio.toggled.connect(self.on_mode_changed)
        self.run_mode_radio.toggled.connect(self.on_mode_changed)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.ok_button = QtWidgets.QPushButton("OK")
        self.cancel_button = QtWidgets.QPushButton("Cancel")

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def create_load_widget(self) -> QtWidgets.QWidget:
        """Create widget for loading existing results"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Available runs list
        layout.addWidget(QtWidgets.QLabel("Available Backtest Runs:"))

        self.runs_list = QtWidgets.QListWidget()
        self.populate_runs_list()
        layout.addWidget(self.runs_list)

        # Run details
        self.run_details = QtWidgets.QTextEdit()
        self.run_details.setMaximumHeight(100)
        self.run_details.setReadOnly(True)
        layout.addWidget(self.run_details)

        # Connect selection
        self.runs_list.itemSelectionChanged.connect(self.on_run_selected)

        widget.setLayout(layout)
        return widget

    def create_run_widget(self) -> QtWidgets.QWidget:
        """Create widget for running new backtest"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Strategy parameters
        layout.addWidget(QtWidgets.QLabel("Strategy Parameters:"))

        params_layout = QtWidgets.QFormLayout()

        self.strategy_combo = QtWidgets.QComboBox()
        self.strategy_combo.addItems(['sma_crossover', 'rsi_momentum'])
        params_layout.addRow("Strategy:", self.strategy_combo)

        self.fast_period = QtWidgets.QSpinBox()
        self.fast_period.setRange(1, 100)
        self.fast_period.setValue(10)
        params_layout.addRow("Fast Period:", self.fast_period)

        self.slow_period = QtWidgets.QSpinBox()
        self.slow_period.setRange(1, 200)
        self.slow_period.setValue(30)
        params_layout.addRow("Slow Period:", self.slow_period)

        self.signal_lag = QtWidgets.QSpinBox()
        self.signal_lag.setRange(0, 10)
        self.signal_lag.setValue(2)
        params_layout.addRow("Signal Lag:", self.signal_lag)

        self.min_exec_time = QtWidgets.QDoubleSpinBox()
        self.min_exec_time.setRange(1.0, 60.0)
        self.min_exec_time.setValue(5.0)
        self.min_exec_time.setSuffix(" minutes")
        params_layout.addRow("Min Execution Time:", self.min_exec_time)

        self.long_only = QtWidgets.QCheckBox()
        params_layout.addRow("Long Only:", self.long_only)

        # Data file selection
        data_layout = QtWidgets.QHBoxLayout()
        self.data_file_edit = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_data_file)

        data_layout.addWidget(self.data_file_edit)
        data_layout.addWidget(self.browse_button)
        params_layout.addRow("Data File:", data_layout)

        # Execution mode
        self.exec_mode = QtWidgets.QComboBox()
        self.exec_mode.addItems(['twap', 'standard'])
        params_layout.addRow("Execution Mode:", self.exec_mode)

        layout.addLayout(params_layout)
        widget.setLayout(layout)
        return widget

    def populate_runs_list(self):
        """Populate list of available backtest runs"""
        try:
            loader = BacktestResultLoader()
            runs = loader.list_available_runs()

            for run in runs:
                item_text = f"{run['strategy_name']} ({run['timestamp']}) - {run['execution_mode']}"
                item = QtWidgets.QListWidgetItem(item_text)
                item.setData(QtCore.Qt.UserRole, run)
                self.runs_list.addItem(item)

        except Exception as e:
            error_item = QtWidgets.QListWidgetItem(f"Error loading runs: {e}")
            self.runs_list.addItem(error_item)

    def on_mode_changed(self):
        """Handle mode selection change"""
        if self.load_mode_radio.isChecked():
            self.stacked_widget.setCurrentIndex(0)
        else:
            self.stacked_widget.setCurrentIndex(1)

    def on_run_selected(self):
        """Handle run selection"""
        current_item = self.runs_list.currentItem()
        if current_item:
            run_data = current_item.data(QtCore.Qt.UserRole)
            if run_data:
                details = f"Strategy: {run_data['strategy_name']}\n"
                details += f"Execution Mode: {run_data['execution_mode']}\n"
                details += f"Parameters: {run_data['parameters']}"
                self.run_details.setText(details)

    def browse_data_file(self):
        """Browse for data file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "CSV Files (*.csv);;Parquet Files (*.parquet);;All Files (*)"
        )
        if file_path:
            self.data_file_edit.setText(file_path)

    def accept(self):
        """Handle OK button"""
        if self.load_mode_radio.isChecked():
            # Load mode
            current_item = self.runs_list.currentItem()
            if not current_item:
                QtWidgets.QMessageBox.warning(self, "Warning", "Please select a backtest run to load.")
                return

            run_data = current_item.data(QtCore.Qt.UserRole)
            if not run_data:
                QtWidgets.QMessageBox.warning(self, "Warning", "Invalid run data.")
                return

            self.selected_mode = "load"
            self.selected_run = run_data

        else:
            # Run mode
            data_file = self.data_file_edit.text().strip()
            if not data_file:
                QtWidgets.QMessageBox.warning(self, "Warning", "Please select a data file.")
                return

            if not os.path.exists(data_file):
                QtWidgets.QMessageBox.warning(self, "Warning", "Data file does not exist.")
                return

            self.selected_mode = "run"
            self.new_backtest_params = {
                'strategy_name': self.strategy_combo.currentText(),
                'parameters': {
                    'fast_period': self.fast_period.value(),
                    'slow_period': self.slow_period.value(),
                    'signal_lag': self.signal_lag.value(),
                    'min_execution_time': self.min_exec_time.value(),
                    'long_only': self.long_only.isChecked(),
                    'position_size': 1.0,
                    'fees': 0.0
                },
                'data_file': data_file,
                'execution_mode': self.exec_mode.currentText()
            }

        super().accept()


class IntegratedChartRunner:
    """
    Main integration class that handles both loading and running backtests
    """

    def __init__(self):
        self.integration = ChartVisualizerIntegration()
        self.loader = BacktestResultLoader()

    def show_mode_selector(self, parent=None) -> Optional[Tuple[str, Any]]:
        """
        Show mode selector dialog

        Returns:
            Tuple of (mode, data) where:
            - mode: 'load' or 'run'
            - data: run_data for load mode, backtest_params for run mode
        """
        dialog = BacktestModeSelector(parent)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            if dialog.selected_mode == "load":
                return ("load", dialog.selected_run)
            else:
                return ("run", dialog.new_backtest_params)
        return None

    def execute_backtest_workflow(self, parent=None) -> Optional[Tuple[TradeCollection, Dict[str, Any]]]:
        """
        Execute the complete backtest workflow

        Returns:
            Tuple of (trades, metadata) or None if cancelled
        """

        result = self.show_mode_selector(parent)
        if not result:
            return None

        mode, data = result

        if mode == "load":
            return self._load_existing_results(data)
        else:
            return self._run_new_backtest(data, parent)

    def _load_existing_results(self, run_data: Dict[str, Any]) -> Optional[Tuple[TradeCollection, Dict[str, Any]]]:
        """Load existing backtest results"""

        print(f"[CHART] Loading existing results: {run_data['run_id']}")

        try:
            trades = self.loader.load_trade_list(run_data['run_id'])
            if not trades:
                print(f"[CHART] No trades found for run: {run_data['run_id']}")
                return None

            metadata = {
                'run_id': run_data['run_id'],
                'strategy_name': run_data['strategy_name'],
                'execution_mode': run_data['execution_mode'],
                'parameters': run_data['parameters'],
                'source': 'loaded_from_csv'
            }

            print(f"[CHART] Loaded {len(trades)} trades from existing results")
            return (trades, metadata)

        except Exception as e:
            print(f"[CHART] Error loading results: {e}")
            return None

    def _run_new_backtest(self, params: Dict[str, Any], parent=None) -> Optional[Tuple[TradeCollection, Dict[str, Any]]]:
        """Run new headless backtest"""

        print(f"[CHART] Running new headless backtest: {params['strategy_name']}")

        # Show progress dialog
        progress = QtWidgets.QProgressDialog("Running headless backtest...", "Cancel", 0, 0, parent)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()

        try:
            # Run headless backtest
            run_id = self.integration.trigger_headless_backtest(
                strategy_name=params['strategy_name'],
                parameters=params['parameters'],
                data_file=params['data_file'],
                execution_mode=params['execution_mode']
            )

            progress.setLabelText("Loading results...")

            # Load the results
            trades = self.loader.load_trade_list(run_id)
            if not trades:
                print(f"[CHART] No trades generated from backtest: {run_id}")
                return None

            metadata = {
                'run_id': run_id,
                'strategy_name': params['strategy_name'],
                'execution_mode': params['execution_mode'],
                'parameters': params['parameters'],
                'source': 'new_headless_backtest'
            }

            print(f"[CHART] Generated {len(trades)} trades from new backtest")
            return (trades, metadata)

        except Exception as e:
            print(f"[CHART] Error running new backtest: {e}")
            return None

        finally:
            progress.close()


def main():
    """Test the integrated chart runner"""

    app = QtWidgets.QApplication([])

    runner = IntegratedChartRunner()
    result = runner.execute_backtest_workflow()

    if result:
        trades, metadata = result
        print(f"Successfully got {len(trades)} trades from {metadata['source']}")
        print(f"Strategy: {metadata['strategy_name']}")
        print(f"Execution mode: {metadata['execution_mode']}")

        # Show first few trades
        for i, trade in enumerate(trades[:3]):
            print(f"Trade {i+1}: {trade.trade_type} {trade.size} @ {trade.price:.2f}")
            if hasattr(trade, 'metadata') and trade.metadata:
                exec_bars = trade.metadata.get('exec_bars', 'N/A')
                print(f"  execBars: {exec_bars}")

    else:
        print("No results - user cancelled or error occurred")

    app.quit()


if __name__ == "__main__":
    main()