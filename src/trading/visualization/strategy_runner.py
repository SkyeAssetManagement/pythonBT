"""Strategy Runner Widget - Run trading strategies directly from chart"""

import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Optional, Dict, Callable
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_momentum import RSIMomentumStrategy
from strategies.enhanced_sma_crossover import EnhancedSMACrossoverStrategy
from data.trade_data import TradeCollection


class StrategyRunner(QtWidgets.QWidget):
    """Widget for running trading strategies on chart data"""

    # Signals
    trades_generated = QtCore.pyqtSignal(TradeCollection)
    indicators_calculated = QtCore.pyqtSignal(dict)  # For indicator overlays

    def __init__(self):
        super().__init__()
        self.chart_data = None
        self.current_strategy = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the strategy runner UI"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        # Title
        title = QtWidgets.QLabel("Strategy Runner")
        title.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        layout.addWidget(title)

        # Strategy selector
        strategy_layout = QtWidgets.QHBoxLayout()
        strategy_label = QtWidgets.QLabel("Strategy:")
        self.strategy_combo = QtWidgets.QComboBox()
        self.strategy_combo.addItems([
            "SMA Crossover",
            "RSI Momentum",
            "Enhanced SMA (with Lag)"
        ])
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strategy_layout.addWidget(strategy_label)
        strategy_layout.addWidget(self.strategy_combo)
        layout.addLayout(strategy_layout)

        # Parameters section
        params_group = QtWidgets.QGroupBox("Parameters")
        self.params_layout = QtWidgets.QFormLayout()
        params_group.setLayout(self.params_layout)
        layout.addWidget(params_group)

        # Initialize with SMA parameters
        self.setup_sma_params()

        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.run_button = QtWidgets.QPushButton("Run Strategy")
        self.run_button.clicked.connect(self.run_strategy)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.clear_button = QtWidgets.QPushButton("Clear Trades")
        self.clear_button.clicked.connect(self.clear_trades)

        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)

        # Status label
        self.status_label = QtWidgets.QLabel("Ready to run strategy")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(self.status_label)

        # Add stretch to push everything to top
        layout.addStretch()

    def clear_params(self):
        """Clear all parameter widgets"""
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def setup_sma_params(self):
        """Setup SMA crossover parameters"""
        self.clear_params()

        # Fast period
        self.fast_period_spin = QtWidgets.QSpinBox()
        self.fast_period_spin.setMinimum(2)
        self.fast_period_spin.setMaximum(200)
        self.fast_period_spin.setValue(10)
        self.params_layout.addRow("Fast Period:", self.fast_period_spin)

        # Slow period
        self.slow_period_spin = QtWidgets.QSpinBox()
        self.slow_period_spin.setMinimum(2)
        self.slow_period_spin.setMaximum(500)
        self.slow_period_spin.setValue(30)
        self.params_layout.addRow("Slow Period:", self.slow_period_spin)

        # Long only checkbox
        self.long_only_check = QtWidgets.QCheckBox()
        self.long_only_check.setChecked(True)
        self.params_layout.addRow("Long Only:", self.long_only_check)

    def setup_rsi_params(self):
        """Setup RSI momentum parameters"""
        self.clear_params()

        # RSI period
        self.rsi_period_spin = QtWidgets.QSpinBox()
        self.rsi_period_spin.setMinimum(2)
        self.rsi_period_spin.setMaximum(100)
        self.rsi_period_spin.setValue(14)
        self.params_layout.addRow("RSI Period:", self.rsi_period_spin)

        # Oversold level
        self.oversold_spin = QtWidgets.QSpinBox()
        self.oversold_spin.setMinimum(10)
        self.oversold_spin.setMaximum(50)
        self.oversold_spin.setValue(30)
        self.params_layout.addRow("Oversold Level:", self.oversold_spin)

        # Overbought level
        self.overbought_spin = QtWidgets.QSpinBox()
        self.overbought_spin.setMinimum(50)
        self.overbought_spin.setMaximum(90)
        self.overbought_spin.setValue(70)
        self.params_layout.addRow("Overbought Level:", self.overbought_spin)

        # Long only checkbox
        self.long_only_check = QtWidgets.QCheckBox()
        self.long_only_check.setChecked(True)
        self.params_layout.addRow("Long Only:", self.long_only_check)

    def on_strategy_changed(self, strategy_name):
        """Handle strategy selection change"""
        if strategy_name == "SMA Crossover" or strategy_name == "Enhanced SMA (with Lag)":
            self.setup_sma_params()
        elif strategy_name == "RSI Momentum":
            self.setup_rsi_params()

        self.status_label.setText(f"Selected {strategy_name} strategy")

    def set_chart_data(self, data_dict):
        """Set chart data for strategy execution"""
        # Convert to DataFrame
        if 'timestamp' in data_dict:
            print(f"[STRATEGY_RUNNER] set_chart_data: Found timestamp in data_dict")
            print(f"[STRATEGY_RUNNER] First timestamp: {data_dict['timestamp'][0] if len(data_dict['timestamp']) > 0 else 'EMPTY'}")
            print(f"[STRATEGY_RUNNER] Timestamp type: {type(data_dict['timestamp'])}")

            df = pd.DataFrame({
                'DateTime': data_dict['timestamp'],
                'Open': data_dict['open'],
                'High': data_dict['high'],
                'Low': data_dict['low'],
                'Close': data_dict['close'],
                'Volume': data_dict.get('volume', np.zeros(len(data_dict['close'])))
            })
            print(f"[STRATEGY_RUNNER] Created DataFrame with columns: {df.columns.tolist()}")
            print(f"[STRATEGY_RUNNER] DataFrame DateTime column type: {df['DateTime'].dtype}")
            print(f"[STRATEGY_RUNNER] First DateTime value: {df['DateTime'].iloc[0] if len(df) > 0 else 'EMPTY'}")
        else:
            print(f"[STRATEGY_RUNNER] set_chart_data: No timestamp found in data_dict")
            df = pd.DataFrame({
                'Open': data_dict['open'],
                'High': data_dict['high'],
                'Low': data_dict['low'],
                'Close': data_dict['close'],
                'Volume': data_dict.get('volume', np.zeros(len(data_dict['close'])))
            })
            print(f"[STRATEGY_RUNNER] Created DataFrame without DateTime, columns: {df.columns.tolist()}")

        self.chart_data = df
        self.status_label.setText(f"Chart data loaded: {len(df):,} bars - Ready to run strategies")
        self.status_label.setStyleSheet("")  # Reset color

    def run_strategy(self):
        """Run the selected strategy"""
        if self.chart_data is None or len(self.chart_data) == 0:
            self.status_label.setText("No chart data available!")
            return

        try:
            strategy_name = self.strategy_combo.currentText()

            # Create strategy instance with parameters
            if strategy_name == "SMA Crossover":
                strategy = SMACrossoverStrategy(
                    fast_period=self.fast_period_spin.value(),
                    slow_period=self.slow_period_spin.value(),
                    long_only=self.long_only_check.isChecked()
                )

                # Calculate indicators for overlay
                sma_fast, sma_slow = strategy.calculate_smas(self.chart_data)
                indicators = {
                    f'SMA_{strategy.fast_period}': sma_fast.values,
                    f'SMA_{strategy.slow_period}': sma_slow.values
                }
                self.indicators_calculated.emit(indicators)

            elif strategy_name == "Enhanced SMA (with Lag)":
                strategy = EnhancedSMACrossoverStrategy(
                    fast_period=self.fast_period_spin.value(),
                    slow_period=self.slow_period_spin.value(),
                    long_only=self.long_only_check.isChecked()
                )

                # Calculate indicators for overlay
                sma_fast, sma_slow = strategy.calculate_smas(self.chart_data)
                indicators = {
                    f'SMA_{strategy.fast_period}': sma_fast.values,
                    f'SMA_{strategy.slow_period}': sma_slow.values
                }
                self.indicators_calculated.emit(indicators)

            elif strategy_name == "RSI Momentum":
                strategy = RSIMomentumStrategy(
                    period=self.rsi_period_spin.value(),
                    oversold=self.oversold_spin.value(),
                    overbought=self.overbought_spin.value(),
                    long_only=self.long_only_check.isChecked()
                )

                # Calculate RSI for overlay (would need separate indicator panel)
                rsi_values = strategy.get_rsi_values(self.chart_data)
                indicators = {
                    f'RSI_{strategy.period}': rsi_values.values
                }
                self.indicators_calculated.emit(indicators)

            self.current_strategy = strategy

            # Generate signals
            self.status_label.setText(f"Generating {strategy_name} signals...")
            signals = strategy.generate_signals(self.chart_data)

            # Convert signals to trades
            trades = strategy.signals_to_trades(signals, self.chart_data)

            # Emit trades
            self.trades_generated.emit(trades)

            # Update status with better feedback
            num_trades = len(trades)

            # Provide clear feedback based on results
            if num_trades == 0:
                self.status_label.setText(
                    f"No trades generated using {strategy_name}\n"
                    f"Try adjusting parameters or checking if the data has enough price movement"
                )
                self.status_label.setStyleSheet("color: orange;")
            elif num_trades > 1000:
                self.status_label.setText(
                    f"Generated {num_trades:,} trades using {strategy_name} (WARNING: Excessive trades!)\n"
                    f"Entries: {len(trades.get_entry_trades())}, Exits: {len(trades.get_exit_trades())}\n"
                    f"Consider using longer periods to reduce trade frequency"
                )
                self.status_label.setStyleSheet("color: orange;")
            else:
                self.status_label.setText(
                    f"Successfully generated {num_trades} trades using {strategy_name}\n"
                    f"Entries: {len(trades.get_entry_trades())}, Exits: {len(trades.get_exit_trades())}"
                )
                self.status_label.setStyleSheet("color: green;")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Strategy execution error: {e}")

    def clear_trades(self):
        """Clear all trades"""
        empty_trades = TradeCollection([])
        self.trades_generated.emit(empty_trades)
        self.status_label.setText("Trades cleared")


if __name__ == "__main__":
    # Test the widget
    app = QtWidgets.QApplication(sys.argv)

    widget = StrategyRunner()
    widget.show()

    # Test with sample data
    sample_data = {
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
    }
    widget.set_chart_data(sample_data)

    sys.exit(app.exec_())