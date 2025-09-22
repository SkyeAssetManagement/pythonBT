#!/usr/bin/env python3
"""
PyQtGraph Data Selector Dialog
===============================
Entry screen for selecting data series, trades, and indicators
before launching the PyQtGraph chart
"""

import sys
import os
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

class DataSelectorDialog(QtWidgets.QDialog):
    """Dialog for selecting data, trades, and indicators"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM - Chart Data Selector")
        self.setModal(True)
        self.resize(800, 600)
        
        # Store selections
        self.selected_data_file = None
        self.selected_trade_source = None
        self.selected_trade_file = None
        self.selected_indicators = []
        self.selected_system = None
        
        # Initialize summary_text early to avoid AttributeError
        self.summary_text = None
        
        # Setup UI
        self.setup_ui()
        
        # Load available files
        self.scan_data_files()
        self.scan_trade_files()
        
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Select Data for Chart Visualization")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)
        
        # Tab widget for different sections
        self.tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Data tab
        self.create_data_tab()
        
        # Trades tab
        self.create_trades_tab()
        
        # Indicators tab
        self.create_indicators_tab()
        
        # Summary tab
        self.create_summary_tab()
        
        # Button box
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Style the OK button
        ok_button = button_box.button(QtWidgets.QDialogButtonBox.Ok)
        ok_button.setText("Launch Chart")
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
        """)
    
    def create_data_tab(self):
        """Create the data selection tab"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Instructions
        instructions = QtWidgets.QLabel(
            "Select a data file to visualize. The chart will display candlesticks for the selected data."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # File selection group
        group = QtWidgets.QGroupBox("Data File Selection")
        group_layout = QtWidgets.QVBoxLayout()
        
        # Radio buttons for quick selection vs browse
        self.quick_select_radio = QtWidgets.QRadioButton("Quick Select from Available Files")
        self.quick_select_radio.setChecked(True)
        group_layout.addWidget(self.quick_select_radio)
        
        # List of available files
        self.data_list = QtWidgets.QListWidget()
        self.data_list.setMaximumHeight(300)
        group_layout.addWidget(self.data_list)
        
        # Browse option
        self.browse_data_radio = QtWidgets.QRadioButton("Browse for File")
        group_layout.addWidget(self.browse_data_radio)
        
        browse_layout = QtWidgets.QHBoxLayout()
        self.data_path_edit = QtWidgets.QLineEdit()
        self.data_path_edit.setEnabled(False)
        browse_layout.addWidget(self.data_path_edit)
        
        self.browse_data_btn = QtWidgets.QPushButton("Browse...")
        self.browse_data_btn.setEnabled(False)
        self.browse_data_btn.clicked.connect(self.browse_data_file)
        browse_layout.addWidget(self.browse_data_btn)
        
        group_layout.addLayout(browse_layout)
        
        # Connect radio buttons
        self.quick_select_radio.toggled.connect(self.on_data_mode_changed)
        self.browse_data_radio.toggled.connect(self.on_data_mode_changed)
        
        # File info
        self.data_info_label = QtWidgets.QLabel("No file selected")
        self.data_info_label.setStyleSheet("color: #666; padding: 10px;")
        group_layout.addWidget(self.data_info_label)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # Connect list selection
        self.data_list.itemSelectionChanged.connect(self.on_data_selected)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "1. Data Selection")
    
    def create_trades_tab(self):
        """Create the trades selection tab"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        instructions = QtWidgets.QLabel(
            "Select how to load trades. You can load from a CSV file, generate from a trading system, "
            "or use sample trades for testing."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Trade source group
        group = QtWidgets.QGroupBox("Trade Source")
        group_layout = QtWidgets.QVBoxLayout()
        
        # Trade source options
        self.no_trades_radio = QtWidgets.QRadioButton("No Trades (Chart Only)")
        self.no_trades_radio.setChecked(True)
        group_layout.addWidget(self.no_trades_radio)
        
        self.sample_trades_radio = QtWidgets.QRadioButton("Generate Sample Trades")
        group_layout.addWidget(self.sample_trades_radio)
        
        self.csv_trades_radio = QtWidgets.QRadioButton("Load from CSV File")
        group_layout.addWidget(self.csv_trades_radio)
        
        # CSV file selection
        csv_frame = QtWidgets.QFrame()
        csv_layout = QtWidgets.QVBoxLayout(csv_frame)
        csv_layout.setContentsMargins(20, 5, 5, 5)
        
        self.trade_list = QtWidgets.QListWidget()
        self.trade_list.setMaximumHeight(150)
        self.trade_list.setEnabled(False)
        csv_layout.addWidget(self.trade_list)
        
        browse_trade_layout = QtWidgets.QHBoxLayout()
        self.trade_path_edit = QtWidgets.QLineEdit()
        self.trade_path_edit.setEnabled(False)
        browse_trade_layout.addWidget(self.trade_path_edit)
        
        self.browse_trade_btn = QtWidgets.QPushButton("Browse...")
        self.browse_trade_btn.setEnabled(False)
        self.browse_trade_btn.clicked.connect(self.browse_trade_file)
        browse_trade_layout.addWidget(self.browse_trade_btn)
        
        csv_layout.addLayout(browse_trade_layout)
        group_layout.addWidget(csv_frame)
        
        # System trades
        self.system_trades_radio = QtWidgets.QRadioButton("Generate from Trading System")
        group_layout.addWidget(self.system_trades_radio)
        
        system_frame = QtWidgets.QFrame()
        system_layout = QtWidgets.QVBoxLayout(system_frame)
        system_layout.setContentsMargins(20, 5, 5, 5)
        
        self.system_combo = QtWidgets.QComboBox()
        self.system_combo.addItems([
            "Simple Moving Average",
            "RSI Momentum",
            "Mean Reversion",
            "Breakout Strategy",
            "Custom Strategy"
        ])
        self.system_combo.setEnabled(False)
        system_layout.addWidget(self.system_combo)
        
        group_layout.addWidget(system_frame)
        
        # Connect radio buttons
        self.csv_trades_radio.toggled.connect(self.on_trade_source_changed)
        self.system_trades_radio.toggled.connect(self.on_trade_source_changed)
        
        # Trade info
        self.trade_info_label = QtWidgets.QLabel("No trades will be loaded")
        self.trade_info_label.setStyleSheet("color: #666; padding: 10px;")
        group_layout.addWidget(self.trade_info_label)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # Connect selections
        self.trade_list.itemSelectionChanged.connect(self.on_trade_file_selected)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "2. Trade Selection")
    
    def create_indicators_tab(self):
        """Create the indicators selection tab"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        instructions = QtWidgets.QLabel(
            "Select technical indicators to overlay on the chart. "
            "These will be calculated from the selected data."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Indicators group
        group = QtWidgets.QGroupBox("Technical Indicators")
        group_layout = QtWidgets.QVBoxLayout()
        
        # Indicator info label (create early)
        self.indicator_info_label = QtWidgets.QLabel("Volume, ATR indicators selected")
        self.indicator_info_label.setStyleSheet("color: #666; padding: 10px;")
        
        # Create checkboxes for common indicators
        indicators = [
            ("SMA 20", "sma_20"),
            ("SMA 50", "sma_50"),
            ("EMA 20", "ema_20"),
            ("Bollinger Bands", "bbands"),
            ("RSI", "rsi"),
            ("MACD", "macd"),
            ("Volume", "volume"),
            ("ATR", "atr")
        ]
        
        self.indicator_checkboxes = {}
        for label, key in indicators:
            checkbox = QtWidgets.QCheckBox(label)
            self.indicator_checkboxes[key] = checkbox
            group_layout.addWidget(checkbox)
            checkbox.stateChanged.connect(self.on_indicator_changed)
        
        # Pre-check Volume and ATR if available
        self.indicator_checkboxes["volume"].setChecked(True)
        self.indicator_checkboxes["atr"].setChecked(True)
        
        # Add info label at the end
        group_layout.addWidget(self.indicator_info_label)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "3. Indicators")
    
    def create_summary_tab(self):
        """Create the summary tab"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Summary text
        self.summary_text = QtWidgets.QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 10px;
                font-family: Consolas, monospace;
            }
        """)
        layout.addWidget(self.summary_text)
        
        # Update summary button
        update_btn = QtWidgets.QPushButton("Update Summary")
        update_btn.clicked.connect(self.update_summary)
        layout.addWidget(update_btn)
        
        self.tab_widget.addTab(tab, "4. Summary")
        
        # Initial summary
        self.update_summary()
    
    def scan_data_files(self):
        """Scan for available data files"""
        data_dirs = [
            "parquetData",
            "dataRaw/range-ATR30x0.05/ES/diffAdjusted",
            "dataRaw/range-ATR30x0.1/ES/diffAdjusted",
            "dataRaw/range-ATR30x0.2/ES/diffAdjusted",
            "data"
        ]
        
        files = []
        for dir_path in data_dirs:
            if os.path.exists(dir_path):
                path = Path(dir_path)
                # Look for parquet and CSV files
                files.extend(path.glob("*.parquet"))
                files.extend(path.glob("*.csv"))
                
                # Also check subdirectories (one level)
                for subdir in path.iterdir():
                    if subdir.is_dir():
                        files.extend(subdir.glob("*.parquet"))
                        files.extend(subdir.glob("*.csv"))
        
        # Add to list widget
        for file_path in files:
            # Show relative path for clarity
            try:
                rel_path = file_path.relative_to(Path.cwd())
            except:
                rel_path = file_path
            
            item = QtWidgets.QListWidgetItem(str(rel_path))
            item.setData(QtCore.Qt.UserRole, str(file_path))
            self.data_list.addItem(item)
        
        # Select first item if available
        if self.data_list.count() > 0:
            self.data_list.setCurrentRow(0)
    
    def scan_trade_files(self):
        """Scan for available trade CSV files"""
        trade_patterns = [
            "*trade*.csv",
            "*Trade*.csv",
            "trades.csv"
        ]
        
        search_dirs = [
            ".",
            "trades",
            "results",
            "output",
            "data"
        ]
        
        files = []
        for dir_path in search_dirs:
            if os.path.exists(dir_path):
                path = Path(dir_path)
                for pattern in trade_patterns:
                    files.extend(path.glob(pattern))
        
        # Add to list widget
        for file_path in files:
            try:
                rel_path = file_path.relative_to(Path.cwd())
            except:
                rel_path = file_path
            
            item = QtWidgets.QListWidgetItem(str(rel_path))
            item.setData(QtCore.Qt.UserRole, str(file_path))
            self.trade_list.addItem(item)
    
    def on_data_mode_changed(self):
        """Handle data selection mode change"""
        if self.quick_select_radio.isChecked():
            self.data_list.setEnabled(True)
            self.data_path_edit.setEnabled(False)
            self.browse_data_btn.setEnabled(False)
        else:
            self.data_list.setEnabled(False)
            self.data_path_edit.setEnabled(True)
            self.browse_data_btn.setEnabled(True)
    
    def on_data_selected(self):
        """Handle data file selection"""
        items = self.data_list.selectedItems()
        if items:
            self.selected_data_file = items[0].data(QtCore.Qt.UserRole)
            self.update_data_info()
            self.update_summary()
    
    def browse_data_file(self):
        """Browse for data file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "Data Files (*.parquet *.csv);;All Files (*.*)"
        )
        
        if file_path:
            self.data_path_edit.setText(file_path)
            self.selected_data_file = file_path
            self.update_data_info()
            self.update_summary()
    
    def update_data_info(self):
        """Update data file info"""
        if self.selected_data_file and os.path.exists(self.selected_data_file):
            try:
                # Get file info
                file_size = os.path.getsize(self.selected_data_file) / (1024 * 1024)  # MB
                
                # Try to get row count
                if self.selected_data_file.endswith('.parquet'):
                    df = pd.read_parquet(self.selected_data_file, columns=['Open'])
                else:
                    df = pd.read_csv(self.selected_data_file, nrows=1)
                    df = pd.read_csv(self.selected_data_file, usecols=[0])
                
                rows = len(df)
                
                self.data_info_label.setText(
                    f"Selected: {Path(self.selected_data_file).name}\n"
                    f"Size: {file_size:.1f} MB | Rows: {rows:,}"
                )
            except Exception as e:
                self.data_info_label.setText(
                    f"Selected: {Path(self.selected_data_file).name}\n"
                    f"Error reading file: {str(e)[:50]}"
                )
        else:
            self.data_info_label.setText("No file selected")
    
    def on_trade_source_changed(self):
        """Handle trade source change"""
        # Enable/disable relevant controls
        csv_selected = self.csv_trades_radio.isChecked()
        system_selected = self.system_trades_radio.isChecked()
        
        self.trade_list.setEnabled(csv_selected)
        self.trade_path_edit.setEnabled(csv_selected)
        self.browse_trade_btn.setEnabled(csv_selected)
        
        self.system_combo.setEnabled(system_selected)
        
        # Update info
        if self.no_trades_radio.isChecked():
            self.trade_info_label.setText("No trades will be loaded")
        elif self.sample_trades_radio.isChecked():
            self.trade_info_label.setText("Sample trades will be generated")
        elif csv_selected:
            self.trade_info_label.setText("Select a CSV file with trades")
        elif system_selected:
            self.trade_info_label.setText(f"Trades from: {self.system_combo.currentText()}")
        
        self.update_summary()
    
    def on_trade_file_selected(self):
        """Handle trade file selection"""
        items = self.trade_list.selectedItems()
        if items:
            self.selected_trade_file = items[0].data(QtCore.Qt.UserRole)
            self.trade_info_label.setText(f"Selected: {Path(self.selected_trade_file).name}")
            self.update_summary()
    
    def browse_trade_file(self):
        """Browse for trade file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Trade CSV File",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            self.trade_path_edit.setText(file_path)
            self.selected_trade_file = file_path
            self.trade_info_label.setText(f"Selected: {Path(file_path).name}")
            self.update_summary()
    
    def on_indicator_changed(self):
        """Handle indicator selection change"""
        selected = []
        for key, checkbox in self.indicator_checkboxes.items():
            if checkbox.isChecked():
                selected.append(checkbox.text())
        
        if selected:
            self.indicator_info_label.setText(f"Selected: {', '.join(selected)}")
        else:
            self.indicator_info_label.setText("No indicators selected")
        
        self.update_summary()
    
    def update_summary(self):
        """Update the summary tab"""
        # Check if summary_text exists yet (may be called during initialization)
        if not self.summary_text:
            return
            
        summary = []
        summary.append("=" * 50)
        summary.append("CHART CONFIGURATION SUMMARY")
        summary.append("=" * 50)
        summary.append("")
        
        # Data file
        summary.append("DATA FILE:")
        if self.selected_data_file:
            summary.append(f"  {self.selected_data_file}")
        else:
            summary.append("  [No file selected]")
        summary.append("")
        
        # Trade source
        summary.append("TRADE SOURCE:")
        if hasattr(self, 'no_trades_radio') and self.no_trades_radio.isChecked():
            summary.append("  No trades")
            self.selected_trade_source = "none"
        elif hasattr(self, 'sample_trades_radio') and self.sample_trades_radio.isChecked():
            summary.append("  Sample trades (auto-generated)")
            self.selected_trade_source = "sample"
        elif hasattr(self, 'csv_trades_radio') and self.csv_trades_radio.isChecked():
            self.selected_trade_source = "csv"
            if self.selected_trade_file:
                summary.append(f"  CSV: {self.selected_trade_file}")
            else:
                summary.append("  CSV: [No file selected]")
        elif hasattr(self, 'system_trades_radio') and self.system_trades_radio.isChecked():
            self.selected_trade_source = "system"
            self.selected_system = self.system_combo.currentText()
            summary.append(f"  System: {self.selected_system}")
        summary.append("")
        
        # Indicators
        summary.append("INDICATORS:")
        self.selected_indicators = []
        if hasattr(self, 'indicator_checkboxes'):
            for key, checkbox in self.indicator_checkboxes.items():
                if checkbox.isChecked():
                    summary.append(f"  ✓ {checkbox.text()}")
                    self.selected_indicators.append(key)
        
        if not self.selected_indicators:
            summary.append("  [None selected]")
        
        summary.append("")
        summary.append("=" * 50)
        
        self.summary_text.setPlainText("\n".join(summary))
    
    def get_configuration(self) -> Dict:
        """Get the selected configuration"""
        return {
            'data_file': self.selected_data_file,
            'trade_source': self.selected_trade_source,
            'trade_file': self.selected_trade_file,
            'system': self.selected_system,
            'indicators': self.selected_indicators
        }


def main():
    """Test the dialog"""
    app = QtWidgets.QApplication(sys.argv)
    
    dialog = DataSelectorDialog()
    
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        config = dialog.get_configuration()
        print("Configuration selected:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print("Dialog cancelled")
    
    sys.exit(0)


if __name__ == "__main__":
    main()