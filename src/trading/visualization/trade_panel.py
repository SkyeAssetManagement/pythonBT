#!/usr/bin/env python3
"""
Trade Panel - Side panel with trade list and source selector
===========================================================

Interactive trade list with CSV/Backtester source selection, auto-sync with chart,
and click navigation. Maximum 20% screen width with continuous scrolling.
"""

import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Optional, Dict, List, Callable
from pathlib import Path
import logging

from trade_data import TradeData, TradeCollection
from csv_trade_loader import CSVTradeLoader
from strategy_runner import StrategyRunner

logger = logging.getLogger(__name__)

class TradeSourceSelector(QtWidgets.QWidget):
    """Widget for selecting trade data source (Backtester vs CSV)"""
    
    # Signals
    trades_loaded = QtCore.pyqtSignal(TradeCollection)
    
    def __init__(self):
        super().__init__()
        self.csv_loader = CSVTradeLoader()
        
        # Chart timestamps for coordinated trade creation
        self.chart_timestamps = None
        self.bar_data = None
        
        self.setup_ui()
    
    def set_chart_timestamps(self, timestamps):
        """Set chart timestamps for coordinated trade generation"""
        self.chart_timestamps = timestamps
    
    def set_bar_data(self, bar_data):
        """Set bar data for realistic trade pricing"""
        self.bar_data = bar_data
        
    def setup_ui(self):
        """Setup the source selector UI"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Title
        title = QtWidgets.QLabel("Trade Data Source")
        title.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        layout.addWidget(title)
        
        # Radio buttons for source selection
        self.backtester_radio = QtWidgets.QRadioButton("Load from Backtester")
        self.csv_radio = QtWidgets.QRadioButton("Load from CSV File")
        # Default to CSV to enforce real trades
        self.backtester_radio.setChecked(False)
        self.csv_radio.setChecked(True)
        
        layout.addWidget(self.backtester_radio)
        layout.addWidget(self.csv_radio)
        
        # CSV file selection
        csv_layout = QtWidgets.QHBoxLayout()
        
        self.file_path_edit = QtWidgets.QLineEdit()
        self.file_path_edit.setPlaceholderText("Select CSV file...")
        self.file_path_edit.setEnabled(False)
        
        self.browse_button = QtWidgets.QPushButton("Browse...")
        self.browse_button.setEnabled(False)
        self.browse_button.clicked.connect(self.browse_csv_file)
        
        csv_layout.addWidget(self.file_path_edit)
        csv_layout.addWidget(self.browse_button)
        layout.addLayout(csv_layout)
        
        # Load button
        self.load_button = QtWidgets.QPushButton("Load Trades")
        self.load_button.clicked.connect(self.load_trades)
        layout.addWidget(self.load_button)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Ready to load trades")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(self.status_label)
        
        # Connect radio button changes
        self.csv_radio.toggled.connect(self.on_source_changed)
        
    def on_source_changed(self, checked):
        """Handle source radio button changes"""
        csv_selected = self.csv_radio.isChecked()
        
        self.file_path_edit.setEnabled(csv_selected)
        self.browse_button.setEnabled(csv_selected)
        
        if csv_selected:
            self.status_label.setText("Select CSV file and click Load Trades")
        else:
            self.status_label.setText("Ready to load from backtester")
    
    def browse_csv_file(self):
        """Open file browser for CSV selection"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select CSV Trade File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
            self.status_label.setText(f"Selected: {Path(file_path).name}")
    
    def load_trades(self):
        """Load trades based on selected source"""
        try:
            if self.csv_radio.isChecked():
                self.load_csv_trades()
            else:
                self.load_backtester_trades()
                
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            logger.error(f"Failed to load trades: {e}")
    
    def load_csv_trades(self):
        """Load trades from CSV file"""
        file_path = self.file_path_edit.text().strip()
        
        if not file_path:
            self.status_label.setText("Please select a CSV file first")
            return
        
        if not Path(file_path).exists():
            self.status_label.setText("Selected file does not exist")
            return
        
        self.status_label.setText("Loading CSV trades...")
        self.load_button.setEnabled(False)
        
        try:
            # Load in separate thread to avoid UI blocking
            QtCore.QTimer.singleShot(10, lambda: self._load_csv_async(file_path))
            
        except Exception as e:
            self.status_label.setText(f"CSV load error: {str(e)}")
            self.load_button.setEnabled(True)
    
    def _load_csv_async(self, file_path: str):
        """Async CSV loading"""
        try:
            trades = self.csv_loader.load_csv_trades(file_path)
            
            self.status_label.setText(
                f"Loaded {len(trades)} trades from CSV\\n"
                f"Date range: {trades.date_range[0].strftime('%Y-%m-%d')} to "
                f"{trades.date_range[1].strftime('%Y-%m-%d')}"
            )
            
            self.trades_loaded.emit(trades)
            
        except Exception as e:
            self.status_label.setText(f"CSV error: {str(e)}")
        finally:
            self.load_button.setEnabled(True)
    
    def load_backtester_trades(self):
        """Load trades from backtester (placeholder for now)"""
        # Explicitly disable sample/backtester auto-generation
        self.status_label.setText("Backtester integration not yet implemented (CSV only)")
        logger.info("Backtester trade loading is disabled; use CSV")
        return

class TradeTableModel(QtCore.QAbstractTableModel):
    """Table model for trade list with efficient updates"""
    
    COLUMNS = [
        ("Trade #", "trade_id"),
        ("DateTime", "timestamp"), 
        ("Type", "trade_type"),
        ("Price", "price"),
        ("Size", "size"),
        ("P&L", "pnl"),
        ("Bar #", "bar_index")  # Added bar number for easy lookup and sync
    ]
    
    def __init__(self):
        super().__init__()
        self.trades = TradeCollection([])
        
    def set_trades(self, trades: TradeCollection):
        """Update trade data"""
        self.beginResetModel()
        self.trades = trades
        self.endResetModel()
        
        logger.debug(f"Updated table model with {len(trades)} trades")
    
    def rowCount(self, parent=QtCore.QModelIndex()):
        """Return number of rows"""
        return len(self.trades)
    
    def columnCount(self, parent=QtCore.QModelIndex()):
        """Return number of columns"""
        return len(self.COLUMNS)
    
    def headerData(self, section, orientation, role):
        """Return header data"""
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.COLUMNS[section][0]
            else:
                return str(section + 1)
        return None
    
    def data(self, index, role):
        """Return data for table cells"""
        if not index.isValid() or index.row() >= len(self.trades):
            return None

        trade = self.trades[index.row()]
        column_attr = self.COLUMNS[index.column()][1]

        if role == QtCore.Qt.DisplayRole:
            value = getattr(trade, column_attr)

            # Format specific columns
            if column_attr == "timestamp":
                # Debug logging for first trade
                if index.row() == 0:
                    print(f"[TRADE_PANEL] First trade timestamp value: {value}")
                    print(f"[TRADE_PANEL] First trade timestamp type: {type(value)}")

                if value is not None:
                    try:
                        # Handle different timestamp types
                        if hasattr(value, 'strftime'):
                            return value.strftime('%y-%m-%d %H:%M:%S')
                        else:
                            # Convert to datetime if needed
                            dt = pd.to_datetime(value)
                            return dt.strftime('%y-%m-%d %H:%M:%S')
                    except Exception as e:
                        print(f"[TRADE_PANEL] Error formatting timestamp: {e}")
                        return "-"
                else:
                    return "-"
            elif column_attr == "trade_id":
                return str(value) if value is not None else "-"
            elif column_attr == "price":
                return f"${value:.2f}"
            elif column_attr == "size":
                return str(value) if value is not None else "1"
            elif column_attr == "pnl" and value is not None:
                return f"${value:.2f}"
            elif column_attr == "pnl" and value is None:
                return "-"
            else:
                return str(value) if value is not None else ""
        
        elif role == QtCore.Qt.TextColorRole:
            # Color code by trade type and P&L
            if column_attr == "trade_type":
                if trade.trade_type in ['BUY', 'COVER']:
                    return QtGui.QColor(0, 200, 0)  # Brighter green
                else:
                    return QtGui.QColor(255, 120, 120)  # Light red for better contrast
            elif column_attr == "pnl" and trade.pnl is not None:
                if trade.pnl > 0:
                    return QtGui.QColor(0, 200, 0)  # Brighter green profit
                elif trade.pnl < 0:
                    return QtGui.QColor(255, 120, 120)  # Light red loss for better contrast
        
        elif role == QtCore.Qt.TextAlignmentRole:
            if column_attr in ["price", "size", "pnl", "trade_id"]:
                return QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
            elif column_attr == "trade_type":
                return QtCore.Qt.AlignCenter
            else:
                return QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        
        return None
    
    def get_trade_at_row(self, row: int) -> Optional[TradeData]:
        """Get trade data for specific row"""
        if 0 <= row < len(self.trades):
            return self.trades[row]
        return None

class TradeListPanel(QtWidgets.QWidget):
    """
    Trade list side panel - 20% screen width maximum
    
    Features:
    - Trade source selector (Backtester vs CSV)
    - Continuous scrolling trade list
    - Auto-sync with chart viewport
    - Click navigation to trades
    """
    
    # Signals
    trade_selected = QtCore.pyqtSignal(object)  # Emitted when trade clicked (TradeData object)
    
    def __init__(self):
        super().__init__()
        
        self.trades = TradeCollection([])
        self.auto_sync_enabled = True
        
        # Chart timestamps for coordinated trade creation
        self.chart_timestamps = None
        self.bar_data = None
        
        self.setup_ui()
        self.setup_connections()
    
    def set_chart_timestamps(self, timestamps):
        """Set chart timestamps for coordinated trade generation"""
        self.chart_timestamps = timestamps
        
        # Also pass timestamps to the source selector
        if hasattr(self, 'source_selector') and self.source_selector:
            self.source_selector.set_chart_timestamps(timestamps)
    
    def set_bar_data(self, bar_data):
        """Set bar data for realistic trade pricing"""
        self.bar_data = bar_data

        # Also pass bar data to the source selector
        if hasattr(self, 'source_selector') and self.source_selector:
            self.source_selector.set_bar_data(bar_data)

        # Pass to strategy runner
        if hasattr(self, 'strategy_runner') and self.strategy_runner:
            self.strategy_runner.set_chart_data(bar_data)
        
    def setup_ui(self):
        """Setup the panel UI"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Tab widget for Source Selector and Strategy Runner
        self.tab_widget = QtWidgets.QTabWidget()

        # Trade source selector tab
        self.source_selector = TradeSourceSelector()
        self.tab_widget.addTab(self.source_selector, "Load Trades")

        # Strategy runner tab
        self.strategy_runner = StrategyRunner()
        self.tab_widget.addTab(self.strategy_runner, "Run Strategy")

        layout.addWidget(self.tab_widget)

        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)
        
        # Trade list section
        list_layout = QtWidgets.QVBoxLayout()
        
        # Trade list header
        header_layout = QtWidgets.QHBoxLayout()
        
        list_title = QtWidgets.QLabel("Trade List")
        list_title.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        header_layout.addWidget(list_title)
        
        # Auto-sync checkbox
        self.auto_sync_cb = QtWidgets.QCheckBox("Auto-sync with chart")
        self.auto_sync_cb.setChecked(True)
        self.auto_sync_cb.toggled.connect(self.set_auto_sync)
        header_layout.addWidget(self.auto_sync_cb)
        
        list_layout.addLayout(header_layout)
        
        # Trade table
        self.table_model = TradeTableModel()
        self.table_view = QtWidgets.QTableView()
        self.table_view.setModel(self.table_model)
        
        # Table configuration
        self.table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(True)
        
        # Grey theme styling - much better visibility
        self.table_view.setStyleSheet("""
            QTableView {
                background-color: #2a2a2a;
                alternate-background-color: #333333;
                color: #ffffff;
                gridline-color: #444444;
                selection-background-color: #4a4a4a;
                border: 1px solid #555555;
            }
            QTableView::item {
                padding: 6px;
                border: none;
            }
            QTableView::item:selected {
                background-color: #555555;
                color: #ffffff;
            }
            QTableView::item:hover {
                background-color: #404040;
            }
            QHeaderView::section {
                background-color: #1e1e1e;
                color: #ffffff;
                padding: 8px;
                border: 1px solid #444444;
                font-weight: bold;
            }
        """)
        
        # Column sizing
        header = self.table_view.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        
        # Vertical header (row numbers)
        self.table_view.verticalHeader().setVisible(False)
        
        list_layout.addWidget(self.table_view)
        
        # Export buttons
        export_layout = QtWidgets.QHBoxLayout()
        
        self.export_all_btn = QtWidgets.QPushButton("Export All")
        self.export_all_btn.setToolTip("Export all trades to CSV")
        self.export_all_btn.clicked.connect(self.export_all_trades)
        
        self.export_visible_btn = QtWidgets.QPushButton("Export Visible")
        self.export_visible_btn.setToolTip("Export currently visible trades to CSV")
        self.export_visible_btn.clicked.connect(self.export_visible_trades)
        
        # Style the export buttons
        button_style = """
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666666;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
                border-color: #888888;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """
        self.export_all_btn.setStyleSheet(button_style)
        self.export_visible_btn.setStyleSheet(button_style)
        
        export_layout.addWidget(self.export_all_btn)
        export_layout.addWidget(self.export_visible_btn)
        export_layout.addStretch()  # Push buttons to the left
        
        # Initially disable export buttons until trades are loaded
        self.export_all_btn.setEnabled(False)
        self.export_visible_btn.setEnabled(False)
        
        list_layout.addLayout(export_layout)
        
        layout.addLayout(list_layout, 1)  # Give table most of the space
        
        # Status bar
        self.status_bar = QtWidgets.QLabel("No trades loaded")
        self.status_bar.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        layout.addWidget(self.status_bar)
        
    def setup_connections(self):
        """Setup signal connections"""
        # Source selector
        self.source_selector.trades_loaded.connect(self.load_trades)

        # Strategy runner
        self.strategy_runner.trades_generated.connect(self.load_trades)

        # Table selection
        self.table_view.doubleClicked.connect(self.on_trade_double_clicked)
        
    def load_trades(self, trades: TradeCollection):
        """Load trades into the panel"""
        print(f"[TRADE_PANEL] load_trades called with {len(trades)} trades")

        # Debug: Check first few trades for timestamp
        for i, trade in enumerate(trades[:3]):
            print(f"[TRADE_PANEL] Trade {i}: timestamp={trade.timestamp}, type={type(trade.timestamp)}")

        self.trades = trades
        self.table_model.set_trades(trades)

        # Simple status without redundant trade type counts
        self.status_bar.setText(f"{len(trades)} trades loaded")
        
        # Enable export buttons when trades are loaded
        if len(trades) > 0:
            self.export_all_btn.setEnabled(True)
            self.export_visible_btn.setEnabled(True)
        else:
            self.export_all_btn.setEnabled(False)
            self.export_visible_btn.setEnabled(False)
        
        logger.info(f"Loaded {len(trades)} trades into panel")
    
    def on_trade_double_clicked(self, index: QtCore.QModelIndex):
        """Handle double-click on trade row"""
        print(f"[TRADE_PANEL] Double-click detected on row {index.row()}")
        if not index.isValid():
            print(f"[TRADE_PANEL] Invalid index")
            return

        trade = self.table_model.get_trade_at_row(index.row())
        if trade:
            print(f"[TRADE_PANEL] Trade found: {trade.trade_type} at bar {trade.bar_index}")
            print(f"[TRADE_PANEL] Emitting trade_selected signal")
            self.trade_selected.emit(trade)
            logger.debug(f"Trade selected: {trade.trade_type} at bar {trade.bar_index}")
        else:
            print(f"[TRADE_PANEL] No trade found at row {index.row()}")
    
    def scroll_to_trade(self, trade: TradeData):
        """Scroll to specific trade in the list (for auto-sync)"""
        if not self.auto_sync_enabled:
            return
        
        # Find trade in model
        for row in range(len(self.trades)):
            if self.trades[row] == trade:
                index = self.table_model.index(row, 0)
                self.table_view.scrollTo(index, QtWidgets.QAbstractItemView.PositionAtTop)
                break
    
    def scroll_to_first_visible_trade(self, start_bar: int, end_bar: int):
        """Scroll to first trade visible in chart range"""
        if not self.auto_sync_enabled or not self.trades:
            return
        
        first_trade = self.trades.get_first_visible_trade(start_bar, end_bar)
        if first_trade:
            self.scroll_to_trade(first_trade)
    
    def set_auto_sync(self, enabled: bool):
        """Enable/disable auto-sync with chart"""
        self.auto_sync_enabled = enabled
        logger.debug(f"Auto-sync {'enabled' if enabled else 'disabled'}")
    
    def get_selected_trade(self) -> Optional[TradeData]:
        """Get currently selected trade"""
        selection = self.table_view.selectionModel().currentIndex()
        if selection.isValid():
            return self.table_model.get_trade_at_row(selection.row())
        return None
    
    def export_all_trades(self):
        """Export all trades to CSV file"""
        if not self.trades or len(self.trades) == 0:
            QtWidgets.QMessageBox.information(self, "Export", "No trades to export.")
            return
        
        # File dialog for save location
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 
            "Export All Trades", 
            f"all_trades_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV files (*.csv)"
        )
        
        if file_path:
            try:
                df = self.trades.to_dataframe()
                df.to_csv(file_path, index=False)
                QtWidgets.QMessageBox.information(
                    self, 
                    "Export Complete", 
                    f"Exported {len(df)} trades to:\n{file_path}"
                )
                logger.info(f"Exported {len(df)} trades to {file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, 
                    "Export Error", 
                    f"Failed to export trades:\n{str(e)}"
                )
                logger.error(f"Export failed: {e}")
    
    def export_visible_trades(self):
        """Export currently visible/filtered trades to CSV file"""
        if not self.trades or len(self.trades) == 0:
            QtWidgets.QMessageBox.information(self, "Export", "No trades to export.")
            return
        
        # TODO: Implement visible range filtering
        # For now, export all trades (same as export_all_trades)
        # In the future, this could filter by current chart viewport
        self.export_all_trades()
        
        # Future implementation would get visible range from parent chart:
        # visible_trades = self.trades.get_trades_in_range(start_bar, end_bar)

def create_test_panel():
    """Create test trade panel for development"""
    app = QtWidgets.QApplication([])
    
    # Create panel
    panel = TradeListPanel()
    panel.setWindowTitle("Trade Panel Test")
    panel.resize(400, 600)
    
    # Connect signals for testing
    def on_trade_selected(trade):
        print(f"Trade selected: {trade.trade_type} {trade.size} @ ${trade.price:.2f} at bar {trade.bar_index}")
    
    panel.trade_selected.connect(on_trade_selected)
    
    # Show panel
    panel.show()
    
    return app, panel

if __name__ == "__main__":
    # Test the trade panel
    print("Testing Trade Panel...")
    
    try:
        app, panel = create_test_panel()
        
        # Load some test trades
        from trade_data import create_sample_trades
        test_trades = create_sample_trades(50, 0, 200)
        panel.load_trades(test_trades)
        
        print(f"Loaded {len(test_trades)} test trades")
        print("Double-click trades to test navigation")
        print("Use auto-sync checkbox to test chart synchronization")
        
        # Run Qt event loop for testing (comment out for automated testing)
        # app.exec_()
        
        print("Trade panel test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()