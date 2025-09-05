# src/dashboard/trade_list_widget.py
# Clickable Trade List Widget with Chart Navigation Integration
# 
# Provides a high-performance trade list that integrates seamlessly with the VisPy chart
# Features:
# - Load trades from VectorBT CSV output
# - Clickable rows that jump to trade location on chart
# - Real-time synchronization with chart viewport
# - High-performance rendering for thousands of trades

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QWidget

# Import time formatting from time_axis_widget
from dashboard.time_axis_widget import TradeListTimeFormatter

@dataclass
class TradeData:
    """
    Structured trade data for efficient processing and display
    Compatible with VectorBT trade list output format
    """
    trade_id: str
    entry_time: int        # Timestamp as integer (nanoseconds or sequential index)
    exit_time: int         # Timestamp as integer  
    side: str              # 'Long' or 'Short'
    entry_price: float     # Average entry price
    exit_price: float      # Average exit price
    size: float            # Position size
    pnl: float             # Profit/Loss
    pnl_pct: float         # PnL percentage
    duration: int          # Trade duration in bars
    
    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable"""
        return self.pnl > 0
    
    @property
    def trade_type_color(self) -> str:
        """Get color for trade type"""
        if self.is_profitable:
            return "#2E7D32" if self.side == 'Long' else "#1976D2"  # Green for long profit, blue for short profit
        else:
            return "#C62828" if self.side == 'Long' else "#D32F2F"  # Red variations for losses

class TradeListWidget(QTableWidget):
    """
    High-performance trade list widget with chart navigation integration
    Optimized for displaying thousands of trades efficiently
    """
    
    # Signal emitted when user clicks on a trade - passes trade data and chart position
    trade_selected = pyqtSignal(object, int)  # TradeData, chart_index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Trade data storage
        self.trades_data: List[TradeData] = []
        self.trades_df: Optional[pd.DataFrame] = None
        
        # Chart integration
        self.chart_navigation_callback: Optional[Callable] = None
        self.timestamp_to_index_mapper: Optional[Callable] = None
        
        # Datetime data for time formatting
        self.datetime_data: Optional[np.ndarray] = None
        
        # UI optimization settings
        self.max_visible_trades = 100000  # Increased limit to show all trades
        
        # Initialize the widget
        self._setup_table_structure()
        self._setup_styling()
        self._connect_signals()
        
        print("Trade list widget initialized with navigation functionality")
        
    def _setup_table_structure(self):
        """Set up table columns and basic structure"""
        
        # Define table columns based on trading dashboard requirements
        self.columns = [
            ("Trade ID", 80),
            ("Time", 120), 
            ("Side", 60),
            ("Entry", 80),
            ("Exit", 80), 
            ("Size", 80),
            ("PnL", 80),
            ("PnL%", 60),
            ("Duration", 70)
        ]
        
        # Set up table structure
        self.setColumnCount(len(self.columns))
        
        # Set column headers and widths
        headers = []
        for i, (header, width) in enumerate(self.columns):
            headers.append(header)
            self.setColumnWidth(i, width)
            
        self.setHorizontalHeaderLabels(headers)
        
        # Configure table behavior
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(False)  # Remove white rows
        self.setSortingEnabled(True)
        
        # Optimize for performance
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.verticalHeader().setVisible(False)
        
        # Auto-resize columns to content
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
    
    def _setup_styling(self):
        """Set up professional styling for the trade list"""
        
        # Professional dark theme styling
        self.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                selection-background-color: #404040;
                gridline-color: #555555;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
            }
            
            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid #444444;
            }
            
            QTableWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            
            QTableWidget::item:hover {
                background-color: #404040;
            }
            
            QHeaderView::section {
                background-color: #404040;
                color: white;
                padding: 6px;
                border: 1px solid #555555;
                font-weight: bold;
                font-size: 9pt;
            }
            
            QHeaderView::section:hover {
                background-color: #505050;
            }
        """)
    
    def _connect_signals(self):
        """Connect signals for trade navigation functionality"""
        
        # Connect cell click signal to navigation handler
        self.cellClicked.connect(self._on_trade_clicked)
        
        # Connect double-click for additional functionality if needed
        self.cellDoubleClicked.connect(self._on_trade_double_clicked)
        
        print("Trade list signals connected for navigation")
    
    def _on_trade_clicked(self, row: int, column: int):
        """Handle single click on trade row"""
        try:
            if row >= len(self.trades_data):
                print(f"Invalid row clicked: {row} (max: {len(self.trades_data)})")
                return
            
            # Get the trade data for this row
            trade_data = self.trades_data[row]
            
            print(f"Trade clicked: {trade_data.trade_id} at row {row}")
            
            # Emit signal with trade data and chart position
            chart_index = int(trade_data.entry_time)  # Use entry time as chart index
            self.trade_selected.emit(trade_data, chart_index)
            
            # Call navigation callback if available  
            if self.chart_navigation_callback:
                success = self.chart_navigation_callback(trade_data.entry_time)
                if success:
                    print(f"SUCCESS: Navigated to trade {trade_data.trade_id}")
                else:
                    print(f"FAILED: Navigation to trade {trade_data.trade_id}")
            else:
                print("No chart navigation callback set")
                
        except Exception as e:
            print(f"ERROR: Failed to handle trade click: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_trade_double_clicked(self, row: int, column: int):
        """Handle double click on trade row (for future enhancements)"""
        try:
            if row >= len(self.trades_data):
                return
            
            trade_data = self.trades_data[row]
            print(f"Trade double-clicked: {trade_data.trade_id} - could show trade details dialog")
            
        except Exception as e:
            print(f"ERROR: Failed to handle trade double-click: {e}")
    
    def set_chart_navigation_callback(self, callback: Callable):
        """Set the callback function for chart navigation"""
        self.chart_navigation_callback = callback
        print(f"Chart navigation callback set: {callback}")
    
    def navigate_to_trade_by_id(self, trade_id: str) -> bool:
        """
        Navigate to trade by ID - for input box functionality
        
        Args:
            trade_id: Trade ID to navigate to (e.g., "T001", "T21801")
            
        Returns:
            bool: True if navigation successful
        """
        try:
            # Find trade by ID
            found_trade = None
            for trade in self.trades_data:
                if trade.trade_id.upper() == trade_id.upper():
                    found_trade = trade
                    break
                    
                # Also check if user entered just a number
                if trade_id.isdigit() and trade.trade_id.upper() == f"T{int(trade_id):03d}":
                    found_trade = trade
                    break
            
            if not found_trade:
                print(f"Trade ID '{trade_id}' not found")
                available_ids = [trade.trade_id for trade in self.trades_data[:10]]
                print(f"Available trade IDs: {available_ids}")
                return False
            
            # Navigate to the trade
            if self.chart_navigation_callback:
                success = self.chart_navigation_callback(found_trade.entry_time)
                if success:
                    # Also select the row in the table
                    for i, trade in enumerate(self.trades_data):
                        if trade.trade_id == found_trade.trade_id:
                            self.selectRow(i)
                            self.scrollToItem(self.item(i, 0))
                            break
                    
                    print(f"SUCCESS: Navigated to trade {trade_id}")
                    return True
                else:
                    print(f"FAILED: Navigation callback returned false for {trade_id}")
                    return False
            else:
                print("No chart navigation callback set")
                return False
                
        except Exception as e:
            print(f"ERROR: Failed to navigate to trade {trade_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_trades_from_csv(self, csv_path: str) -> bool:
        """
        Load trades from VectorBT CSV output file
        Handles both standard and custom trade list formats
        """
        try:
            # Read trade CSV file
            df = pd.read_csv(csv_path)
            
            print(f"   INFO: Loading trades from {csv_path}")
            print(f"   INFO: Found {len(df)} trades in CSV")
            print(f"   INFO: CSV columns: {list(df.columns)}")
            
            return self.load_trades_from_dataframe(df)
            
        except FileNotFoundError:
            print(f"   ERROR: Trade CSV file not found: {csv_path}")
            return False
        except Exception as e:
            print(f"   ERROR: Failed to load trades from CSV: {e}")
            return False
    
    def load_trades_from_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Load trades from pandas DataFrame (VectorBT format)
        Handles multiple trade list formats automatically
        """
        try:
            if df.empty:
                print(f"   WARNING: No trades found in DataFrame")
                return False
            
            self.trades_df = df.copy()
            
            # Detect and process different trade list formats
            if self._is_vectorbt_format(df):
                trades_data = self._process_vectorbt_format(df)
            elif self._is_standard_format(df):
                trades_data = self._process_standard_format(df)
            else:
                trades_data = self._process_generic_format(df)
            
            if trades_data:
                self.trades_data = trades_data
                self._populate_table()
                
                print(f"   SUCCESS: Loaded {len(self.trades_data)} trades into trade list")
                return True
            else:
                print(f"   ERROR: Could not process trade data format")
                return False
                
        except Exception as e:
            print(f"   ERROR: Failed to process trade DataFrame: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_datetime_data(self, datetime_data: np.ndarray):
        """Set datetime data for time formatting"""
        self.datetime_data = datetime_data
        # Refresh the table if it has data
        if self.trades_data:
            self._populate_table()
    
    def _is_vectorbt_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is in VectorBT trade list format"""
        vectorbt_columns = ['EntryTime', 'ExitTime', 'Direction', 'Avg Entry Price', 'Avg Exit Price', 'Size', 'PnL']
        return all(col in df.columns for col in vectorbt_columns[:4])  # Check core columns
    
    def _is_standard_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is in standard trade format"""
        standard_columns = ['entry_time', 'exit_time', 'side', 'entry_price', 'exit_price']
        return all(col in df.columns for col in standard_columns[:3])  # Check core columns
    
    def _process_vectorbt_format(self, df: pd.DataFrame) -> List[TradeData]:
        """Process VectorBT trade list format"""
        trades = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Extract core trade data
            entry_time = int(row['EntryTime']) if not pd.isna(row['EntryTime']) else 0
            exit_time = int(row['ExitTime']) if not pd.isna(row['ExitTime']) else entry_time + 1
            
            side = str(row.get('Direction', 'Long')).strip()
            entry_price = float(row.get('Avg Entry Price', 0))
            exit_price = float(row.get('Avg Exit Price', entry_price))
            size = float(row.get('Size', 1.0))
            pnl = float(row.get('PnL', 0))
            
            # Calculate additional metrics
            pnl_pct = (pnl / (entry_price * abs(size)) * 100) if entry_price > 0 and size != 0 else 0
            duration = max(1, exit_time - entry_time)  # Ensure positive duration
            
            trade = TradeData(
                trade_id=f"T{i+1:03d}",
                entry_time=entry_time,
                exit_time=exit_time,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                duration=duration
            )
            
            trades.append(trade)
        
        return trades
    
    def _process_standard_format(self, df: pd.DataFrame) -> List[TradeData]:
        """Process standard trade format"""
        trades = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            entry_time = int(row['entry_time'])
            exit_time = int(row.get('exit_time', entry_time + 1))
            
            trade = TradeData(
                trade_id=str(row.get('trade_id', f"T{i+1}")),
                entry_time=entry_time,
                exit_time=exit_time,
                side=str(row.get('side', 'Long')),
                entry_price=float(row.get('entry_price', 0)),
                exit_price=float(row.get('exit_price', 0)),
                size=float(row.get('size', 1)),
                pnl=float(row.get('pnl', 0)),
                pnl_pct=float(row.get('pnl_pct', 0)),
                duration=int(row.get('duration', 1))
            )
            
            trades.append(trade)
        
        return trades
    
    def _process_generic_format(self, df: pd.DataFrame) -> List[TradeData]:
        """Process generic CSV format - best effort conversion"""
        trades = []
        
        print(f"   INFO: Processing generic trade format, available columns: {list(df.columns)}")
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Try to extract time information from various column names
            entry_time = 0
            for col in ['entry_time', 'EntryTime', 'open_time', 'timestamp', 'time']:
                if col in row and not pd.isna(row[col]):
                    entry_time = int(row[col])
                    break
            
            if entry_time == 0:
                entry_time = i  # Use row index as fallback
            
            exit_time = entry_time + 1  # Default duration of 1 bar
            for col in ['exit_time', 'ExitTime', 'close_time']:
                if col in row and not pd.isna(row[col]):
                    exit_time = int(row[col])
                    break
            
            # Extract other data with fallbacks
            side = 'Long'  # Default
            for col in ['side', 'direction', 'Direction', 'type']:
                if col in row:
                    side = str(row[col])
                    break
            
            entry_price = 0.0
            for col in ['entry_price', 'Avg Entry Price', 'open_price', 'price']:
                if col in row and not pd.isna(row[col]):
                    entry_price = float(row[col])
                    break
                    
            exit_price = entry_price
            for col in ['exit_price', 'Avg Exit Price', 'close_price']:
                if col in row and not pd.isna(row[col]):
                    exit_price = float(row[col])
                    break
            
            pnl = 0.0
            for col in ['pnl', 'PnL', 'profit', 'return']:
                if col in row and not pd.isna(row[col]):
                    pnl = float(row[col])
                    break
            
            size = 1.0
            for col in ['size', 'Size', 'quantity', 'amount']:
                if col in row and not pd.isna(row[col]):
                    size = float(row[col])
                    break
            
            pnl_pct = (pnl / (entry_price * abs(size)) * 100) if entry_price > 0 and size != 0 else 0
            duration = max(1, exit_time - entry_time)
            
            trade = TradeData(
                trade_id=f"T{i+1:03d}",
                entry_time=entry_time,
                exit_time=exit_time,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                duration=duration
            )
            
            trades.append(trade)
        
        return trades
    
    def _populate_table(self):
        """Populate the table with trade data efficiently"""
        
        if not self.trades_data:
            return
        
        # Limit rows for performance (show most recent trades)
        trades_to_show = self.trades_data[-self.max_visible_trades:] if len(self.trades_data) > self.max_visible_trades else self.trades_data
        
        # Set table size
        self.setRowCount(len(trades_to_show))
        
        # Populate table data
        for row_idx, trade in enumerate(trades_to_show):
            # Trade ID
            self._set_table_item(row_idx, 0, trade.trade_id, alignment=Qt.AlignCenter)
            
            # Entry Time (format with HH:MM YYYY-MM-DD)
            # Check if entry_time is a timestamp (large number) or bar index (small number)
            if trade.entry_time > 1e15:  # Nanosecond timestamp
                time_str = TradeListTimeFormatter.format_timestamp(trade.entry_time)
            elif self.datetime_data is not None and trade.entry_time < len(self.datetime_data):
                # Bar index - look up timestamp in datetime_data
                time_str = TradeListTimeFormatter.format_timestamp(self.datetime_data[trade.entry_time])
            else:
                # Fallback to raw value
                time_str = TradeListTimeFormatter.format_timestamp(trade.entry_time)
            self._set_table_item(row_idx, 1, time_str, alignment=Qt.AlignCenter)
            
            # Side (Long/Short with color coding)
            side_item = self._set_table_item(row_idx, 2, trade.side, alignment=Qt.AlignCenter)
            side_color = QtGui.QColor("#4CAF50") if trade.side == 'Long' else QtGui.QColor("#FF9800")
            side_item.setBackground(side_color)
            side_item.setForeground(QtGui.QColor("white"))
            
            # Entry Price
            self._set_table_item(row_idx, 3, f"{trade.entry_price:.5f}", alignment=Qt.AlignRight)
            
            # Exit Price  
            self._set_table_item(row_idx, 4, f"{trade.exit_price:.5f}", alignment=Qt.AlignRight)
            
            # Size
            self._set_table_item(row_idx, 5, f"{trade.size:.2f}", alignment=Qt.AlignRight)
            
            # PnL with color coding
            pnl_text = f"{trade.pnl:+.2f}"
            pnl_item = self._set_table_item(row_idx, 6, pnl_text, alignment=Qt.AlignRight)
            pnl_color = QtGui.QColor("#2E7D32") if trade.pnl > 0 else QtGui.QColor("#C62828")
            pnl_item.setForeground(pnl_color)
            
            # PnL% with color coding
            pnl_pct_text = f"{trade.pnl_pct:+.1f}%"
            pnl_pct_item = self._set_table_item(row_idx, 7, pnl_pct_text, alignment=Qt.AlignRight)
            pnl_pct_item.setForeground(pnl_color)
            
            # Duration
            self._set_table_item(row_idx, 8, str(trade.duration), alignment=Qt.AlignCenter)
        
        # Auto-resize columns to content
        self.resizeColumnsToContents()
        
        print(f"   INFO: Trade table populated with {len(trades_to_show)} trades")
    
    def _set_table_item(self, row: int, col: int, text: str, alignment: Qt.Alignment = Qt.AlignLeft) -> QTableWidgetItem:
        """Helper to create and set table items with consistent formatting"""
        item = QTableWidgetItem(text)
        item.setTextAlignment(alignment)
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)  # Read-only
        self.setItem(row, col, item)
        return item
    
    def _on_trade_clicked(self, row: int, column: int):
        """Handle trade row click - emit signal for chart navigation"""
        print(f"TRADE CLICK DEBUG: Row {row}, Column {column} clicked")
        if 0 <= row < len(self.trades_data):
            # Get trade data (accounting for potential pagination)
            visible_start_idx = max(0, len(self.trades_data) - self.max_visible_trades)
            trade_idx = visible_start_idx + row
            trade = self.trades_data[trade_idx]
            
            print(f"TRADE CLICK DEBUG: Found trade {trade.trade_id} at index {trade_idx}")
            
            # Convert trade timestamp to chart index
            chart_index = self._get_chart_index_for_trade(trade)
            
            print(f"TRADE CLICK DEBUG: Chart index: {chart_index}")
            print(f"TRADE CLICK DEBUG: Emitting signal with trade {trade.trade_id}")
            
            # Emit signal for chart navigation
            self.trade_selected.emit(trade, chart_index)
            
            print(f"   INFO: Trade {trade.trade_id} selected - navigating to chart index {chart_index}")
    
    def _on_trade_double_clicked(self, row: int, column: int):
        """Handle trade double-click for immediate chart focus"""
        self._on_trade_clicked(row, column)  # Same behavior for now
    
    def _get_chart_index_for_trade(self, trade: TradeData) -> int:
        """Convert trade timestamp to chart index for navigation"""
        if self.timestamp_to_index_mapper:
            return self.timestamp_to_index_mapper(trade.entry_time)
        else:
            # Fallback: assume timestamp is already an index
            return int(trade.entry_time)
    
    def set_chart_navigation_callback(self, callback: Callable[[int], None]):
        """Set callback function for chart navigation"""
        self.chart_navigation_callback = callback
    
    def set_timestamp_mapper(self, mapper: Callable[[int], int]):
        """Set function to convert timestamps to chart indices"""
        self.timestamp_to_index_mapper = mapper
    
    def navigate_to_trade(self, trade_id: str) -> bool:
        """Programmatically navigate to a specific trade"""
        for i, trade in enumerate(self.trades_data):
            if trade.trade_id == trade_id:
                chart_index = self._get_chart_index_for_trade(trade)
                self.trade_selected.emit(trade, chart_index)
                
                # Select the row in the table
                visible_start_idx = max(0, len(self.trades_data) - self.max_visible_trades)
                table_row = i - visible_start_idx
                if 0 <= table_row < self.rowCount():
                    self.selectRow(table_row)
                
                return True
        
        print(f"   WARNING: Trade {trade_id} not found")
        return False
    
    def get_trade_statistics(self) -> Dict[str, float]:
        """Get trade statistics for display"""
        if not self.trades_data:
            return {}
        
        profitable_trades = [t for t in self.trades_data if t.is_profitable]
        losing_trades = [t for t in self.trades_data if not t.is_profitable]
        
        total_pnl = sum(t.pnl for t in self.trades_data)
        total_trades = len(self.trades_data)
        win_rate = len(profitable_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }


class TradeListContainer(QWidget):
    """
    Container widget for the trade list with additional controls and statistics
    Provides the full trade list panel as shown in the target screenshot
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.trade_list_widget = TradeListWidget()
        self.stats_labels = {}
        
        self._setup_layout()
        self._connect_signals()
    
    def _setup_layout(self):
        """Set up the container layout with trade list and statistics"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title label
        title_label = QtWidgets.QLabel("Trade List")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 10pt;
                padding: 4px;
                background-color: #404040;
                border: 1px solid #555555;
            }
        """)
        layout.addWidget(title_label)
        
        # Trade list widget
        layout.addWidget(self.trade_list_widget)
        
        # Statistics panel
        stats_widget = self._create_statistics_panel()
        layout.addWidget(stats_widget)
        
        # Set stretch factors - trade list gets most space
        layout.setStretchFactor(self.trade_list_widget, 10)
        layout.setStretchFactor(stats_widget, 1)
    
    def _create_statistics_panel(self) -> QWidget:
        """Create statistics panel showing trade performance metrics"""
        stats_container = QWidget()
        stats_container.setMaximumHeight(80)
        
        layout = QtWidgets.QGridLayout(stats_container)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create statistic labels
        stat_items = [
            ("Trades:", "total_trades"),
            ("PnL:", "total_pnl"), 
            ("Win%:", "win_rate"),
            ("Avg Win:", "avg_win"),
            ("Avg Loss:", "avg_loss"),
            ("PF:", "profit_factor")
        ]
        
        for i, (label_text, key) in enumerate(stat_items):
            row = i // 3
            col = (i % 3) * 2
            
            # Label
            label = QtWidgets.QLabel(label_text)
            label.setStyleSheet("color: #cccccc; font-size: 8pt;")
            layout.addWidget(label, row, col)
            
            # Value
            value_label = QtWidgets.QLabel("0")
            value_label.setStyleSheet("color: white; font-size: 8pt; font-weight: bold;")
            layout.addWidget(value_label, row, col + 1)
            
            self.stats_labels[key] = value_label
        
        stats_container.setStyleSheet("""
            QWidget {
                background-color: #333333;
                border: 1px solid #555555;
            }
        """)
        
        return stats_container
    
    def _connect_signals(self):
        """Connect signals from trade list widget"""
        self.trade_list_widget.trade_selected.connect(self._on_trade_selected)
    
    def _on_trade_selected(self, trade_data, chart_index):
        """Handle trade selection and forward to parent"""
        # Update any UI feedback here if needed
        pass
    
    def load_trades(self, trades_source) -> bool:
        """Load trades from CSV file or DataFrame"""
        if isinstance(trades_source, str):
            success = self.trade_list_widget.load_trades_from_csv(trades_source)
        elif isinstance(trades_source, pd.DataFrame):
            success = self.trade_list_widget.load_trades_from_dataframe(trades_source)
        else:
            print(f"   ERROR: Unsupported trades source type: {type(trades_source)}")
            return False
        
        if success:
            self._update_statistics()
            
        return success
    
    def _update_statistics(self):
        """Update the statistics panel with current trade data"""
        stats = self.trade_list_widget.get_trade_statistics()
        
        # Update each statistic label
        self.stats_labels['total_trades'].setText(str(int(stats.get('total_trades', 0))))
        self.stats_labels['total_pnl'].setText(f"{stats.get('total_pnl', 0):+.2f}")
        self.stats_labels['win_rate'].setText(f"{stats.get('win_rate', 0):.1f}%")
        self.stats_labels['avg_win'].setText(f"+{stats.get('avg_win', 0):.2f}")
        self.stats_labels['avg_loss'].setText(f"{stats.get('avg_loss', 0):.2f}")
        self.stats_labels['profit_factor'].setText(f"{stats.get('profit_factor', 0):.2f}")
        
        # Color coding for PnL
        total_pnl = stats.get('total_pnl', 0)
        pnl_color = "#2E7D32" if total_pnl > 0 else "#C62828"
        self.stats_labels['total_pnl'].setStyleSheet(f"color: {pnl_color}; font-size: 8pt; font-weight: bold;")
    
    # Expose trade list widget methods
    def set_chart_navigation_callback(self, callback):
        """Set callback for chart navigation"""
        self.trade_list_widget.set_chart_navigation_callback(callback)
    
    def set_timestamp_mapper(self, mapper):
        """Set timestamp to chart index mapper"""
        self.trade_list_widget.set_timestamp_mapper(mapper)
    
    @property
    def trade_selected(self):
        """Expose trade selection signal"""
        return self.trade_list_widget.trade_selected


# Testing and validation functions
def test_trade_list_widget():
    """Test the trade list widget with synthetic data"""
    print(f"\n=== TESTING TRADE LIST WIDGET ===")
    
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create test trade data
    test_trades = []
    for i in range(20):
        pnl = np.random.normal(10, 50)  # Random PnL
        entry_price = 1.2000 + np.random.normal(0, 0.01)
        exit_price = entry_price + np.random.normal(0, 0.005)
        
        trade = TradeData(
            trade_id=f"T{i+1:03d}",
            entry_time=i * 100,
            exit_time=(i * 100) + np.random.randint(1, 50),
            side="Long" if i % 3 != 0 else "Short",
            entry_price=entry_price,
            exit_price=exit_price,
            size=np.random.uniform(0.1, 2.0),
            pnl=pnl,
            pnl_pct=pnl / (entry_price * 1.0) * 100,
            duration=np.random.randint(1, 50)
        )
        test_trades.append(trade)
    
    # Create and test widget
    container = TradeListContainer()
    container.trade_list_widget.trades_data = test_trades
    container.trade_list_widget._populate_table()
    container._update_statistics()
    
    # Set up test window
    container.setWindowTitle("Trade List Widget Test")
    container.resize(600, 400)
    container.show()
    
    print(f"   SUCCESS: Trade list widget displayed with {len(test_trades)} test trades")
    print(f"   INFO: Close window to continue testing")
    
    # Note: In production this would be: app.exec_()
    return True


if __name__ == "__main__":
    # Run standalone test
    test_trade_list_widget()