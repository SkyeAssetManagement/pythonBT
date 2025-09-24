#!/usr/bin/env python3
"""
Enhanced Trade Panel - Displays P&L as percentages instead of dollar amounts
Compatible with existing trade_panel.py but with percentage P&L display
Minimal changes to preserve chart rendering
"""

import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Optional, Dict, List
import logging
import sys
import os

# Import existing components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_panel import TradeSourceSelector, TradeListPanel
from trade_data import TradeData, TradeCollection

logger = logging.getLogger(__name__)


class EnhancedTradeTableModel(QtCore.QAbstractTableModel):
    """Enhanced table model that displays P&L as percentages"""

    COLUMNS = [
        ("Trade #", "trade_id"),
        ("DateTime", "timestamp"),
        ("Type", "trade_type"),
        ("Price", "price"),
        ("Size", "size"),
        ("P&L %", "pnl_percent"),  # Changed to percentage display
        ("Cum P&L %", "cumulative_pnl_percent"),  # Added cumulative P&L
        ("Bar #", "bar_index")
    ]

    def __init__(self):
        super().__init__()
        self.trades = TradeCollection([])
        self.cumulative_pnl_percent = []  # Track cumulative P&L
        self.sort_column = None
        self.sort_order = QtCore.Qt.AscendingOrder

    def set_trades(self, trades: TradeCollection):
        """Update trade data and calculate cumulative P&L"""
        self.beginResetModel()
        self.trades = trades
        self._calculate_cumulative_pnl()
        self.endResetModel()

        logger.debug(f"Updated table model with {len(trades)} trades")

    def _calculate_cumulative_pnl(self):
        """Calculate cumulative P&L percentages (properly compounded returns)"""
        self.cumulative_pnl_percent = []
        cumulative_multiplier = 1.0  # Start with $1

        for trade in self.trades:
            # Get P&L percentage (based on $1 invested)
            pnl_percent = self._get_pnl_percent(trade)
            if pnl_percent is not None:
                # Properly compound the return: new value = old value * (1 + return)
                cumulative_multiplier *= (1 + pnl_percent)
            # Store as percentage gain/loss from initial capital
            cumulative_return = cumulative_multiplier - 1.0
            self.cumulative_pnl_percent.append(cumulative_return)

    def _get_pnl_percent(self, trade: TradeData) -> Optional[float]:
        """Get P&L as percentage (already calculated based on $1 invested)"""
        # Check if trade has pnl_percent attribute (new format)
        if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
            return trade.pnl_percent

        # Otherwise try to calculate from dollar P&L
        # Note: Legacy pnl field represents points, not percentage
        if hasattr(trade, 'pnl') and trade.pnl is not None:
            # Legacy pnl is in price points (e.g., 100 points)
            # We need to convert to percentage assuming $1 position size
            # For that we need the entry price
            if hasattr(trade, 'price') and trade.price is not None:
                # Approximate: if pnl is 100 points and price is 4200,
                # then percentage return is 100/4200 = 2.38%
                return trade.pnl / trade.price
            else:
                # Without price info, can't convert properly
                # Return as tiny fraction assuming high price stock
                return trade.pnl / 4000  # Assume ~$4000 stock price

        return None

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
            # Handle special percentage columns
            if column_attr == "pnl_percent":
                pnl_percent = self._get_pnl_percent(trade)
                if pnl_percent is not None:
                    if pnl_percent != 0:
                        # Convert decimal to percentage for display
                        pnl_display = pnl_percent * 100
                        sign = '+' if pnl_display >= 0 else ''
                        return f"{sign}{pnl_display:.2f}%"
                    else:
                        return "0.00%"
                return "-"

            elif column_attr == "cumulative_pnl_percent":
                if index.row() < len(self.cumulative_pnl_percent):
                    cum_pnl = self.cumulative_pnl_percent[index.row()]
                    if cum_pnl != 0:
                        # Convert decimal to percentage for display
                        cum_pnl_display = cum_pnl * 100
                        sign = '+' if cum_pnl_display >= 0 else ''
                        return f"{sign}{cum_pnl_display:.2f}%"
                    else:
                        return "0.00%"
                return "-"

            # Standard columns
            elif column_attr == "timestamp":
                value = getattr(trade, column_attr, None)
                if value is not None:
                    try:
                        if hasattr(value, 'strftime'):
                            return value.strftime('%y-%m-%d %H:%M:%S')
                        else:
                            dt = pd.to_datetime(value)
                            return dt.strftime('%y-%m-%d %H:%M:%S')
                    except:
                        return "-"
                return "-"

            elif column_attr == "trade_id":
                value = getattr(trade, column_attr, None)
                return str(value) if value is not None else "-"

            elif column_attr == "price":
                value = getattr(trade, column_attr, None)
                return f"${value:.2f}" if value is not None else "-"

            elif column_attr == "size":
                value = getattr(trade, column_attr, 1)
                return str(value)

            else:
                value = getattr(trade, column_attr, None)
                return str(value) if value is not None else ""

        elif role == QtCore.Qt.TextColorRole:
            # Color code by trade type and P&L
            if column_attr == "trade_type":
                if trade.trade_type in ['BUY', 'COVER']:
                    return QtGui.QColor(0, 200, 0)  # Green
                else:
                    return QtGui.QColor(255, 120, 120)  # Light red

            elif column_attr == "pnl_percent":
                pnl_percent = self._get_pnl_percent(trade)
                if pnl_percent is not None:
                    if pnl_percent > 0:
                        return QtGui.QColor(0, 200, 0)  # Green profit
                    elif pnl_percent < 0:
                        return QtGui.QColor(255, 120, 120)  # Red loss

            elif column_attr == "cumulative_pnl_percent":
                if index.row() < len(self.cumulative_pnl_percent):
                    cum_pnl = self.cumulative_pnl_percent[index.row()]
                    if cum_pnl > 0:
                        return QtGui.QColor(0, 200, 0)
                    elif cum_pnl < 0:
                        return QtGui.QColor(255, 120, 120)

        elif role == QtCore.Qt.TextAlignmentRole:
            if column_attr in ["price", "size", "pnl_percent", "cumulative_pnl_percent", "trade_id"]:
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


class EnhancedTradeListPanel(TradeListPanel):
    """
    Enhanced trade panel with percentage P&L display and summary statistics
    Inherits from existing TradeListPanel to maintain compatibility
    """

    def __init__(self):
        super().__init__()
        # Replace the table model with enhanced version
        self.setup_enhanced_components()

    def setup_enhanced_components(self):
        """Replace standard components with enhanced versions"""
        # Replace table model with enhanced version
        self.table_model = EnhancedTradeTableModel()
        self.table_view.setModel(self.table_model)

        # Enable sorting on the table
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().sectionClicked.connect(self.on_header_clicked)

        # Add summary panel
        self.add_summary_panel()

    def add_summary_panel(self):
        """Add P&L summary statistics panel at the bottom of the trade list"""
        # Find the main layout
        main_layout = self.layout()

        # Create summary widget with better visibility
        summary_widget = QtWidgets.QGroupBox("Backtest Summary")
        summary_widget.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                color: #ffffff;
            }
        """)
        summary_layout = QtWidgets.QGridLayout(summary_widget)
        summary_layout.setContentsMargins(10, 10, 10, 10)
        summary_layout.setSpacing(5)

        # Summary labels with better styling
        self.total_trades_label = QtWidgets.QLabel("Total Trades: 0")
        self.win_rate_label = QtWidgets.QLabel("Win Rate: 0.00%")
        self.total_pnl_label = QtWidgets.QLabel("Total P&L: 0.00%")
        self.avg_pnl_label = QtWidgets.QLabel("Avg P&L: 0.00%")
        # Trade type counts
        self.longs_label = QtWidgets.QLabel("Longs: 0")
        self.shorts_label = QtWidgets.QLabel("Shorts: 0")
        # Commission and lag info
        self.commission_label = QtWidgets.QLabel("Commission: $0.00")
        self.execution_label = QtWidgets.QLabel("Avg Lag: 0.0 bars")

        # Style the labels for better visibility
        label_style = "QLabel { font-size: 11pt; padding: 5px; color: #ffffff; background-color: #333333; border-radius: 3px; }"
        for label in [self.total_trades_label, self.win_rate_label,
                     self.total_pnl_label, self.avg_pnl_label,
                     self.longs_label, self.shorts_label,
                     self.commission_label, self.execution_label]:
            label.setStyleSheet(label_style)

        # Add to grid layout with better spacing (4 rows now)
        summary_layout.addWidget(self.total_trades_label, 0, 0)
        summary_layout.addWidget(self.win_rate_label, 0, 1)
        summary_layout.addWidget(self.longs_label, 1, 0)
        summary_layout.addWidget(self.shorts_label, 1, 1)
        summary_layout.addWidget(self.total_pnl_label, 2, 0)
        summary_layout.addWidget(self.avg_pnl_label, 2, 1)
        summary_layout.addWidget(self.commission_label, 3, 0)
        summary_layout.addWidget(self.execution_label, 3, 1)

        # Make summary widget 50% larger
        summary_widget.setMinimumHeight(120)
        summary_widget.setMaximumHeight(150)

        # Insert summary panel at the bottom
        main_layout.addWidget(summary_widget)

        # Adjust layout to maximize trade list space
        # Give maximum stretch to the table view (typically item 1)
        if main_layout.count() > 1:
            main_layout.setStretch(1, 10)  # Trade table gets most space
        # Minimal stretch for summary
        main_layout.setStretch(main_layout.count() - 1, 0)

    def on_trades_loaded(self, trades: TradeCollection):
        """Handle trades loaded - override to update summary"""
        super().on_trades_loaded(trades)
        self.update_summary_stats()

    def update_summary_stats(self):
        """Calculate and display summary statistics"""
        if not self.trades or len(self.trades) == 0:
            self.total_trades_label.setText("Total Trades: 0")
            self.win_rate_label.setText("Win Rate: 0.00%")
            self.longs_label.setText("Longs: 0")
            self.shorts_label.setText("Shorts: 0")
            self.total_pnl_label.setText("Total P&L: 0.00%")
            self.avg_pnl_label.setText("Avg P&L: 0.00%")
            self.commission_label.setText("Commission: $0.00")
            self.execution_label.setText("Avg Lag: 0.0 bars")
            return

        # Calculate statistics
        total_trades = len(self.trades)
        trades_with_pnl = []
        total_commission = 0.0
        lag_values = []

        # Count longs and shorts
        longs_count = 0
        shorts_count = 0

        for trade in self.trades:
            # Count trade types correctly:
            # BUY = long entry, SELL = long exit
            # SHORT = short entry, COVER = short exit
            if hasattr(trade, 'trade_type'):
                trade_type = trade.trade_type.upper()
                if trade_type in ['BUY', 'SELL']:
                    longs_count += 1
                elif trade_type in ['SHORT', 'COVER']:
                    shorts_count += 1

            # Get P&L percentage (based on $1 invested)
            pnl_percent = None
            if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
                pnl_percent = trade.pnl_percent
            elif hasattr(trade, 'pnl') and trade.pnl is not None:
                # For compatibility - treat pnl as percentage if it's the old format
                pnl_percent = trade.pnl

            if pnl_percent is not None:
                trades_with_pnl.append(pnl_percent)

            # Sum commissions if available
            if hasattr(trade, 'fees') and trade.fees is not None:
                total_commission += trade.fees

            # Track execution lag
            if hasattr(trade, 'lag') and trade.lag is not None:
                lag_values.append(trade.lag)

        # Calculate metrics
        if trades_with_pnl:
            # Only count closed trades for win rate
            closed_trades = [p for p in trades_with_pnl if p != 0]
            wins = [p for p in closed_trades if p > 0]
            win_rate = (len(wins) / len(closed_trades)) * 100 if closed_trades else 0

            # Properly calculate total P&L as compounded return
            cumulative_multiplier = 1.0
            for pnl in trades_with_pnl:
                cumulative_multiplier *= (1 + pnl)
            total_pnl = cumulative_multiplier - 1.0  # Convert back to percentage gain/loss

            # Average P&L per trade (arithmetic mean is OK for individual trade performance)
            avg_pnl = sum(trades_with_pnl) / len(trades_with_pnl) if trades_with_pnl else 0
        else:
            win_rate = 0.00
            total_pnl = 0.00
            avg_pnl = 0.00

        # Calculate average lag
        avg_lag = sum(lag_values) / len(lag_values) if lag_values else 0

        # Update labels with proper formatting
        self.total_trades_label.setText(f"Total Trades: {total_trades}")
        self.win_rate_label.setText(f"Win Rate: {win_rate:.2f}%")
        self.longs_label.setText(f"Longs: {longs_count}")
        self.shorts_label.setText(f"Shorts: {shorts_count}")

        # Color code P&L labels
        total_color = "green" if total_pnl >= 0 else "red"
        avg_color = "green" if avg_pnl >= 0 else "red"

        # Convert decimal to percentage for display
        total_pnl_display = total_pnl * 100
        avg_pnl_display = avg_pnl * 100

        total_sign = '+' if total_pnl_display >= 0 else ''
        avg_sign = '+' if avg_pnl_display >= 0 else ''

        self.total_pnl_label.setText(f"Total P&L: {total_sign}{total_pnl_display:.2f}%")
        self.total_pnl_label.setStyleSheet(f"QLabel {{ color: {total_color}; font-size: 10pt; padding: 2px; }}")

        self.avg_pnl_label.setText(f"Avg P&L: {avg_sign}{avg_pnl_display:.2f}%")
        self.avg_pnl_label.setStyleSheet(f"QLabel {{ color: {avg_color}; font-size: 10pt; padding: 2px; }}")

        # Update commission and execution info with correct formatting
        self.commission_label.setText(f"Commission: ${total_commission:.2f}")
        # Display average lag from actual trade data
        if lag_values:
            # Calculate and display actual average lag from trades
            self.execution_label.setText(f"Avg Lag: {avg_lag:.1f} bars")
        else:
            # No explicit lag data in trades - try to read from config
            try:
                import yaml
                import os
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'tradingCode', 'config.yaml')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        lag = config.get('backtest', {}).get('signal_lag', 1)
                        self.execution_label.setText(f"Avg Lag: {lag:.1f} bars")
                else:
                    self.execution_label.setText("Avg Lag: 1.0 bars")
            except:
                self.execution_label.setText("Avg Lag: 1.0 bars")

    def load_trades(self, trades: TradeCollection):
        """Load trades and update display - override parent method"""
        super().load_trades(trades)
        self.update_summary_stats()

    def set_trades(self, trades: TradeCollection):
        """Set trades and update display - convenience method"""
        self.load_trades(trades)

    def on_header_clicked(self, logical_index: int):
        """Handle header click for sorting"""
        # Toggle sort order if clicking same column
        if self.table_model.sort_column == logical_index:
            if self.table_model.sort_order == QtCore.Qt.AscendingOrder:
                self.table_model.sort_order = QtCore.Qt.DescendingOrder
            else:
                self.table_model.sort_order = QtCore.Qt.AscendingOrder
        else:
            self.table_model.sort_column = logical_index
            self.table_model.sort_order = QtCore.Qt.AscendingOrder

        # Sort the trades
        self.sort_trades(logical_index, self.table_model.sort_order)

    def sort_trades(self, column: int, order: QtCore.Qt.SortOrder):
        """Sort trades by specified column"""
        if not self.trades or len(self.trades) == 0:
            return

        column_attr = self.table_model.COLUMNS[column][1]

        # Create list of (value, trade) pairs for sorting
        sort_data = []
        for i, trade in enumerate(self.trades):
            if column_attr == "pnl_percent":
                value = self.table_model._get_pnl_percent(trade)
                if value is None:
                    value = 0.0
            elif column_attr == "cumulative_pnl_percent":
                if i < len(self.table_model.cumulative_pnl_percent):
                    value = self.table_model.cumulative_pnl_percent[i]
                else:
                    value = 0.0
            elif column_attr == "timestamp":
                value = getattr(trade, column_attr, None)
                if value is not None:
                    try:
                        if not hasattr(value, 'timestamp'):
                            value = pd.to_datetime(value)
                        # Convert to timestamp for sorting
                        value = value.timestamp() if hasattr(value, 'timestamp') else 0
                    except:
                        value = 0
                else:
                    value = 0
            else:
                value = getattr(trade, column_attr, None)
                if value is None:
                    value = 0 if column_attr in ["trade_id", "bar_index", "price", "size"] else ""

            sort_data.append((value, trade))

        # Sort the data
        reverse = (order == QtCore.Qt.DescendingOrder)
        try:
            sort_data.sort(key=lambda x: x[0], reverse=reverse)
        except TypeError:
            # If sorting fails due to mixed types, convert to strings
            sort_data.sort(key=lambda x: str(x[0]), reverse=reverse)

        # Extract sorted trades
        sorted_trades = [trade for _, trade in sort_data]

        # Update the model with sorted trades
        self.table_model.set_trades(TradeCollection(sorted_trades))
        self.trades = TradeCollection(sorted_trades)

        # Update summary stats after sorting
        self.update_summary_stats()


# Export the enhanced panel as the default
TradeListPanel = EnhancedTradeListPanel