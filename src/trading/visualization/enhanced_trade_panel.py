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

    def set_trades(self, trades: TradeCollection):
        """Update trade data and calculate cumulative P&L"""
        self.beginResetModel()
        self.trades = trades
        self._calculate_cumulative_pnl()
        self.endResetModel()

        logger.debug(f"Updated table model with {len(trades)} trades")

    def _calculate_cumulative_pnl(self):
        """Calculate cumulative P&L percentages"""
        self.cumulative_pnl_percent = []
        cumulative = 0.0

        for trade in self.trades:
            # Convert dollar P&L to percentage if needed
            pnl_percent = self._get_pnl_percent(trade)
            if pnl_percent is not None:
                cumulative += pnl_percent
            self.cumulative_pnl_percent.append(cumulative)

    def _get_pnl_percent(self, trade: TradeData) -> Optional[float]:
        """Convert P&L to percentage format"""
        # Check if trade has pnl_percent attribute (new format)
        if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
            return trade.pnl_percent

        # Otherwise try to calculate from dollar P&L
        if hasattr(trade, 'pnl') and trade.pnl is not None:
            # Assume $1 position size for percentage calculation
            # This gives clean percentage display
            return trade.pnl * 100  # Convert to percentage

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
                    sign = '+' if pnl_percent >= 0 else ''
                    return f"{sign}{pnl_percent:.2f}%"
                return "-"

            elif column_attr == "cumulative_pnl_percent":
                if index.row() < len(self.cumulative_pnl_percent):
                    cum_pnl = self.cumulative_pnl_percent[index.row()]
                    sign = '+' if cum_pnl >= 0 else ''
                    return f"{sign}{cum_pnl:.2f}%"
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

        # Add summary panel
        self.add_summary_panel()

    def add_summary_panel(self):
        """Add P&L summary statistics panel"""
        # Find the main layout
        main_layout = self.layout()

        # Create summary widget
        summary_widget = QtWidgets.QGroupBox("Trade Summary")
        summary_layout = QtWidgets.QGridLayout(summary_widget)

        # Summary labels
        self.total_trades_label = QtWidgets.QLabel("Total Trades: 0")
        self.win_rate_label = QtWidgets.QLabel("Win Rate: 0.0%")
        self.total_pnl_label = QtWidgets.QLabel("Total P&L: 0.00%")
        self.avg_pnl_label = QtWidgets.QLabel("Avg P&L: 0.00%")

        # Style the labels
        label_style = "QLabel { font-size: 10pt; padding: 2px; }"
        for label in [self.total_trades_label, self.win_rate_label,
                     self.total_pnl_label, self.avg_pnl_label]:
            label.setStyleSheet(label_style)

        # Add to grid layout
        summary_layout.addWidget(self.total_trades_label, 0, 0)
        summary_layout.addWidget(self.win_rate_label, 0, 1)
        summary_layout.addWidget(self.total_pnl_label, 1, 0)
        summary_layout.addWidget(self.avg_pnl_label, 1, 1)

        # Insert summary panel at the bottom
        main_layout.addWidget(summary_widget)

        # Adjust layout stretch factors
        main_layout.setStretchFactor(summary_widget, 0)  # Don't stretch summary

    def on_trades_loaded(self, trades: TradeCollection):
        """Handle trades loaded - override to update summary"""
        super().on_trades_loaded(trades)
        self.update_summary_stats()

    def update_summary_stats(self):
        """Calculate and display summary statistics"""
        if not self.trades or len(self.trades) == 0:
            self.total_trades_label.setText("Total Trades: 0")
            self.win_rate_label.setText("Win Rate: 0.0%")
            self.total_pnl_label.setText("Total P&L: 0.00%")
            self.avg_pnl_label.setText("Avg P&L: 0.00%")
            return

        # Calculate statistics
        total_trades = len(self.trades)
        trades_with_pnl = []

        for trade in self.trades:
            # Get P&L percentage
            pnl_percent = None
            if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
                pnl_percent = trade.pnl_percent
            elif hasattr(trade, 'pnl') and trade.pnl is not None:
                pnl_percent = trade.pnl * 100

            if pnl_percent is not None:
                trades_with_pnl.append(pnl_percent)

        # Calculate metrics
        if trades_with_pnl:
            wins = [p for p in trades_with_pnl if p > 0]
            win_rate = (len(wins) / len(trades_with_pnl)) * 100 if trades_with_pnl else 0
            total_pnl = sum(trades_with_pnl)
            avg_pnl = total_pnl / len(trades_with_pnl)
        else:
            win_rate = 0
            total_pnl = 0
            avg_pnl = 0

        # Update labels
        self.total_trades_label.setText(f"Total Trades: {total_trades}")
        self.win_rate_label.setText(f"Win Rate: {win_rate:.1f}%")

        # Color code P&L labels
        total_color = "green" if total_pnl >= 0 else "red"
        avg_color = "green" if avg_pnl >= 0 else "red"

        total_sign = '+' if total_pnl >= 0 else ''
        avg_sign = '+' if avg_pnl >= 0 else ''

        self.total_pnl_label.setText(f"Total P&L: {total_sign}{total_pnl:.2f}%")
        self.total_pnl_label.setStyleSheet(f"QLabel {{ color: {total_color}; font-size: 10pt; padding: 2px; }}")

        self.avg_pnl_label.setText(f"Avg P&L: {avg_sign}{avg_pnl:.2f}%")
        self.avg_pnl_label.setStyleSheet(f"QLabel {{ color: {avg_color}; font-size: 10pt; padding: 2px; }}")

    def set_trades(self, trades: TradeCollection):
        """Set trades and update display"""
        super().set_trades(trades)
        self.update_summary_stats()


# Export the enhanced panel as the default
TradeListPanel = EnhancedTradeListPanel