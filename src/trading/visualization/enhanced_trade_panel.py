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
        """Calculate cumulative P&L percentages (cumulative sum of returns)"""
        self.cumulative_pnl_percent = []
        cumulative = 0.0

        for trade in self.trades:
            # Get P&L percentage (based on $1 invested)
            pnl_percent = self._get_pnl_percent(trade)
            if pnl_percent is not None:
                cumulative += pnl_percent
            self.cumulative_pnl_percent.append(cumulative)

    def _get_pnl_percent(self, trade: TradeData) -> Optional[float]:
        """Get P&L as percentage (already calculated based on $1 invested)"""
        # Check if trade has pnl_percent attribute (new format)
        if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
            return trade.pnl_percent

        # Otherwise try to calculate from dollar P&L
        # Note: Legacy pnl field represents points, not dollars
        if hasattr(trade, 'pnl') and trade.pnl is not None:
            # For compatibility - treat pnl as percentage if it's the old format
            return trade.pnl

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
                        sign = '+' if pnl_percent >= 0 else ''
                        return f"{sign}{pnl_percent:.2f}%"
                    else:
                        return "0.00%"
                return "-"

            elif column_attr == "cumulative_pnl_percent":
                if index.row() < len(self.cumulative_pnl_percent):
                    cum_pnl = self.cumulative_pnl_percent[index.row()]
                    if cum_pnl != 0:
                        sign = '+' if cum_pnl >= 0 else ''
                        return f"{sign}{cum_pnl:.2f}%"
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
        # Additional labels for commission and execution info
        self.commission_label = QtWidgets.QLabel("Commission: $0.00")
        self.execution_label = QtWidgets.QLabel("Avg Lag: 0 bars")

        # Style the labels for better visibility
        label_style = "QLabel { font-size: 11pt; padding: 5px; color: #ffffff; background-color: #333333; border-radius: 3px; }"
        for label in [self.total_trades_label, self.win_rate_label,
                     self.total_pnl_label, self.avg_pnl_label,
                     self.commission_label, self.execution_label]:
            label.setStyleSheet(label_style)

        # Add to grid layout with better spacing
        summary_layout.addWidget(self.total_trades_label, 0, 0)
        summary_layout.addWidget(self.win_rate_label, 0, 1)
        summary_layout.addWidget(self.total_pnl_label, 1, 0)
        summary_layout.addWidget(self.avg_pnl_label, 1, 1)
        summary_layout.addWidget(self.commission_label, 2, 0)
        summary_layout.addWidget(self.execution_label, 2, 1)

        # Set minimum height to ensure visibility (increased for extra row)
        summary_widget.setMinimumHeight(120)
        summary_widget.setMaximumHeight(150)

        # Insert summary panel at the bottom
        main_layout.addWidget(summary_widget)

        # Adjust layout stretch factors - ensure summary is visible
        # Set stretch for items above to allow them to expand
        for i in range(main_layout.count() - 1):
            main_layout.setStretch(i, 1)
        # Don't stretch summary panel
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
            self.total_pnl_label.setText("Total P&L: 0.00%")
            self.avg_pnl_label.setText("Avg P&L: 0.00%")
            self.commission_label.setText("Commission: $0.00")
            self.execution_label.setText("Avg Lag: 0 bars")
            return

        # Calculate statistics
        total_trades = len(self.trades)
        trades_with_pnl = []
        total_commission = 0.0
        lag_values = []

        for trade in self.trades:
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
            # Total P&L is cumulative percentage return
            total_pnl = sum(trades_with_pnl)
            avg_pnl = total_pnl / len(trades_with_pnl) if trades_with_pnl else 0
        else:
            win_rate = 0.00
            total_pnl = 0.00
            avg_pnl = 0.00

        # Calculate average lag
        avg_lag = sum(lag_values) / len(lag_values) if lag_values else 0

        # Update labels with proper formatting
        self.total_trades_label.setText(f"Total Trades: {total_trades}")
        self.win_rate_label.setText(f"Win Rate: {win_rate:.2f}%")

        # Color code P&L labels
        total_color = "green" if total_pnl >= 0 else "red"
        avg_color = "green" if avg_pnl >= 0 else "red"

        total_sign = '+' if total_pnl >= 0 else ''
        avg_sign = '+' if avg_pnl >= 0 else ''

        self.total_pnl_label.setText(f"Total P&L: {total_sign}{total_pnl:.2f}%")
        self.total_pnl_label.setStyleSheet(f"QLabel {{ color: {total_color}; font-size: 10pt; padding: 2px; }}")

        self.avg_pnl_label.setText(f"Avg P&L: {avg_sign}{avg_pnl:.2f}%")
        self.avg_pnl_label.setStyleSheet(f"QLabel {{ color: {avg_color}; font-size: 10pt; padding: 2px; }}")

        # Update commission and execution info
        self.commission_label.setText(f"Commission: ${total_commission:.2f}")
        self.execution_label.setText(f"Avg Lag: {avg_lag:.1f} bars")

    def load_trades(self, trades: TradeCollection):
        """Load trades and update display - override parent method"""
        super().load_trades(trades)
        self.update_summary_stats()

    def set_trades(self, trades: TradeCollection):
        """Set trades and update display - convenience method"""
        self.load_trades(trades)


# Export the enhanced panel as the default
TradeListPanel = EnhancedTradeListPanel