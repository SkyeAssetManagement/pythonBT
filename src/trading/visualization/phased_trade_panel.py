"""
Phased Entry Trade Panel - Enhanced trade visualization with phased entry support
Shows individual phases, average entry prices, and phase-specific metrics
"""

import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Optional, Dict, List, Tuple
import logging
import sys
import os

# Import existing components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from enhanced_trade_panel import EnhancedTradeTableModel, EnhancedTradePanel
from trade_data import TradeData, TradeCollection

logger = logging.getLogger(__name__)


class PhasedTradeTableModel(EnhancedTradeTableModel):
    """Enhanced table model with support for phased entry display"""

    COLUMNS = [
        ("Trade #", "trade_id"),
        ("Phase", "phase_number"),
        ("DateTime", "timestamp"),
        ("Type", "trade_type"),
        ("Price", "price"),
        ("Size", "size"),
        ("P&L %", "pnl_percent"),
        ("Cum P&L %", "cumulative_pnl_percent"),
        ("Avg Entry", "average_entry_price"),
        ("Total Size", "total_position_size"),
        ("Bar #", "bar_index")
    ]

    def __init__(self):
        super().__init__()
        self.show_phases = True
        self.group_by_position = True

    def set_phase_display(self, show_phases: bool):
        """Toggle between showing individual phases vs grouped positions"""
        if self.show_phases != show_phases:
            self.show_phases = show_phases
            self.beginResetModel()
            self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        """Return number of rows based on display mode"""
        if not self.trades:
            return 0

        if self.show_phases:
            # Show all trades including individual phases
            return len(self.trades)
        else:
            # Show only complete positions (group phases)
            return self._count_positions()

    def _count_positions(self) -> int:
        """Count unique positions (group phases together)"""
        if not self.trades:
            return 0

        positions = set()
        for trade in self.trades:
            # Group by trade_id or a combination that identifies the position
            position_key = getattr(trade, 'position_id', trade.trade_id)
            if hasattr(trade, 'is_phased') and trade.is_phased and hasattr(trade, 'phase_number'):
                # For phased trades, use base trade ID
                base_id = trade.trade_id // 100 if trade.phase_number > 1 else trade.trade_id
                position_key = base_id
            positions.add(position_key)

        return len(positions)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        """Return data for display with phased entry support"""
        if not index.isValid() or not self.trades:
            return None

        row = index.row()
        col = index.column()

        if row >= len(self.trades):
            return None

        trade = self.trades[row]
        column_attr = self.COLUMNS[col][1]

        if role == QtCore.Qt.DisplayRole:
            return self._format_display_value(trade, column_attr, row)
        elif role == QtCore.Qt.BackgroundRole:
            return self._get_background_color(trade, column_attr)
        elif role == QtCore.Qt.ForegroundRole:
            return self._get_foreground_color(trade, column_attr)
        elif role == QtCore.Qt.TextAlignmentRole:
            return self._get_alignment(column_attr)

        return None

    def _format_display_value(self, trade: TradeData, column_attr: str, row: int):
        """Format display values with phased entry support"""
        if column_attr == "phase_number":
            if hasattr(trade, 'phase_number'):
                return f"{trade.phase_number}"
            else:
                return "1"  # Single entry

        elif column_attr == "average_entry_price":
            if hasattr(trade, 'average_entry_price') and trade.average_entry_price:
                return f"{trade.average_entry_price:.2f}"
            else:
                return f"{trade.price:.2f}"  # Use trade price as fallback

        elif column_attr == "total_position_size":
            if hasattr(trade, 'total_position_size') and trade.total_position_size:
                return f"{trade.total_position_size:.2f}"
            else:
                return f"{getattr(trade, 'size', 0):.2f}"  # Use trade size as fallback

        elif column_attr == "pnl_percent":
            pnl = self._get_pnl_percent(trade)
            if pnl is not None:
                return f"{pnl:.2f}%"
            else:
                return "-"

        elif column_attr == "cumulative_pnl_percent":
            if row < len(self.cumulative_pnl_percent):
                return f"{self.cumulative_pnl_percent[row]:.2f}%"
            else:
                return "-"

        elif column_attr == "timestamp":
            if hasattr(trade, 'timestamp') and trade.timestamp:
                if hasattr(trade.timestamp, 'strftime'):
                    return trade.timestamp.strftime('%H:%M:%S')
                else:
                    return str(trade.timestamp)
            else:
                return "-"

        elif column_attr == "trade_type":
            trade_type = getattr(trade, column_attr, "UNKNOWN")
            # Add phase indicator for phased trades
            if hasattr(trade, 'is_phased') and trade.is_phased:
                phase_num = getattr(trade, 'phase_number', 1)
                return f"{trade_type} (P{phase_num})"
            return trade_type

        elif column_attr == "price":
            price = getattr(trade, column_attr, 0)
            return f"{price:.2f}"

        elif column_attr == "size":
            size = getattr(trade, column_attr, 0)
            return f"{size:.2f}"

        else:
            # Standard attribute access
            value = getattr(trade, column_attr, "-")
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)

    def _get_background_color(self, trade: TradeData, column_attr: str) -> Optional[QtGui.QColor]:
        """Get background color for phased entry visualization"""
        if column_attr == "phase_number" and hasattr(trade, 'phase_number'):
            # Color-code phases
            phase = trade.phase_number
            if phase == 1:
                return QtGui.QColor(220, 255, 220)  # Light green for first phase
            elif phase == 2:
                return QtGui.QColor(255, 255, 200)  # Light yellow for second phase
            elif phase == 3:
                return QtGui.QColor(255, 220, 220)  # Light red for third phase
            else:
                return QtGui.QColor(240, 240, 240)  # Light gray for additional phases

        elif column_attr == "pnl_percent":
            pnl = self._get_pnl_percent(trade)
            if pnl is not None:
                if pnl > 0:
                    return QtGui.QColor(200, 255, 200)  # Light green for profit
                elif pnl < 0:
                    return QtGui.QColor(255, 200, 200)  # Light red for loss

        return None

    def _get_foreground_color(self, trade: TradeData, column_attr: str) -> Optional[QtGui.QColor]:
        """Get foreground color for better readability"""
        if column_attr == "trade_type":
            trade_type = getattr(trade, column_attr, "")
            if trade_type in ["BUY", "SHORT"]:
                return QtGui.QColor(0, 120, 0)  # Dark green for entries
            elif trade_type in ["SELL", "COVER"]:
                return QtGui.QColor(180, 0, 0)  # Dark red for exits

        elif column_attr == "pnl_percent":
            pnl = self._get_pnl_percent(trade)
            if pnl is not None:
                if pnl > 0:
                    return QtGui.QColor(0, 120, 0)  # Dark green for profit
                elif pnl < 0:
                    return QtGui.QColor(180, 0, 0)  # Dark red for loss

        return None


class PhasedTradeStatsWidget(QtWidgets.QWidget):
    """Statistics widget with phased entry metrics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the statistics UI"""
        layout = QtWidgets.QVBoxLayout(self)

        # Title
        title = QtWidgets.QLabel("Phased Entry Statistics")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        layout.addWidget(title)

        # Stats grid
        stats_layout = QtWidgets.QGridLayout()

        # Standard stats
        self.total_trades_label = QtWidgets.QLabel("Total Trades: -")
        self.phased_trades_label = QtWidgets.QLabel("Phased Trades: -")
        self.avg_phases_label = QtWidgets.QLabel("Avg Phases: -")
        self.completion_rate_label = QtWidgets.QLabel("Completion Rate: -")

        # P&L stats
        self.total_pnl_label = QtWidgets.QLabel("Total P&L: -")
        self.win_rate_label = QtWidgets.QLabel("Win Rate: -")
        self.avg_win_label = QtWidgets.QLabel("Avg Win: -")
        self.avg_loss_label = QtWidgets.QLabel("Avg Loss: -")

        # Phase-specific stats
        self.phase_pnl_label = QtWidgets.QLabel("Phase P&L Breakdown:")
        self.phase_breakdown = QtWidgets.QLabel("-")

        # Add to grid
        stats_layout.addWidget(self.total_trades_label, 0, 0)
        stats_layout.addWidget(self.phased_trades_label, 0, 1)
        stats_layout.addWidget(self.avg_phases_label, 1, 0)
        stats_layout.addWidget(self.completion_rate_label, 1, 1)
        stats_layout.addWidget(self.total_pnl_label, 2, 0)
        stats_layout.addWidget(self.win_rate_label, 2, 1)
        stats_layout.addWidget(self.avg_win_label, 3, 0)
        stats_layout.addWidget(self.avg_loss_label, 3, 1)

        layout.addLayout(stats_layout)

        # Phase breakdown
        layout.addWidget(self.phase_pnl_label)
        layout.addWidget(self.phase_breakdown)

        layout.addStretch()

    def update_stats(self, trades: TradeCollection):
        """Update statistics display"""
        if not trades or len(trades) == 0:
            self._clear_stats()
            return

        # Basic stats
        total_trades = len(trades)
        phased_trades = len([t for t in trades if hasattr(t, 'is_phased') and t.is_phased])

        # Calculate phased statistics
        phase_stats = self._calculate_phase_stats(trades)

        # Update labels
        self.total_trades_label.setText(f"Total Trades: {total_trades}")
        self.phased_trades_label.setText(f"Phased Trades: {phased_trades}")
        self.avg_phases_label.setText(f"Avg Phases: {phase_stats['avg_phases']:.1f}")
        self.completion_rate_label.setText(f"Completion Rate: {phase_stats['completion_rate']:.1f}%")

        # P&L stats
        pnl_stats = self._calculate_pnl_stats(trades)
        self.total_pnl_label.setText(f"Total P&L: {pnl_stats['total']:.2f}%")
        self.win_rate_label.setText(f"Win Rate: {pnl_stats['win_rate']:.1f}%")
        self.avg_win_label.setText(f"Avg Win: {pnl_stats['avg_win']:.2f}%")
        self.avg_loss_label.setText(f"Avg Loss: {pnl_stats['avg_loss']:.2f}%")

        # Phase breakdown
        breakdown_text = self._format_phase_breakdown(phase_stats['phase_pnl'])
        self.phase_breakdown.setText(breakdown_text)

    def _calculate_phase_stats(self, trades: TradeCollection) -> Dict:
        """Calculate phase-specific statistics"""
        phase_counts = {}
        phase_pnl = {}
        total_positions = 0
        completed_positions = 0

        for trade in trades:
            if hasattr(trade, 'phase_number'):
                phase = trade.phase_number
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

                # Track P&L by phase
                pnl = self._get_trade_pnl(trade)
                if pnl is not None:
                    if phase not in phase_pnl:
                        phase_pnl[phase] = []
                    phase_pnl[phase].append(pnl)

        # Calculate averages
        avg_phases = sum(phase_counts.values()) / len(phase_counts) if phase_counts else 0
        completion_rate = 100.0  # Simplified for now

        return {
            'avg_phases': avg_phases,
            'completion_rate': completion_rate,
            'phase_counts': phase_counts,
            'phase_pnl': phase_pnl
        }

    def _calculate_pnl_stats(self, trades: TradeCollection) -> Dict:
        """Calculate P&L statistics"""
        pnls = []
        for trade in trades:
            pnl = self._get_trade_pnl(trade)
            if pnl is not None:
                pnls.append(pnl)

        if not pnls:
            return {
                'total': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            'total': sum(pnls),
            'win_rate': len(wins) / len(pnls) * 100 if pnls else 0,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0
        }

    def _get_trade_pnl(self, trade: TradeData) -> Optional[float]:
        """Get P&L from trade"""
        if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
            return trade.pnl_percent
        elif hasattr(trade, 'pnl') and trade.pnl is not None:
            return trade.pnl
        return None

    def _format_phase_breakdown(self, phase_pnl: Dict) -> str:
        """Format phase P&L breakdown"""
        if not phase_pnl:
            return "-"

        lines = []
        for phase in sorted(phase_pnl.keys()):
            pnls = phase_pnl[phase]
            if pnls:
                avg_pnl = sum(pnls) / len(pnls)
                count = len(pnls)
                lines.append(f"Phase {phase}: {avg_pnl:.2f}% ({count} trades)")

        return "\n".join(lines) if lines else "-"

    def _clear_stats(self):
        """Clear all statistics"""
        self.total_trades_label.setText("Total Trades: -")
        self.phased_trades_label.setText("Phased Trades: -")
        self.avg_phases_label.setText("Avg Phases: -")
        self.completion_rate_label.setText("Completion Rate: -")
        self.total_pnl_label.setText("Total P&L: -")
        self.win_rate_label.setText("Win Rate: -")
        self.avg_win_label.setText("Avg Win: -")
        self.avg_loss_label.setText("Avg Loss: -")
        self.phase_breakdown.setText("-")


class PhasedTradePanel(EnhancedTradePanel):
    """Enhanced trade panel with phased entry support"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_phased_ui()

    def setup_phased_ui(self):
        """Add phased entry specific UI elements"""
        # Replace the table model
        self.table_model = PhasedTradeTableModel()
        self.table_view.setModel(self.table_model)

        # Add phase display toggle
        self.phase_toggle = QtWidgets.QCheckBox("Show Individual Phases")
        self.phase_toggle.setChecked(True)
        self.phase_toggle.stateChanged.connect(self.on_phase_display_changed)

        # Insert toggle before the table
        layout = self.layout()
        table_index = layout.indexOf(self.table_view)
        layout.insertWidget(table_index, self.phase_toggle)

        # Replace stats widget with phased version
        if hasattr(self, 'stats_widget'):
            self.stats_widget.setParent(None)

        self.stats_widget = PhasedTradeStatsWidget()
        layout.addWidget(self.stats_widget)

        # Adjust column widths for new columns
        self.table_view.setColumnWidth(1, 50)   # Phase column
        self.table_view.setColumnWidth(8, 80)   # Avg Entry column
        self.table_view.setColumnWidth(9, 80)   # Total Size column

    def on_phase_display_changed(self, state):
        """Handle phase display toggle"""
        show_phases = state == QtCore.Qt.Checked
        self.table_model.set_phase_display(show_phases)

    def update_trades(self, trades: TradeCollection):
        """Update trades with phased entry support"""
        super().update_trades(trades)

        # Update phased statistics
        if hasattr(self, 'stats_widget'):
            self.stats_widget.update_stats(trades)

    def get_selected_trades(self) -> List[TradeData]:
        """Get selected trades (considering phase grouping)"""
        selection = self.table_view.selectionModel().selectedRows()
        selected_trades = []

        for index in selection:
            row = index.row()
            if 0 <= row < len(self.table_model.trades):
                selected_trades.append(self.table_model.trades[row])

        return selected_trades