"""
Phased Entry Trade Marks - Enhanced chart overlays for phased entries
Shows different markers for each phase with connecting lines
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from typing import List, Dict, Tuple, Optional
import sys
import os

# Import existing components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_data import TradeData, TradeCollection
from simple_white_x_trades import SimpleWhiteXTrades


class PhasedTradeMarks:
    """Enhanced trade marks with phased entry visualization"""

    def __init__(self, plot_widget):
        self.plot_widget = plot_widget
        self.phase_markers = {}  # Store markers by phase
        self.connection_lines = []  # Lines connecting phases
        self.position_groups = {}  # Group trades by position

        # Phase marker styles
        self.phase_styles = {
            1: {'symbol': 't', 'size': 12, 'pen': pg.mkPen('green', width=2), 'brush': pg.mkBrush('lightgreen')},
            2: {'symbol': 'd', 'size': 10, 'pen': pg.mkPen('orange', width=2), 'brush': pg.mkBrush('lightyellow')},
            3: {'symbol': 's', 'size': 8, 'pen': pg.mkPen('red', width=2), 'brush': pg.mkBrush('lightcoral')},
            4: {'symbol': 'o', 'size': 6, 'pen': pg.mkPen('purple', width=2), 'brush': pg.mkBrush('plum')},
        }

        # Default style for additional phases
        self.default_style = {'symbol': '+', 'size': 6, 'pen': pg.mkPen('gray', width=2), 'brush': pg.mkBrush('lightgray')}

    def clear_all_marks(self):
        """Clear all trade marks from the chart"""
        # Clear phase markers
        for phase_items in self.phase_markers.values():
            for item in phase_items:
                self.plot_widget.removeItem(item)

        # Clear connection lines
        for line in self.connection_lines:
            self.plot_widget.removeItem(line)

        self.phase_markers.clear()
        self.connection_lines.clear()
        self.position_groups.clear()

    def update_trades(self, trades: TradeCollection, df=None):
        """Update trade marks with phased entry support"""
        self.clear_all_marks()

        if not trades or len(trades) == 0:
            return

        # Group trades by position
        self._group_trades_by_position(trades)

        # Draw markers for each trade
        for trade in trades:
            self._draw_trade_marker(trade, df)

        # Draw connection lines between phases
        self._draw_connection_lines()

    def _group_trades_by_position(self, trades: TradeCollection):
        """Group trades by position for line connections"""
        self.position_groups.clear()

        for trade in trades:
            # Determine position key
            position_key = self._get_position_key(trade)

            if position_key not in self.position_groups:
                self.position_groups[position_key] = []

            self.position_groups[position_key].append(trade)

        # Sort each group by phase number or bar index
        for position_key in self.position_groups:
            self.position_groups[position_key].sort(key=lambda t: (
                getattr(t, 'phase_number', 1),
                t.bar_index
            ))

    def _get_position_key(self, trade: TradeData) -> str:
        """Get position key for grouping trades"""
        if hasattr(trade, 'is_phased') and trade.is_phased:
            # For phased trades, group by base trade ID
            base_id = trade.trade_id // 100 if hasattr(trade, 'phase_number') and trade.phase_number > 1 else trade.trade_id
            return f"position_{base_id}"
        else:
            # For regular trades, each trade is its own position
            return f"trade_{trade.trade_id}"

    def _draw_trade_marker(self, trade: TradeData, df=None):
        """Draw a marker for a single trade"""
        if not hasattr(trade, 'bar_index') or trade.bar_index < 0:
            return

        # Get phase number
        phase_number = getattr(trade, 'phase_number', 1)

        # Get marker style
        style = self.phase_styles.get(phase_number, self.default_style)

        # Adjust marker size based on position size (if available)
        base_size = style['size']
        if hasattr(trade, 'size') and trade.size:
            size_multiplier = min(2.0, max(0.5, trade.size / 100))  # Scale between 0.5x and 2x
            marker_size = int(base_size * size_multiplier)
        else:
            marker_size = base_size

        # Get position for marker
        x_pos = trade.bar_index
        y_pos = trade.price

        # Create marker
        marker = pg.ScatterPlotItem(
            pos=[(x_pos, y_pos)],
            size=marker_size,
            symbol=style['symbol'],
            pen=style['pen'],
            brush=style['brush']
        )

        # Store marker for later cleanup
        if phase_number not in self.phase_markers:
            self.phase_markers[phase_number] = []
        self.phase_markers[phase_number].append(marker)

        # Add to plot
        self.plot_widget.addItem(marker)

        # Add tooltip with trade information
        self._add_trade_tooltip(marker, trade)

    def _add_trade_tooltip(self, marker, trade: TradeData):
        """Add tooltip information to marker"""
        tooltip_text = self._create_tooltip_text(trade)

        # Create custom tooltip (simplified implementation)
        # In a full implementation, you would override mouseMoveEvent
        marker.setToolTip(tooltip_text)

    def _create_tooltip_text(self, trade: TradeData) -> str:
        """Create tooltip text for a trade"""
        lines = []

        # Basic trade info
        lines.append(f"Trade #{trade.trade_id}")
        lines.append(f"Type: {trade.trade_type}")
        lines.append(f"Price: ${trade.price:.2f}")

        # Phase information
        if hasattr(trade, 'phase_number'):
            lines.append(f"Phase: {trade.phase_number}")

        if hasattr(trade, 'total_phases'):
            lines.append(f"Total Phases: {trade.total_phases}")

        # Size information
        if hasattr(trade, 'size'):
            lines.append(f"Size: {trade.size:.2f}")

        if hasattr(trade, 'total_position_size'):
            lines.append(f"Total Position: {trade.total_position_size:.2f}")

        # Average entry price for exits
        if hasattr(trade, 'average_entry_price') and trade.average_entry_price:
            lines.append(f"Avg Entry: ${trade.average_entry_price:.2f}")

        # P&L information
        if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
            lines.append(f"P&L: {trade.pnl_percent:.2f}%")

        # Timestamp
        if hasattr(trade, 'timestamp') and trade.timestamp:
            if hasattr(trade.timestamp, 'strftime'):
                lines.append(f"Time: {trade.timestamp.strftime('%H:%M:%S')}")

        return "\\n".join(lines)

    def _draw_connection_lines(self):
        """Draw lines connecting phases of the same position"""
        for position_key, trades in self.position_groups.items():
            if len(trades) < 2:
                continue  # No connections needed for single trades

            # Only connect entry trades (phases of the same position)
            entry_trades = []
            for trade in trades:
                if trade.trade_type in ['BUY', 'SHORT']:
                    entry_trades.append(trade)

            if len(entry_trades) < 2:
                continue

            # Create connection line
            self._draw_position_connection_line(entry_trades)

    def _draw_position_connection_line(self, trades: List[TradeData]):
        """Draw connection line for a single position's phases"""
        if len(trades) < 2:
            return

        # Prepare line data
        x_coords = []
        y_coords = []

        for trade in trades:
            x_coords.append(trade.bar_index)
            y_coords.append(trade.price)

        # Create line
        line = pg.PlotDataItem(
            x=x_coords,
            y=y_coords,
            pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine),
            connect='all'
        )

        self.connection_lines.append(line)
        self.plot_widget.addItem(line)

    def highlight_position(self, position_key: str):
        """Highlight all trades in a position"""
        if position_key not in self.position_groups:
            return

        trades = self.position_groups[position_key]
        for trade in trades:
            # Implementation would highlight the corresponding markers
            # This is a placeholder for the highlighting logic
            pass

    def get_marker_info_at_position(self, x: int, y: float, tolerance: float = 5.0) -> Optional[TradeData]:
        """Get trade information at chart position"""
        # Find the closest trade marker to the given position
        closest_trade = None
        min_distance = float('inf')

        for position_key, trades in self.position_groups.items():
            for trade in trades:
                # Calculate distance
                x_dist = abs(trade.bar_index - x)
                y_dist = abs(trade.price - y)
                distance = np.sqrt(x_dist**2 + (y_dist / y * 100)**2)  # Normalize y distance

                if distance < tolerance and distance < min_distance:
                    min_distance = distance
                    closest_trade = trade

        return closest_trade

    def get_phase_statistics(self) -> Dict:
        """Get statistics about phase distribution"""
        phase_counts = {}
        total_trades = 0

        for trades in self.position_groups.values():
            for trade in trades:
                phase = getattr(trade, 'phase_number', 1)
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                total_trades += 1

        return {
            'phase_counts': phase_counts,
            'total_trades': total_trades,
            'total_positions': len(self.position_groups),
            'avg_phases_per_position': total_trades / len(self.position_groups) if self.position_groups else 0
        }


class PhasedTradeOverlay:
    """Complete overlay system combining markers and additional visualizations"""

    def __init__(self, plot_widget):
        self.plot_widget = plot_widget
        self.trade_marks = PhasedTradeMarks(plot_widget)
        self.legend = None
        self.info_text = None

    def update_trades(self, trades: TradeCollection, df=None):
        """Update all trade visualizations"""
        self.trade_marks.update_trades(trades, df)
        self._update_legend()
        self._update_info_display(trades)

    def _update_legend(self):
        """Update the phase legend"""
        if self.legend:
            self.plot_widget.removeItem(self.legend)

        # Create legend for phase markers
        legend = pg.LegendItem(offset=(10, 10))
        legend.setParentItem(self.plot_widget.plotItem)

        for phase, style in self.trade_marks.phase_styles.items():
            # Create a sample marker for the legend
            sample = pg.ScatterPlotItem(
                pos=[(0, 0)],
                size=style['size'],
                symbol=style['symbol'],
                pen=style['pen'],
                brush=style['brush']
            )
            legend.addItem(sample, f'Phase {phase}')

        self.legend = legend

    def _update_info_display(self, trades: TradeCollection):
        """Update information display"""
        if self.info_text:
            self.plot_widget.removeItem(self.info_text)

        # Get phase statistics
        stats = self.trade_marks.get_phase_statistics()

        # Create info text
        info_lines = [
            f"Total Positions: {stats['total_positions']}",
            f"Total Trades: {stats['total_trades']}",
            f"Avg Phases: {stats['avg_phases_per_position']:.1f}"
        ]

        info_text = "\\n".join(info_lines)

        # Add text to plot (positioned at top-right)
        text_item = pg.TextItem(
            text=info_text,
            anchor=(1, 0),
            color='white',
            fill=pg.mkBrush(0, 0, 0, 100)
        )

        # Position in top-right corner
        view_box = self.plot_widget.plotItem.viewBox
        text_item.setParentItem(self.plot_widget.plotItem)

        self.info_text = text_item

    def clear_all(self):
        """Clear all overlays"""
        self.trade_marks.clear_all_marks()

        if self.legend:
            self.plot_widget.removeItem(self.legend)
            self.legend = None

        if self.info_text:
            self.plot_widget.removeItem(self.info_text)
            self.info_text = None