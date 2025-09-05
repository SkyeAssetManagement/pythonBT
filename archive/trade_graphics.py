#!/usr/bin/env python3
"""
Trade Graphics - PyQtGraph graphics items for trade visualization
================================================================

High-performance trade arrow and price dot visualization with LOD rendering.
Optimized for thousands of trades with smooth zoom/pan performance.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from typing import List, Dict, Optional, Tuple
import logging

from trade_data import TradeData, TradeCollection

logger = logging.getLogger(__name__)

class TradeArrowItem(pg.GraphicsObject):
    """
    Efficient trade arrow rendering with LOD (Level of Detail)
    
    Arrow types:
    - BUY: Filled green up arrow below bar
    - COVER: Hollow green up arrow below bar  
    - SHORT: Filled red down arrow above bar
    - SELL: Hollow red down arrow above bar
    """
    
    def __init__(self, trades: TradeCollection, bar_data: Dict[str, np.ndarray]):
        """
        Initialize trade arrows
        
        Args:
            trades: TradeCollection with trade data
            bar_data: Dictionary with 'high', 'low' arrays for positioning arrows
        """
        pg.GraphicsObject.__init__(self)
        
        self.trades = trades
        self.bar_data = bar_data
        self.lod_threshold = 2000  # Show all arrows below this many visible bars
        
        # Pre-calculate arrow data for performance
        self.arrow_data = self._prepare_arrow_data()
        
        # Generate rendering cache
        self.generatePicture()
        
        logger.debug(f"Created TradeArrowItem with {len(trades)} trades")
    
    def _prepare_arrow_data(self) -> Dict[str, List]:
        """Pre-calculate arrow positions and properties"""
        arrow_data = {
            'positions': [],
            'types': [],
            'colors': [],
            'sizes': []
        }
        
        # Color scheme
        colors = {
            'BUY': QtGui.QColor(0, 255, 136, 200),      # Bright green, filled
            'COVER': QtGui.QColor(0, 255, 136, 120),    # Bright green, hollow  
            'SHORT': QtGui.QColor(255, 68, 68, 200),    # Bright red, filled
            'SELL': QtGui.QColor(255, 68, 68, 120)      # Bright red, hollow
        }
        
        for trade in self.trades:
            bar_idx = trade.bar_index
            
            # Skip trades without bar data
            if (bar_idx < 0 or 
                bar_idx >= len(self.bar_data['high']) or
                bar_idx >= len(self.bar_data['low'])):
                continue
            
            # Get bar high/low for positioning
            bar_high = self.bar_data['high'][bar_idx]
            bar_low = self.bar_data['low'][bar_idx]
            bar_range = bar_high - bar_low
            
            # Position arrows relative to bar
            if trade.trade_type in ['BUY', 'COVER']:
                # Below bar for long entries/exits
                arrow_y = bar_low - (bar_range * 0.15)
                arrow_type = 'up'
            else:  # SHORT, SELL
                # Above bar for short entries/exits  
                arrow_y = bar_high + (bar_range * 0.15)
                arrow_type = 'down'
            
            arrow_data['positions'].append((bar_idx, arrow_y))
            arrow_data['types'].append(arrow_type)
            arrow_data['colors'].append(colors[trade.trade_type])
            
            # Size based on trade size (with reasonable bounds)
            size = min(max(trade.size * 2, 8), 20)
            arrow_data['sizes'].append(size)
        
        return arrow_data
    
    def generatePicture(self):
        """Generate QPicture for efficient rendering"""
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        
        # Draw all arrows
        for i, (pos, arrow_type, color, size) in enumerate(zip(
            self.arrow_data['positions'],
            self.arrow_data['types'], 
            self.arrow_data['colors'],
            self.arrow_data['sizes']
        )):
            x, y = pos
            self._draw_arrow(painter, x, y, arrow_type, color, size, i)
        
        painter.end()
        
        logger.debug(f"Generated picture with {len(self.arrow_data['positions'])} arrows")
    
    def _draw_arrow(self, painter: QtGui.QPainter, x: float, y: float, 
                   arrow_type: str, color: QtGui.QColor, size: float, trade_idx: int):
        """Draw single arrow"""
        # Get corresponding trade for fill determination
        trade = self.trades[trade_idx]
        is_filled = trade.trade_type in ['BUY', 'SHORT']  # Entry trades are filled
        
        # Set pen and brush
        pen = QtGui.QPen(color, 1.5)
        brush = QtGui.QBrush(color) if is_filled else QtGui.QBrush()
        
        painter.setPen(pen)
        painter.setBrush(brush)
        
        # Create arrow polygon
        if arrow_type == 'up':
            # Up arrow (triangle pointing up)
            points = [
                QtCore.QPointF(x, y - size/2),           # Top point
                QtCore.QPointF(x - size/3, y + size/2),  # Bottom left
                QtCore.QPointF(x + size/3, y + size/2)   # Bottom right
            ]
        else:
            # Down arrow (triangle pointing down)  
            points = [
                QtCore.QPointF(x, y + size/2),           # Bottom point
                QtCore.QPointF(x - size/3, y - size/2),  # Top left
                QtCore.QPointF(x + size/3, y - size/2)   # Top right
            ]
        
        # Draw arrow
        polygon = QtGui.QPolygonF(points)
        painter.drawPolygon(polygon)
    
    def paint(self, painter, option, widget):
        """Paint the arrows"""
        # Check if we need LOD rendering
        view_rect = option.exposedRect
        if view_rect.width() > self.lod_threshold:
            # LOD: Skip some arrows when zoomed out
            self._paint_lod(painter, option)
        else:
            # Full detail rendering
            painter.drawPicture(0, 0, self.picture)
    
    def _paint_lod(self, painter, option):
        """Level of Detail rendering for zoomed out views"""
        # For LOD, we could:
        # 1. Skip every nth arrow
        # 2. Draw simplified markers
        # 3. Group nearby arrows
        
        # For now, just draw every 3rd arrow when heavily zoomed out
        painter.setOpacity(0.7)  # Make slightly transparent
        
        # Draw subset of arrows
        for i in range(0, len(self.arrow_data['positions']), 3):
            pos = self.arrow_data['positions'][i]
            arrow_type = self.arrow_data['types'][i]  
            color = self.arrow_data['colors'][i]
            size = max(self.arrow_data['sizes'][i] * 0.7, 6)  # Smaller arrows
            
            x, y = pos
            self._draw_arrow(painter, x, y, arrow_type, color, size, i)
        
        painter.setOpacity(1.0)
    
    def boundingRect(self):
        """Return bounding rectangle"""
        if not self.arrow_data['positions']:
            return QtCore.QRectF()
        
        # Calculate bounds from arrow positions
        x_coords = [pos[0] for pos in self.arrow_data['positions']]
        y_coords = [pos[1] for pos in self.arrow_data['positions']]
        
        if not x_coords or not y_coords:
            return QtCore.QRectF()
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding for arrow sizes
        max_size = max(self.arrow_data['sizes']) if self.arrow_data['sizes'] else 20
        padding = max_size
        
        return QtCore.QRectF(
            x_min - padding, y_min - padding,
            (x_max - x_min) + 2 * padding, (y_max - y_min) + 2 * padding
        )

class TradePriceDotsItem(pg.GraphicsObject):
    """
    White dots at exact execution prices on candlesticks
    
    Shows precise execution price with hover detection for tooltips.
    """
    
    def __init__(self, trades: TradeCollection):
        """
        Initialize price dots
        
        Args:
            trades: TradeCollection with trade data
        """
        pg.GraphicsObject.__init__(self)
        
        self.trades = trades
        self.dot_data = self._prepare_dot_data()
        self.generatePicture()
        
        logger.debug(f"Created TradePriceDotsItem with {len(trades)} dots")
    
    def _prepare_dot_data(self) -> List[Tuple[float, float, TradeData]]:
        """Prepare dot position data"""
        dot_data = []
        
        for trade in self.trades:
            x = float(trade.bar_index)
            y = float(trade.price)
            dot_data.append((x, y, trade))
        
        return dot_data
    
    def generatePicture(self):
        """Generate QPicture for efficient rendering"""
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        
        # White dots with black outline for visibility
        pen = QtGui.QPen(QtCore.Qt.black, 1)
        brush = QtGui.QBrush(QtCore.Qt.white)
        
        painter.setPen(pen)
        painter.setBrush(brush)
        
        # Draw all dots
        dot_radius = 2.5
        for x, y, trade in self.dot_data:
            painter.drawEllipse(
                QtCore.QPointF(x, y), 
                dot_radius, dot_radius
            )
        
        painter.end()
    
    def paint(self, painter, option, widget):
        """Paint the price dots"""
        painter.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        """Return bounding rectangle"""
        if not self.dot_data:
            return QtCore.QRectF()
        
        x_coords = [x for x, y, trade in self.dot_data]
        y_coords = [y for x, y, trade in self.dot_data]
        
        if not x_coords or not y_coords:
            return QtCore.QRectF()
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding for dot size
        padding = 5
        
        return QtCore.QRectF(
            x_min - padding, y_min - padding,
            (x_max - x_min) + 2 * padding, (y_max - y_min) + 2 * padding
        )
    
    def get_trade_at_point(self, x: float, y: float, tolerance: float = 5.0) -> Optional[TradeData]:
        """
        Get trade at specific point for hover detection
        
        Args:
            x, y: Point coordinates
            tolerance: Search tolerance in pixels
            
        Returns:
            TradeData if found within tolerance, None otherwise
        """
        for dot_x, dot_y, trade in self.dot_data:
            if (abs(dot_x - x) <= tolerance and 
                abs(dot_y - y) <= tolerance):
                return trade
        return None

def create_test_graphics():
    """Create test trade graphics for development"""
    from trade_data import create_sample_trades
    
    # Create sample data
    trades = create_sample_trades(50, 0, 200)
    
    # Create fake bar data  
    n_bars = 200
    bar_data = {
        'high': 4000 + np.random.randn(n_bars) * 10 + np.arange(n_bars) * 0.1,
        'low': 3990 + np.random.randn(n_bars) * 10 + np.arange(n_bars) * 0.1
    }
    
    # Ensure high > low
    for i in range(n_bars):
        if bar_data['high'][i] <= bar_data['low'][i]:
            bar_data['high'][i] = bar_data['low'][i] + 5
    
    # Create graphics items
    arrows = TradeArrowItem(trades, bar_data)
    dots = TradePriceDotsItem(trades)
    
    print(f"Created test graphics:")
    print(f"  - {len(trades)} trade arrows")  
    print(f"  - {len(trades)} price dots")
    print(f"  - Arrow bounds: {arrows.boundingRect()}")
    print(f"  - Dots bounds: {dots.boundingRect()}")
    
    return arrows, dots, trades, bar_data

if __name__ == "__main__":
    # Test the graphics items
    print("Testing Trade Graphics...")
    
    try:
        arrows, dots, trades, bar_data = create_test_graphics()
        
        # Test trade lookup
        first_trade = trades[0]
        found_trade = dots.get_trade_at_point(
            first_trade.bar_index, 
            first_trade.price, 
            tolerance=1.0
        )
        
        if found_trade:
            print(f"[OK] Trade lookup working: Found trade at bar {found_trade.bar_index}")
        else:
            print("[FAIL] Trade lookup failed")
        
        # Test statistics  
        stats = trades.get_statistics()
        print(f"Trade statistics: {stats['trade_types']}")
        
        print("Trade graphics test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()