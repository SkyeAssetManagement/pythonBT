#!/usr/bin/env python3
"""
Step 1: Minimal candlestick test - reproduce the working Screenshot 2025-08-04 105740.png
Goal: Get proper thin candlesticks with wicks, not fat ovals
"""

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

class SimpleCandlestickItem(pg.GraphicsObject):
    """Simple candlestick implementation that matches the working screenshot"""
    
    def __init__(self, ohlc_data):
        super().__init__()
        self.data = ohlc_data
        self.picture = None
        self._generate_picture()
    
    def _generate_picture(self):
        """Generate candlestick picture - SIMPLE approach like the working version"""
        
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        
        # CRITICAL: Use proper candlestick rendering like the working version
        n_bars = len(self.data)
        
        for i in range(n_bars):
            x = float(i)
            o, h, l, c = self.data[i]
            
            # Skip invalid data
            if not all(np.isfinite([o, h, l, c])) or any(v <= 0 for v in [o, h, l, c]):
                continue
            
            # WORKING PATTERN: Thin candlestick bodies like the target screenshot
            body_width = 0.6  # Fixed width for consistency
            
            # 1. Draw the wick first (thin line from low to high)
            wick_pen = QtGui.QPen(QtCore.Qt.black, 1)
            painter.setPen(wick_pen)
            painter.drawLine(
                QtCore.QPointF(x, l),  # From low
                QtCore.QPointF(x, h)   # To high
            )
            
            # 2. Draw the body (rectangle from open to close)
            body_rect = QtCore.QRectF(
                x - body_width/2,    # x (centered)
                min(o, c),          # y (bottom)
                body_width,         # width
                abs(c - o)          # height
            )
            
            # Color based on direction
            if c >= o:  # Up candle
                painter.setBrush(QtGui.QBrush(QtCore.Qt.white))  # White fill
                painter.setPen(QtGui.QPen(QtCore.Qt.green, 1))   # Green border
            else:       # Down candle
                painter.setBrush(QtGui.QBrush(QtCore.Qt.red))    # Red fill
                painter.setPen(QtGui.QPen(QtCore.Qt.red, 1))     # Red border
            
            painter.drawRect(body_rect)
        
        painter.end()
    
    def paint(self, painter, option, widget):
        """Paint the candlesticks"""
        if self.picture:
            painter.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        """Return bounding rectangle"""
        if len(self.data) == 0:
            return QtCore.QRectF()
        
        highs = [row[1] for row in self.data]
        lows = [row[2] for row in self.data]
        
        return QtCore.QRectF(
            0, min(lows),
            len(self.data), max(highs) - min(lows)
        )

def create_test_data(n_bars=50):
    """Create simple test OHLC data"""
    np.random.seed(42)  # Reproducible data
    
    # Start with a base price
    base_price = 100.0
    
    # Generate price movements
    returns = np.random.randn(n_bars) * 0.02  # 2% volatility
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    ohlc_data = []
    for i in range(n_bars):
        c = close_prices[i]
        
        # Generate realistic OHLC from close
        body_size = abs(np.random.randn()) * 0.5  # Body size
        wick_size = abs(np.random.randn()) * 0.3  # Wick size
        
        if np.random.rand() > 0.5:  # Up candle
            o = c - body_size
            h = c + wick_size
            l = o - wick_size
        else:                       # Down candle
            o = c + body_size
            h = o + wick_size
            l = c - wick_size
        
        # Ensure valid OHLC relationships
        h = max(h, max(o, c))
        l = min(l, min(o, c))
        
        ohlc_data.append([o, h, l, c])
    
    return ohlc_data

def test_step1():
    """Step 1: Test minimal candlestick rendering"""
    
    print("Step 1: Testing minimal candlestick rendering...")
    
    app = QtWidgets.QApplication(sys.argv if sys.argv else ['step1'])
    
    # Create plot widget
    plot = pg.PlotWidget()
    plot.setWindowTitle("Step 1: Minimal Candlestick Test")
    plot.resize(1000, 600)
    plot.setBackground('w')  # White background like target
    
    # Create test data
    print("Creating test OHLC data...")
    ohlc_data = create_test_data(50)  # Small dataset
    
    # Create candlestick item
    print("Creating candlestick item...")
    candle_item = SimpleCandlestickItem(ohlc_data)
    
    # Add to plot
    plot.addItem(candle_item)
    
    # Set up axes like the working version
    plot.setLabel('left', 'Price')
    plot.setLabel('bottom', 'Time')
    
    # Show
    print("Showing plot...")
    plot.show()
    
    # Process events
    for i in range(10):
        app.processEvents()
    
    print("Step 1 complete - check if candlesticks look like the target screenshot")
    print("Close window to continue...")
    
    app.exec_()

if __name__ == "__main__":
    test_step1()