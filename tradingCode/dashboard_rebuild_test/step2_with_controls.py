#!/usr/bin/env python3
"""
Step 2: Add keyboard controls for testing pan/zoom without mouse
Goal: Test that candlesticks render properly during pan/zoom operations

Keyboard Controls:
- Left Arrow: Pan left
- Right Arrow: Pan right  
- Up Arrow: Zoom in
- Down Arrow: Zoom out
- R: Reset view
- Q: Quit
"""

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

class SimpleCandlestickItem(pg.GraphicsObject):
    """Simple candlestick implementation with debugging"""
    
    def __init__(self, ohlc_data):
        super().__init__()
        self.data = ohlc_data
        self.picture = None
        self._generate_picture()
    
    def _generate_picture(self):
        """Generate candlestick picture - SIMPLE approach"""
        
        print(f"Generating candlesticks for {len(self.data)} bars...")
        
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        
        n_bars = len(self.data)
        valid_candles = 0
        
        for i in range(n_bars):
            x = float(i)
            o, h, l, c = self.data[i]
            
            # Skip invalid data
            if not all(np.isfinite([o, h, l, c])) or any(v <= 0 for v in [o, h, l, c]):
                continue
            
            valid_candles += 1
            
            # Fixed width for consistency
            body_width = 0.6
            
            # 1. Draw the wick (thin line from low to high)
            wick_pen = QtGui.QPen(QtCore.Qt.black, 1)
            painter.setPen(wick_pen)
            painter.drawLine(
                QtCore.QPointF(x, l),
                QtCore.QPointF(x, h)
            )
            
            # 2. Draw the body
            body_height = abs(c - o)
            if body_height < 0.01:  # Minimum height for doji
                body_height = 0.01
            
            body_rect = QtCore.QRectF(
                x - body_width/2,
                min(o, c),
                body_width,
                body_height
            )
            
            # Color based on direction
            if c >= o:  # Up candle
                painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
                painter.setPen(QtGui.QPen(QtCore.Qt.green, 1))
            else:       # Down candle
                painter.setBrush(QtGui.QBrush(QtCore.Qt.red))
                painter.setPen(QtGui.QPen(QtCore.Qt.red, 1))
            
            painter.drawRect(body_rect)
        
        painter.end()
        print(f"Generated {valid_candles} valid candlesticks")
    
    def paint(self, painter, option, widget):
        """Paint the candlesticks"""
        if self.picture:
            painter.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        """Return bounding rectangle"""
        if len(self.data) == 0:
            return QtCore.QRectF()
        
        highs = [row[1] for row in self.data if all(np.isfinite(row)) and all(v > 0 for v in row)]
        lows = [row[2] for row in self.data if all(np.isfinite(row)) and all(v > 0 for v in row)]
        
        if not highs or not lows:
            return QtCore.QRectF()
        
        return QtCore.QRectF(
            0, min(lows),
            len(self.data), max(highs) - min(lows)
        )

class ControlledPlotWidget(pg.PlotWidget):
    """Plot widget with keyboard controls"""
    
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)  # Enable keyboard focus
        
        # Pan/zoom parameters
        self.pan_step = 5  # bars
        self.zoom_factor = 1.2
        
        print("Keyboard controls:")
        print("- Left Arrow: Pan left")
        print("- Right Arrow: Pan right")
        print("- Up Arrow: Zoom in")
        print("- Down Arrow: Zoom out")
        print("- R: Reset view")
        print("- Q: Quit")
    
    def keyPressEvent(self, event):
        """Handle keyboard controls"""
        key = event.key()
        
        if key == QtCore.Qt.Key_Left:
            self.pan_left()
        elif key == QtCore.Qt.Key_Right:
            self.pan_right()
        elif key == QtCore.Qt.Key_Up:
            self.zoom_in()
        elif key == QtCore.Qt.Key_Down:
            self.zoom_out()
        elif key == QtCore.Qt.Key_R:
            self.reset_view()
        elif key == QtCore.Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)
    
    def pan_left(self):
        """Pan left"""
        x_range = self.viewRange()[0]
        span = x_range[1] - x_range[0]
        new_range = [x_range[0] - self.pan_step, x_range[1] - self.pan_step]
        self.setXRange(*new_range, padding=0)
        print(f"Panned left to {new_range}")
    
    def pan_right(self):
        """Pan right"""
        x_range = self.viewRange()[0]
        span = x_range[1] - x_range[0]
        new_range = [x_range[0] + self.pan_step, x_range[1] + self.pan_step]
        self.setXRange(*new_range, padding=0)
        print(f"Panned right to {new_range}")
    
    def zoom_in(self):
        """Zoom in"""
        x_range = self.viewRange()[0]
        center = (x_range[0] + x_range[1]) / 2
        span = x_range[1] - x_range[0]
        new_span = span / self.zoom_factor
        new_range = [center - new_span/2, center + new_span/2]
        self.setXRange(*new_range, padding=0)
        print(f"Zoomed in to {new_range}")
    
    def zoom_out(self):
        """Zoom out"""
        x_range = self.viewRange()[0]
        center = (x_range[0] + x_range[1]) / 2
        span = x_range[1] - x_range[0]
        new_span = span * self.zoom_factor
        new_range = [center - new_span/2, center + new_span/2]
        self.setXRange(*new_range, padding=0)
        print(f"Zoomed out to {new_range}")
    
    def reset_view(self):
        """Reset to full view"""
        self.autoRange()
        print("Reset view to full range")

def create_test_data(n_bars=100):
    """Create test OHLC data that looks like real financial data"""
    np.random.seed(42)
    
    # Start with a base price around 3000 (like ES futures)
    base_price = 3000.0
    
    # Generate realistic price movements
    returns = np.random.randn(n_bars) * 0.01  # 1% daily volatility
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    ohlc_data = []
    for i in range(n_bars):
        c = close_prices[i]
        
        # Generate realistic intraday range
        daily_range = c * 0.02 * abs(np.random.randn())  # 2% daily range
        
        # Create OHLC
        if np.random.rand() > 0.5:  # Up day
            o = c - abs(np.random.randn()) * daily_range * 0.5
            h = max(o, c) + abs(np.random.randn()) * daily_range * 0.3
            l = min(o, c) - abs(np.random.randn()) * daily_range * 0.2
        else:  # Down day
            o = c + abs(np.random.randn()) * daily_range * 0.5
            h = max(o, c) + abs(np.random.randn()) * daily_range * 0.2
            l = min(o, c) - abs(np.random.randn()) * daily_range * 0.3
        
        # Ensure valid OHLC
        h = max(h, o, c)
        l = min(l, o, c)
        
        ohlc_data.append([o, h, l, c])
    
    return ohlc_data

def test_step2():
    """Step 2: Test candlesticks with keyboard controls"""
    
    print("Step 2: Testing candlesticks with keyboard controls...")
    
    app = QtWidgets.QApplication(sys.argv if sys.argv else ['step2'])
    
    # Create controlled plot widget
    plot = ControlledPlotWidget()
    plot.setWindowTitle("Step 2: Candlesticks with Keyboard Controls")
    plot.resize(1200, 700)
    plot.setBackground('w')
    
    # Create test data
    print("Creating test OHLC data...")
    ohlc_data = create_test_data(100)
    
    # Create candlestick item
    print("Creating candlestick item...")
    candle_item = SimpleCandlestickItem(ohlc_data)
    
    # Add to plot
    plot.addItem(candle_item)
    
    # Set up axes
    plot.setLabel('left', 'Price ($)')
    plot.setLabel('bottom', 'Bar Index')
    
    # Set initial view
    plot.setXRange(0, 50, padding=0)  # Show first 50 bars
    
    # Show
    print("Showing plot with keyboard controls...")
    plot.show()
    plot.setFocus()  # Give keyboard focus
    
    # Process events
    for i in range(10):
        app.processEvents()
    
    print("Step 2 ready - use keyboard controls to test pan/zoom")
    print("Press 'Q' to quit when done testing")
    
    try:
        app.exec_()
        print("Step 2 completed")
        return True
    except Exception as e:
        print(f"Step 2 error: {e}")
        return False

if __name__ == "__main__":
    test_step2()