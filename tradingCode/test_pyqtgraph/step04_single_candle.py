"""
Step 4: Draw single candlestick manually
Goal: Create one OHLC candlestick using basic PyQtGraph drawing primitives
"""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

def save_widget_screenshot(widget, filename):
    """Save a screenshot of the widget"""
    try:
        pixmap = widget.grab()
        success = pixmap.save(filename, 'PNG')
        print(f"Screenshot saved: {filename} (Success: {success})")
        return success
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return False

class SingleCandleWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        
        # Set white background
        self.setBackground('white')
        
        # Set axis labels
        self.setLabel('left', 'Price ($)', color='black', size='14pt')
        self.setLabel('right', 'Price ($)', color='black', size='14pt')
        self.setLabel('bottom', 'Time (bars)', color='black', size='14pt')
        self.setLabel('top', 'Single Candlestick - Step 4', color='black', size='16pt')
        
        # Configure all axes for maximum visibility
        for axis_name in ['left', 'right', 'bottom', 'top']:
            self.showAxis(axis_name, True)
            axis = self.getAxis(axis_name)
            if axis:
                # Thick black axes
                axis.setPen(pg.mkPen('black', width=3))
                axis.setTextPen(pg.mkPen('black', width=2))
                axis.setStyle(showValues=True, tickLength=10)
                
                # Set appropriate dimensions
                if axis_name in ['left', 'right']:
                    axis.setWidth(100)
                else:
                    axis.setHeight(60)
                
                axis.show()
        
        # Create single candlestick
        self.create_single_candlestick()
        
        print("Step 4: Single candlestick widget created")

    def create_single_candlestick(self):
        """Create a single OHLC candlestick manually"""
        
        # OHLC data for one candle (centered at x=50)
        x_pos = 50
        open_price = 145.0
        high_price = 152.0
        low_price = 143.0
        close_price = 149.0
        
        print(f"Creating candlestick: O={open_price}, H={high_price}, L={low_price}, C={close_price}")
        
        # Determine candle color (green if close > open, red if close < open)
        is_bullish = close_price > open_price
        candle_color = 'green' if is_bullish else 'red'
        
        # 1. Draw the wick (high-low line)
        wick_line = pg.PlotDataItem(
            [x_pos, x_pos], 
            [low_price, high_price],
            pen=pg.mkPen('black', width=2)
        )
        self.addItem(wick_line)
        
        # 2. Draw the body (open-close rectangle)
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        body_height = body_top - body_bottom
        body_width = 0.8  # Width of candle body
        
        # Create rectangle for candle body
        if is_bullish:
            # Green/bullish candle - hollow (white fill with green border)
            brush = pg.mkBrush('white')
            pen = pg.mkPen('green', width=2)
        else:
            # Red/bearish candle - filled (red)
            brush = pg.mkBrush('red')
            pen = pg.mkPen('red', width=2)
        
        # Use BarGraphItem for the candle body
        body_bar = pg.BarGraphItem(
            x=[x_pos], 
            height=[body_height],
            width=body_width,
            y0=[body_bottom],
            brush=brush,
            pen=pen
        )
        self.addItem(body_bar)
        
        # 3. Add text labels for OHLC values
        text_items = [
            (x_pos - 15, high_price + 1, f"H: ${high_price:.0f}"),
            (x_pos - 15, low_price - 2, f"L: ${low_price:.0f}"),
            (x_pos + 5, open_price, f"O: ${open_price:.0f}"),
            (x_pos + 5, close_price, f"C: ${close_price:.0f}")
        ]
        
        for x, y, text in text_items:
            text_item = pg.TextItem(text, color='black', anchor=(0, 0.5))
            text_item.setPos(x, y)
            self.addItem(text_item)
        
        # Set appropriate view range to center the candle
        self.setXRange(20, 80)
        self.setYRange(140, 155)
        
        print(f"Single {candle_color} candlestick created at position {x_pos}")

def test_step4():
    """Test Step 4: Single candlestick"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Step 4: Single Candlestick")
    win.resize(1000, 700)
    
    # Create single candle widget
    candle_widget = SingleCandleWidget()
    win.setCentralWidget(candle_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    for i in range(10):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "step04_single_candle_result.png")
    
    print("\n=== STEP 4 VERIFICATION ===")
    print("Required elements to verify in screenshot:")
    print("- All previous elements (white background, visible axes)")
    print("- One candlestick centered in the chart")
    print("- Black wick line from low to high price")
    print("- Rectangular candle body (green/hollow or red/filled)")
    print("- OHLC text labels showing the values")
    print("- Proper proportions (body width ~0.8, wick is thin line)")
    print("- Color indicates bullish (green/hollow) or bearish (red/filled)")
    print("\nCheck step04_single_candle_result.png for verification")
    
    return app, win

if __name__ == "__main__":
    app, win = test_step4()
    
    # Keep open for inspection
    QtCore.QTimer.singleShot(8000, app.quit)
    app.exec_()