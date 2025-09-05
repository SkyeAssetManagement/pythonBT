"""
Step 5: Create array of 10 simple candlesticks
Goal: Build array of 10 OHLC candlesticks with varying colors and sizes
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

class TenCandlesWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        
        # Set white background
        self.setBackground('white')
        
        # Set axis labels
        self.setLabel('left', 'Price ($)', color='black', size='14pt')
        self.setLabel('right', 'Price ($)', color='black', size='14pt')
        self.setLabel('bottom', 'Time (bars)', color='black', size='14pt')
        self.setLabel('top', '10 Candlesticks - Step 5', color='black', size='16pt')
        
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
        
        # Create 10 candlesticks
        self.create_ten_candlesticks()
        
        print("Step 5: Ten candlesticks widget created")

    def create_ten_candlesticks(self):
        """Create 10 OHLC candlesticks with realistic data"""
        
        # Generate 10 bars of OHLC data
        num_candles = 10
        base_price = 150.0
        
        # Generate realistic price progression
        np.random.seed(42)  # For reproducible results
        
        # Start with base price and create random walk
        price_changes = np.random.normal(0, 2.0, num_candles)  # Daily price changes
        close_prices = base_price + np.cumsum(price_changes)
        
        # Generate OHLC data for each candle
        ohlc_data = []
        for i in range(num_candles):
            close = close_prices[i]
            
            # Generate open (previous close or starting price)
            if i == 0:
                open_price = base_price
            else:
                open_price = close_prices[i-1]
            
            # Generate high and low with some volatility
            volatility = np.random.uniform(1.0, 4.0)
            high = max(open_price, close) + np.random.uniform(0.5, volatility)
            low = min(open_price, close) - np.random.uniform(0.5, volatility)
            
            ohlc_data.append({
                'x': i,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close
            })
        
        print(f"Generated OHLC data for {num_candles} candles")
        
        # Draw all candlesticks
        for candle in ohlc_data:
            self.draw_candlestick(candle)
        
        # Set appropriate view range
        all_highs = [c['high'] for c in ohlc_data]
        all_lows = [c['low'] for c in ohlc_data]
        
        y_min = min(all_lows)
        y_max = max(all_highs)
        y_padding = (y_max - y_min) * 0.1
        
        self.setXRange(-0.5, num_candles - 0.5)
        self.setYRange(y_min - y_padding, y_max + y_padding)
        
        print(f"Chart range set: X=0-{num_candles-1}, Y=${y_min:.1f}-${y_max:.1f}")

    def draw_candlestick(self, candle):
        """Draw a single candlestick from OHLC data"""
        
        x = candle['x']
        open_price = candle['open']
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        # Determine candle color
        is_bullish = close > open_price
        
        # 1. Draw the wick (high-low line)
        wick_line = pg.PlotDataItem(
            [x, x], 
            [low, high],
            pen=pg.mkPen('black', width=2)
        )
        self.addItem(wick_line)
        
        # 2. Draw the body (open-close rectangle)
        body_top = max(open_price, close)
        body_bottom = min(open_price, close)
        body_height = body_top - body_bottom
        
        # Ensure minimum body height for visibility
        if body_height < 0.1:
            body_height = 0.1
            body_top = body_bottom + body_height
        
        body_width = 0.7  # Width of candle body
        
        # Set colors based on bullish/bearish
        if is_bullish:
            # Green/bullish candle - hollow (white fill with green border)
            brush = pg.mkBrush('white')
            pen = pg.mkPen('green', width=2)
        else:
            # Red/bearish candle - filled (red)
            brush = pg.mkBrush('red')
            pen = pg.mkPen('red', width=2)
        
        # Create candle body using BarGraphItem
        body_bar = pg.BarGraphItem(
            x=[x], 
            height=[body_height],
            width=body_width,
            y0=[body_bottom],
            brush=brush,
            pen=pen
        )
        self.addItem(body_bar)
        
        # Debug output
        color_str = "Green" if is_bullish else "Red"
        print(f"Candle {x}: {color_str} O=${open_price:.1f} H=${high:.1f} L=${low:.1f} C=${close:.1f}")

def test_step5():
    """Test Step 5: Ten candlesticks"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Step 5: Ten Candlesticks")
    win.resize(1000, 700)
    
    # Create ten candles widget
    candles_widget = TenCandlesWidget()
    win.setCentralWidget(candles_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    for i in range(10):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "step05_ten_candles_result.png")
    
    print("\n=== STEP 5 VERIFICATION ===")
    print("Required elements to verify in screenshot:")
    print("- All previous elements (white background, visible axes)")
    print("- 10 candlesticks arranged horizontally (positions 0-9)")
    print("- Mix of green (bullish) and red (bearish) candles")
    print("- Each candle has: black wick line + colored body")
    print("- Green candles: white/hollow body with green border")
    print("- Red candles: solid red body")
    print("- Realistic price progression across the 10 candles")
    print("- Proper spacing and proportions")
    print("\nCheck step05_ten_candles_result.png for verification")
    
    return app, win

if __name__ == "__main__":
    app, win = test_step5()
    
    # Keep open for inspection
    QtCore.QTimer.singleShot(8000, app.quit)
    app.exec_()