"""
Step 7: OHLC Data Structure
Goal: Create proper OHLC data arrays and structured data management
"""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from dataclasses import dataclass
from typing import List

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

@dataclass
class OHLCBar:
    """Single OHLC bar data structure"""
    index: int
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

class OHLCDataManager:
    """Manager for OHLC data arrays"""
    
    def __init__(self, num_bars: int = 50):
        self.num_bars = num_bars
        self.generate_data()
        
    def generate_data(self):
        """Generate realistic OHLC data arrays"""
        
        # Set seed for reproducible results
        np.random.seed(123)
        
        # Generate timestamps (1-minute bars)
        start_timestamp = 1609459200  # Jan 1, 2021 00:00:00 UTC
        self.timestamps = np.arange(start_timestamp, start_timestamp + self.num_bars * 60, 60)
        
        # Generate price data using random walk
        base_price = 145.0
        volatility = 0.02  # 2% daily volatility
        drift = 0.0001     # Small upward drift
        
        # Generate returns and compute prices
        returns = np.random.normal(drift, volatility, self.num_bars)
        log_prices = np.log(base_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)
        
        # Generate opens (previous close + gap)
        gaps = np.random.normal(0, 0.001, self.num_bars)  # Small opening gaps
        open_prices = np.zeros_like(close_prices)
        open_prices[0] = base_price
        for i in range(1, self.num_bars):
            open_prices[i] = close_prices[i-1] * (1 + gaps[i])
        
        # Generate highs and lows
        high_factors = np.random.exponential(0.005, self.num_bars)  # High spikes
        low_factors = np.random.exponential(0.005, self.num_bars)   # Low dips
        
        self.highs = np.maximum(open_prices, close_prices) * (1 + high_factors)
        self.lows = np.minimum(open_prices, close_prices) * (1 - low_factors)
        self.opens = open_prices
        self.closes = close_prices
        
        # Generate volume
        base_volume = 10000
        volume_volatility = 0.5
        volume_factors = np.random.lognormal(0, volume_volatility, self.num_bars)
        self.volumes = (base_volume * volume_factors).astype(int)
        
        print(f"Generated {self.num_bars} OHLC bars:")
        print(f"  Price range: ${self.lows.min():.2f} - ${self.highs.max():.2f}")
        print(f"  Volume range: {self.volumes.min():,} - {self.volumes.max():,}")
        
    def get_bar(self, index: int) -> OHLCBar:
        """Get single OHLC bar by index"""
        if 0 <= index < self.num_bars:
            return OHLCBar(
                index=index,
                timestamp=self.timestamps[index],
                open=self.opens[index],
                high=self.highs[index],
                low=self.lows[index],
                close=self.closes[index],
                volume=self.volumes[index]
            )
        raise IndexError(f"Bar index {index} out of range (0-{self.num_bars-1})")
        
    def get_bars(self, start: int = 0, end: int = None) -> List[OHLCBar]:
        """Get range of OHLC bars"""
        if end is None:
            end = self.num_bars
        return [self.get_bar(i) for i in range(start, min(end, self.num_bars))]

class OHLCChartWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        
        # Set white background
        self.setBackground('white')
        
        # Set axis labels
        self.setLabel('left', 'Price ($)', color='black', size='14pt')
        self.setLabel('right', 'Price ($)', color='black', size='14pt')
        self.setLabel('bottom', 'Time (bars)', color='black', size='14pt')
        self.setLabel('top', 'OHLC Data Structure - Step 7', color='black', size='16pt')
        
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
        
        # Create OHLC data manager
        self.data_manager = OHLCDataManager(50)
        
        # Create candlestick chart from structured data
        self.create_candlestick_chart()
        
        print("Step 7: OHLC data structure widget created")

    def create_candlestick_chart(self):
        """Create candlestick chart from OHLC data structure"""
        
        # Get all bars
        bars = self.data_manager.get_bars()
        
        # Draw each candlestick
        for bar in bars:
            self.draw_candlestick_from_bar(bar)
        
        # Set appropriate view range
        all_highs = [bar.high for bar in bars]
        all_lows = [bar.low for bar in bars]
        
        y_min = min(all_lows)
        y_max = max(all_highs)
        y_padding = (y_max - y_min) * 0.05
        
        self.setXRange(-1, len(bars))
        self.setYRange(y_min - y_padding, y_max + y_padding)
        
        print(f"Chart created with {len(bars)} candlesticks")
        
        # Print sample data for verification
        print("\nSample OHLC data (first 5 bars):")
        for i in range(min(5, len(bars))):
            bar = bars[i]
            print(f"  Bar {i}: O=${bar.open:.2f} H=${bar.high:.2f} L=${bar.low:.2f} C=${bar.close:.2f} V={bar.volume:,}")

    def draw_candlestick_from_bar(self, bar: OHLCBar):
        """Draw a single candlestick from OHLCBar data"""
        
        x = bar.index
        
        # Determine candle color
        is_bullish = bar.close > bar.open
        
        # 1. Draw the wick (high-low line)
        wick_line = pg.PlotDataItem(
            [x, x], 
            [bar.low, bar.high],
            pen=pg.mkPen('black', width=1)
        )
        self.addItem(wick_line)
        
        # 2. Draw the body (open-close rectangle)
        body_top = max(bar.open, bar.close)
        body_bottom = min(bar.open, bar.close)
        body_height = body_top - body_bottom
        
        # Ensure minimum body height for visibility
        if body_height < (bar.high - bar.low) * 0.01:
            body_height = (bar.high - bar.low) * 0.01
            body_top = body_bottom + body_height
        
        body_width = 0.6  # Width of candle body
        
        # Set colors based on bullish/bearish
        if is_bullish:
            # Green/bullish candle - hollow (white fill with green border)
            brush = pg.mkBrush('white')
            pen = pg.mkPen('green', width=1)
        else:
            # Red/bearish candle - filled (red)
            brush = pg.mkBrush('red')
            pen = pg.mkPen('red', width=1)
        
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

def test_step7():
    """Test Step 7: OHLC Data Structure"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Step 7: OHLC Data Structure")
    win.resize(1200, 700)
    
    # Create OHLC chart widget
    ohlc_widget = OHLCChartWidget()
    win.setCentralWidget(ohlc_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    for i in range(10):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "step07_ohlc_data_result.png")
    
    print("\n=== STEP 7 VERIFICATION ===")
    print("Required elements to verify in screenshot:")
    print("- All previous elements (white background, visible axes)")
    print("- 50 candlesticks arranged horizontally")
    print("- Realistic price movement with proper OHLC relationships")
    print("- Green (bullish) and red (bearish) candles")
    print("- Proper data structure: H >= max(O,C), L <= min(O,C)")
    print("- Smooth chart appearance with no gaps or errors")
    print("- Appropriate scaling and proportions")
    print("\nCheck step07_ohlc_data_result.png for verification")
    
    return app, win

if __name__ == "__main__":
    app, win = test_step7()
    
    # Keep open for inspection
    QtCore.QTimer.singleShot(8000, app.quit)
    app.exec_()