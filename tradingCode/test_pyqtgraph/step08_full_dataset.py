"""
Step 8: Full Candlestick Dataset (100+ candles)
Goal: Create a larger dataset with 200 candles and test performance
"""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from dataclasses import dataclass
from typing import List
import time

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

class LargeOHLCDataManager:
    """Manager for large OHLC datasets"""
    
    def __init__(self, num_bars: int = 200):
        self.num_bars = num_bars
        self.generate_large_dataset()
        
    def generate_large_dataset(self):
        """Generate realistic OHLC data for 200+ bars"""
        
        print(f"Generating large dataset with {self.num_bars} bars...")
        start_time = time.time()
        
        # Set seed for reproducible results
        np.random.seed(456)
        
        # Generate timestamps (1-minute bars)
        start_timestamp = 1609459200  # Jan 1, 2021 00:00:00 UTC
        self.timestamps = np.arange(start_timestamp, start_timestamp + self.num_bars * 60, 60)
        
        # Generate more complex price data with trends and cycles
        base_price = 150.0
        
        # Create multiple market regimes
        regime_length = self.num_bars // 4
        regimes = []
        
        # Regime 1: Uptrend with low volatility
        trend1 = np.linspace(0, 0.2, regime_length)  # 20% gain
        vol1 = np.full(regime_length, 0.015)  # 1.5% volatility
        
        # Regime 2: High volatility sideways
        trend2 = np.random.normal(0, 0.01, regime_length)
        vol2 = np.full(regime_length, 0.035)  # 3.5% volatility
        
        # Regime 3: Downtrend
        trend3 = np.linspace(0, -0.15, regime_length)  # 15% decline
        vol3 = np.full(regime_length, 0.025)  # 2.5% volatility
        
        # Regime 4: Recovery uptrend
        remaining = self.num_bars - 3 * regime_length
        trend4 = np.linspace(0, 0.25, remaining)  # 25% recovery
        vol4 = np.full(remaining, 0.02)  # 2% volatility
        
        # Combine regimes
        trend = np.concatenate([trend1, trend2, trend3, trend4])
        volatility = np.concatenate([vol1, vol2, vol3, vol4])
        
        # Generate returns with regime-specific characteristics
        returns = np.random.normal(0, volatility) + np.diff(np.concatenate([[0], trend]))
        
        # Compute close prices
        log_prices = np.log(base_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)
        
        # Generate opens with overnight gaps
        gaps = np.random.normal(0, 0.002, self.num_bars)  # Small overnight gaps
        open_prices = np.zeros_like(close_prices)
        open_prices[0] = base_price
        for i in range(1, self.num_bars):
            open_prices[i] = close_prices[i-1] * (1 + gaps[i])
        
        # Generate intraday highs and lows with more realistic behavior
        # Higher volatility during certain hours, lower during others
        high_factors = np.random.exponential(0.008, self.num_bars)
        low_factors = np.random.exponential(0.008, self.num_bars)
        
        # Add some extreme moves (gaps, spikes)
        extreme_indices = np.random.choice(self.num_bars, size=self.num_bars//20, replace=False)
        high_factors[extreme_indices] *= 2  # Double the high moves
        low_factors[extreme_indices] *= 2   # Double the low moves
        
        self.highs = np.maximum(open_prices, close_prices) * (1 + high_factors)
        self.lows = np.minimum(open_prices, close_prices) * (1 - low_factors)
        self.opens = open_prices
        self.closes = close_prices
        
        # Generate volume with realistic patterns
        base_volume = 15000
        # Volume tends to be higher during big moves
        price_changes = np.abs(close_prices - open_prices) / open_prices
        volume_multiplier = 1 + price_changes * 3  # Higher volume on big moves
        volume_noise = np.random.lognormal(0, 0.6, self.num_bars)
        self.volumes = (base_volume * volume_multiplier * volume_noise).astype(int)
        
        generation_time = time.time() - start_time
        print(f"Generated {self.num_bars} OHLC bars in {generation_time:.3f}s:")
        print(f"  Price range: ${self.lows.min():.2f} - ${self.highs.max():.2f}")
        print(f"  Volume range: {self.volumes.min():,} - {self.volumes.max():,}")
        print(f"  Total price change: {((close_prices[-1] / close_prices[0]) - 1) * 100:.1f}%")
        
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

class FullDatasetChartWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        
        # Set white background
        self.setBackground('white')
        
        # Set axis labels
        self.setLabel('left', 'Price ($)', color='black', size='14pt')
        self.setLabel('right', 'Price ($)', color='black', size='14pt')
        self.setLabel('bottom', 'Time (bars)', color='black', size='14pt')
        self.setLabel('top', 'Full Dataset (200 bars) - Step 8', color='black', size='16pt')
        
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
        
        # Create large OHLC data manager
        self.data_manager = LargeOHLCDataManager(200)
        
        # Create candlestick chart from structured data
        self.create_full_candlestick_chart()
        
        print("Step 8: Full dataset widget created")

    def create_full_candlestick_chart(self):
        """Create candlestick chart from full OHLC dataset"""
        
        print("Rendering full candlestick chart...")
        start_time = time.time()
        
        # Get all bars
        bars = self.data_manager.get_bars()
        
        # Draw each candlestick
        for i, bar in enumerate(bars):
            if i % 50 == 0:  # Progress indicator
                print(f"  Rendering candle {i}/{len(bars)}")
            self.draw_candlestick_from_bar(bar)
        
        # Set appropriate view range
        all_highs = [bar.high for bar in bars]
        all_lows = [bar.low for bar in bars]
        
        y_min = min(all_lows)
        y_max = max(all_highs)
        y_padding = (y_max - y_min) * 0.05
        
        self.setXRange(-5, len(bars) + 5)
        self.setYRange(y_min - y_padding, y_max + y_padding)
        
        render_time = time.time() - start_time
        print(f"Chart rendered in {render_time:.3f}s with {len(bars)} candlesticks")
        
        # Performance statistics
        bullish_count = sum(1 for bar in bars if bar.close > bar.open)
        bearish_count = len(bars) - bullish_count
        print(f"Market composition: {bullish_count} bullish, {bearish_count} bearish candles")

    def draw_candlestick_from_bar(self, bar: OHLCBar):
        """Draw a single candlestick from OHLCBar data"""
        
        x = bar.index
        
        # Determine candle color
        is_bullish = bar.close > bar.open
        
        # 1. Draw the wick (high-low line) - thinner for large datasets
        wick_line = pg.PlotDataItem(
            [x, x], 
            [bar.low, bar.high],
            pen=pg.mkPen('black', width=0.5)
        )
        self.addItem(wick_line)
        
        # 2. Draw the body (open-close rectangle)
        body_top = max(bar.open, bar.close)
        body_bottom = min(bar.open, bar.close)
        body_height = body_top - body_bottom
        
        # Ensure minimum body height for visibility
        if body_height < (bar.high - bar.low) * 0.02:
            body_height = (bar.high - bar.low) * 0.02
            body_top = body_bottom + body_height
        
        body_width = 0.4  # Narrower for large datasets
        
        # Set colors based on bullish/bearish
        if is_bullish:
            # Green/bullish candle - hollow (white fill with green border)
            brush = pg.mkBrush('white')
            pen = pg.mkPen('green', width=0.5)
        else:
            # Red/bearish candle - filled (red)
            brush = pg.mkBrush('red')
            pen = pg.mkPen('red', width=0.5)
        
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

def test_step8():
    """Test Step 8: Full Dataset (200 candles)"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Step 8: Full Dataset (200 Candlesticks)")
    win.resize(1400, 800)  # Larger window for more data
    
    # Create full dataset chart widget
    full_widget = FullDatasetChartWidget()
    win.setCentralWidget(full_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    for i in range(15):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "step08_full_dataset_result.png")
    
    print("\n=== STEP 8 VERIFICATION ===")
    print("Required elements to verify in screenshot:")
    print("- All previous elements (white background, visible axes)")
    print("- 200 candlesticks showing complete market cycles")
    print("- Multiple market regimes: uptrend, sideways, downtrend, recovery")
    print("- Good performance despite large dataset")
    print("- Appropriate scaling to fit all data")
    print("- Mix of candle sizes and colors throughout")
    print("- Clear market trends and patterns visible")
    print("\nCheck step08_full_dataset_result.png for verification")
    
    return app, win

if __name__ == "__main__":
    app, win = test_step8()
    
    # Keep open for inspection
    QtCore.QTimer.singleShot(10000, app.quit)
    app.exec_()