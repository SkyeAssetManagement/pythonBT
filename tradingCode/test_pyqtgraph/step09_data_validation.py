"""
Step 9: Data validation and null data handling
Goal: Handle invalid/null OHLC data to prevent black bars covering candles
"""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from dataclasses import dataclass
from typing import List, Optional
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
    """Single OHLC bar data structure with validation"""
    index: int
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_valid: bool = True
    
    def __post_init__(self):
        """Validate OHLC data relationships"""
        self.is_valid = self.validate_ohlc()
    
    def validate_ohlc(self) -> bool:
        """Validate OHLC data for correctness"""
        try:
            # Check for NaN or infinite values
            values = [self.open, self.high, self.low, self.close, self.volume]
            if any(not np.isfinite(v) for v in values):
                return False
            
            # Check for negative or zero prices
            if any(v <= 0 for v in [self.open, self.high, self.low, self.close]):
                return False
            
            # Check OHLC relationships
            if self.high < max(self.open, self.close):
                return False
            if self.low > min(self.open, self.close):
                return False
            if self.high < self.low:
                return False
            
            # Check volume
            if self.volume < 0:
                return False
                
            return True
        except:
            return False

class DataValidationManager:
    """Manager for OHLC data with built-in validation and corruption simulation"""
    
    def __init__(self, num_bars: int = 100):
        self.num_bars = num_bars
        self.generate_data_with_corruption()
        
    def generate_data_with_corruption(self):
        """Generate OHLC data with intentional corruption to test validation"""
        
        print(f"Generating dataset with {self.num_bars} bars and intentional data corruption...")
        
        # Set seed for reproducible results
        np.random.seed(789)
        
        # Generate timestamps
        start_timestamp = 1609459200
        self.timestamps = np.arange(start_timestamp, start_timestamp + self.num_bars * 60, 60)
        
        # Generate clean OHLC data first
        base_price = 160.0
        returns = np.random.normal(0.0005, 0.02, self.num_bars)
        log_prices = np.log(base_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)
        
        # Generate opens
        gaps = np.random.normal(0, 0.001, self.num_bars)
        open_prices = np.zeros_like(close_prices)
        open_prices[0] = base_price
        for i in range(1, self.num_bars):
            open_prices[i] = close_prices[i-1] * (1 + gaps[i])
        
        # Generate highs and lows
        high_factors = np.random.exponential(0.005, self.num_bars)
        low_factors = np.random.exponential(0.005, self.num_bars)
        
        highs = np.maximum(open_prices, close_prices) * (1 + high_factors)
        lows = np.minimum(open_prices, close_prices) * (1 - low_factors)
        opens = open_prices
        closes = close_prices
        
        # Generate volume
        base_volume = 12000
        volumes = base_volume * np.random.lognormal(0, 0.5, self.num_bars)
        
        # Now introduce various types of data corruption
        corruption_indices = []
        
        # 1. NaN values (5 bars)
        nan_indices = np.random.choice(self.num_bars, 5, replace=False)
        for idx in nan_indices:
            field = np.random.choice(['open', 'high', 'low', 'close'])
            if field == 'open':
                opens[idx] = np.nan
            elif field == 'high':
                highs[idx] = np.nan
            elif field == 'low':
                lows[idx] = np.nan
            else:
                closes[idx] = np.nan
            corruption_indices.append((idx, f"NaN {field}"))
        
        # 2. Negative values (3 bars)
        neg_indices = np.random.choice([i for i in range(self.num_bars) if i not in nan_indices], 3, replace=False)
        for idx in neg_indices:
            field = np.random.choice(['open', 'close'])
            if field == 'open':
                opens[idx] = -abs(opens[idx])
            else:
                closes[idx] = -abs(closes[idx])
            corruption_indices.append((idx, f"Negative {field}"))
        
        # 3. Invalid OHLC relationships (5 bars)
        invalid_indices = np.random.choice([i for i in range(self.num_bars) if i not in nan_indices and i not in neg_indices], 5, replace=False)
        for idx in invalid_indices:
            # Make high lower than close
            highs[idx] = closes[idx] * 0.95
            corruption_indices.append((idx, "High < Close"))
        
        # 4. Zero values (2 bars)
        zero_indices = np.random.choice([i for i in range(self.num_bars) if i not in nan_indices and i not in neg_indices and i not in invalid_indices], 2, replace=False)
        for idx in zero_indices:
            opens[idx] = 0.0
            corruption_indices.append((idx, "Zero open"))
        
        # 5. Infinite values (2 bars)
        inf_indices = np.random.choice([i for i in range(self.num_bars) if i not in nan_indices and i not in neg_indices and i not in invalid_indices and i not in zero_indices], 2, replace=False)
        for idx in inf_indices:
            highs[idx] = np.inf
            corruption_indices.append((idx, "Infinite high"))
        
        # Store data arrays
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.volumes = volumes.astype(int)
        
        # Print corruption summary
        print(f"Introduced {len(corruption_indices)} data corruptions:")
        for idx, desc in sorted(corruption_indices):
            print(f"  Bar {idx}: {desc}")
        
        print(f"Clean price range: ${np.nanmin(lows):.2f} - ${np.nanmax(highs):.2f}")
        
    def get_bar(self, index: int) -> OHLCBar:
        """Get single OHLC bar by index with validation"""
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
        
    def get_valid_bars(self) -> List[OHLCBar]:
        """Get only valid OHLC bars, filtering out corrupted data"""
        valid_bars = []
        invalid_count = 0
        
        for i in range(self.num_bars):
            bar = self.get_bar(i)
            if bar.is_valid:
                valid_bars.append(bar)
            else:
                invalid_count += 1
                print(f"Filtered invalid bar {i}: O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f}")
        
        print(f"Filtered {invalid_count} invalid bars, {len(valid_bars)} valid bars remaining")
        return valid_bars

class DataValidationChartWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        
        # Set white background
        self.setBackground('white')
        
        # Set axis labels
        self.setLabel('left', 'Price ($)', color='black', size='14pt')
        self.setLabel('right', 'Price ($)', color='black', size='14pt')
        self.setLabel('bottom', 'Time (bars)', color='black', size='14pt')
        self.setLabel('top', 'Data Validation (No Black Bars) - Step 9', color='black', size='16pt')
        
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
        
        # Create data validation manager
        self.data_manager = DataValidationManager(100)
        
        # Create candlestick chart with validation
        self.create_validated_candlestick_chart()
        
        print("Step 9: Data validation widget created")

    def create_validated_candlestick_chart(self):
        """Create candlestick chart with data validation (no black bars)"""
        
        print("Rendering validated candlestick chart...")
        start_time = time.time()
        
        # Get only valid bars (this filters out corrupted data)
        valid_bars = self.data_manager.get_valid_bars()
        
        # Draw each valid candlestick
        for bar in valid_bars:
            self.draw_validated_candlestick(bar)
        
        # Set appropriate view range based on valid data only
        if valid_bars:
            all_highs = [bar.high for bar in valid_bars]
            all_lows = [bar.low for bar in valid_bars]
            
            y_min = min(all_lows)
            y_max = max(all_highs)
            y_padding = (y_max - y_min) * 0.05
            
            # X range shows gaps where invalid data was removed
            x_indices = [bar.index for bar in valid_bars]
            x_min = min(x_indices)
            x_max = max(x_indices)
            
            self.setXRange(x_min - 2, x_max + 2)
            self.setYRange(y_min - y_padding, y_max + y_padding)
        
        render_time = time.time() - start_time
        print(f"Validated chart rendered in {render_time:.3f}s with {len(valid_bars)} valid candlesticks")

    def draw_validated_candlestick(self, bar: OHLCBar):
        """Draw a candlestick only if data is valid (prevents black bars)"""
        
        # Double-check validation before drawing
        if not bar.is_valid:
            print(f"Skipping invalid bar {bar.index}")
            return
        
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
        if body_height < (bar.high - bar.low) * 0.02:
            body_height = (bar.high - bar.low) * 0.02
            body_top = body_bottom + body_height
        
        body_width = 0.6
        
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

def test_step9():
    """Test Step 9: Data Validation (No Black Bars)"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Step 9: Data Validation - No Black Bars")
    win.resize(1200, 800)
    
    # Create data validation chart widget
    validation_widget = DataValidationChartWidget()
    win.setCentralWidget(validation_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    for i in range(15):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "step09_data_validation_result.png")
    
    print("\n=== STEP 9 VERIFICATION ===")
    print("Required elements to verify in screenshot:")
    print("- All previous elements (white background, visible axes)")
    print("- NO BLACK BARS covering candlesticks")
    print("- Only valid candlesticks rendered (invalid data filtered out)")
    print("- Some gaps in sequence where invalid bars were removed")
    print("- Clean chart appearance with proper OHLC relationships")
    print("- Normal green/red candlestick coloring")
    print("- No visual artifacts from corrupted data")
    print("\nThis demonstrates robust handling of bad data that would")
    print("otherwise cause black bars in the original dashboard.")
    print("\nCheck step09_data_validation_result.png for verification")
    
    return app, win

if __name__ == "__main__":
    app, win = test_step9()
    
    # Keep open for inspection
    QtCore.QTimer.singleShot(10000, app.quit)
    app.exec_()