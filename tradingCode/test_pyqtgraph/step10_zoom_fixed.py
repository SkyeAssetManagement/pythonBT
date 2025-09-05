"""
Step 10: Basic zoom functionality - FIXED VERSION
Goal: Implement Ctrl+scroll wheel zooming WITHOUT losing axis labels
Solution: Override wheelEvent in PlotWidget instead of replacing ViewBox
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

class ZoomableOHLCDataManager:
    """Manager for OHLC data optimized for zooming"""
    
    def __init__(self, num_bars: int = 150):
        self.num_bars = num_bars
        self.generate_zoom_test_data()
        
    def generate_zoom_test_data(self):
        """Generate OHLC data optimized for zoom testing"""
        
        print(f"Generating zoom test data with {self.num_bars} bars...")
        
        # Set seed for reproducible results
        np.random.seed(1000)
        
        # Generate timestamps
        start_timestamp = 1609459200
        self.timestamps = np.arange(start_timestamp, start_timestamp + self.num_bars * 60, 60)
        
        # Generate clean OHLC data with clear patterns for zoom testing
        base_price = 180.0
        
        # Create distinct price patterns that are easy to see when zooming
        pattern_length = self.num_bars // 3
        
        # Pattern 1: Clear uptrend
        trend1 = np.linspace(0, 0.3, pattern_length)  # 30% gain
        vol1 = np.full(pattern_length, 0.01)  # Low volatility
        
        # Pattern 2: High volatility consolidation
        trend2 = np.random.normal(0, 0.005, pattern_length)
        vol2 = np.full(pattern_length, 0.04)  # High volatility
        
        # Pattern 3: Downtrend with recovery
        remaining = self.num_bars - 2 * pattern_length
        trend3_down = np.linspace(0, -0.2, remaining//2)  # Down 20%
        trend3_up = np.linspace(0, 0.15, remaining - remaining//2)  # Recover 15%
        trend3 = np.concatenate([trend3_down, trend3_up])
        vol3 = np.full(remaining, 0.02)
        
        # Combine patterns
        trend = np.concatenate([trend1, trend2, trend3])
        volatility = np.concatenate([vol1, vol2, vol3])
        
        # Generate returns
        returns = np.random.normal(0, volatility) + np.diff(np.concatenate([[0], trend]))
        
        # Compute prices
        log_prices = np.log(base_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)
        
        # Generate opens
        gaps = np.random.normal(0, 0.001, self.num_bars)
        open_prices = np.zeros_like(close_prices)
        open_prices[0] = base_price
        for i in range(1, self.num_bars):
            open_prices[i] = close_prices[i-1] * (1 + gaps[i])
        
        # Generate highs and lows
        high_factors = np.random.exponential(0.006, self.num_bars)
        low_factors = np.random.exponential(0.006, self.num_bars)
        
        self.highs = np.maximum(open_prices, close_prices) * (1 + high_factors)
        self.lows = np.minimum(open_prices, close_prices) * (1 - low_factors)
        self.opens = open_prices
        self.closes = close_prices
        
        # Generate volume
        base_volume = 18000
        volume_noise = np.random.lognormal(0, 0.4, self.num_bars)
        self.volumes = (base_volume * volume_noise).astype(int)
        
        print(f"Price range: ${self.lows.min():.2f} - ${self.highs.max():.2f}")
        
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

class ZoomableChartWidget(pg.PlotWidget):
    """PlotWidget with custom zoom functionality that preserves axes"""
    
    def __init__(self):
        super().__init__()
        
        # Set white background
        self.setBackground('white')
        
        # Set axis labels
        self.setLabel('left', 'Price ($)', color='black', size='14pt')
        self.setLabel('right', 'Price ($)', color='black', size='14pt')
        self.setLabel('bottom', 'Time (bars)', color='black', size='14pt')
        self.setLabel('top', 'Zoomable Chart (FIXED) - Step 10', color='black', size='16pt')
        
        # Configure all axes for maximum visibility (using the working pattern from previous steps)
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
        
        # DON'T replace the ViewBox - instead, override the wheel event directly
        # The existing ViewBox is already properly linked to the axes
        
        # Create zoomable OHLC data manager
        self.data_manager = ZoomableOHLCDataManager(150)
        
        # Create candlestick chart
        self.create_zoomable_candlestick_chart()
        
        # Add zoom instructions
        self.add_zoom_instructions()
        
        print("Step 10 FIXED: Zoomable chart widget created with preserved axes")

    def wheelEvent(self, ev):
        """Custom wheel event handler that preserves axis functionality"""
        
        # Check if Ctrl key is pressed
        modifiers = ev.modifiers()
        ctrl_pressed = modifiers & QtCore.Qt.ControlModifier
        
        if ctrl_pressed:
            # Ctrl+scroll: Custom zoom behavior
            zoom_factor = 1.2 if ev.angleDelta().y() > 0 else 1/1.2
            
            # Get current view range from the ViewBox
            current_range = self.plotItem.vb.viewRange()
            x_range = current_range[0]
            y_range = current_range[1]
            
            # Calculate zoom center (center of current view)
            zoom_center_x = (x_range[0] + x_range[1]) / 2
            zoom_center_y = (y_range[0] + y_range[1]) / 2
            
            # Calculate new ranges
            x_span = x_range[1] - x_range[0]
            y_span = y_range[1] - y_range[0]
            
            new_x_span = x_span / zoom_factor
            new_y_span = y_span / zoom_factor
            
            new_x_range = [
                zoom_center_x - new_x_span/2,
                zoom_center_x + new_x_span/2
            ]
            new_y_range = [
                zoom_center_y - new_y_span/2,
                zoom_center_y + new_y_span/2
            ]
            
            # Apply zoom using the existing ViewBox (preserves axis linking)
            self.plotItem.vb.setRange(xRange=new_x_range, yRange=new_y_range, padding=0)
            
            print(f"Zoomed: X={new_x_range[0]:.1f}-{new_x_range[1]:.1f}, Y=${new_y_range[0]:.1f}-${new_y_range[1]:.1f}")
            
            # Accept the event so it doesn't propagate
            ev.accept()
            
        else:
            # Regular scroll: Let the parent handle it (or do nothing)
            # For now, do nothing - panning will be added in Step 11
            print("Regular scroll - no action (panning will be added in Step 11)")
            ev.accept()

    def create_zoomable_candlestick_chart(self):
        """Create candlestick chart optimized for zooming"""
        
        print("Rendering zoomable candlestick chart...")
        start_time = time.time()
        
        # Get all bars
        bars = self.data_manager.get_bars()
        
        # Draw each candlestick
        for bar in bars:
            self.draw_zoomable_candlestick(bar)
        
        # Set initial view range to show all data
        all_highs = [bar.high for bar in bars]
        all_lows = [bar.low for bar in bars]
        
        y_min = min(all_lows)
        y_max = max(all_highs)
        y_padding = (y_max - y_min) * 0.05
        
        self.setXRange(-5, len(bars) + 5)
        self.setYRange(y_min - y_padding, y_max + y_padding)
        
        render_time = time.time() - start_time
        print(f"Zoomable chart rendered in {render_time:.3f}s with {len(bars)} candlesticks")
        print("Initial view shows all data - use Ctrl+scroll to zoom")

    def draw_zoomable_candlestick(self, bar: OHLCBar):
        """Draw a candlestick optimized for zooming"""
        
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
        
        body_width = 0.7  # Slightly wider for better zoom visibility
        
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

    def add_zoom_instructions(self):
        """Add text instructions for zoom functionality"""
        instructions = pg.TextItem(
            text="FIXED: Hold Ctrl + Scroll Wheel to Zoom In/Out (Axes Preserved!)",
            color='blue',
            fill=pg.mkBrush(255, 255, 255, 200),
            border=pg.mkPen('blue', width=2),
            anchor=(1, 0)  # Top-right anchor
        )
        
        self.addItem(instructions, ignoreBounds=True)
        
        # Position at top-right
        view_range = self.viewRange()
        x_max = view_range[0][1] - 5
        y_max = view_range[1][1] - 5
        instructions.setPos(x_max, y_max)
        instructions.setZValue(1000)

def test_step10_fixed():
    """Test Step 10 FIXED: Basic Zoom Functionality with Preserved Axes"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Step 10 FIXED: Basic Zoom Functionality - Axes Preserved")
    win.resize(1300, 800)
    
    # Create zoomable chart widget
    zoom_widget = ZoomableChartWidget()
    win.setCentralWidget(zoom_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    for i in range(15):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "step10_zoom_fixed_result.png")
    
    print("\n=== STEP 10 FIXED VERIFICATION ===")
    print("Required elements to verify in screenshot:")
    print("- All previous elements (white background, VISIBLE AXES with labels)")
    print("- 150 candlesticks showing clear patterns for zoom testing")
    print("- Blue instruction text: 'FIXED: Hold Ctrl + Scroll Wheel to Zoom'")
    print("- Chart initially shows all data (full zoom out)")
    print("- MOST IMPORTANTLY: All axis labels and tick marks are visible!")
    print("\nFIX EXPLANATION:")
    print("- Instead of replacing ViewBox with setCentralItem(), we override wheelEvent()")
    print("- This preserves the existing PlotItem's axis linking")
    print("- Axes remain properly connected and visible")
    print("\nTo test zoom functionality:")
    print("1. Hold Ctrl key")
    print("2. Scroll wheel up to zoom in")
    print("3. Scroll wheel down to zoom out")
    print("4. Observe that both X and Y axes zoom together WITH LABELS")
    print("\nCheck step10_zoom_fixed_result.png for verification")
    
    return app, win

if __name__ == "__main__":
    app, win = test_step10_fixed()
    
    # Keep open for extended zoom testing
    QtCore.QTimer.singleShot(15000, app.quit)  # 15 seconds for testing
    app.exec_()