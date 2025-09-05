"""
Step 13: Fix crosshair styling - thin, dotted, black
Goal: Change crosshairs from thick red to thin dotted black lines
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

class CrosshairTestDataManager:
    """Manager for OHLC data with crosshair testing"""
    
    def __init__(self, num_bars: int = 100):
        self.num_bars = num_bars
        self.generate_data()
        
    def generate_data(self):
        """Generate OHLC data for testing"""
        
        print(f"Generating crosshair test data with {self.num_bars} bars...")
        
        # Set seed for reproducible results
        np.random.seed(4000)
        
        # Generate timestamps (1-minute bars starting from a specific date)
        start_timestamp = 1609459200  # Jan 1, 2021 00:00:00 UTC
        self.timestamps = np.arange(start_timestamp, start_timestamp + self.num_bars * 60, 60)
        
        # Generate price data with clear trends
        base_price = 200.0
        
        # Create realistic price movement
        returns = np.random.normal(0.0008, 0.025, self.num_bars)  # Slight upward bias
        log_prices = np.log(base_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)
        
        # Generate opens
        gaps = np.random.normal(0, 0.002, self.num_bars)
        open_prices = np.zeros_like(close_prices)
        open_prices[0] = base_price
        for i in range(1, self.num_bars):
            open_prices[i] = close_prices[i-1] * (1 + gaps[i])
        
        # Generate highs and lows
        high_factors = np.random.exponential(0.01, self.num_bars)
        low_factors = np.random.exponential(0.01, self.num_bars)
        
        self.highs = np.maximum(open_prices, close_prices) * (1 + high_factors)
        self.lows = np.minimum(open_prices, close_prices) * (1 - low_factors)
        self.opens = open_prices
        self.closes = close_prices
        
        # Generate volume
        base_volume = 25000
        volume_noise = np.random.lognormal(0, 0.6, self.num_bars)
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

class CrosshairFixChartWidget(pg.PlotWidget):
    """Chart widget with fixed crosshair styling - thin, dotted, black"""
    
    def __init__(self):
        super().__init__()
        
        # Set white background
        self.setBackground('white')
        
        # Remove background gridlines
        self.showGrid(x=False, y=False)
        
        # Set axis labels
        self.setLabel('left', 'Price ($)', color='black', size='14pt')
        self.setLabel('right', 'Price ($)', color='black', size='14pt')
        self.setLabel('bottom', 'Time (bars)', color='black', size='14pt')
        self.setLabel('top', 'Step 13: Crosshair Fix - Thin, Dotted, Black', color='black', size='16pt')
        
        # Configure all axes for maximum visibility
        for axis_name in ['left', 'right', 'bottom', 'top']:
            self.showAxis(axis_name, True)
            axis = self.getAxis(axis_name)
            if axis:
                axis.setPen(pg.mkPen('black', width=3))
                axis.setTextPen(pg.mkPen('black', width=2))
                axis.setStyle(showValues=True, tickLength=10)
                
                if axis_name in ['left', 'right']:
                    axis.setWidth(100)
                else:
                    axis.setHeight(60)
                
                axis.show()
        
        # Create data manager
        self.data_manager = CrosshairTestDataManager(100)
        
        # Setup OLD crosshairs first (for comparison)
        self.setup_old_crosshairs()
        
        # Create candlestick chart
        self.create_candlestick_chart()
        
        print("Step 13: Crosshair fix chart widget created")

    def setup_old_crosshairs(self):
        """Setup OLD crosshairs (thick red) for comparison"""
        
        print("Setting up OLD crosshairs (thick red) for comparison...")
        
        # OLD STYLE - thick, bright red crosshairs
        self.crosshair_v_old = pg.InfiniteLine(
            angle=90, 
            movable=False,
            pen=pg.mkPen('red', width=3, style=QtCore.Qt.SolidLine)  # OLD: thick red
        )
        self.crosshair_h_old = pg.InfiniteLine(
            angle=0, 
            movable=False,
            pen=pg.mkPen('red', width=3, style=QtCore.Qt.SolidLine)  # OLD: thick red
        )
        
        # Add to plot
        self.addItem(self.crosshair_v_old, ignoreBounds=True)
        self.addItem(self.crosshair_h_old, ignoreBounds=True)
        self.crosshair_v_old.setZValue(900)  # Lower Z so new ones are on top
        self.crosshair_h_old.setZValue(900)
        
        # Position at a specific location
        self.crosshair_v_old.setPos(30)
        self.crosshair_h_old.setPos(220)
        self.crosshair_v_old.show()
        self.crosshair_h_old.show()
        
        # Now setup NEW crosshairs (thin dotted black)
        self.setup_new_crosshairs()
        
    def setup_new_crosshairs(self):
        """Setup NEW crosshairs (thin, dotted, black) - FIXED VERSION"""
        
        print("Setting up NEW crosshairs (thin, dotted, black)...")
        
        # NEW STYLE - thin, dotted, black crosshairs (FIXED)
        self.crosshair_v_new = pg.InfiniteLine(
            angle=90, 
            movable=False,
            pen=pg.mkPen('black', width=1, style=QtCore.Qt.DotLine)  # NEW: thin dotted black
        )
        self.crosshair_h_new = pg.InfiniteLine(
            angle=0, 
            movable=False,
            pen=pg.mkPen('black', width=1, style=QtCore.Qt.DotLine)  # NEW: thin dotted black
        )
        
        # Add to plot with high Z-value (on top)
        self.addItem(self.crosshair_v_new, ignoreBounds=True)
        self.addItem(self.crosshair_h_new, ignoreBounds=True)
        self.crosshair_v_new.setZValue(1000)  # Higher Z so they're on top
        self.crosshair_h_new.setZValue(1000)
        
        # Position at a different location to compare
        self.crosshair_v_new.setPos(70)
        self.crosshair_h_new.setPos(250)
        self.crosshair_v_new.show()
        self.crosshair_h_new.show()
        
        # Create data box for new crosshairs
        self.crosshair_data_box = pg.TextItem(
            text="NEW: Thin, Dotted, Black Crosshairs",
            color='black',
            fill=pg.mkBrush(255, 255, 0, 200),  # Yellow background
            border=pg.mkPen('black', width=2),
            anchor=(0, 1)  # Bottom-left anchor
        )
        
        self.addItem(self.crosshair_data_box, ignoreBounds=True)
        self.crosshair_data_box.setZValue(1001)
        self.crosshair_data_box.setPos(72, 245)
        self.crosshair_data_box.show()
        
        # Create comparison label for old crosshairs
        self.old_crosshair_label = pg.TextItem(
            text="OLD: Thick, Red Crosshairs",
            color='white',
            fill=pg.mkBrush(255, 0, 0, 180),  # Red background
            border=pg.mkPen('red', width=2),
            anchor=(0, 1)  # Bottom-left anchor
        )
        
        self.addItem(self.old_crosshair_label, ignoreBounds=True)
        self.old_crosshair_label.setZValue(1001)
        self.old_crosshair_label.setPos(32, 215)
        self.old_crosshair_label.show()
        
        # Use SignalProxy pattern for mouse tracking of NEW crosshairs
        self.mouse_proxy = pg.SignalProxy(
            self.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self._on_mouse_moved
        )
        
        print("NEW crosshairs setup complete with mouse tracking")

    def _on_mouse_moved(self, evt):
        """Handle mouse movement for NEW crosshairs (thin, dotted, black)"""
        try:
            pos = evt[0]  # Get position from SignalProxy
            if self.plotItem.vb.sceneBoundingRect().contains(pos):
                mousePoint = self.plotItem.vb.mapSceneToView(pos)
                x_coord = mousePoint.x()
                y_coord = mousePoint.y()
                
                # Update NEW crosshair position
                self.crosshair_v_new.setPos(x_coord)
                self.crosshair_h_new.setPos(y_coord)
                
                # Update data box text with coordinates
                data_text = f"NEW Crosshairs: X={x_coord:.1f}, Y=${y_coord:.2f}"
                self.crosshair_data_box.setText(data_text)
                self.crosshair_data_box.setPos(x_coord + 2, y_coord - 5)
                
        except Exception as e:
            print(f"Error in mouse handler: {e}")

    def create_candlestick_chart(self):
        """Create candlestick chart"""
        
        print("Rendering candlestick chart...")
        start_time = time.time()
        
        # Get all bars
        bars = self.data_manager.get_bars()
        
        # Draw each candlestick
        for bar in bars:
            self.draw_candlestick(bar)
        
        # Set initial view range
        self.setXRange(-3, len(bars) + 3)
        
        render_time = time.time() - start_time
        print(f"Chart rendered in {render_time:.3f}s with {len(bars)} candlesticks")

    def draw_candlestick(self, bar: OHLCBar):
        """Draw a single candlestick"""
        
        x = bar.index
        
        # Enhanced validation
        if not (np.isfinite(bar.open) and np.isfinite(bar.high) and 
                np.isfinite(bar.low) and np.isfinite(bar.close) and
                bar.high >= max(bar.open, bar.close) and 
                bar.low <= min(bar.open, bar.close)):
            return
        
        # Determine candle color
        is_bullish = bar.close > bar.open
        
        # 1. Draw the wick
        wick_line = pg.PlotDataItem(
            [x, x], 
            [bar.low, bar.high],
            pen=pg.mkPen('black', width=1)
        )
        self.addItem(wick_line)
        
        # 2. Draw the body
        body_top = max(bar.open, bar.close)
        body_bottom = min(bar.open, bar.close)
        body_height = body_top - body_bottom
        
        # Ensure minimum body height
        if body_height < (bar.high - bar.low) * 0.02:
            body_height = (bar.high - bar.low) * 0.02
            body_top = body_bottom + body_height
        
        body_width = 0.7
        
        # Set colors (red/green)
        if is_bullish:
            brush = pg.mkBrush('white')
            pen = pg.mkPen('green', width=1)
        else:
            brush = pg.mkBrush('red')
            pen = pg.mkPen('red', width=1)
        
        # Create candle body
        body_bar = pg.BarGraphItem(
            x=[x], 
            height=[body_height],
            width=body_width,
            y0=[body_bottom],
            brush=brush,
            pen=pen
        )
        self.addItem(body_bar)

def test_crosshair_fix():
    """Test Step 13: Crosshair styling fix"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Step 13: Crosshair Fix - Thin, Dotted, Black vs Thick, Red")
    win.resize(1600, 900)
    
    # Create crosshair fix chart widget
    chart_widget = CrosshairFixChartWidget()
    win.setCentralWidget(chart_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    for i in range(25):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "step13_crosshair_fix_result.png")
    
    print("\n=== STEP 13 CROSSHAIR FIX VERIFICATION ===")
    print("Visual comparison:")
    print("- OLD: Thick red crosshairs (fixed position)")
    print("- NEW: Thin dotted black crosshairs (follow mouse)")
    print("")
    print("Test interactions:")
    print("- Move mouse over chart to see NEW crosshairs follow")
    print("- Compare OLD (thick red) vs NEW (thin dotted black)")
    print("- NEW crosshairs should be barely visible but functional")
    print("")
    print("Expected result:")
    print("- NEW crosshairs: thin (width=1), dotted (DotLine), black color")
    print("- OLD crosshairs: thick (width=3), solid (SolidLine), red color")
    print("")
    print("Check step13_crosshair_fix_result.png for verification")
    
    return app, win

if __name__ == "__main__":
    app, win = test_crosshair_fix()
    
    # Keep open for extended testing
    QtCore.QTimer.singleShot(30000, app.quit)  # 30 seconds for testing
    app.exec_()