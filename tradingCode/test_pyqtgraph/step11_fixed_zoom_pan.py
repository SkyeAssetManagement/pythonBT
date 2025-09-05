"""
Step 11: Fixed zoom/pan functionality
Goal: 
1. Y-axis: no zoom/scroll, auto-fit to price range
2. X-axis: Ctrl+wheel = zoom, wheel = pan
3. Working crosshair
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

class FixedOHLCDataManager:
    """Manager for OHLC data with fixed zoom/pan"""
    
    def __init__(self, num_bars: int = 150):
        self.num_bars = num_bars
        self.generate_data()
        
    def generate_data(self):
        """Generate OHLC data for testing"""
        
        print(f"Generating data with {self.num_bars} bars...")
        
        # Set seed for reproducible results
        np.random.seed(2000)
        
        # Generate timestamps
        start_timestamp = 1609459200
        self.timestamps = np.arange(start_timestamp, start_timestamp + self.num_bars * 60, 60)
        
        # Generate price data with clear trends
        base_price = 180.0
        
        # Create price movement
        returns = np.random.normal(0.001, 0.02, self.num_bars)
        log_prices = np.log(base_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)
        
        # Generate opens
        gaps = np.random.normal(0, 0.001, self.num_bars)
        open_prices = np.zeros_like(close_prices)
        open_prices[0] = base_price
        for i in range(1, self.num_bars):
            open_prices[i] = close_prices[i-1] * (1 + gaps[i])
        
        # Generate highs and lows
        high_factors = np.random.exponential(0.008, self.num_bars)
        low_factors = np.random.exponential(0.008, self.num_bars)
        
        self.highs = np.maximum(open_prices, close_prices) * (1 + high_factors)
        self.lows = np.minimum(open_prices, close_prices) * (1 - low_factors)
        self.opens = open_prices
        self.closes = close_prices
        
        # Generate volume
        base_volume = 15000
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
    
    def get_visible_price_range(self, start_idx: int, end_idx: int) -> tuple:
        """Get min/max price range for visible bars"""
        start_idx = max(0, int(start_idx))
        end_idx = min(self.num_bars, int(end_idx))
        
        if start_idx >= end_idx:
            return 0, 100
        
        visible_highs = self.highs[start_idx:end_idx]
        visible_lows = self.lows[start_idx:end_idx]
        
        if len(visible_highs) == 0 or len(visible_lows) == 0:
            return 0, 100
            
        y_min = np.min(visible_lows)
        y_max = np.max(visible_highs)
        y_padding = (y_max - y_min) * 0.05  # 5% padding
        
        return y_min - y_padding, y_max + y_padding

class FixedZoomPanChartWidget(pg.PlotWidget):
    """Chart widget with fixed zoom/pan behavior"""
    
    def __init__(self):
        super().__init__()
        
        # Set white background
        self.setBackground('white')
        
        # Set axis labels
        self.setLabel('left', 'Price ($)', color='black', size='14pt')
        self.setLabel('right', 'Price ($)', color='black', size='14pt')
        self.setLabel('bottom', 'Time (bars)', color='black', size='14pt')
        self.setLabel('top', 'Fixed Zoom/Pan + Crosshair - Step 11', color='black', size='16pt')
        
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
        
        # Create data manager
        self.data_manager = FixedOHLCDataManager(150)
        
        # Setup crosshair FIRST (before adding candlesticks)
        self.setup_crosshair()
        
        # Create candlestick chart
        self.create_candlestick_chart()
        
        # Connect to range changes for Y-axis auto-scaling
        self.plotItem.vb.sigRangeChanged.connect(self._on_range_changed)
        
        # Add instructions
        self.add_instructions()
        
        print("Step 11: Fixed zoom/pan chart widget created")

    def setup_crosshair(self):
        """Setup crosshair using successful Test 3 patterns"""
        
        # Create thick, bright crosshairs (Test 3 pattern)
        self.crosshair_v = pg.InfiniteLine(
            angle=90, 
            movable=False,
            pen=pg.mkPen('red', width=4, style=QtCore.Qt.SolidLine)
        )
        self.crosshair_h = pg.InfiniteLine(
            angle=0, 
            movable=False,
            pen=pg.mkPen('red', width=4, style=QtCore.Qt.SolidLine)
        )
        
        # Add to plot with high Z-value
        self.addItem(self.crosshair_v, ignoreBounds=True)
        self.addItem(self.crosshair_h, ignoreBounds=True)
        self.crosshair_v.setZValue(1000)
        self.crosshair_h.setZValue(1000)
        
        # Set initial position and show
        self.crosshair_v.setPos(75)
        self.crosshair_h.setPos(180)
        self.crosshair_v.show()
        self.crosshair_h.show()
        
        # Use SignalProxy pattern from Test 3 (the key to success!)
        self.mouse_proxy = pg.SignalProxy(
            self.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self._on_mouse_moved
        )
        
        print("Crosshair setup using successful Test 3 pattern")

    def wheelEvent(self, ev):
        """Fixed wheel event handler with proper zoom/pan behavior"""
        
        # Check if Ctrl key is pressed
        modifiers = ev.modifiers()
        ctrl_pressed = modifiers & QtCore.Qt.ControlModifier
        
        # Get current X range
        current_range = self.plotItem.vb.viewRange()
        x_range = current_range[0]
        
        if ctrl_pressed:
            # Ctrl+scroll: X-axis zoom only
            zoom_factor = 1.2 if ev.angleDelta().y() > 0 else 1/1.2
            
            # Calculate zoom center (center of current view)
            zoom_center_x = (x_range[0] + x_range[1]) / 2
            
            # Calculate new X range
            x_span = x_range[1] - x_range[0]
            new_x_span = x_span / zoom_factor
            
            new_x_range = [
                zoom_center_x - new_x_span/2,
                zoom_center_x + new_x_span/2
            ]
            
            # Apply X zoom only - Y will auto-adjust
            self.plotItem.vb.setXRange(*new_x_range, padding=0)
            
            print(f"X-Zoomed to: {new_x_range[0]:.1f}-{new_x_range[1]:.1f}")
            
        else:
            # Regular scroll: X-axis pan only
            x_span = x_range[1] - x_range[0]
            pan_amount = x_span * 0.15  # 15% of visible range
            
            # Pan left or right based on wheel direction
            if ev.angleDelta().y() > 0:  # Wheel up - pan left
                new_x_range = [x_range[0] - pan_amount, x_range[1] - pan_amount]
            else:  # Wheel down - pan right
                new_x_range = [x_range[0] + pan_amount, x_range[1] + pan_amount]
            
            # Apply X pan only - Y will auto-adjust
            self.plotItem.vb.setXRange(*new_x_range, padding=0)
            
            print(f"X-Panned to: {new_x_range[0]:.1f}-{new_x_range[1]:.1f}")
        
        ev.accept()

    def _on_range_changed(self):
        """Handle range changes - auto-fit Y axis to visible price data"""
        
        # Get current X range
        x_range = self.plotItem.vb.viewRange()[0]
        
        # Get price range for visible bars
        y_min, y_max = self.data_manager.get_visible_price_range(x_range[0], x_range[1])
        
        # Update Y range without triggering recursion
        self.plotItem.vb.blockSignals(True)
        self.plotItem.vb.setYRange(y_min, y_max, padding=0)
        self.plotItem.vb.blockSignals(False)
        
        print(f"Y-axis auto-fit to visible range: ${y_min:.2f} - ${y_max:.2f}")

    def _on_mouse_moved(self, evt):
        """Handle mouse movement for crosshair (Test 3 pattern)"""
        try:
            pos = evt[0]  # Get position from SignalProxy
            if self.plotItem.vb.sceneBoundingRect().contains(pos):
                mousePoint = self.plotItem.vb.mapSceneToView(pos)
                x_coord = mousePoint.x()
                y_coord = mousePoint.y()
                
                # Update crosshair position
                self.crosshair_v.setPos(x_coord)
                self.crosshair_h.setPos(y_coord)
                
                print(f"Crosshair: ({x_coord:.1f}, ${y_coord:.2f})")
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
        
        # Set initial view range to show all data
        self.setXRange(-5, len(bars) + 5)
        # Y range will be set by _on_range_changed
        
        render_time = time.time() - start_time
        print(f"Chart rendered in {render_time:.3f}s with {len(bars)} candlesticks")

    def draw_candlestick(self, bar: OHLCBar):
        """Draw a single candlestick"""
        
        x = bar.index
        
        # Enhanced validation from Step 9
        if not (np.isfinite(bar.open) and np.isfinite(bar.high) and 
                np.isfinite(bar.low) and np.isfinite(bar.close) and
                bar.high >= max(bar.open, bar.close) and 
                bar.low <= min(bar.open, bar.close)):
            return
        
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
        
        body_width = 0.7
        
        # Set colors (red/green as requested)
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

    def add_instructions(self):
        """Add text instructions"""
        instructions = pg.TextItem(
            text="FIXED: Ctrl+Wheel=X-Zoom, Wheel=X-Pan, Y=Auto-fit, Red Crosshair Active",
            color='blue',
            fill=pg.mkBrush(255, 255, 255, 200),
            border=pg.mkPen('blue', width=2),
            anchor=(0, 0)  # Top-left anchor
        )
        
        self.addItem(instructions, ignoreBounds=True)
        
        # Position at top-left
        instructions.setPos(5, 200)  # Fixed position
        instructions.setZValue(2000)

def test_fixed_zoom_pan():
    """Test Step 11: Fixed zoom/pan functionality"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Step 11: Fixed Zoom/Pan + Working Crosshair")
    win.resize(1400, 800)
    
    # Create fixed zoom/pan chart widget
    chart_widget = FixedZoomPanChartWidget()
    win.setCentralWidget(chart_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    for i in range(20):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "step11_fixed_zoom_pan_result.png")
    
    print("\n=== STEP 11 FIXED VERIFICATION ===")
    print("Required functionality:")
    print("1. Y-axis: NO zoom/scroll, auto-fits to visible price range")
    print("2. X-axis: Ctrl+wheel = zoom, wheel = pan")
    print("3. Red crosshair working and visible")
    print("4. All axis labels visible")
    print("5. Red/green candlesticks")
    print("")
    print("Test instructions:")
    print("- Move mouse over chart to see thick red crosshair")
    print("- Hold Ctrl + scroll wheel to zoom X-axis (Y auto-adjusts)")
    print("- Regular scroll wheel to pan X-axis (Y auto-adjusts)")
    print("- Y-axis should always show min/max of visible price data")
    print("")
    print("Check step11_fixed_zoom_pan_result.png for verification")
    
    return app, win

if __name__ == "__main__":
    app, win = test_fixed_zoom_pan()
    
    # Keep open for extended testing
    QtCore.QTimer.singleShot(20000, app.quit)  # 20 seconds for testing
    app.exec_()