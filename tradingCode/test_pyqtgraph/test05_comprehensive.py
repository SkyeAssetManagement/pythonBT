"""
Test 5: Comprehensive test combining all working patterns
Goal: Combine successful axis, crosshair, and text overlay patterns
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

class ComprehensiveTradingChart(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        
        # Setup basic plot with white background
        self.setBackground('white')
        self.setLabel('left', 'Price', color='black', size='14pt')
        self.setLabel('bottom', 'Time', color='black', size='14pt')
        self.setLabel('top', 'Trading Chart - All Features', color='black', size='16pt')
        
        # Configure ALL axes for maximum visibility (from Test 2 success)
        for axis_name in ['left', 'right', 'bottom', 'top']:
            self.showAxis(axis_name, True)
            axis = self.getAxis(axis_name)
            axis.setPen(pg.mkPen('black', width=3))  # Thick axis lines
            axis.setTextPen(pg.mkPen('black', width=2))  # Thick text
            axis.setStyle(showValues=True, tickLength=10)
            
            # Set appropriate widths/heights for visibility
            if axis_name in ['left', 'right']:
                axis.setWidth(100)  # Wide Y-axes
            else:
                axis.setHeight(60)  # Tall X-axes
            
            axis.show()  # Force show
        
        # Create candlestick-like trading data
        self.setup_trading_data()
        
        # Setup crosshairs (from Test 3 success)
        self.setup_crosshairs()
        
        # Setup OHLCV text display (from Test 4 success)
        self.setup_ohlcv_display()
        
        print("Comprehensive trading chart created with all features")
        
    def setup_trading_data(self):
        """Create realistic trading data"""
        # Generate OHLCV-like data
        self.num_bars = 100
        self.times = np.arange(self.num_bars)
        
        # Simulate price movement
        base_price = 100.0
        price_changes = np.cumsum(np.random.randn(self.num_bars) * 0.5)
        close_prices = base_price + price_changes
        
        # Create OHLC data
        self.ohlc_data = []
        for i in range(self.num_bars):
            close = close_prices[i]
            high = close + abs(np.random.randn() * 0.8)
            low = close - abs(np.random.randn() * 0.8)
            open_price = close + np.random.randn() * 0.3
            volume = 1000 + abs(np.random.randn() * 500)
            
            self.ohlc_data.append({
                'time': i,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        # Plot the close prices as a line
        closes = [bar['close'] for bar in self.ohlc_data]
        self.plot(self.times, closes, pen='blue', name='Close Price')
        
        # Add some trade arrows
        buy_times = [20, 40, 70]
        sell_times = [30, 60, 90]
        
        buy_prices = [self.ohlc_data[i]['close'] for i in buy_times]
        sell_prices = [self.ohlc_data[i]['close'] for i in sell_times]
        
        # Green up arrows for buys
        self.buy_scatter = pg.ScatterPlotItem(
            pos=list(zip(buy_times, buy_prices)),
            symbol='t1',  # Up triangle
            brush='#00CC00',
            size=15,
            pen=pg.mkPen('darkgreen', width=2)
        )
        self.addItem(self.buy_scatter)
        
        # Red down arrows for sells  
        self.sell_scatter = pg.ScatterPlotItem(
            pos=list(zip(sell_times, sell_prices)),
            symbol='t',  # Down triangle
            brush='#FF0000', 
            size=15,
            pen=pg.mkPen('darkred', width=2)
        )
        self.addItem(self.sell_scatter)
        
        print(f"Created {self.num_bars} bars of trading data with buy/sell signals")
        
    def setup_crosshairs(self):
        """Setup crosshairs using Test 3 success pattern"""
        # Create thick, bright crosshairs
        self.vLine = pg.InfiniteLine(
            angle=90, 
            movable=False,
            pen=pg.mkPen('red', width=4, style=QtCore.Qt.SolidLine)
        )
        self.hLine = pg.InfiniteLine(
            angle=0, 
            movable=False,
            pen=pg.mkPen('red', width=4, style=QtCore.Qt.SolidLine)
        )
        
        # Add to plot with high Z-value
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)
        self.vLine.setZValue(1000)
        self.hLine.setZValue(1000)
        
        # Set initial position
        mid_time = self.num_bars // 2
        mid_price = self.ohlc_data[mid_time]['close']
        self.vLine.setPos(mid_time)
        self.hLine.setPos(mid_price)
        
        # Force show
        self.vLine.show()
        self.hLine.show()
        
        # Connect mouse events using SignalProxy (Test 3 pattern)
        self.proxy = pg.SignalProxy(
            self.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self.mouseMoved
        )
        
        print("Crosshairs setup complete with thick red lines")
        
    def setup_ohlcv_display(self):
        """Setup OHLCV display using Test 4 success pattern"""
        # Create OHLCV text display at top of chart
        self.ohlcv_text = pg.TextItem(
            text="OHLCV: Loading...",
            color='white',
            fill=pg.mkBrush(0, 0, 0, 180),  # Semi-transparent black background
            border=pg.mkPen('white', width=2),
            anchor=(0, 0)  # Top-left anchor
        )
        
        # Add to plot with high Z-value and ignore bounds
        self.addItem(self.ohlcv_text, ignoreBounds=True)
        self.ohlcv_text.setZValue(2000)  # Higher than crosshairs
        
        # Position at top-left of chart
        view_range = self.viewRange()
        x_min = view_range[0][0]
        y_max = view_range[1][1]
        self.ohlcv_text.setPos(x_min + 2, y_max - 2)
        
        # Update with current data
        self.update_ohlcv_display(self.num_bars // 2)  # Start with middle bar
        
        print("OHLCV display setup complete")
        
    def update_ohlcv_display(self, bar_index):
        """Update OHLCV display for given bar"""
        if 0 <= bar_index < len(self.ohlc_data):
            bar = self.ohlc_data[bar_index]
            
            ohlcv_text = (
                f"Time: {bar['time']:3.0f} | "
                f"O: {bar['open']:6.2f} | "
                f"H: {bar['high']:6.2f} | "
                f"L: {bar['low']:6.2f} | "
                f"C: {bar['close']:6.2f} | "
                f"V: {bar['volume']:4.0f}"
            )
            
            self.ohlcv_text.setText(ohlcv_text)
            
    def mouseMoved(self, evt):
        """Handle mouse movement for crosshairs and OHLCV update"""
        pos = evt[0]  # Get position from SignalProxy
        if self.plotItem.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.plotItem.vb.mapSceneToView(pos)
            
            # Update crosshair position
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            
            # Update OHLCV display for nearest bar
            bar_index = int(round(mousePoint.x()))
            self.update_ohlcv_display(bar_index)
            
            # Update crosshair position display
            print(f"Crosshair: ({mousePoint.x():.1f}, {mousePoint.y():.2f}) | Bar: {bar_index}")

def test_comprehensive():
    """Test comprehensive functionality combining all successful patterns"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Test 5: Comprehensive Trading Chart - All Features")
    win.resize(1000, 700)
    
    # Create comprehensive widget
    chart_widget = ComprehensiveTradingChart()
    win.setCentralWidget(chart_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    for i in range(15):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "test05_comprehensive_result.png")
    
    print("\n=== TEST 5 SUMMARY ===")
    print("Expected: Complete trading chart with:")
    print("  - Visible axes on all sides with labels")
    print("  - Thick red crosshairs responding to mouse")
    print("  - OHLCV data display at top updating with mouse position")
    print("  - Green up arrows (buys) and red down arrows (sells)")
    print("  - Price line chart with trading data")
    print("Check test05_comprehensive_result.png for actual result")
    print("Move mouse over chart to test all interactive features")
    
    return app, win

if __name__ == "__main__":
    app, win = test_comprehensive()
    
    # Keep open for interaction testing
    QtCore.QTimer.singleShot(15000, app.quit)  # 15 seconds for testing
    app.exec_()