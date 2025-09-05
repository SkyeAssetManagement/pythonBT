"""
Step 2: Add simple line chart with test data
Goal: Build on Step 1 by adding a blue price line with realistic movement
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

class LineChartWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        
        # Set white background
        self.setBackground('white')
        
        # Set axis labels
        self.setLabel('left', 'Price ($)', color='black', size='14pt')
        self.setLabel('right', 'Price ($)', color='black', size='14pt')
        self.setLabel('bottom', 'Time (bars)', color='black', size='14pt')
        self.setLabel('top', 'Trading Chart - Step 2', color='black', size='16pt')
        
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
        
        # Generate realistic price data
        self.create_price_data()
        
        # Add price line to chart
        self.add_price_line()
        
        print("Step 2: Line chart widget created with price data")

    def create_price_data(self):
        """Generate realistic price movement data"""
        num_bars = 100
        
        # Time axis (bar indices)
        self.x_data = np.arange(num_bars)
        
        # Generate realistic price movement using random walk
        base_price = 150.0
        daily_returns = np.random.normal(0.001, 0.02, num_bars)  # 0.1% mean, 2% volatility
        price_changes = np.cumsum(daily_returns)
        
        # Create price series
        self.y_data = base_price * (1 + price_changes)
        
        print(f"Generated {num_bars} price points from ${self.y_data.min():.2f} to ${self.y_data.max():.2f}")
        
    def add_price_line(self):
        """Add price line to the chart"""
        # Create blue price line
        pen = pg.mkPen('blue', width=2)
        self.price_line = self.plot(
            self.x_data, 
            self.y_data,
            pen=pen,
            name='Close Price'
        )
        
        # Set appropriate ranges based on data
        x_padding = 2
        y_range = self.y_data.max() - self.y_data.min()
        y_padding = y_range * 0.05  # 5% padding
        
        self.setXRange(0 - x_padding, len(self.x_data) - 1 + x_padding)
        self.setYRange(self.y_data.min() - y_padding, self.y_data.max() + y_padding)
        
        print(f"Price line added: X range 0-{len(self.x_data)-1}, Y range ${self.y_data.min():.2f}-${self.y_data.max():.2f}")

def test_step2():
    """Test Step 2: Line chart with price data"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Step 2: Line Chart with Price Data")
    win.resize(1000, 700)
    
    # Create line chart widget
    chart_widget = LineChartWidget()
    win.setCentralWidget(chart_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    for i in range(10):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "step02_line_chart_result.png")
    
    print("\n=== STEP 2 VERIFICATION ===")
    print("Required elements to verify in screenshot:")
    print("- All Step 1 elements (white background, visible axes, labels)")
    print("- Blue price line showing realistic stock price movement")
    print("- Price line spans full width of chart area")
    print("- Y-axis shows price values around $140-160 range")
    print("- X-axis shows 0-100 bar indices")
    print("- Smooth price movement with some volatility")
    print("\nCheck step02_line_chart_result.png for verification")
    
    return app, win

if __name__ == "__main__":
    app, win = test_step2()
    
    # Keep open for inspection
    QtCore.QTimer.singleShot(8000, app.quit)
    app.exec_()