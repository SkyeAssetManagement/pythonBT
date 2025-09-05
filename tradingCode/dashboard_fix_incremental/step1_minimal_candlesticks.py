"""
Step 1: Minimal candlestick dashboard with proper thin candlesticks and speed measurement
Focus: Fix the fat blob candlesticks issue from the broken screenshot
"""

import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import pandas as pd

# Configure PyQtGraph for performance
pg.setConfigOptions(
    antialias=False,
    useOpenGL=True,
    imageAxisOrder='row-major',
    leftButtonPan=True,
    foreground='k',
    background='w'
)

class SimpleCandlestickItem(pg.GraphicsObject):
    """Ultra-simple candlestick item with proper thin rendering"""
    
    def __init__(self, ohlc_data):
        super().__init__()
        self.data = ohlc_data
        self.picture = None
        self._generate_picture()
    
    def _generate_picture(self):
        """Generate candlestick picture with THIN candlesticks"""
        start_time = time.time()
        
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)  # Faster rendering
        
        # CRITICAL: Use thin pens for proper candlestick appearance
        wick_pen = QtGui.QPen(QtCore.Qt.black, 1)  # Thin black wicks
        up_pen = QtGui.QPen(QtCore.Qt.black, 1)    # Thin black borders
        down_pen = QtGui.QPen(QtCore.Qt.black, 1)  # Thin black borders
        
        # Proper brush colors
        up_brush = QtGui.QBrush(QtCore.Qt.white)   # White for up candles
        down_brush = QtGui.QBrush(QtCore.Qt.red)   # Red for down candles
        
        # Calculate THIN candle width
        candle_width = 0.6  # THIN width for proper appearance
        
        for i in range(len(self.data)):
            x, open_price, high, low, close = self.data[i]
            
            # Skip invalid data
            if not all(np.isfinite([open_price, high, low, close])):
                continue
                
            # 1. Draw wick (thin vertical line from low to high)
            painter.setPen(wick_pen)
            wick_line = QtCore.QLineF(x, low, x, high)
            painter.drawLine(wick_line)
            
            # 2. Draw body (thin rectangle)
            body_top = max(open_price, close)
            body_bottom = min(open_price, close)
            body_height = abs(close - open_price)
            
            # Ensure minimum height for doji candles
            if body_height < (high - low) * 0.01:
                body_height = (high - low) * 0.01
            
            # Create THIN rectangle
            body_rect = QtCore.QRectF(
                x - candle_width/2,  # x position (centered)
                body_bottom,         # y position (bottom)
                candle_width,        # THIN width
                body_height          # actual height
            )
            
            # Set colors and draw
            if close >= open_price:  # Up candle
                painter.setBrush(up_brush)
                painter.setPen(up_pen)
            else:  # Down candle
                painter.setBrush(down_brush)
                painter.setPen(down_pen)
            
            painter.drawRect(body_rect)
        
        painter.end()
        
        render_time = time.time() - start_time
        print(f"   Rendered {len(self.data)} candlesticks in {render_time:.3f}s")
    
    def paint(self, painter, option, widget):
        """Paint the candlesticks"""
        if self.picture:
            painter.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        """Return bounding rectangle"""
        if len(self.data) == 0:
            return QtCore.QRectF()
        
        min_price = min(row[3] for row in self.data)  # low prices
        max_price = max(row[2] for row in self.data)  # high prices
        
        return QtCore.QRectF(
            0, min_price,
            len(self.data), max_price - min_price
        )

class MinimalDashboard(QtWidgets.QMainWindow):
    """Minimal dashboard focused on proper candlestick display"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Step 1: Minimal Candlestick Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Create chart
        self.chart = pg.PlotWidget()
        self.chart.setBackground('w')  # White background
        self.chart.showGrid(x=True, y=True)  # Show grid
        self.chart.setLabel('left', 'Price', color='black')
        self.chart.setLabel('bottom', 'Time', color='black')
        
        layout.addWidget(self.chart)
        
        # Add status label for timing
        self.status_label = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Store for cleanup
        self.candle_item = None
    
    def load_sample_data(self, num_bars=100):
        """Load sample OHLC data for testing"""
        print(f"Loading {num_bars} sample bars...")
        start_time = time.time()
        
        # Generate realistic sample data
        np.random.seed(42)
        
        # Start price
        price = 100.0
        data = []
        
        for i in range(num_bars):
            # Random walk for realistic price movement
            price_change = np.random.normal(0, 0.5)
            
            # OHLC calculation
            open_price = price
            close_price = price + price_change
            
            # High/Low with some spread
            high = max(open_price, close_price) + abs(np.random.normal(0, 0.2))
            low = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
            
            data.append([i, open_price, high, low, close_price])
            price = close_price
        
        load_time = time.time() - start_time
        print(f"   Generated {num_bars} bars in {load_time:.3f}s")
        
        return data
    
    def display_candlesticks(self, ohlc_data):
        """Display candlesticks on chart"""
        print(f"Displaying {len(ohlc_data)} candlesticks...")
        start_time = time.time()
        
        # Clear existing items
        self.chart.clear()
        
        # Create and add candlestick item
        self.candle_item = SimpleCandlestickItem(ohlc_data)
        self.chart.addItem(self.candle_item)
        
        # Set appropriate view range
        if ohlc_data:
            min_price = min(row[3] for row in ohlc_data)  # low prices
            max_price = max(row[2] for row in ohlc_data)  # high prices
            price_padding = (max_price - min_price) * 0.05
            
            self.chart.setXRange(0, len(ohlc_data)-1)
            self.chart.setYRange(min_price - price_padding, max_price + price_padding)
        
        display_time = time.time() - start_time
        print(f"   Displayed candlesticks in {display_time:.3f}s")
        
        self.status_label.setText(f"Loaded {len(ohlc_data)} bars - Render: {display_time:.3f}s")
    
    def keyPressEvent(self, event):
        """Handle keyboard controls"""
        if event.key() == Qt.Key_1:
            self.test_small_dataset()
        elif event.key() == Qt.Key_2:
            self.test_medium_dataset()
        elif event.key() == Qt.Key_3:
            self.test_large_dataset()
        elif event.key() == Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)
    
    def test_small_dataset(self):
        """Test with 100 bars"""
        print("\n=== Testing Small Dataset (100 bars) ===")
        data = self.load_sample_data(100)
        self.display_candlesticks(data)
    
    def test_medium_dataset(self):
        """Test with 1000 bars"""
        print("\n=== Testing Medium Dataset (1000 bars) ===")
        data = self.load_sample_data(1000)
        self.display_candlesticks(data)
    
    def test_large_dataset(self):
        """Test with 10000 bars"""
        print("\n=== Testing Large Dataset (10000 bars) ===")
        data = self.load_sample_data(10000)
        self.display_candlesticks(data)

def main():
    """Main function to run the minimal dashboard"""
    print("="*60)
    print("STEP 1: MINIMAL CANDLESTICK DASHBOARD")
    print("="*60)
    print("Objective: Fix fat blob candlesticks with proper thin rendering")
    print("Controls:")
    print("  1 = Test 100 bars")
    print("  2 = Test 1000 bars") 
    print("  3 = Test 10000 bars")
    print("  Q = Quit")
    print("="*60)
    
    app = QtWidgets.QApplication(sys.argv)
    
    # Create and show dashboard
    dashboard = MinimalDashboard()
    dashboard.show()
    
    # Start with small test
    dashboard.test_small_dataset()
    
    return app.exec_()

if __name__ == "__main__":
    main()