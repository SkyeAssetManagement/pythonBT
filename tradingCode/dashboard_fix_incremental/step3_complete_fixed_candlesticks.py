"""
Step 3: Complete solution with fixed thin candlesticks for integration
Focus: Production-ready candlestick widget that fixes the blob issue
"""

import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Tuple, List, Optional

# Configure PyQtGraph for performance and proper appearance
pg.setConfigOptions(
    antialias=False,
    useOpenGL=True,
    imageAxisOrder='row-major',
    leftButtonPan=True,
    foreground='k',
    background='w'
)

class FixedCandlestickItem(pg.GraphicsObject):
    """
    FIXED candlestick renderer that produces proper thin candlesticks 
    instead of fat blobs. This is the solution to the display problem.
    """
    
    def __init__(self, ohlc_data, max_initial_render=1000):
        super().__init__()
        self.full_data = np.array(ohlc_data) if ohlc_data else np.array([])
        self.max_initial_render = max_initial_render
        self.visible_range = None
        self.picture = None
        self.last_render_time = 0
        
        # Generate initial picture with limited data for fast startup
        if len(self.full_data) > 0:
            self._generate_picture()
    
    def _generate_picture(self, view_range: Optional[Tuple[int, int]] = None):
        """Generate candlestick picture with PROPER thin rendering"""
        start_time = time.time()
        
        # Determine what data to render
        if view_range:
            start_idx = max(0, int(view_range[0] - 50))  # Add buffer
            end_idx = min(len(self.full_data), int(view_range[1] + 50))
        else:
            # Initial render - limit to recent data for speed
            if len(self.full_data) > self.max_initial_render:
                start_idx = len(self.full_data) - self.max_initial_render
                end_idx = len(self.full_data)
            else:
                start_idx = 0
                end_idx = len(self.full_data)
        
        if start_idx >= end_idx or len(self.full_data) == 0:
            return
        
        data_slice = self.full_data[start_idx:end_idx]
        self.visible_range = (start_idx, end_idx)
        
        # Create picture
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        
        # CRITICAL FIX: Calculate proper candle width based on zoom level
        visible_bars = len(data_slice)
        
        if visible_bars <= 50:
            candle_width = 0.8      # Wide for very zoomed view
        elif visible_bars <= 200:
            candle_width = 0.6      # Medium for normal view
        elif visible_bars <= 1000:
            candle_width = 0.4      # Narrow for overview
        else:
            candle_width = 0.2      # Very narrow for wide overview
        
        # PERFORMANCE: Pre-separate candles by type for batch rendering
        up_candles = []
        down_candles = []
        all_wicks = []
        
        for i, (x, open_price, high, low, close) in enumerate(data_slice):
            # Adjust X coordinate to screen position
            screen_x = start_idx + i
            
            # Skip invalid data
            if not all(np.isfinite([open_price, high, low, close])):
                continue
            if high <= 0 or low <= 0 or high < low:
                continue
            
            # FIXED: Wick coordinates (thin vertical line from low to high)
            all_wicks.append((screen_x, low, screen_x, high))
            
            # FIXED: Body coordinates with proper height calculation
            body_bottom = min(open_price, close)
            body_height = abs(close - open_price)
            
            # Handle doji candles (open == close)
            min_height = (high - low) * 0.005  # 0.5% of total range
            if body_height < min_height:
                body_height = min_height
            
            # Create THIN rectangle coordinates
            candle_rect = (
                screen_x - candle_width/2,  # x (centered)
                body_bottom,                # y (bottom of body)
                candle_width,               # width (THIN!)
                body_height                 # height
            )
            
            # Separate by candle direction
            if close >= open_price:  # Up candle
                up_candles.append(candle_rect)
            else:  # Down candle
                down_candles.append(candle_rect)
        
        # BATCH RENDERING for maximum performance
        
        # 1. Draw all wicks first (thin black lines)
        if all_wicks:
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            for x1, y1, x2, y2 in all_wicks:
                painter.drawLine(QtCore.QLineF(x1, y1, x2, y2))
        
        # 2. Draw up candles (white fill, black outline)
        if up_candles:
            painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            for x, y, width, height in up_candles:
                painter.drawRect(QtCore.QRectF(x, y, width, height))
        
        # 3. Draw down candles (red fill, black outline)
        if down_candles:
            painter.setBrush(QtGui.QBrush(QtCore.Qt.red))
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            for x, y, width, height in down_candles:
                painter.drawRect(QtCore.QRectF(x, y, width, height))
        
        painter.end()
        
        # Performance tracking
        self.last_render_time = time.time() - start_time
        print(f"FIXED CANDLESTICKS: Rendered {len(data_slice)} bars in {self.last_render_time:.4f}s "
              f"(width: {candle_width:.2f})")
    
    def update_view_range(self, view_range: Tuple[int, int]):
        """Update visible range and re-render if needed"""
        if self.visible_range is None or view_range != self.visible_range:
            self._generate_picture(view_range)
            self.update()
    
    def paint(self, painter, option, widget):
        """Paint the candlesticks"""
        if self.picture:
            painter.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        """Return bounding rectangle"""
        if len(self.full_data) == 0:
            return QtCore.QRectF()
        
        # Use visible range if available, otherwise full data
        if self.visible_range:
            start_idx, end_idx = self.visible_range
            data_slice = self.full_data[start_idx:end_idx]
        else:
            data_slice = self.full_data
            start_idx = 0
        
        if len(data_slice) == 0:
            return QtCore.QRectF()
        
        min_price = np.min(data_slice[:, 3])  # lows
        max_price = np.max(data_slice[:, 2])  # highs
        
        return QtCore.QRectF(
            start_idx, min_price,
            len(data_slice), max_price - min_price
        )

class TestDashboard(QtWidgets.QMainWindow):
    """Test dashboard to verify the candlestick fix"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STEP 3: FIXED CANDLESTICKS - No More Blobs!")
        self.setGeometry(100, 100, 1400, 900)
        
        self.full_data = None
        self.candle_item = None
        
        self._setup_ui()
        self._generate_test_data()
    
    def _setup_ui(self):
        """Setup UI"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Status bar
        self.status_label = QtWidgets.QLabel("Ready to test fixed candlesticks")
        layout.addWidget(self.status_label)
        
        # Chart
        self.chart = pg.PlotWidget()
        self.chart.setBackground('w')  # White background like AmiBroker
        self.chart.showGrid(x=True, y=True, alpha=0.3)
        self.chart.setLabel('left', 'Price', color='black')
        self.chart.setLabel('bottom', 'Time', color='black')
        
        # Enable keyboard focus
        self.chart.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        layout.addWidget(self.chart)
        
        # Controls
        controls = QtWidgets.QLabel(
            "FIXED CANDLESTICKS TEST - Controls: 1=100 bars | 2=1K bars | 3=10K bars | "
            "4=100K bars | Arrow Keys=Pan | +/-=Zoom | Q=Quit"
        )
        controls.setStyleSheet("background: #e0f0e0; padding: 8px; font-weight: bold;")
        layout.addWidget(controls)
        
        # Connect range change handler
        self.chart.sigRangeChanged.connect(self._on_range_changed)
    
    def _generate_test_data(self):
        """Generate test data to verify the fix"""
        print("Generating test data...")
        
        # Generate realistic price data
        np.random.seed(42)
        bars = 100000  # Large dataset to test performance
        
        price = 4000.0  # Starting price (like ES)
        data = []
        
        for i in range(bars):
            # Random walk
            price_change = np.random.normal(0, 2.0)
            
            open_price = price
            close_price = price + price_change
            
            # Add some volatility for high/low
            volatility = abs(np.random.normal(0, 1.0))
            high = max(open_price, close_price) + volatility
            low = min(open_price, close_price) - volatility
            
            data.append([i, open_price, high, low, close_price])
            price = close_price
        
        self.full_data = data
        print(f"Generated {len(data):,} test bars")
        
        # Start with 1000 bars
        self._display_data(1000)
    
    def _display_data(self, num_bars):
        """Display specific number of bars"""
        if not self.full_data:
            return
        
        print(f"\\nDisplaying {num_bars:,} bars with FIXED candlesticks...")
        start_time = time.time()
        
        # Clear chart
        self.chart.clear()
        
        # Get data range
        total_bars = len(self.full_data)
        start_idx = max(0, total_bars - num_bars)
        display_data = self.full_data[start_idx:]
        
        # Create FIXED candlestick item
        self.candle_item = FixedCandlestickItem(display_data)
        self.chart.addItem(self.candle_item)
        
        # Set view range
        if display_data:
            min_price = min(row[3] for row in display_data)  # lows
            max_price = max(row[2] for row in display_data)  # highs
            padding = (max_price - min_price) * 0.05
            
            self.chart.setXRange(start_idx, total_bars - 1, padding=0)
            self.chart.setYRange(min_price - padding, max_price + padding, padding=0)
        
        display_time = time.time() - start_time
        
        self.status_label.setText(
            f"FIXED CANDLESTICKS: {num_bars:,} bars displayed in {display_time:.4f}s - "
            f"Should be THIN, not fat blobs!"
        )
        
        print(f"Display completed in {display_time:.4f}s")
    
    def _on_range_changed(self):
        """Handle range changes for dynamic rendering"""
        if not self.candle_item:
            return
        
        view_range = self.chart.viewRange()[0]
        start_idx = int(max(0, view_range[0]))
        end_idx = int(min(len(self.full_data), view_range[1]))
        
        self.candle_item.update_view_range((start_idx, end_idx))
    
    def keyPressEvent(self, event):
        """Handle keyboard controls"""
        key = event.key()
        
        if key == QtCore.Qt.Key_1:
            self._display_data(100)
        elif key == QtCore.Qt.Key_2:
            self._display_data(1000)
        elif key == QtCore.Qt.Key_3:
            self._display_data(10000)
        elif key == QtCore.Qt.Key_4:
            self._display_data(100000)
        elif key == QtCore.Qt.Key_Q:
            self.close()
        elif key == QtCore.Qt.Key_Left:
            self._pan(-0.1)
        elif key == QtCore.Qt.Key_Right:
            self._pan(0.1)
        elif key == QtCore.Qt.Key_Plus:
            self._zoom(0.8)
        elif key == QtCore.Qt.Key_Minus:
            self._zoom(1.25)
        else:
            super().keyPressEvent(event)
    
    def _pan(self, factor):
        """Pan chart"""
        x_range = self.chart.viewRange()[0]
        span = x_range[1] - x_range[0]
        shift = span * factor
        self.chart.setXRange(x_range[0] + shift, x_range[1] + shift, padding=0)
    
    def _zoom(self, factor):
        """Zoom chart"""
        x_range = self.chart.viewRange()[0]
        center = (x_range[0] + x_range[1]) / 2
        span = (x_range[1] - x_range[0]) * factor
        self.chart.setXRange(center - span/2, center + span/2, padding=0)

def main():
    """Test the fixed candlesticks"""
    print("="*80)
    print("STEP 3: COMPLETE CANDLESTICK FIX TEST")
    print("="*80)
    print("This tests the FIXED candlestick renderer that should produce")
    print("THIN candlesticks instead of fat blobs.")
    print("="*80)
    
    app = QtWidgets.QApplication(sys.argv)
    
    dashboard = TestDashboard()
    dashboard.show()
    
    return app.exec_()

if __name__ == "__main__":
    main()