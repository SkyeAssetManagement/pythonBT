"""
Step 1: Dashboard with built-in screenshot functionality
This solves the screenshot problem by adding PNG export directly to the dashboard
"""

import sys
import time
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from pathlib import Path

# Configure PyQtGraph for performance
pg.setConfigOptions(
    antialias=False,
    useOpenGL=True,
    imageAxisOrder='row-major',
    leftButtonPan=True,
    foreground='k',
    background='w'
)

class ScreenshotCandlestickItem(pg.GraphicsObject):
    """Fixed candlestick item with proper thin rendering"""
    
    def __init__(self, ohlc_data, visible_range=None):
        super().__init__()
        self.full_data = ohlc_data
        self.visible_range = visible_range or (0, len(ohlc_data))
        self.picture = None
        self._generate_picture()
    
    def set_visible_range(self, start_idx, end_idx):
        """Update visible range for efficient rendering"""
        self.visible_range = (max(0, start_idx), min(len(self.full_data), end_idx))
        self._generate_picture()
        self.update()
    
    def _generate_picture(self):
        """Generate THIN candlesticks (fix for fat blob issue)"""
        start_time = time.time()
        
        start_idx, end_idx = self.visible_range
        visible_data = self.full_data[start_idx:end_idx]
        
        if len(visible_data) == 0:
            return
        
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        
        # CRITICAL FIX: Calculate THIN candle width
        visible_bars = len(visible_data)
        if visible_bars <= 50:
            candle_width = 0.6    # Thin even when zoomed
        elif visible_bars <= 200:
            candle_width = 0.4    # Medium-thin
        elif visible_bars <= 1000:
            candle_width = 0.3    # Narrow
        else:
            candle_width = 0.2    # Very narrow
        
        # Apply additional thinning factor to fix blob issue
        candle_width *= 0.7  # Make 30% thinner
        
        print(f"Rendering {visible_bars} candles with width {candle_width:.3f}")
        
        # Pre-separate candles for batch rendering
        up_candles = []
        down_candles = []
        all_wicks = []
        
        for i, (x, open_price, high, low, close) in enumerate(visible_data):
            screen_x = start_idx + i
            
            # Skip invalid data
            if not all(np.isfinite([open_price, high, low, close])):
                continue
            if high <= 0 or low <= 0 or high < low:
                continue
            
            # Wick (thin vertical line)
            all_wicks.append((screen_x, low, screen_x, high))
            
            # Body (THIN rectangle)
            body_bottom = min(open_price, close)
            body_height = abs(close - open_price)
            
            # Handle doji candles
            min_height = (high - low) * 0.005
            if body_height < min_height:
                body_height = min_height
            
            candle_rect = (
                screen_x - candle_width/2,  # x (centered)
                body_bottom,                # y
                candle_width,               # width (THIN!)
                body_height                 # height
            )
            
            if close >= open_price:
                up_candles.append(candle_rect)
            else:
                down_candles.append(candle_rect)
        
        # Draw wicks first
        if all_wicks:
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            for x1, y1, x2, y2 in all_wicks:
                painter.drawLine(QtCore.QLineF(x1, y1, x2, y2))
        
        # Draw up candles (white fill)
        if up_candles:
            painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            for x, y, width, height in up_candles:
                painter.drawRect(QtCore.QRectF(x, y, width, height))
        
        # Draw down candles (red fill)
        if down_candles:
            painter.setBrush(QtGui.QBrush(QtCore.Qt.red))
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            for x, y, width, height in down_candles:
                painter.drawRect(QtCore.QRectF(x, y, width, height))
        
        painter.end()
        
        render_time = time.time() - start_time
        print(f"   Rendered in {render_time:.4f}s")
    
    def paint(self, painter, option, widget):
        if self.picture:
            painter.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        if not self.full_data:
            return QtCore.QRectF()
        
        start_idx, end_idx = self.visible_range
        visible_data = self.full_data[start_idx:end_idx]
        
        if not visible_data:
            return QtCore.QRectF()
        
        min_price = min(row[3] for row in visible_data)  # lows
        max_price = max(row[2] for row in visible_data)  # highs
        
        return QtCore.QRectF(
            start_idx, min_price,
            end_idx - start_idx, max_price - min_price
        )

class ScreenshotDashboard(QtWidgets.QMainWindow):
    """Dashboard with built-in screenshot functionality"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard with Screenshot - Thin Candlesticks Test")
        self.setGeometry(100, 100, 1400, 900)
        
        self.full_data = None
        self.candle_item = None
        self.screenshot_counter = 1
        
        self._setup_ui()
        self._load_data()
    
    def _setup_ui(self):
        """Setup UI with screenshot capability"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Info panel
        info_panel = QtWidgets.QHBoxLayout()
        self.info_label = QtWidgets.QLabel("Loading data...")
        self.screenshot_button = QtWidgets.QPushButton("Take Screenshot (S)")
        self.screenshot_button.clicked.connect(self.take_screenshot)
        
        info_panel.addWidget(self.info_label)
        info_panel.addStretch()
        info_panel.addWidget(self.screenshot_button)
        layout.addLayout(info_panel)
        
        # Chart
        self.chart = pg.PlotWidget()
        self.chart.setBackground('w')  # White background
        self.chart.showGrid(x=True, y=True, alpha=0.3)
        self.chart.setLabel('left', 'Price', color='black', size='12pt')
        self.chart.setLabel('bottom', 'Time', color='black', size='12pt')
        
        # Enable keyboard focus
        self.chart.setFocusPolicy(Qt.StrongFocus)
        
        layout.addWidget(self.chart)
        
        # Controls
        controls_text = (
            "SCREENSHOT TEST CONTROLS: "
            "S=Screenshot | 1=100 bars | 2=1K bars | 3=10K bars | "
            "Arrows=Pan/Zoom | Q=Quit"
        )
        self.controls_label = QtWidgets.QLabel(controls_text)
        self.controls_label.setStyleSheet("background: #e0f0ff; padding: 8px; font-weight: bold;")
        layout.addWidget(self.controls_label)
        
        # Range change handler
        self.chart.sigRangeChanged.connect(self.on_range_changed)
    
    def _load_data(self):
        """Load ES data or create synthetic data"""
        print("Loading data for screenshot test...")
        
        # Try to load real ES data first
        es_file = Path(__file__).parent.parent.parent / "dataRaw" / "1m" / "ES" / "Current" / "ES-NONE-1m-EST-NoPad.csv"
        
        if es_file.exists():
            print(f"Loading real ES data: {es_file}")
            df = pd.read_csv(es_file)
            
            # Convert to OHLC format (sample first 10000 for speed)
            sample_size = min(10000, len(df))
            df_sample = df.head(sample_size)
            
            ohlc_data = []
            for i, row in df_sample.iterrows():
                ohlc_data.append([
                    i,
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close'])
                ])
            
            self.full_data = ohlc_data
            self.info_label.setText(f"Real ES Data: {len(self.full_data):,} bars loaded")
            print(f"Loaded {len(self.full_data)} real ES bars")
            
        else:
            print("ES data not found, creating synthetic data...")
            # Create synthetic data
            n_bars = 1000
            np.random.seed(42)
            price = 4000.0
            ohlc_data = []
            
            for i in range(n_bars):
                open_price = price
                price_change = np.random.normal(0, 2.0)
                close_price = price + price_change
                
                high = max(open_price, close_price) + abs(np.random.normal(0, 1.0))
                low = min(open_price, close_price) - abs(np.random.normal(0, 1.0))
                
                ohlc_data.append([i, open_price, high, low, close_price])
                price = close_price
            
            self.full_data = ohlc_data
            self.info_label.setText(f"Synthetic Data: {len(self.full_data):,} bars loaded")
            print(f"Created {len(self.full_data)} synthetic bars")
        
        # Start with 100 bars
        self.display_data(100)
    
    def display_data(self, num_bars):
        """Display specific number of bars"""
        if not self.full_data:
            return
        
        print(f"\nDisplaying {num_bars:,} bars...")
        start_time = time.time()
        
        # Clear chart
        self.chart.clear()
        
        # Get data range
        total_bars = len(self.full_data)
        start_idx = max(0, total_bars - num_bars)
        display_data = self.full_data[start_idx:]
        
        # Create candlestick item
        self.candle_item = ScreenshotCandlestickItem(display_data)
        self.chart.addItem(self.candle_item)
        
        # Set view range
        if display_data:
            min_price = min(row[3] for row in display_data)  # lows
            max_price = max(row[2] for row in display_data)  # highs
            padding = (max_price - min_price) * 0.05
            
            self.chart.setXRange(start_idx, total_bars - 1, padding=0)
            self.chart.setYRange(min_price - padding, max_price + padding, padding=0)
        
        display_time = time.time() - start_time
        print(f"Display completed in {display_time:.4f}s")
        
        # Update status
        self.info_label.setText(f"{num_bars:,} bars displayed in {display_time:.4f}s")
    
    def take_screenshot(self):
        """Take screenshot using Qt's built-in functionality"""
        print(f"\nTaking screenshot #{self.screenshot_counter}...")
        
        try:
            # Get the chart widget
            chart_widget = self.chart
            
            # Create pixmap from widget
            pixmap = chart_widget.grab()
            
            # Save to file
            screenshot_file = Path(__file__).parent / f"step1_screenshot_{self.screenshot_counter:02d}.png"
            success = pixmap.save(str(screenshot_file))
            
            if success:
                print(f"[OK] Screenshot saved: {screenshot_file}")
                self.screenshot_button.setText(f"Screenshot Saved #{self.screenshot_counter}")
                self.screenshot_counter += 1
                
                # Reset button text after 2 seconds
                QtCore.QTimer.singleShot(2000, lambda: self.screenshot_button.setText("Take Screenshot (S)"))
                
            else:
                print("[X] Failed to save screenshot")
                self.screenshot_button.setText("Screenshot Failed")
                
        except Exception as e:
            print(f"Screenshot error: {e}")
            self.screenshot_button.setText("Screenshot Error")
    
    def on_range_changed(self):
        """Handle range changes for dynamic rendering"""
        if not self.candle_item:
            return
        
        view_range = self.chart.viewRange()[0]
        start_idx = int(max(0, view_range[0]))
        end_idx = int(min(len(self.full_data), view_range[1]))
        
        self.candle_item.set_visible_range(start_idx, end_idx)
    
    def keyPressEvent(self, event):
        """Handle keyboard controls"""
        key = event.key()
        
        if key == Qt.Key_S:
            self.take_screenshot()
        elif key == Qt.Key_1:
            self.display_data(100)
            QtCore.QTimer.singleShot(500, self.take_screenshot)  # Auto-screenshot
        elif key == Qt.Key_2:
            self.display_data(1000)
            QtCore.QTimer.singleShot(500, self.take_screenshot)  # Auto-screenshot
        elif key == Qt.Key_3:
            self.display_data(10000)
            QtCore.QTimer.singleShot(500, self.take_screenshot)  # Auto-screenshot
        elif key == Qt.Key_Q:
            self.close()
        elif key == Qt.Key_Left:
            self.pan_chart(-0.1)
        elif key == Qt.Key_Right:
            self.pan_chart(0.1)
        elif key == Qt.Key_Up:
            self.zoom_chart(0.8)
        elif key == Qt.Key_Down:
            self.zoom_chart(1.25)
        else:
            super().keyPressEvent(event)
    
    def pan_chart(self, factor):
        """Pan chart"""
        x_range = self.chart.viewRange()[0]
        span = x_range[1] - x_range[0]
        shift = span * factor
        self.chart.setXRange(x_range[0] + shift, x_range[1] + shift, padding=0)
        print(f"Panned chart, factor: {factor}")
    
    def zoom_chart(self, factor):
        """Zoom chart"""
        x_range = self.chart.viewRange()[0]
        center = (x_range[0] + x_range[1]) / 2
        span = (x_range[1] - x_range[0]) * factor
        self.chart.setXRange(center - span/2, center + span/2, padding=0)
        print(f"Zoomed chart, factor: {factor}")

def main():
    """Main function with screenshot testing"""
    print("="*70)
    print("STEP 1: DASHBOARD WITH SCREENSHOT FUNCTIONALITY")
    print("="*70)
    print("This tests the candlestick fix with automatic screenshots")
    print("Controls:")
    print("  S = Take screenshot manually")
    print("  1 = 100 bars + auto-screenshot")
    print("  2 = 1000 bars + auto-screenshot") 
    print("  3 = 10000 bars + auto-screenshot")
    print("  Arrow keys = Pan/zoom")
    print("  Q = Quit")
    print("="*70)
    
    app = QtWidgets.QApplication(sys.argv)
    
    dashboard = ScreenshotDashboard()
    dashboard.show()
    
    # Take initial screenshot after 1 second
    QtCore.QTimer.singleShot(1000, dashboard.take_screenshot)
    
    return app.exec_()

if __name__ == "__main__":
    main()