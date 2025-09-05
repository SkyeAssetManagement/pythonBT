"""
STEP-BY-STEP FIX for candlestick rendering
Based on visual feedback, I can see the exact issues and fix them properly
"""
import sys
import time
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPen, QBrush, QPicture, QPainter
import pyqtgraph as pg
import numpy as np


class FixedCandlestickItem(pg.GraphicsObject):
    """
    FIXED candlestick implementation based on visual debugging
    
    Issues found from screenshot:
    1. Wicks not visible (black on black?)
    2. Bodies too wide and solid black
    3. No color differentiation
    4. Bodies merge together
    """
    
    def __init__(self, data_buffer):
        super().__init__()
        print("*** CREATING FIXED CANDLESTICK ITEM ***")
        
        self.data = data_buffer
        self.picture = None
        
        # Generate the picture with FIXES
        self._generate_fixed_picture()
        
        print(f"*** FIXED CANDLESTICK: Created for {len(data_buffer)} bars ***")
    
    def _generate_fixed_picture(self):
        """Generate FIXED candlestick picture with proper rendering"""
        print("*** GENERATING FIXED CANDLESTICK PICTURE ***")
        
        if len(self.data) == 0:
            return
        
        # Create picture
        self.picture = QPicture()
        painter = QPainter(self.picture)
        
        # FIXED rendering approach
        candle_width = 0.4  # NARROWER width so they don't merge
        
        num_drawn = 0
        
        for i in range(len(self.data)):
            x = i
            open_price = self.data.open[i]
            high = self.data.high[i]
            low = self.data.low[i]
            close = self.data.close[i]
            
            # Skip invalid data
            if not all(np.isfinite([open_price, high, low, close])):
                continue
            
            # FIX 1: Draw wick FIRST with VISIBLE color and width
            painter.setPen(QPen(Qt.black, 2))  # Thicker black line for visibility
            from PyQt5.QtCore import QPointF
            painter.drawLine(QPointF(float(x), float(low)), QPointF(float(x), float(high)))  # Vertical line from low to high
            
            # FIX 2: Calculate body properly
            body_bottom = min(open_price, close)
            body_top = max(open_price, close)
            body_height = body_top - body_bottom
            
            # FIX 3: Ensure minimum visible height
            if body_height < 0.5:  # Minimum height for visibility
                body_center = (open_price + close) / 2
                body_height = 0.5
                body_bottom = body_center - body_height / 2
                body_top = body_bottom + body_height
            
            # FIX 4: Create rectangle with proper positioning
            rect_x = x - candle_width / 2
            rect_y = body_bottom
            rect_width = candle_width
            rect_height = body_height
            
            # FIX 5: Set colors EXPLICITLY and CORRECTLY
            if close >= open_price:
                # UP candle: WHITE body with BLACK border
                painter.setBrush(QBrush(Qt.white))
                painter.setPen(QPen(Qt.black, 1))
                print(f"  Bar {i}: UP candle (white)")
            else:
                # DOWN candle: RED body with BLACK border
                painter.setBrush(QBrush(Qt.red))
                painter.setPen(QPen(Qt.black, 1))
                print(f"  Bar {i}: DOWN candle (red)")
            
            # FIX 6: Draw rectangle with explicit coordinates
            from PyQt5.QtCore import QRectF
            rect = QRectF(float(rect_x), float(rect_y), float(rect_width), float(rect_height))
            painter.drawRect(rect)
            
            num_drawn += 1
        
        painter.end()
        
        print(f"*** FIXED CANDLESTICK: Drew {num_drawn} candlesticks ***")
    
    def paint(self, painter, option, widget):
        """Paint the candlesticks"""
        if self.picture:
            painter.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        """Return bounding rectangle"""
        if len(self.data) == 0:
            from PyQt5.QtCore import QRectF
            return QRectF()
        
        min_price = float(np.min(self.data.low))
        max_price = float(np.max(self.data.high))
        
        from PyQt5.QtCore import QRectF
        return QRectF(0, min_price, len(self.data), max_price - min_price)


def test_fixed_candlesticks():
    """Test the FIXED candlestick implementation"""
    
    print("=== TESTING FIXED CANDLESTICKS ===")
    
    # Create Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        # Import data structure
        from src.dashboard.data_structures import ChartDataBuffer
        import pyautogui
        
        print("SUCCESS: Imports loaded")
        
        # Create CLEAR test data with obvious patterns
        n_bars = 10  # Even smaller for crystal clear debugging
        timestamps = np.arange(n_bars, dtype=np.int64) * int(60 * 1e9)
        
        # Create ALTERNATING up/down pattern for clear visual verification
        base_price = 100.0  # Use round numbers for clarity
        opens = np.full(n_bars, base_price)
        closes = np.array([base_price + (5 if i % 2 == 0 else -5) for i in range(n_bars)])
        highs = np.maximum(opens, closes) + 3  # Clear wicks
        lows = np.minimum(opens, closes) - 3   # Clear wicks
        volumes = np.full(n_bars, 1000.0)
        
        print(f"SUCCESS: Created {n_bars} test bars")
        print(f"  Pattern: Alternating up/down candles")
        print(f"  Opens: all {base_price}")
        print(f"  Closes: {closes}")
        print(f"  Expected: White, Red, White, Red, White...")
        
        # Create data buffer
        data_buffer = ChartDataBuffer(
            timestamps=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            volume=volumes
        )
        
        print("SUCCESS: Data buffer created")
        
        # Create plot widget with better settings
        plot_widget = pg.PlotWidget()
        plot_widget.setWindowTitle("FIXED CANDLESTICKS - Should Show Proper OHLC")
        plot_widget.setLabel('left', 'Price')
        plot_widget.setLabel('bottom', 'Bar Index')
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setBackground('w')  # White background
        plot_widget.resize(1000, 700)  # Larger window
        
        print("SUCCESS: Plot widget created")
        
        # Create FIXED candlestick item
        print("Creating FIXED CandlestickItem...")
        candle_item = FixedCandlestickItem(data_buffer)
        plot_widget.addItem(candle_item)
        
        print("SUCCESS: FIXED candlestick item added")
        
        # Show window
        plot_widget.show()
        plot_widget.raise_()
        plot_widget.activateWindow()
        
        print("SUCCESS: Window shown")
        
        # Setup screenshot directory
        screenshot_dir = Path("debug_screenshots")
        screenshot_dir.mkdir(exist_ok=True)
        
        def take_fixed_screenshot(suffix=""):
            try:
                timestamp = time.strftime("%H%M%S")
                filename = f"FIXED_candlestick_{timestamp}{suffix}.png"
                filepath = screenshot_dir / filename
                
                screenshot = pyautogui.screenshot()
                screenshot.save(str(filepath))
                print(f"FIXED SCREENSHOT: {filepath}")
                return str(filepath)
            except Exception as e:
                print(f"SCREENSHOT FAILED: {e}")
                return None
        
        # Take screenshots to verify the fix
        QTimer.singleShot(2000, lambda: take_fixed_screenshot("_2s"))
        QTimer.singleShot(4000, lambda: take_fixed_screenshot("_4s"))
        QTimer.singleShot(6000, lambda: take_fixed_screenshot("_6s"))
        
        # Close after 12 seconds
        QTimer.singleShot(12000, app.quit)
        
        print("RUNNING: FIXED candlesticks test (12 seconds)")
        print("  Should show alternating white/red candles with black wicks")
        print("  Screenshots will verify the fix worked")
        
        app.exec_()
        
        print("COMPLETE: Fixed candlesticks test finished")
        print("Check debug_screenshots/ for FIXED_candlestick_*.png files")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("FIXED Candlestick Rendering Test")
    print("This should show proper OHLC candlesticks instead of black blobs")
    print()
    
    test_fixed_candlesticks()