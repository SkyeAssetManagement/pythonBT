"""
Simple test of FIXED candlesticks with proper scaling
"""
import sys
import time
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPen, QBrush, QPicture, QPainter
import pyqtgraph as pg
import numpy as np


class SuperSimpleCandlestickItem(pg.GraphicsObject):
    """Ultra-simple fixed candlestick item for debugging"""
    
    def __init__(self, ohlc_data):
        super().__init__()
        self.ohlc_data = ohlc_data
        self.picture = None
        self._generate_picture()
    
    def _generate_picture(self):
        """Generate candlestick picture"""
        self.picture = QPicture()
        painter = QPainter(self.picture)
        
        from PyQt5.QtCore import QPointF, QRectF
        
        for i, (o, h, l, c) in enumerate(self.ohlc_data):
            x = float(i)
            
            # Draw wick (thin black line)
            painter.setPen(QPen(Qt.black, 1))
            painter.drawLine(QPointF(x, l), QPointF(x, h))
            
            # Draw body
            body_bottom = min(o, c)
            body_top = max(o, c)
            body_height = max(body_top - body_bottom, 0.1)  # Minimum height
            
            if c >= o:
                # Up candle - white body
                painter.setBrush(QBrush(Qt.white))
                painter.setPen(QPen(Qt.black, 1))
            else:
                # Down candle - red body
                painter.setBrush(QBrush(Qt.red))
                painter.setPen(QPen(Qt.black, 1))
            
            rect = QRectF(x - 0.3, body_bottom, 0.6, body_height)
            painter.drawRect(rect)
        
        painter.end()
    
    def paint(self, painter, option, widget):
        if self.picture:
            painter.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        from PyQt5.QtCore import QRectF
        if not self.ohlc_data:
            return QRectF()
        
        all_prices = [price for ohlc in self.ohlc_data for price in ohlc[1:]]  # Skip timestamp
        min_price = min(all_prices)
        max_price = max(all_prices)
        
        return QRectF(0, min_price, len(self.ohlc_data), max_price - min_price)


def test_super_simple_candlesticks():
    """Test with the simplest possible approach"""
    
    print("=== SUPER SIMPLE CANDLESTICK TEST ===")
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        import pyautogui
        
        # Create super simple test data (OHLC tuples)
        ohlc_data = [
            (100, 103, 98, 102),   # Up candle
            (102, 105, 100, 99),   # Down candle  
            (99, 102, 97, 101),    # Up candle
            (101, 104, 98, 96),    # Down candle
            (96, 99, 94, 98),      # Up candle
        ]
        
        print(f"Created {len(ohlc_data)} test candles")
        for i, (o, h, l, c) in enumerate(ohlc_data):
            direction = "UP" if c >= o else "DOWN"
            print(f"  Candle {i}: O={o} H={h} L={l} C={c} ({direction})")
        
        # Create plot
        plot_widget = pg.PlotWidget()
        plot_widget.setWindowTitle("SUPER SIMPLE CANDLESTICKS")
        plot_widget.setLabel('left', 'Price')
        plot_widget.setLabel('bottom', 'Candle Index')
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setBackground('w')  # White background
        plot_widget.resize(800, 600)
        
        # Create candlestick item
        candle_item = SuperSimpleCandlestickItem(ohlc_data)
        plot_widget.addItem(candle_item)
        
        # AUTO-SCALE to fit data
        plot_widget.autoRange()
        
        # Show window
        plot_widget.show()
        plot_widget.raise_()
        plot_widget.activateWindow()
        
        print("Window shown - should see 5 candlesticks")
        
        # Screenshot directory
        screenshot_dir = Path("debug_screenshots")
        screenshot_dir.mkdir(exist_ok=True)
        
        def take_screenshot(suffix=""):
            timestamp = time.strftime("%H%M%S")
            filename = f"SIMPLE_candlesticks_{timestamp}{suffix}.png"
            filepath = screenshot_dir / filename
            screenshot = pyautogui.screenshot()
            screenshot.save(str(filepath))
            print(f"SCREENSHOT: {filepath}")
        
        # Take screenshots
        QTimer.singleShot(2000, lambda: take_screenshot("_2s"))
        QTimer.singleShot(4000, lambda: take_screenshot("_4s"))
        QTimer.singleShot(6000, lambda: take_screenshot("_6s"))
        
        # Auto-close
        QTimer.singleShot(10000, app.quit)
        
        print("Running for 10 seconds...")
        app.exec_()
        
        print("Test complete - check screenshots")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_super_simple_candlesticks()