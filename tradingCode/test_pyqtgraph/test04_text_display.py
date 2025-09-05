"""
Test 4: Text display overlay test
Goal: Create visible text overlay at top of chart (OHLCV style)
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

class TextDisplayTestWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        
        # Setup basic plot
        self.setBackground('white')
        self.setLabel('left', 'Price', color='black', size='14pt')
        self.setLabel('bottom', 'Time', color='black', size='14pt')
        
        # Configure axes
        for axis_name in ['left', 'right', 'bottom']:
            self.showAxis(axis_name, True)
            axis = self.getAxis(axis_name)
            axis.setPen(pg.mkPen('black', width=1))
            axis.setTextPen(pg.mkPen('black'))
            axis.setStyle(showValues=True)
        
        # Add candlestick-like data
        x = np.arange(100)
        y = 100 + np.cumsum(np.random.randn(100) * 0.5)
        self.plot(x, y, pen='blue', name='Price Data')
        
        # Test different text display methods
        self.test_text_methods()
        
        print("Text display methods tested")
        
    def test_text_methods(self):
        """Test various text display approaches"""
        
        # Method 1: TextItem with different positions and styles
        self.text1 = pg.TextItem(
            text="Method 1: Blue BG OHLCV Data Here", 
            color='white',
            fill=pg.mkBrush(0, 100, 200, 180),
            border=pg.mkPen('white', width=2),
            anchor=(0, 0)
        )
        self.addItem(self.text1, ignoreBounds=True)
        self.text1.setPos(10, 110)  # Top left
        self.text1.setZValue(1000)
        
        # Method 2: TextItem with different styling  
        self.text2 = pg.TextItem(
            text="Method 2: Green BG | O: 123.45 | H: 125.67 | L: 121.23 | C: 124.56",
            color='black',
            fill=pg.mkBrush(100, 200, 100, 200),
            border=pg.mkPen('black', width=1),
            anchor=(0, 0)
        )
        self.addItem(self.text2, ignoreBounds=True)
        self.text2.setPos(10, 105)  # Slightly lower
        self.text2.setZValue(1001)
        
        # Method 3: Simple text with high contrast
        self.text3 = pg.TextItem(
            text="Method 3: HIGH CONTRAST TEXT",
            color='red',
            fill=pg.mkBrush(255, 255, 0, 220),  # Yellow background
            border=pg.mkPen('red', width=3),
            anchor=(0, 0)
        )
        self.addItem(self.text3, ignoreBounds=True)
        self.text3.setPos(10, 100)
        self.text3.setZValue(1002)
        
        # Method 4: Position at different locations
        positions = [
            (200, 110, "Top Middle"),
            (400, 110, "Top Right"),
            (10, 95, "Lower Left")
        ]
        
        self.extra_texts = []
        for i, (x, y, label) in enumerate(positions):
            text = pg.TextItem(
                text=f"{label}: Test Text {i+4}",
                color='white',
                fill=pg.mkBrush(i*80, 100, 255-i*50, 200),
                border=pg.mkPen('white', width=1),
                anchor=(0, 0)
            )
            self.addItem(text, ignoreBounds=True)
            text.setPos(x, y)
            text.setZValue(1003 + i)
            self.extra_texts.append(text)
        
        print("Created 7 different text overlays at various positions")

def test_text_display():
    """Test text display functionality"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Test 4: Text Display Overlay")
    win.resize(800, 600)
    
    # Create text display widget
    text_widget = TextDisplayTestWidget()
    win.setCentralWidget(text_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events
    for i in range(15):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "test04_text_display_result.png")
    
    print("\n=== TEST 4 SUMMARY ===")
    print("Expected: Multiple colored text overlays at different positions")
    print("Check test04_text_display_result.png for actual result")
    print("Should see OHLCV-style data display at top of chart")
    
    return app, win

if __name__ == "__main__":
    app, win = test_text_display()
    
    # Keep open for inspection
    QtCore.QTimer.singleShot(8000, app.quit)
    app.exec_()