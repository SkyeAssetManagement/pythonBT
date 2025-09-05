"""
Test 3: Crosshair visibility test
Goal: Create visible crosshairs that respond to mouse movement
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

class CrosshairTestWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        
        # Setup basic plot
        self.setBackground('white')
        self.setLabel('left', 'Y Values', color='black', size='16pt')
        self.setLabel('bottom', 'X Values', color='black', size='16pt') 
        self.setLabel('top', 'Crosshair Test', color='black', size='18pt')
        
        # Configure axes for visibility
        for axis_name in ['left', 'right', 'bottom', 'top']:
            self.showAxis(axis_name, True)
            axis = self.getAxis(axis_name)
            axis.setPen(pg.mkPen('black', width=2))
            axis.setTextPen(pg.mkPen('black'))
            axis.setStyle(showValues=True)
            if axis_name in ['left', 'right']:
                axis.setWidth(80)
            else:
                axis.setHeight(50)
        
        # Add test data
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + 0.1 * np.random.random(50)
        self.plot(x, y, pen='blue', symbol='o', symbolSize=5, name='Test Data')
        
        # Create crosshairs - BRIGHT AND THICK
        self.vLine = pg.InfiniteLine(angle=90, movable=False, 
                                   pen=pg.mkPen('red', width=4, style=QtCore.Qt.SolidLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False,
                                   pen=pg.mkPen('red', width=4, style=QtCore.Qt.SolidLine))
        
        # Add to plot
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)
        
        # Set initial position (center of view)
        self.vLine.setPos(5)
        self.hLine.setPos(0)
        
        # Force show
        self.vLine.show()
        self.hLine.show()
        
        # Connect mouse events
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        
        print("Crosshairs created with thick red lines at position (5, 0)")
        
    def mouseMoved(self, evt):
        """Handle mouse movement"""
        pos = evt[0]  # Get position from SignalProxy
        if self.plotItem.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.plotItem.vb.mapSceneToView(pos)
            
            # Update crosshair position
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            
            print(f"Crosshair moved to: ({mousePoint.x():.2f}, {mousePoint.y():.2f})")

def test_crosshair():
    """Test crosshair functionality"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Test 3: Crosshair Visibility")
    win.resize(800, 600)
    
    # Create crosshair widget
    crosshair_widget = CrosshairTestWidget()
    win.setCentralWidget(crosshair_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events
    for i in range(10):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Save screenshot
    save_widget_screenshot(win, "test03_crosshair_result.png")
    
    print("\n=== TEST 3 SUMMARY ===")
    print("Expected: Thick red crosshairs at center, responding to mouse")
    print("Check test03_crosshair_result.png for actual result")
    print("Move mouse over plot to test interaction")
    
    return app, win

if __name__ == "__main__":
    app, win = test_crosshair()
    
    # Keep open for interaction testing
    QtCore.QTimer.singleShot(10000, app.quit)
    app.exec_()