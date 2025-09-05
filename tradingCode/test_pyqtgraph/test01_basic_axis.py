"""
Test 1: Basic PyQtGraph axis visibility
Goal: Create the most basic plot with visible axis labels and save screenshot
"""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
import datetime

def save_widget_screenshot(widget, filename):
    """Save a screenshot of the widget"""
    try:
        # Get the widget's pixmap
        pixmap = widget.grab()
        
        # Save the pixmap
        success = pixmap.save(filename, 'PNG')
        print(f"Screenshot saved: {filename} (Success: {success})")
        return success
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return False

def test_basic_axis():
    """Test basic axis visibility"""
    
    # Create application
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Test 1: Basic Axis Labels")
    win.resize(800, 600)
    
    # Create plot widget
    plot_widget = pg.PlotWidget()
    win.setCentralWidget(plot_widget)
    
    # Configure plot
    plot_widget.setLabel('left', 'Y-axis Label', color='red', size='16pt')
    plot_widget.setLabel('bottom', 'X-axis Label', color='blue', size='16pt')
    plot_widget.setLabel('top', 'Chart Title', color='black', size='18pt')
    
    # Add simple data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plot_widget.plot(x, y, pen='r', name='Test Data')
    
    # Force show all axes
    plot_widget.showAxis('left', True)
    plot_widget.showAxis('right', True)
    plot_widget.showAxis('bottom', True)
    plot_widget.showAxis('top', True)
    
    # Style axes
    for axis_name in ['left', 'right', 'bottom', 'top']:
        axis = plot_widget.getAxis(axis_name)
        axis.setPen('black')
        axis.setTextPen('black')
        axis.setStyle(showValues=True)
        axis.show()
        print(f"Configured {axis_name} axis")
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events to ensure rendering
    app.processEvents()
    QtCore.QThread.msleep(1000)  # Wait 1 second
    app.processEvents()
    
    # Save screenshot
    save_widget_screenshot(win, "test01_basic_axis_result.png")
    
    print("Test 1 complete. Check test01_basic_axis_result.png")
    print("EXPECTED: Should see red Y-axis label, blue X-axis label, black title")
    print("ACTUAL: Check the saved image to verify")
    
    # Keep window open for manual inspection
    return app, win

if __name__ == "__main__":
    app, win = test_basic_axis()
    
    # Run for 5 seconds then close
    QtCore.QTimer.singleShot(5000, app.quit)
    app.exec_()