"""
Test 2: Debug axis system thoroughly
Goal: Understand why axis labels aren't showing and force them to appear
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

def debug_axis_properties(plot_widget):
    """Debug axis properties"""
    print("\n=== AXIS DEBUG INFO ===")
    
    for axis_name in ['left', 'right', 'bottom', 'top']:
        try:
            axis = plot_widget.getAxis(axis_name)
            print(f"\n{axis_name.upper()} AXIS:")
            print(f"  Visible: {axis.isVisible()}")
            print(f"  Width/Height: {axis.width() if axis_name in ['left', 'right'] else axis.height()}")
            print(f"  Style showValues: {getattr(axis, 'style', {}).get('showValues', 'N/A')}")
            print(f"  Label text: {axis.labelText}")
            print(f"  Pen: {axis.pen()}")
            print(f"  TextPen: {axis.textPen()}")
            
            # Check if axis has any ticks
            if hasattr(axis, 'tickValues'):
                ticks = axis.tickValues(0, 100, 100)
                print(f"  Sample ticks: {ticks}")
                
        except Exception as e:
            print(f"  ERROR: {e}")

def test_axis_debug():
    """Test with maximum debugging"""
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window with white background
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Test 2: Axis Debug")
    win.resize(800, 600)
    win.setStyleSheet("background-color: white;")
    
    # Create plot widget with explicit background
    plot_widget = pg.PlotWidget()
    plot_widget.setBackground('white')
    win.setCentralWidget(plot_widget)
    
    # Set labels BEFORE configuring axes
    plot_widget.setLabel('left', 'Y AXIS LABEL', color='red', size='20pt')
    plot_widget.setLabel('bottom', 'X AXIS LABEL', color='blue', size='20pt')
    plot_widget.setLabel('top', 'TITLE LABEL', color='green', size='24pt')
    
    # Add test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plot_widget.plot(x, y, pen=pg.mkPen('red', width=3), name='Test Sine Wave')
    
    # Configure axes with extreme visibility settings
    axes_config = {
        'left': True,
        'right': True,
        'bottom': True,
        'top': True
    }
    
    for axis_name, show in axes_config.items():
        plot_widget.showAxis(axis_name, show)
        axis = plot_widget.getAxis(axis_name)
        
        # Force axis styling
        axis.setPen(pg.mkPen('black', width=3))
        axis.setTextPen(pg.mkPen('black', width=2))
        axis.setStyle(showValues=True, tickLength=10)
        
        # Set fixed dimensions
        if axis_name in ['left', 'right']:
            axis.setWidth(100)  # Very wide for visibility
        else:
            axis.setHeight(60)  # Very tall for visibility
        
        # Force show
        axis.show()
        axis.setVisible(True)
        axis.update()
        
        print(f"Configured {axis_name} axis with extreme visibility")
    
    # Force plot update
    plot_widget.plotItem.updateButtons()
    plot_widget.plotItem.showButtons()
    plot_widget.update()
    plot_widget.repaint()
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    # Process events extensively
    for i in range(10):
        app.processEvents()
        QtCore.QThread.msleep(100)
    
    # Debug axis properties
    debug_axis_properties(plot_widget)
    
    # Save screenshot
    save_widget_screenshot(win, "test02_axis_debug_result.png")
    
    print("\n=== TEST 2 SUMMARY ===")
    print("Expected: Large axis labels, thick black axis lines, numbers")
    print("Check test02_axis_debug_result.png for actual result")
    
    return app, win

if __name__ == "__main__":
    app, win = test_axis_debug()
    
    # Keep open longer for inspection
    QtCore.QTimer.singleShot(8000, app.quit)
    app.exec_()