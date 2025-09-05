#!/usr/bin/env python3
"""
Debug crosshair step by step - minimal test
"""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

class MinimalCrosshairTest(pg.PlotWidget):
    """Minimal crosshair test to debug step by step"""
    
    def __init__(self):
        super().__init__()
        
        print("1. Creating minimal crosshair test...")
        
        # Set white background like working examples
        self.setBackground('white')
        self.setLabel('left', 'Price', color='black')
        self.setLabel('bottom', 'Time', color='black')
        
        print("2. Adding test data...")
        # Add simple test data
        x = np.arange(100)
        y = np.random.random(100) * 100 + 100
        self.plot(x, y, pen='blue')
        
        print("3. Creating crosshairs...")
        # Create crosshairs exactly like working example
        self.crosshair_v = pg.InfiniteLine(
            angle=90, 
            movable=False,
            pen=pg.mkPen('black', width=1, style=QtCore.Qt.DotLine)
        )
        self.crosshair_h = pg.InfiniteLine(
            angle=0, 
            movable=False,
            pen=pg.mkPen('black', width=1, style=QtCore.Qt.DotLine)
        )
        
        # Add to plot
        self.addItem(self.crosshair_v, ignoreBounds=True)
        self.addItem(self.crosshair_h, ignoreBounds=True)
        self.crosshair_v.setZValue(1000)
        self.crosshair_h.setZValue(1000)
        
        # Show initially like working example
        self.crosshair_v.setPos(50)
        self.crosshair_h.setPos(150)
        self.crosshair_v.show()
        self.crosshair_h.show()
        
        print("4. Setting up mouse tracking immediately...")
        # Connect mouse events immediately like working example
        self._setup_mouse_tracking()
        
    def _setup_mouse_tracking(self):
        """Setup mouse tracking exactly like working example"""
        print("5. Connecting mouse signals...")
        try:
            # Connect immediately like working example (no scene check)
            self.mouse_proxy = pg.SignalProxy(
                self.scene().sigMouseMoved, 
                rateLimit=60, 
                slot=self._on_mouse_moved
            )
            print("6. Mouse proxy connected successfully!")
        except Exception as e:
            print(f"6. Error connecting mouse: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_mouse_moved(self, evt):
        """Handle mouse movement exactly like working example"""
        try:
            print("7. Mouse moved event received!")
            pos = evt[0]  # Get position from SignalProxy like working example
            if self.plotItem.vb.sceneBoundingRect().contains(pos):
                mousePoint = self.plotItem.vb.mapSceneToView(pos)
                x_coord = mousePoint.x()
                y_coord = mousePoint.y()
                
                print(f"8. Mouse coordinates: X={x_coord:.1f}, Y={y_coord:.1f}")
                
                # Show and update crosshairs
                self.crosshair_v.setPos(x_coord)
                self.crosshair_h.setPos(y_coord)
                self.crosshair_v.show()
                self.crosshair_h.show()
                
            else:
                print("8. Mouse outside plot area")
                self.crosshair_v.hide()
                self.crosshair_h.hide()
                
        except Exception as e:
            print(f"Error in mouse handler: {e}")
            import traceback
            traceback.print_exc()

def test_minimal_crosshair():
    """Test minimal crosshair functionality"""
    
    print("MINIMAL CROSSHAIR DEBUG TEST")
    print("=" * 40)
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Minimal Crosshair Debug Test")
    win.resize(800, 600)
    
    # Create minimal crosshair widget
    crosshair_widget = MinimalCrosshairTest()
    win.setCentralWidget(crosshair_widget)
    
    # Show window
    win.show()
    win.raise_()
    win.activateWindow()
    
    print("Test window shown. Move mouse over chart to test crosshair.")
    print("Close window to continue...")
    
    # Run for limited time
    QtCore.QTimer.singleShot(15000, app.quit)  # 15 seconds
    app.exec_()
    
    print("Minimal crosshair test completed.")

if __name__ == "__main__":
    test_minimal_crosshair()