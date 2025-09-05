#!/usr/bin/env python3
"""
Test the new pan/zoom controls:
- Ctrl+scroll for zoom  
- Scroll for pan
"""

import sys
from PyQt5.QtWidgets import QApplication
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data

def test_pan_zoom_controls():
    """Test the new pan/zoom behavior"""
    print("="*60)
    print("TESTING PAN/ZOOM CONTROLS")
    print("="*60)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    # Create test data
    ohlcv_data, trades_csv_path = create_ultimate_test_data()
    
    print(f"Test data created:")
    print(f"  Bars: {len(ohlcv_data['close'])}")
    print(f"  Trades CSV: {trades_csv_path}")
    
    # Create dashboard
    dashboard = FinalTradingDashboard()
    dashboard.setWindowTitle("PAN/ZOOM CONTROLS TEST")
    dashboard.resize(1920, 1200)
    dashboard.show()
    
    # Load data
    success = dashboard.load_ultimate_dataset(ohlcv_data, trades_csv_path)
    print(f"Dataset loaded: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        print("\n" + "="*50)
        print("NEW CONTROLS:")
        print("  Ctrl+Scroll: ZOOM in/out")  
        print("  Scroll:      PAN left/right")
        print("  Mouse drag:  PAN (still works)")
        print("="*50)
        print("\nTest the new scroll behavior:")
        print("1. Try scrolling WITHOUT Ctrl - should pan left/right")
        print("2. Try scrolling WITH Ctrl - should zoom in/out")
        print("3. Mouse drag should still work for panning")
        
        app.exec_()
    
    print("Pan/zoom controls test complete!")

if __name__ == "__main__":
    test_pan_zoom_controls()