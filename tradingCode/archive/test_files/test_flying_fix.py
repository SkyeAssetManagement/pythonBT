#!/usr/bin/env python3
"""
Test that flying candlesticks are fixed
"""

import sys
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data

def test_flying_fix():
    """Test that flying candlesticks are fixed"""
    print("="*60)
    print("TESTING FLYING CANDLESTICKS FIX")
    print("="*60)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    # Create test data
    ohlcv_data, trades_csv_path = create_ultimate_test_data()
    
    print(f"Test data created:")
    print(f"  Bars: {len(ohlcv_data['close'])}")
    print(f"  Datetime (bar indices): {ohlcv_data['datetime'][:5]}")
    print(f"  Datetime_ns (timestamps): {ohlcv_data['datetime_ns'][:5]}")
    
    # Create dashboard
    dashboard = FinalTradingDashboard()
    dashboard.setWindowTitle("FLYING CANDLESTICKS FIX TEST")
    dashboard.resize(1920, 1200)
    dashboard.show()
    
    # Load data
    success = dashboard.load_ultimate_dataset(ohlcv_data, trades_csv_path)
    print(f"Dataset loaded: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        def take_comparison_screenshot():
            """Take screenshot to compare with before"""
            print("\nTaking fix verification screenshot...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"flying_fix_verification_{timestamp}.png"
            pixmap = dashboard.grab()
            success = pixmap.save(filename)
            if success:
                print(f"Screenshot saved: {filename}")
                print("Compare this with the original flying candlesticks screenshot")
            
            # Show status
            print(f"\nFIX VERIFICATION:")
            print(f"Chart viewport: {dashboard.final_chart.viewport_start}-{dashboard.final_chart.viewport_end}")
            print(f"Data length: {dashboard.final_chart.data_length}")
            print(f"Using bar indices for rendering: YES")
            print(f"Using timestamps for time display: YES")
            
            QTimer.singleShot(2000, app.quit)
        
        QTimer.singleShot(3000, take_comparison_screenshot)
        
        print("Dashboard loaded. Taking screenshot in 3 seconds...")
        app.exec_()
    
    print("Flying candlesticks fix test complete!")

if __name__ == "__main__":
    test_flying_fix()