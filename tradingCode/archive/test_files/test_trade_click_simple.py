#!/usr/bin/env python3
"""
Simple test for trade clicking to verify it works correctly
"""

import sys
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data

def test_trade_clicking():
    """Test that trade clicking navigates to correct location"""
    print("="*60)
    print("TESTING TRADE CLICK NAVIGATION TO LOCATION")
    print("="*60)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    # Create test data
    ohlcv_data, trades_csv_path = create_ultimate_test_data()
    
    # Create dashboard
    dashboard = FinalTradingDashboard()
    dashboard.setWindowTitle("TEST: Trade Click Navigation")
    dashboard.resize(1920, 1200)
    dashboard.show()
    
    # Load data
    success = dashboard.load_ultimate_dataset(ohlcv_data, trades_csv_path)
    print(f"Dataset loaded: {'SUCCESS' if success else 'FAILED'}")
    
    if success and dashboard.trade_list:
        trades = dashboard.trade_list.trade_list_widget.trades_data
        print(f"Available trades: {len(trades)}")
        
        # Test clicking on different trades
        def test_trade_sequence():
            for i in [0, 5, 10]:  # Test first, sixth, and eleventh trade
                if i < len(trades):
                    trade = trades[i]
                    print(f"\nTEST {i+1}: Clicking trade {trade.trade_id}")
                    print(f"  Entry Time: {trade.entry_time}")
                    print(f"  Exit Time: {trade.exit_time}")
                    print(f"  Current chart viewport: {dashboard.final_chart.viewport_start}-{dashboard.final_chart.viewport_end}")
                    
                    # Click the trade
                    dashboard.trade_list.trade_list_widget._on_trade_clicked(i % dashboard.trade_list.trade_list_widget.rowCount(), 0)
                    
                    print(f"  New chart viewport: {dashboard.final_chart.viewport_start}-{dashboard.final_chart.viewport_end}")
                    
                    # Check if viewport includes trade entry
                    if dashboard.final_chart.viewport_start <= trade.entry_time <= dashboard.final_chart.viewport_end:
                        print(f"  ✓ SUCCESS: Trade {trade.trade_id} entry visible in viewport")
                    else:
                        print(f"  ✗ FAILED: Trade {trade.trade_id} entry NOT visible in viewport")
            
            # Take screenshot
            print("\nTaking screenshot...")
            timestamp = time.strftime("%Y%m%d_%H%M%S") 
            filename = f"trade_click_test_{timestamp}.png"
            pixmap = dashboard.grab()
            success = pixmap.save(filename)
            if success:
                print(f"Screenshot saved: {filename}")
        
        # Run test after 2 seconds
        QTimer.singleShot(2000, test_trade_sequence)
        QTimer.singleShot(8000, app.quit)
        
        app.exec_()
    
    print("Trade click test complete")

if __name__ == "__main__":
    test_trade_clicking()