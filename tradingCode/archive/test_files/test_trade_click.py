#!/usr/bin/env python3
"""
Quick test to verify trade click navigation works
"""

import sys
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data

def test_trade_click():
    """Test trade clicking with detailed logging"""
    print("="*60)
    print("TESTING TRADE CLICK NAVIGATION")
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
    
    if success:
        # Add additional debugging to trade selection
        def debug_trade_selected(trade_data, chart_index):
            print(f"\n" + "="*50)
            print(f"TRADE CLICK DETECTED!")
            print(f"Trade ID: {trade_data.trade_id}")
            print(f"Entry Time: {trade_data.entry_time}")
            print(f"Exit Time: {trade_data.exit_time}")
            print(f"Side: {trade_data.side}")
            print(f"PnL: {trade_data.pnl:+.2f}")
            print(f"Chart Index: {chart_index}")
            print(f"="*50)
            
            # Call the original handler
            dashboard._on_ultimate_trade_selected(trade_data, chart_index)
        
        # Reconnect with debug wrapper
        if dashboard.trade_list:
            dashboard.trade_list.trade_selected.disconnect()
            dashboard.trade_list.trade_selected.connect(debug_trade_selected)
            print("DEBUG: Trade click handler connected with debugging")
        
        def test_programmatic_click():
            """Simulate clicking on the first trade after 3 seconds"""
            if dashboard.trade_list and dashboard.trade_list.trade_list_widget.trades_data:
                first_trade = dashboard.trade_list.trade_list_widget.trades_data[0]
                print(f"\nAUTO-CLICKING first trade: {first_trade.trade_id}")
                dashboard.trade_list.trade_list_widget._on_trade_clicked(0, 0)
        
        # Schedule programmatic click
        QTimer.singleShot(3000, test_programmatic_click)
        
        # Schedule app exit
        QTimer.singleShot(10000, app.quit)
        
        print("\nInstructions:")
        print("1. Click on any trade in the trade list")
        print("2. Watch console for debug output")
        print("3. Check if chart and equity curve jump to trade location")
        print("4. Application will auto-exit in 10 seconds")
        print("\nStarting test...")
        
        app.exec_()
    
    print("\nTrade click test complete")

if __name__ == "__main__":
    test_trade_click()