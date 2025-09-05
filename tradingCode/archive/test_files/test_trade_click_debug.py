#!/usr/bin/env python3
"""
Test trade clicking functionality with debug output
"""

import sys
from PyQt5.QtWidgets import QApplication
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data

def test_trade_clicking():
    """Test that trade clicking works and debug the issue"""
    print("="*60)
    print("TESTING TRADE CLICKING WITH DEBUG")
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
    dashboard.setWindowTitle("TRADE CLICKING DEBUG TEST")
    dashboard.resize(1920, 1200)
    dashboard.show()
    
    # Load data
    success = dashboard.load_ultimate_dataset(ohlcv_data, trades_csv_path)
    print(f"Dataset loaded: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        print(f"\nTrade list contains {len(dashboard.trades_data)} trades")
        print("Try clicking on different trades in the trade list")
        print("Watch the console for debug messages")
        print("Expected behavior:")
        print("  1. First click should work and show debug messages")
        print("  2. Subsequent clicks should also work")
        print("  3. Chart should navigate to each trade location")
        
        app.exec_()
    
    print("Trade clicking debug test complete!")

if __name__ == "__main__":
    test_trade_clicking()