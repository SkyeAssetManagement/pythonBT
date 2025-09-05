# tradingCode/test_trade_markers.py  
# Test trade marker triangles on price chart
# Verify Step 2: Green up arrows below low for buy/cover, red down arrows above high for sell/short

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import test functions
sys.path.insert(0, str(Path(__file__).parent))
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data
from PyQt5.QtWidgets import QApplication

def create_test_trades_data():
    """Create test trades data to verify marker positioning"""
    
    trades_data = []
    
    # Test trades with different scenarios
    test_scenarios = [
        # Long trades (should show green arrows below low for entry, colored arrows above high for exit)
        {'entry_time': 100, 'exit_time': 150, 'direction': 'Long', 'pnl': 500, 'side': 'Buy'},
        {'entry_time': 200, 'exit_time': 250, 'direction': 'Long', 'pnl': -200, 'side': 'Buy'},
        {'entry_time': 300, 'exit_time': 350, 'direction': 'Long', 'pnl': 800, 'side': 'Buy'},
        
        # Short trades (should show red arrows above high for entry, colored arrows below low for exit)
        {'entry_time': 400, 'exit_time': 450, 'direction': 'Short', 'pnl': 300, 'side': 'Sell'},
        {'entry_time': 500, 'exit_time': 550, 'direction': 'Short', 'pnl': -150, 'side': 'Sell'},
        {'entry_time': 600, 'exit_time': 650, 'direction': 'Short', 'pnl': 600, 'side': 'Sell'},
    ]
    
    for i, scenario in enumerate(test_scenarios):
        trade = {
            'Trade ID': f'T{i+1:03d}',
            'entry_time': scenario['entry_time'],
            'exit_time': scenario['exit_time'], 
            'direction': scenario['direction'],
            'pnl': scenario['pnl'],
            'side': scenario['side'],
            'entry_price': 5000 + np.random.normal(0, 50),
            'exit_price': 5000 + np.random.normal(0, 50),
            'size': 1.0
        }
        trades_data.append(trade)
    
    return trades_data

def test_trade_markers():
    """Test trade marker triangles on the price chart"""
    print("TESTING TRADE MARKER TRIANGLES - STEP 2")
    print("=" * 60)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Create test OHLCV data
        print("Creating test OHLCV data...")
        ohlcv_data, trades_csv_path = create_ultimate_test_data()
        
        # Create test trades data (override with our custom test trades)
        print("Creating test trades data...")
        trades_data = create_test_trades_data()
        
        print(f"Generated data: {len(ohlcv_data['close'])} candlesticks, {len(trades_data)} trades")
        
        # Create FinalTradingDashboard
        dashboard = FinalTradingDashboard()
        dashboard.setWindowTitle("Trade Marker Test - Step 2 Verification")
        
        # Load test data
        print("Loading test data into dashboard...")
        
        # Load OHLCV data
        success = dashboard.final_chart.load_ohlcv_data(ohlcv_data)
        if not success:
            print("ERROR: Failed to load OHLCV data")
            return False
        
        # Load trades data
        success = dashboard.final_chart.load_trades_data(trades_data) 
        if not success:
            print("ERROR: Failed to load trades data")
            return False
        
        # Setup viewport to show our test trades
        dashboard.final_chart.viewport_start = 50
        dashboard.final_chart.viewport_end = 750
        
        # Force regenerate geometry to ensure markers are visible
        dashboard.final_chart._generate_candlestick_geometry()
        dashboard.final_chart._generate_trade_markers()
        
        # Show dashboard
        dashboard.resize(1600, 1000)
        dashboard.show()
        
        # Take screenshot after a brief delay
        import time
        time.sleep(2)  # Allow time for rendering
        
        screenshot_path = "C:\\Users\\skyeAM\\SkyeAM Dropbox\\SAMresearch\\ABtoPython\\2025-08-08_1\\step2_trade_markers_test.png"
        pixmap = dashboard.grab()
        pixmap.save(screenshot_path)
        print(f"Screenshot saved: {screenshot_path}")
        
        print("\nSUCCESS: Trade marker test displayed")
        print("\nVERIFICATION EXPECTED:")
        print("• Long trades (T001, T002, T003):")
        print("  - Entry: GREEN triangles pointing UP, positioned 2% BELOW candle lows")
        print("  - Exit: Colored triangles pointing DOWN, positioned 2% ABOVE candle highs")
        print("    (Green for profit, Red for loss)")
        print("\n• Short trades (T004, T005, T006):")
        print("  - Entry: RED triangles pointing DOWN, positioned 2% ABOVE candle highs")
        print("  - Exit: Colored triangles pointing UP, positioned 2% BELOW candle lows")
        print("    (Green for profit, Red for loss)")
        print("\n• Triangle shapes should be clearly visible and properly positioned")
        print("• Colors should indicate entry direction and exit P&L")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Trade marker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trade_markers()
    
    if success:
        print("\n" + "=" * 60)
        print("STEP 2 TEST COMPLETED - TRADE MARKER TRIANGLES")
        print("=" * 60)
        print("VERIFICATION REQUIRED:")
        print("1. Check screenshot: step2_trade_markers_test.png")
        print("2. Verify triangle positions relative to candle highs/lows")
        print("3. Confirm color coding (green=buy/cover, red=sell/short)")
        print("4. Check triangles point in correct directions")
        print("\nIf trade markers display correctly, Step 2 is COMPLETE!")
    else:
        print("Step 2 test failed - needs additional work")