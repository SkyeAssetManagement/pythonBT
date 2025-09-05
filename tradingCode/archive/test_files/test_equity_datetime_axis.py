# tradingCode/test_equity_datetime_axis.py
# Test equity curve with proper datetime x-axis formatting
# Quick test to verify Step 1 fix is working

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dashboard.equity_curve_widget import EquityCurveWidget
from PyQt5.QtWidgets import QApplication

def create_test_data_with_timestamps():
    """Create test equity curve data with realistic timestamps"""
    
    # Create 1000 data points over 6 months
    start_date = datetime(2024, 1, 1, 9, 30)  # Market open
    timestamps = []
    equity_values = []
    
    # Generate minute-by-minute data for trading hours
    current_date = start_date
    initial_equity = 10000.0
    current_equity = initial_equity
    
    for i in range(1000):
        # Skip weekends and after-hours
        if current_date.weekday() >= 5:  # Weekend
            current_date += timedelta(days=2)
            current_date = current_date.replace(hour=9, minute=30)
        
        if current_date.hour >= 16:  # After market close
            current_date = current_date.replace(hour=9, minute=30) + timedelta(days=1)
            
        timestamps.append(current_date.timestamp())  # Unix timestamp
        
        # Generate realistic equity curve with volatility
        daily_return = np.random.normal(0.0001, 0.005)  # Small positive drift with volatility
        current_equity *= (1 + daily_return)
        equity_values.append(current_equity)
        
        # Advance time by 1 minute
        current_date += timedelta(minutes=1)
    
    return np.array(equity_values), np.array(timestamps)

def test_equity_datetime_axis():
    """Test the equity curve widget with datetime axis"""
    print("TESTING EQUITY CURVE DATETIME X-AXIS - STEP 1 FIX")
    print("=" * 60)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Create test data
        equity_data, timestamps = create_test_data_with_timestamps()
        print(f"Generated test data: {len(equity_data)} points")
        print(f"Time range: {datetime.fromtimestamp(timestamps[0])} to {datetime.fromtimestamp(timestamps[-1])}")
        
        # Create equity curve widget
        widget = EquityCurveWidget()
        widget.setWindowTitle("Equity Curve DateTime Axis Test - Step 1 Fix")
        
        # Load data with timestamps
        success = widget.load_equity_data(equity_data, timestamps)
        if not success:
            print("ERROR: Failed to load equity data")
            return False
        
        # Show widget
        widget.resize(1200, 400)
        widget.show()
        
        print("\nSUCCESS: Equity curve widget displayed with datetime axis")
        print("EXPECTED BEHAVIOR:")
        print("• X-axis should show dates/times instead of bar numbers")
        print("• Dates should format automatically (daily, monthly, etc.)")
        print("• Tooltip should show actual timestamps when hovering")
        print("\nPlease verify the x-axis shows proper date/time formatting")
        
        # Take screenshot to verify
        import time
        time.sleep(2)  # Wait for render
        
        screenshot_path = "C:\\Users\\skyeAM\\SkyeAM Dropbox\\SAMresearch\\ABtoPython\\2025-08-08_1\\step1_equity_datetime_test.png"
        
        # Take screenshot using widget
        pixmap = widget.grab()
        pixmap.save(screenshot_path)
        print(f"Screenshot saved: {screenshot_path}")
        
        # Keep window open for verification
        widget.show()
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_equity_datetime_axis()
    
    if success:
        print("\n" + "=" * 60)
        print("STEP 1 TEST COMPLETED - EQUITY CURVE DATETIME AXIS")
        print("=" * 60)
        print("VERIFICATION REQUIRED:")
        print("1. Check screenshot: step1_equity_datetime_test.png")
        print("2. X-axis should display dates/times, not bar numbers")
        print("3. Times should be properly formatted and readable")
        print("\nIf datetime formatting is working, Step 1 is COMPLETE!")
    else:
        print("Step 1 test failed - needs additional work")