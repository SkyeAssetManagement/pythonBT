#!/usr/bin/env python3
"""
Debug the import error in main.py context
"""

import sys
import os

def test_imports_from_main_context():
    """Test imports exactly as main.py would do them"""
    
    print("=== DEBUGGING IMPORT ERROR ===")
    print("Testing imports from main.py context...")
    
    # This is what main.py does
    print(f"Current working directory: {os.getcwd()}")
    print(f"sys.path before: {sys.path[:3]}...")
    
    # Test 1: Try importing dashboard manager directly like main.py does
    print("\n1. Testing dashboard manager import (like main.py)...")
    try:
        from src.dashboard.dashboard_manager import launch_dashboard, get_dashboard_manager
        print("   SUCCESS: dashboard_manager import works")
    except Exception as e:
        print(f"   ERROR: dashboard_manager import failed: {e}")
        return False
    
    # Test 2: Try creating a chart widget directly
    print("\n2. Testing chart widget creation...")
    try:
        # Add src to path like our fixes should do
        src_path = os.path.join(os.getcwd(), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from dashboard.chart_widget import TradingChart
        from dashboard.data_structures import ChartDataBuffer
        
        print("   SUCCESS: chart widget imports work")
        
        # Test creating a chart widget
        chart = TradingChart()
        print("   SUCCESS: chart widget created")
        
    except Exception as e:
        print(f"   ERROR: chart widget failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Try creating data and setting it
    print("\n3. Testing data creation and setting...")
    try:
        import numpy as np
        
        # Create minimal test data
        n_bars = 10
        timestamps = np.arange(1609459200000000000, 1609459200000000000 + n_bars * 60 * 1000000000, 60 * 1000000000)
        opens = np.ones(n_bars) * 100
        highs = np.ones(n_bars) * 101
        lows = np.ones(n_bars) * 99
        closes = np.ones(n_bars) * 100.5
        volumes = np.ones(n_bars, dtype=np.int64) * 1000
        
        data_buffer = ChartDataBuffer(
            timestamps=timestamps,
            open=opens, high=highs, low=lows, close=closes,
            volume=volumes
        )
        
        print("   SUCCESS: data buffer created")
        
        # This is where the error likely occurs
        chart.set_data(data_buffer)
        print("   SUCCESS: data set on chart (candlestick generation worked!)")
        
    except Exception as e:
        print(f"   ERROR: data setting failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== ALL IMPORTS WORKING ===")
    return True

if __name__ == "__main__":
    success = test_imports_from_main_context()
    if success:
        print("Import debugging complete - no issues found!")
    else:
        print("Import issues detected - check errors above")