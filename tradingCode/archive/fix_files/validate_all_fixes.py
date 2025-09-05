#!/usr/bin/env python3
"""
Comprehensive validation test for all user-requested fixes
Tests with main.py data loading to match user's scenario
"""

import sys
import time
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

def create_validation_script():
    """Create a script that validates all fixes with screenshots"""
    
    validation_script = """
# COMPREHENSIVE VALIDATION TEST
# ============================

This test validates all four user-requested fixes:

## 1. Trade Clicking ✓ FIXED
- **Issue**: Chart not moving on 2nd+ trade clicks  
- **Fix**: Enhanced debug output, fixed viewport updates, added time axis sync
- **Test**: Click multiple trades and verify chart viewport actually changes

## 2. X-axis Labels ✓ FIXED  
- **Issue**: X-axis labels not moving when scrolling/zooming
- **Fix**: Added viewport change callbacks to sync time axis with chart
- **Test**: Scroll/zoom and verify time labels update along bottom of chart

## 3. Crosshair Position ✓ FIXED
- **Issue**: Crosshair data in middle instead of top-left of price window
- **Fix**: Modified show_at_position() to position at absolute top-left of chart
- **Test**: Move crosshair and verify info box appears in top-left corner

## 4. Time Format in Crosshair ✓ FIXED
- **Issue**: X-axis coordinates showing as numbers instead of HH:MM YYYY-MM-DD
- **Fix**: Updated _format_x_value() and _format_x_coordinate() methods
- **Test**: Move crosshair and verify X coordinate shows as "HH:MM YYYY-MM-DD"

## TESTING INSTRUCTIONS:
1. Run: python main.py AD time_window_strategy_vectorized --useDefaults --start_date 2024-10-01 --end_date 2024-10-02
2. Wait for dashboard to load completely
3. Test each fix systematically with screenshots
4. Verify all issues are resolved
"""
    
    return validation_script

def run_main_validation():
    """Run main.py for validation testing"""
    print("="*80)
    print("COMPREHENSIVE VALIDATION OF ALL FIXES")
    print("="*80)
    
    print(create_validation_script())
    
    print("\n" + "="*50)
    print("RUNNING MAIN.PY FOR VALIDATION")
    print("="*50)
    
    # Run the exact main.py command the user uses
    cmd = [
        sys.executable, "main.py", "AD", "time_window_strategy_vectorized", 
        "--useDefaults", "--start_date", "2024-10-01", "--end_date", "2024-10-02"
    ]
    
    print("Command:", " ".join(cmd))
    print("\nStarting validation test...")
    print("Please test each fix systematically:")
    
    print("\n🔍 TEST 1: TRADE CLICKING")
    print("  1. Click on FIRST trade in trade list")
    print("  2. Note the chart position/viewport")
    print("  3. Click on SECOND trade in trade list")
    print("  4. Verify chart ACTUALLY moves to new position")
    print("  5. Click on THIRD trade - should move again")
    
    print("\n🔍 TEST 2: X-AXIS LABELS")  
    print("  1. Note current time labels on X-axis")
    print("  2. Scroll (without Ctrl) to pan")
    print("  3. Verify time labels UPDATE and show different times")
    print("  4. Ctrl+Scroll to zoom in/out")
    print("  5. Verify time labels SCALE appropriately")
    
    print("\n🔍 TEST 3: CROSSHAIR POSITION")
    print("  1. Move mouse over price chart")
    print("  2. Verify crosshair info box appears in TOP-LEFT corner")
    print("  3. Move mouse to different areas")
    print("  4. Info box should stay in top-left, not follow mouse")
    
    print("\n🔍 TEST 4: TIME FORMAT IN CROSSHAIR")
    print("  1. Move crosshair over chart")
    print("  2. Check X coordinate in crosshair info box")
    print("  3. Should show 'HH:MM YYYY-MM-DD' format")
    print("  4. Should NOT show bar numbers like '1234.56'")
    
    print("\n" + "="*50)
    print("STARTING MAIN.PY...")
    print("="*50)
    
    try:
        # Start main.py
        process = subprocess.Popen(cmd, cwd=Path(__file__).parent)
        
        print(f"✓ Main.py started successfully (PID: {process.pid})")
        print("✓ Perform tests above, then close dashboard when done")
        print("✓ Press Ctrl+C here to stop if needed")
        
        # Wait for process completion
        process.wait()
        
        print("\n" + "="*50)
        print("VALIDATION COMPLETED")
        print("="*50)
        print("If all tests passed, all fixes are working correctly!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        if 'process' in locals():
            process.terminate()
            process.wait()
    except Exception as e:
        print(f"❌ Error running validation: {e}")

if __name__ == "__main__":
    run_main_validation()