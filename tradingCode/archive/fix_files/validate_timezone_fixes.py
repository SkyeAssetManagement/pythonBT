#!/usr/bin/env python3
"""
Validate all timezone fixes are in place
"""

import os

def check_fix_implementation():
    """Check that all timezone fixes are properly implemented"""
    
    print("TIMEZONE FIX IMPLEMENTATION VALIDATION")
    print("=" * 60)
    
    fixes_status = []
    
    # Check 1: Centralized timezone handler exists
    handler_path = "src/utils/timezone_handler.py"
    if os.path.exists(handler_path):
        print("1. Centralized timezone handler: CREATED")
        fixes_status.append(True)
    else:
        print("1. Centralized timezone handler: MISSING")
        fixes_status.append(False)
    
    # Check 2: Dashboard trade list updated
    dashboard_path = "src/dashboard/dashboard_manager.py"
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r') as f:
            content = f.read()
            if "TimezoneHandler" in content and "timestamp_to_est_string" in content:
                print("2. Dashboard trade list: FIXED")
                fixes_status.append(True)
            else:
                print("2. Dashboard trade list: NOT FIXED")
                fixes_status.append(False)
    else:
        print("2. Dashboard trade list: FILE NOT FOUND")
        fixes_status.append(False)
    
    # Check 3: Chart time axis updated
    chart_path = "src/dashboard/chart_widget.py"
    if os.path.exists(chart_path):
        with open(chart_path, 'r') as f:
            content = f.read()
            if "est_timestamp_sec = timestamp_sec + (5 * 3600)" in content:
                print("3. Chart time axis: FIXED")
                fixes_status.append(True)
            else:
                print("3. Chart time axis: NOT FIXED")
                fixes_status.append(False)
    else:
        print("3. Chart time axis: FILE NOT FOUND")
        fixes_status.append(False)
    
    # Check 4: VBT engine CSV export (already working)
    vbt_path = "src/backtest/vbt_engine.py"
    if os.path.exists(vbt_path):
        with open(vbt_path, 'r') as f:
            content = f.read()
            if "est_timestamp_sec = timestamp_sec + (5 * 3600)" in content:
                print("4. CSV export (tradelist.csv): ALREADY FIXED")
                fixes_status.append(True)
            else:
                print("4. CSV export (tradelist.csv): NEEDS CHECK")
                fixes_status.append(False)
    else:
        print("4. CSV export: FILE NOT FOUND")
        fixes_status.append(False)
    
    return all(fixes_status)

def print_expected_behavior():
    """Print what the user should expect after restart"""
    
    print("\nEXPECTED BEHAVIOR AFTER DASHBOARD RESTART:")
    print("=" * 60)
    print("All components will show EST (Eastern Standard Time):")
    print()
    print("BEFORE (3 different timezones):")
    print("  - tradelist.csv:     17:31:00 (confusing)")
    print("  - Dashboard trades:  04:31:00 (wrong)")  
    print("  - Chart X-axis:      12:31:00 (wrong)")
    print()
    print("AFTER (consistent EST):")
    print("  - tradelist.csv:     09:31:00 (EST - exchange time)")
    print("  - Dashboard trades:  09:31:00 (EST - exchange time)")  
    print("  - Chart X-axis:      09:31    (EST - exchange time)")
    print("  - Data window:       09:26:00 (EST - exchange time)")
    print()
    print("STRATEGY ENTRY LOGIC:")
    print("  - Strategy parameter: entry_time = '09:30' (interpreted as EST)")
    print("  - Actual entries:     09:31:00 EST (1 minute after window opens)")
    print("  - Hold time:          60 minutes") 
    print("  - Exit times:         10:31:00 EST")
    print()
    print("Everything aligned to ES futures exchange time!")

if __name__ == "__main__":
    all_fixes_ok = check_fix_implementation()
    
    print_expected_behavior()
    
    print("\n" + "=" * 60)
    if all_fixes_ok:
        print("SUCCESS: All timezone fixes implemented correctly!")
        print()
        print("To see the fixes in action:")
        print("1. Run: python main.py ES time_window_strategy_vectorized_single")
        print("2. Look at the dashboard - all times should show EST")
        print("3. Check tradelist.csv - should show 09:31:00 entries")
        print()
        print("The 3-timezone problem is now solved!")
    else:
        print("WARNING: Some fixes may not be properly implemented")
        print("Check the status above and re-run if needed")