#!/usr/bin/env python3
"""
Debug trade clicking with automatic screenshots using main.py data
"""

import sys
import time
import os
import subprocess
from pathlib import Path

def run_main_with_screenshot_validation():
    """Run main.py and take screenshots after each trade click for validation"""
    print("="*80)
    print("DEBUGGING TRADE CLICKS WITH MAIN.PY + SCREENSHOTS")
    print("="*80)
    
    # Clean up old screenshots
    screenshot_pattern = "trade_click_test_*.png"
    for old_screenshot in Path(".").glob(screenshot_pattern):
        old_screenshot.unlink()
        
    print("Cleaned up old screenshots")
    
    # The exact command the user uses
    cmd = [
        sys.executable, "main.py", "AD", "time_window_strategy_vectorized", 
        "--useDefaults", "--start_date", "2024-10-01", "--end_date", "2024-10-02"
    ]
    
    print("Starting main.py with command:")
    print(" ".join(cmd))
    print()
    
    print("TESTING PROCEDURE:")
    print("1. Wait for dashboard to load completely")
    print("2. Take initial screenshot")  
    print("3. Click FIRST trade in trade list")
    print("4. Take screenshot after 1st click")
    print("5. Click SECOND trade in trade list")
    print("6. Take screenshot after 2nd click") 
    print("7. Click THIRD trade in trade list")
    print("8. Take screenshot after 3rd click")
    print("9. Compare screenshots to see if chart actually moves")
    print()
    
    print("Expected behavior:")
    print("- Chart should visually move to different positions for each trade")
    print("- Viewport numbers in console should change")
    print("- Candlesticks should show different time periods")
    print()
    
    print("Screenshots will be saved as:")
    print("- trade_click_test_initial.png")
    print("- trade_click_test_after_trade_1.png")
    print("- trade_click_test_after_trade_2.png") 
    print("- trade_click_test_after_trade_3.png")
    print()
    
    print("Starting main.py now...")
    print("="*50)
    
    try:
        # Start main.py process
        process = subprocess.Popen(
            cmd, 
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"Main.py started with PID: {process.pid}")
        print("Monitoring output for trade clicks...")
        print()
        
        # Monitor output for trade click events
        trade_click_count = 0
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
                
            if output:
                print(output.strip())
                
                # Detect trade click events
                if "ULTIMATE SYNC: Trade" in output and "selected" in output:
                    trade_click_count += 1
                    print(f"\n>>> DETECTED TRADE CLICK #{trade_click_count} <<<")
                    print("Waiting 2 seconds for chart update...")
                    time.sleep(2)
                    
                    # Take screenshot using Windows built-in screenshot
                    timestamp = time.strftime("%H%M%S")
                    screenshot_name = f"trade_click_test_after_trade_{trade_click_count}_{timestamp}.png"
                    
                    print(f"Taking screenshot: {screenshot_name}")
                    screenshot_cmd = [
                        "powershell", "-Command",
                        f"Add-Type -AssemblyName System.Windows.Forms; "
                        f"[System.Windows.Forms.SendKeys]::SendWait('%{{PRTSC}}'); "
                        f"Start-Sleep 1"
                    ]
                    
                    try:
                        subprocess.run(screenshot_cmd, check=True, capture_output=True)
                        print(f"Screenshot taken: {screenshot_name}")
                    except Exception as e:
                        print(f"Screenshot failed: {e}")
                    
                    print("Continue clicking trades to test more...")
                    print()
                    
                    if trade_click_count >= 3:
                        print("Collected 3 trade click screenshots!")
                        print("Review the screenshots to verify chart movement.")
                        break
                        
        # Wait for process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"Error running test: {e}")
    
    print("\nTest completed!")
    print("Check the generated screenshots to verify trade clicking behavior.")

if __name__ == "__main__":
    run_main_with_screenshot_validation()