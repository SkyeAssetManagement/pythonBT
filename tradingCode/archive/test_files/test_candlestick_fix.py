"""
Test script to verify the candlestick fix is working
Runs the dashboard and automatically closes it to check for errors
"""

import subprocess
import sys
import time
import os
import signal
from threading import Timer

def run_test():
    print("="*60)
    print("TESTING CANDLESTICK FIX")
    print("="*60)
    
    # Command to run
    cmd = [
        sys.executable, "main.py", 
        "ES", "time_window_strategy_vectorized", 
        "--useDefaults", "--start", "2020-01-01"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("\nLooking for SimpleCandlestickItem debug messages...")
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Monitor output for 15 seconds
        start_time = time.time()
        debug_found = False
        error_found = False
        lines_captured = 0
        
        while time.time() - start_time < 15:
            if process.poll() is not None:
                print("Process finished naturally")
                break
            
            line = process.stdout.readline()
            if line:
                lines_captured += 1
                line = line.strip()
                print(f"[{lines_captured:03d}] {line}")
                
                # Check for our debug messages
                if "SIMPLE CANDLESTICK" in line:
                    debug_found = True
                    print(f"*** SUCCESS: {line}")
                
                # Check for errors
                if "AttributeError" in line and "update_range" in line:
                    error_found = True
                    print(f"*** ERROR FOUND: {line}")
                
                # Check if dashboard launched
                if "Dashboard launched successfully" in line:
                    print("*** DASHBOARD LAUNCHED - Will auto-close in 3 seconds ***")
                    time.sleep(3)
                    break
        
        # Kill the process if still running
        if process.poll() is None:
            print("Terminating process...")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
        
        print(f"\n" + "="*60)
        print("TEST RESULTS:")
        print("="*60)
        print(f"Lines captured: {lines_captured}")
        print(f"SimpleCandlestickItem debug found: {debug_found}")
        print(f"update_range error found: {error_found}")
        
        if debug_found and not error_found:
            print("*** SUCCESS: Fix appears to be working! ***")
            return True
        elif error_found:
            print("*** FAILURE: update_range error still present ***")
            return False
        else:
            print("*** UNCLEAR: No debug output found ***")
            return False
            
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)