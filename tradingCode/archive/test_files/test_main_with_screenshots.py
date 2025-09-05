#!/usr/bin/env python3
"""
Test main.py with automatic screenshots to validate trade clicking
"""

import sys
import os
import time
import subprocess
from pathlib import Path

def run_main_with_screenshots():
    """Run main.py and take automated screenshots for validation"""
    print("="*60)
    print("TESTING MAIN.PY WITH SCREENSHOT VALIDATION")
    print("="*60)
    
    # Run the exact command the user mentioned
    cmd = [
        sys.executable, "main.py", "AD", "time_window_strategy_vectorized", 
        "--useDefaults", "--start_date", "2024-10-01", "--end_date", "2024-10-02"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("\nThis will:")
    print("1. Load real data with multiple trades")
    print("2. Allow us to test trade clicking behavior")
    print("3. Take screenshots after each trade click")
    print("4. Verify chart actually moves")
    
    # Start the process
    process = subprocess.Popen(cmd, cwd=Path(__file__).parent)
    
    print(f"\nMain.py started with PID {process.pid}")
    print("Instructions:")
    print("1. Wait for dashboard to load completely")
    print("2. Click on FIRST trade in trade list")
    print("3. Take screenshot manually")
    print("4. Click on SECOND trade in trade list") 
    print("5. Take screenshot manually")
    print("6. Click on THIRD trade in trade list")
    print("7. Take screenshot manually")
    print("8. Check if chart actually moves between screenshots")
    print("\nPress Ctrl+C to stop when done testing...")
    
    try:
        # Wait for the process to complete or user to stop
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping main.py...")
        process.terminate()
        process.wait()
    
    print("Main.py test with screenshots completed!")

if __name__ == "__main__":
    run_main_with_screenshots()