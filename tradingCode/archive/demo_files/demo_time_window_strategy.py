#!/usr/bin/env python3
"""
Demo: Time Window Strategy
Shows how to use the time window strategy with main.py
"""

import subprocess
import sys
from pathlib import Path

def demo_time_window_strategy():
    """Demo the time window strategy with main.py"""
    
    print("=== TIME WINDOW STRATEGY DEMO ===")
    print()
    print("STRATEGY OVERVIEW:")
    print("- Enters positions gradually over a 5-minute window")
    print("- Entry time: configurable (e.g., 09:30 = enter 09:31-09:35)")
    print("- Entry price: (High + Low) / 2")
    print("- Position building: 20% per minute over 5 minutes")
    print("- Hold time: configurable minutes")
    print("- Exit: gradual over 5 minutes at (High + Low) / 2")
    print("- Direction: long or short")
    print()
    
    # Strategy parameters explained
    print("PARAMETERS:")
    print("- entry_time: '09:30', '10:00', '14:30', etc.")
    print("- direction: 'long' or 'short'")
    print("- hold_time: 30, 60, 90, 120+ minutes")
    print("- entry_spread: 3-10 minutes (default 5)")
    print("- max_trades_per_day: 1, 2, or 3")
    print()
    
    print("EXAMPLE USAGE:")
    print("python main.py AD time_window_strategy")
    print()
    
    # Ask if user wants to run demo
    user_input = input("Run demo with AD data? (y/N): ").strip().lower()
    
    if user_input == 'y':
        print("\\nRunning time window strategy demo...")
        print("This will test multiple parameter combinations and show optimization results")
        
        try:
            result = subprocess.run([
                sys.executable, "main.py", "AD", "time_window_strategy", "--no-viz"
            ], capture_output=True, text=True, timeout=180)
            
            print("\\n=== DEMO RESULTS ===")
            if result.stdout:
                print(result.stdout)
            
            if result.stderr:
                print("\\n=== ERRORS ===")
                print(result.stderr)
            
            if result.returncode == 0:
                print("\\nSUCCESS: Time window strategy demo completed!")
                print("Strategy is ready for optimization and live use")
            else:
                print(f"\\nDemo completed with return code: {result.returncode}")
            
        except subprocess.TimeoutExpired:
            print("\\nDemo timed out - strategy may be running too many combinations")
            print("Consider reducing parameter combinations for faster testing")
        except Exception as e:
            print(f"\\nDemo failed: {e}")
    
    else:
        print("\\nDemo skipped.")
        print("To run manually: python main.py AD time_window_strategy")
    
    print("\\n=== STRATEGY READY ===")
    print("The Time Window Strategy is now available as 'time_window_strategy'")
    print("Use with any symbol that has minute-level data")
    print("Perfect for testing time-based entry/exit strategies")

if __name__ == "__main__":
    demo_time_window_strategy()