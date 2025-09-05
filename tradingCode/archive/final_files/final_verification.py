"""
Final verification that the candlestick fix is working
"""

import subprocess
import sys
import time
import threading

def run_dashboard_test():
    """Run the dashboard and monitor for errors"""
    print("="*70)
    print("FINAL VERIFICATION: CANDLESTICK FIX")
    print("="*70)
    
    cmd = [
        sys.executable, "main.py", 
        "ES", "time_window_strategy_vectorized", 
        "--useDefaults", "--start", "2020-01-01"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nMonitoring output for:")
    print("- SimpleCandlestickItem creation")
    print("- Drawing success")
    print("- Any AttributeError with update_range")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Track what we find
        simple_candlestick_created = False
        candlesticks_drawn = False
        update_range_error = False
        dashboard_launched = False
        
        start_time = time.time()
        while time.time() - start_time < 20:  # 20 second timeout
            if process.poll() is not None:
                break
                
            line = process.stdout.readline()
            if not line:
                continue
                
            line = line.strip()
            print(line)
            
            # Check for key indicators
            if "SIMPLE CANDLESTICK ITEM CREATED" in line:
                simple_candlestick_created = True
                print("[OK] SimpleCandlestickItem created!")
            
            if "SIMPLE CANDLESTICK: Drew" in line:
                candlesticks_drawn = True
                print("[OK] Candlesticks drawn successfully!")
            
            if "AttributeError" in line and "update_range" in line:
                update_range_error = True
                print("[X] update_range error found!")
            
            if "Dashboard launched successfully" in line:
                dashboard_launched = True
                print("[OK] Dashboard launched successfully!")
                # Give it a moment then terminate
                time.sleep(2)
                break
        
        # Clean up process
        if process.poll() is None:
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()
        
        # Results
        print("\n" + "="*70)
        print("VERIFICATION RESULTS:")
        print("="*70)
        print(f"SimpleCandlestickItem created: {'[OK]' if simple_candlestick_created else '[X]'}")
        print(f"Candlesticks drawn: {'[OK]' if candlesticks_drawn else '[X]'}")
        print(f"Dashboard launched: {'[OK]' if dashboard_launched else '[X]'}")
        print(f"update_range error: {'[X]' if update_range_error else '[OK] (No errors)'}")
        
        if simple_candlestick_created and candlesticks_drawn and not update_range_error:
            print("\n*** SUCCESS: CANDLESTICK FIX IS WORKING! ***")
            print("The black blob issue should be resolved.")
            print("Candlesticks should now appear as proper white/red rectangles.")
            return True
        else:
            print("\n*** ISSUES DETECTED ***")
            if not simple_candlestick_created:
                print("- SimpleCandlestickItem not being created")
            if not candlesticks_drawn:
                print("- Candlesticks not being drawn")
            if update_range_error:
                print("- update_range method error still present")
            return False
            
    except Exception as e:
        print(f"Error during verification: {e}")
        return False

if __name__ == "__main__":
    success = run_dashboard_test()
    print(f"\nFinal result: {'PASS' if success else 'FAIL'}")