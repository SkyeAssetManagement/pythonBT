"""
Test the exact command with debug output to see what's happening with candlesticks
"""

import subprocess
import sys
import time
from pathlib import Path

def run_with_debug():
    """Run the exact command and capture debug output"""
    print("="*70)
    print("TESTING EXACT COMMAND WITH DEBUG OUTPUT")
    print("="*70)
    
    cmd = [
        sys.executable, "main.py", 
        "ES", "time_window_strategy_vectorized", 
        "--useDefaults", "--start", "2020-01-01"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("Looking for CANDLESTICK DEBUG messages...")
    print("="*70)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path(__file__).parent,
            bufsize=1
        )
        
        candlestick_messages = []
        other_messages = []
        
        start_time = time.time()
        while time.time() - start_time < 45:  # Wait longer for dashboard
            if process.poll() is not None:
                break
                
            line = process.stdout.readline()
            if line:
                line = line.strip()
                if "CANDLESTICK DEBUG" in line:
                    candlestick_messages.append(line)
                    print(f">>> {line}")
                elif any(keyword in line.lower() for keyword in ['error', 'exception', 'failed']):
                    other_messages.append(line)
                    print(f"!!! {line}")
                elif "dashboard" in line.lower() or "chart" in line.lower():
                    other_messages.append(line)
                    print(f"    {line}")
        
        # Clean up
        if process.poll() is None:
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
        
        print("="*70)
        print("ANALYSIS")
        print("="*70)
        print(f"Candlestick debug messages: {len(candlestick_messages)}")
        print(f"Other relevant messages: {len(other_messages)}")
        
        if candlestick_messages:
            print("\\nCANDLESTICK DEBUG SUMMARY:")
            for msg in candlestick_messages[-10:]:  # Last 10 messages
                print(f"  {msg}")
        else:
            print("\\n*** NO CANDLESTICK DEBUG MESSAGES FOUND ***")
            print("This suggests candlestick rendering is not being called!")
        
        if "error" in ''.join(other_messages).lower():
            print("\\nERRORS DETECTED:")
            for msg in other_messages:
                if "error" in msg.lower():
                    print(f"  {msg}")
        
        return candlestick_messages, other_messages
        
    except Exception as e:
        print(f"Failed to run command: {e}")
        return [], []

def main():
    """Main debug function"""
    candlestick_msgs, other_msgs = run_with_debug()
    
    print("\\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if not candlestick_msgs:
        print("PROBLEM IDENTIFIED: Candlestick rendering is not being called at all!")
        print("This explains why only dots and arrows are visible in the dashboard.")
        print("The issue is not with the candlestick width, but with the rendering pipeline.")
    else:
        print("Candlestick rendering IS being called.")
        print("The issue might be with the drawing parameters or coordinates.")

if __name__ == "__main__":
    main()