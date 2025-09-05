#!/usr/bin/env python3
"""
Test main.py with import fixes by calling it programmatically
"""

import subprocess
import sys
import time

def test_main_with_fixes():
    """Test main.py with the import fixes"""
    
    print("=== TESTING MAIN.PY WITH IMPORT FIXES ===")
    print("Running main.py GC simpleSMA for 15 seconds...")
    
    try:
        # Start main.py
        process = subprocess.Popen([
            sys.executable, "main.py", "GC", "simpleSMA"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Capture output for 15 seconds
        output_lines = []
        start_time = time.time()
        
        while time.time() - start_time < 15:
            try:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    print(f"OUTPUT: {line.strip()}")
                elif process.poll() is not None:
                    # Process ended
                    break
            except:
                break
        
        # Terminate if still running
        if process.poll() is None:
            print("\nTIMEOUT: Process still running (likely showing dashboard)")
            process.terminate()
            remaining_output, _ = process.communicate()
            if remaining_output:
                output_lines.extend(remaining_output.strip().split('\n'))
        
        # Analyze output
        full_output = '\n'.join(output_lines)
        
        print(f"\n=== ANALYSIS ===")
        
        # Check for the specific error we're trying to fix
        if "No module named 'dashboard'" in full_output:
            print("ERROR: Still getting 'No module named dashboard' error")
            return False
        else:
            print("SUCCESS: No 'No module named dashboard' error found")
        
        # Check for other indicators
        if "Error generating candlestick picture" in full_output:
            print("WARNING: Still getting candlestick generation errors")
        else:
            print("SUCCESS: No candlestick generation errors")
        
        if "Dashboard" in full_output and "launched" in full_output:
            print("SUCCESS: Dashboard launch detected")
        
        if "QPaintDevice: Cannot destroy paint device" in full_output:
            print("WARNING: Qt paint device warning present")
        else:
            print("SUCCESS: No Qt paint device warnings")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_main_with_fixes()
    if success:
        print("\nIMPORT FIXES APPEAR TO BE WORKING!")
    else:
        print("\nIMPORT FIXES NEED MORE WORK")