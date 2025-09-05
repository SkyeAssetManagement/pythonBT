"""
Test the candlestick fix with the exact user command and take a screenshot
"""

import subprocess
import time
import sys
from pathlib import Path

def test_fix_with_screenshot():
    """Test the fix by running the exact command and taking a screenshot"""
    print("="*70)
    print("TESTING CANDLESTICK FIX WITH EXACT COMMAND")
    print("="*70)
    
    # The exact command the user runs
    cmd = [
        sys.executable, "main.py", 
        "ES", "time_window_strategy_vectorized", 
        "--useDefaults", "--start", "2020-01-01"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("This should now show proper candlesticks, not tiny dots!")
    print("="*70)
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        
        print("Dashboard launching...")
        print("Wait 10 seconds for dashboard to fully load...")
        
        # Wait for dashboard to load
        time.sleep(10)
        
        # Take screenshot using Windows built-in tool
        if sys.platform == 'win32':
            screenshot_cmd = [
                'powershell', '-Command',
                '''
                Add-Type -AssemblyName System.Windows.Forms
                Add-Type -AssemblyName System.Drawing
                $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
                $bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
                $graphics = [System.Drawing.Graphics]::FromImage($bmp)
                $graphics.CopyFromScreen(0, 0, 0, 0, $bmp.Size)
                $bmp.Save("CANDLESTICK_FIX_TEST_RESULT.png")
                $graphics.Dispose()
                $bmp.Dispose()
                Write-Host "Screenshot saved"
                '''
            ]
            
            print("Taking screenshot...")
            subprocess.run(screenshot_cmd, cwd=Path(__file__).parent, timeout=10)
            
            screenshot_file = Path(__file__).parent / "CANDLESTICK_FIX_TEST_RESULT.png"
            if screenshot_file.exists():
                print(f"SUCCESS: Screenshot saved to {screenshot_file}")
                print("\\nPlease check the screenshot:")
                print("1. Candlesticks should be visible (not tiny dots)")
                print("2. Should have proper thin rectangular shapes")
                print("3. Should have visible wicks (vertical lines)")
                print("4. White bodies for up candles, red for down candles")
            else:
                print("Failed to save screenshot")
        
        # Let it run for a bit more
        print("\\nDashboard should be running...")
        print("You can now:")
        print("1. Check if candlesticks are visible")
        print("2. Try zooming in/out")
        print("3. Use pan controls")
        print("4. Compare to your reference screenshot")
        print("\\nPress Ctrl+C to stop when you're done testing")
        
        # Wait indefinitely until user stops it
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\\nStopping dashboard...")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main function"""
    print("Testing candlestick fix with exact user command...")
    success = test_fix_with_screenshot()
    
    if success:
        print("\\nTest completed. Check the dashboard and screenshot!")
    else:
        print("\\nTest failed - check error messages above")

if __name__ == "__main__":
    main()