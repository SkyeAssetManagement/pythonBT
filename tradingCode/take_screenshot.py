"""
Simple screenshot capture to debug dashboard
"""
import time
import subprocess
import sys
from pathlib import Path

def take_screenshot():
    """Take a screenshot using Windows built-in tools"""
    
    # Wait a moment for the dashboard to be ready
    time.sleep(2)
    
    # Use PowerShell to take a screenshot
    script_dir = Path(__file__).parent
    screenshot_path = script_dir / "current_dashboard.png"
    
    # PowerShell command to take screenshot
    powershell_cmd = f"""
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
    $bitmap.Save('{screenshot_path}')
    $graphics.Dispose()
    $bitmap.Dispose()
    """
    
    try:
        subprocess.run([
            "powershell", "-Command", powershell_cmd
        ], check=True, capture_output=True, text=True)
        
        print(f"Screenshot saved to: {screenshot_path}")
        return str(screenshot_path)
        
    except subprocess.CalledProcessError as e:
        print(f"Screenshot failed: {e}")
        return None

if __name__ == "__main__":
    screenshot_path = take_screenshot()
    if screenshot_path:
        print("SUCCESS: Screenshot captured")
    else:
        print("ERROR: Screenshot failed")