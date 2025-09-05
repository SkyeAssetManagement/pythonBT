"""
Script to launch GUI and take screenshot for visual feedback
"""
import time
import subprocess
import pyautogui
import os
import sys

def take_gui_screenshot():
    """Launch GUI and take screenshot after delay"""
    
    # Launch the GUI in background
    print("Launching GUI...")
    process = subprocess.Popen([sys.executable, 'launch.py'], 
                              cwd=r'C:\Users\jd\OM3',
                              shell=True)
    
    # Wait for GUI to fully load
    print("Waiting for GUI to load...")
    time.sleep(5)
    
    # Take screenshot
    print("Taking screenshot...")
    screenshot = pyautogui.screenshot()
    
    # Save screenshot
    screenshot_path = r'C:\Users\jd\OM3\gui_screenshot.png'
    screenshot.save(screenshot_path)
    print(f"Screenshot saved to: {screenshot_path}")
    
    # Also take a screenshot of just the active window if possible
    try:
        # Get the active window region
        active_window = pyautogui.getActiveWindow()
        if active_window:
            x, y, width, height = active_window.left, active_window.top, active_window.width, active_window.height
            window_screenshot = pyautogui.screenshot(region=(x, y, width, height))
            window_path = r'C:\Users\jd\OM3\gui_window_screenshot.png'
            window_screenshot.save(window_path)
            print(f"Window screenshot saved to: {window_path}")
    except Exception as e:
        print(f"Could not capture active window: {e}")
    
    return process

if __name__ == "__main__":
    process = take_gui_screenshot()
    
    # Keep the GUI running for a bit
    print("\nGUI is running. Press Enter to close it...")
    input()
    
    # Terminate the GUI process
    process.terminate()
    print("GUI closed.")