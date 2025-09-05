import time
import pyautogui
import os

# Wait for GUI to fully load
time.sleep(3)

# Take screenshot
screenshot = pyautogui.screenshot()

# Save screenshot
screenshot_path = "gui_screenshot.png"
screenshot.save(screenshot_path)

print(f"Screenshot saved to: {os.path.abspath(screenshot_path)}")