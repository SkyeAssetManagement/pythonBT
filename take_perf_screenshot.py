"""
Click on Performance Stats & Charts tab and take screenshot
"""
import time
import pyautogui

# Wait a moment
time.sleep(1)

# Click on the Performance Stats & Charts tab (approximate position)
# The tab is around the middle-top of the screen
pyautogui.click(200, 50)  # Adjust based on tab position
time.sleep(2)

# Take screenshot
screenshot = pyautogui.screenshot()
screenshot.save('gui_perf_stats_tab.png')
print('Performance Stats tab screenshot saved!')

# Also click on PermuteAlpha tab
pyautogui.click(300, 50)
time.sleep(2)

screenshot2 = pyautogui.screenshot()
screenshot2.save('gui_permute_alpha_tab.png')
print('PermuteAlpha tab screenshot saved!')