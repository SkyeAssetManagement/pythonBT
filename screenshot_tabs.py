"""
Take screenshots of specific tabs in the GUI
"""
import time
import pyautogui
import tkinter as tk
from OMtree_gui import OMtreeGUI

def take_tab_screenshots():
    # Create and run GUI
    root = tk.Tk()
    app = OMtreeGUI(root)
    
    def capture_tabs():
        print("Capturing tab screenshots...")
        
        # Navigate to Performance Stats & Charts tab
        app.notebook.select(2)  # Index 2 is Performance Stats & Charts
        root.update()
        time.sleep(1)
        
        # Take screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save('gui_performance_stats.png')
        print("Performance Stats tab saved")
        
        # Navigate to PermuteAlpha tab
        app.notebook.select(4)  # Index 4 is PermuteAlpha
        root.update()
        time.sleep(1)
        
        screenshot = pyautogui.screenshot()
        screenshot.save('gui_permute_alpha.png')
        print("PermuteAlpha tab saved")
        
        # Schedule close
        root.after(1000, root.quit)
    
    # Schedule screenshot capture after GUI loads
    root.after(2000, capture_tabs)
    
    # Run the GUI
    root.mainloop()
    print("Done!")

if __name__ == "__main__":
    take_tab_screenshots()