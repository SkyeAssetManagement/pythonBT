"""
Take screenshot of the GUI after navigating to Performance Stats tab
"""
import time
import pyautogui
import tkinter as tk
from OMtree_gui import OMtreeGUI

def take_gui_screenshots():
    # Create and run GUI
    root = tk.Tk()
    app = OMtreeGUI(root)
    
    # Function to take screenshots after GUI loads
    def capture_screenshots():
        print("Taking screenshots...")
        
        # Take full screen screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save('gui_fullscreen.png')
        print("Full screen saved")
        
        # Navigate to Performance Stats & Charts tab
        try:
            # Select the Performance Stats tab (index 2)
            app.notebook.select(2)
            root.update()
            time.sleep(1)
            
            # Take another screenshot
            screenshot2 = pyautogui.screenshot()
            screenshot2.save('gui_performance_tab.png')
            print("Performance tab screenshot saved")
            
            # Try each view option
            views = ['tradestats', 'charts', 'feature_timeline']
            for view in views:
                app.chart_var.set(view)
                app.switch_view()
                root.update()
                time.sleep(1)
                
                screenshot = pyautogui.screenshot()
                screenshot.save(f'gui_{view}.png')
                print(f"{view} screenshot saved")
                
        except Exception as e:
            print(f"Error navigating tabs: {e}")
        
        # Schedule GUI close
        root.after(2000, root.quit)
    
    # Schedule screenshot capture after GUI loads
    root.after(3000, capture_screenshots)
    
    # Run the GUI
    root.mainloop()

if __name__ == "__main__":
    take_gui_screenshots()