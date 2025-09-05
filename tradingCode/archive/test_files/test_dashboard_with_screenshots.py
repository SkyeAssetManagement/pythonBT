#!/usr/bin/env python3
"""Test dashboard and capture screenshots for each step verification"""

import sys
import os
import time
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
import traceback

def test_and_capture_dashboard():
    """Run dashboard and capture screenshot after it loads"""
    
    print("DASHBOARD SCREENSHOT TEST")
    print("=" * 50)
    
    try:
        # Import the main function that runs the full pipeline
        from main import main
        
        # Create a custom class to capture the dashboard
        class DashboardCapture:
            def __init__(self):
                self.dashboard = None
                self.app = None
                
            def run_and_capture(self):
                """Run main.py and capture dashboard"""
                print("Running main.py ES simpleSMA...")
                
                # Set up arguments
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument('symbol', default='ES')
                parser.add_argument('strategy', default='simpleSMA')
                parser.add_argument('--config', default=None)
                parser.add_argument('--start_date', default=None)
                parser.add_argument('--end_date', default=None)
                parser.add_argument('--no-viz', action='store_true', default=False)
                parser.add_argument('--useDefaults', action='store_true', default=False)
                parser.add_argument('--mplfinance', action='store_true', default=False)
                parser.add_argument('--intraday', action='store_true', default=False)
                
                args = parser.parse_args(['ES', 'simpleSMA'])
                
                # Monkey-patch to capture dashboard before showing
                original_show = QMainWindow.show
                captured_dashboard = []
                
                def capture_show(self):
                    """Capture dashboard and schedule screenshot"""
                    captured_dashboard.append(self)
                    
                    # Schedule screenshot after 3 seconds
                    QTimer.singleShot(3000, lambda: self.capture_screenshot())
                    
                    # Call original show
                    return original_show(self)
                
                QMainWindow.show = capture_show
                
                # Add screenshot method to dashboard
                def capture_screenshot(self):
                    """Capture and save screenshot"""
                    try:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        folder = "C:/Users/skyeAM/SkyeAM Dropbox/SAMresearch/ABtoPython/2025-08-08_2"
                        filename = f"{folder}/step_verification_{timestamp}.png"
                        
                        # Try to use dashboard's export method
                        if hasattr(self, 'export_dashboard_image'):
                            self.export_dashboard_image(filename)
                            print(f"Screenshot saved: {filename}")
                        else:
                            # Fallback to widget grab
                            pixmap = self.grab()
                            pixmap.save(filename)
                            print(f"Screenshot saved (fallback): {filename}")
                        
                        # Close after screenshot
                        QTimer.singleShot(1000, lambda: QApplication.quit())
                        
                    except Exception as e:
                        print(f"Screenshot error: {e}")
                        QApplication.quit()
                
                QMainWindow.capture_screenshot = capture_screenshot
                
                # Run main
                try:
                    main(args)
                except SystemExit:
                    # Normal exit from dashboard
                    pass
                
                if captured_dashboard:
                    print(f"Dashboard captured: {captured_dashboard[0]}")
                    return True
                    
                return False
                
        capture = DashboardCapture()
        return capture.run_and_capture()
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_and_capture_dashboard()
    
    if success:
        print("\nDASHBOARD TEST COMPLETE")
        print("Check 2025-08-08_2 folder for screenshot")
    else:
        print("\nDASHBOARD TEST FAILED")