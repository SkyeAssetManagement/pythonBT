#!/usr/bin/env python3
"""
Simple screenshot helper for validating dashboard trade clicks
"""

import sys
import time
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
import datetime

class ScreenshotHelper:
    """Helper to capture screenshots during trade testing"""
    
    def __init__(self, dashboard, output_dir="screenshots"):
        self.dashboard = dashboard
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.screenshot_count = 0
        
    def capture_screenshot(self, description=""):
        """Capture a screenshot of the dashboard"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_{timestamp}_{self.screenshot_count:03d}_{description}.png"
            filepath = self.output_dir / filename
            
            # Capture the dashboard window
            pixmap = self.dashboard.grab()
            success = pixmap.save(str(filepath))
            
            if success:
                print(f"SCREENSHOT: Saved {filename}")
                self.screenshot_count += 1
                return str(filepath)
            else:
                print(f"ERROR: Failed to save screenshot {filename}")
                return None
                
        except Exception as e:
            print(f"ERROR: Screenshot capture failed: {e}")
            return None

def manual_trade_clicking_test():
    """Manual test for trade clicking with screenshot capture"""
    print("="*80)
    print("MANUAL TRADE CLICKING TEST WITH SCREENSHOTS")
    print("="*80)
    
    # Import main components
    try:
        from main import main
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    # Set up sys.argv for main.py
    sys.argv = [
        "main.py", "AD", "time_window_strategy_vectorized", 
        "--useDefaults", "--start_date", "2024-10-01", "--end_date", "2024-10-02"
    ]
    
    # Create QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Call main() - this should return the dashboard
        dashboard = main(
            symbol="AD",
            strategy_name="time_window_strategy_vectorized",
            config_path="config.yaml",
            start_date="2024-10-01", 
            end_date="2024-10-02",
            use_defaults=True
        )
        
        if not dashboard:
            print("ERROR: Dashboard creation failed!")
            return
            
        print(f"SUCCESS: Dashboard loaded!")
        
        # Create screenshot helper
        screenshot_helper = ScreenshotHelper(dashboard)
        
        # Take initial screenshot after dashboard loads
        def take_initial_screenshot():
            print("Taking initial dashboard screenshot...")
            screenshot_helper.capture_screenshot("initial_load")
            print()
            print_manual_instructions(dashboard)
        
        # Wait 3 seconds then take initial screenshot
        QTimer.singleShot(3000, take_initial_screenshot)
        
        # Add screenshot buttons or keyboard shortcuts if needed
        def setup_manual_controls():
            """Set up manual screenshot controls"""
            
            def capture_current():
                viewport_info = ""
                if hasattr(dashboard, 'final_chart'):
                    chart = dashboard.final_chart
                    viewport_info = f"viewport_{chart.viewport_start}_{chart.viewport_end}"
                
                screenshot_helper.capture_screenshot(f"manual_{viewport_info}")
            
            # You could add keyboard shortcuts here if needed
            # For now, we'll rely on the console instructions
            
        setup_manual_controls()
        
        # Run the application
        app.exec_()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def print_manual_instructions(dashboard):
    """Print manual testing instructions"""
    print("MANUAL TESTING INSTRUCTIONS:")
    print("-" * 50)
    print("1. Take note of the current chart position")
    print("2. Click on the FIRST trade in the trade list")
    print("3. Watch the chart - does it move to a new position?")
    print("4. Check console for debug messages showing viewport changes")
    print("5. Click on the SECOND trade in the trade list") 
    print("6. Watch the chart - does it move again to show the second trade?")
    print("7. Click on the THIRD trade in the trade list")
    print("8. Watch the chart - does it move again to show the third trade?")
    print()
    print("WHAT TO LOOK FOR:")
    print("- Console messages: 'TRADE NAV: Setting viewport X-Y'")
    print("- Console messages: 'VIEWPORT CHANGE: Updating time axis...'")
    print("- Visual chart movement to different price levels")
    print("- Time axis labels updating when chart moves")
    print("- Crosshair showing correct time format")
    print()
    
    if hasattr(dashboard, 'trades_data') and dashboard.trades_data:
        print("AVAILABLE TRADES:")
        for i, trade in enumerate(dashboard.trades_data[:5]):
            print(f"  Trade {i+1}: {trade.trade_id} at bar {trade.entry_time} (Side: {trade.side})")
    
    print()
    print("SCREENSHOTS: Check the 'screenshots' folder for captured images")
    print("Close the dashboard window when testing is complete.")
    print()

if __name__ == "__main__":
    manual_trade_clicking_test()