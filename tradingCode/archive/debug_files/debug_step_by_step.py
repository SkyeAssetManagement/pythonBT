# debug_step_by_step.py
# Systematic debugging of dashboard issues

import sys
import time
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

def take_screenshot_after_delay():
    """Take screenshot after dashboard loads"""
    app = QApplication.instance()
    if app:
        # Find the dashboard window
        for widget in app.topLevelWidgets():
            if isinstance(widget, FinalTradingDashboard):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"debug_dashboard_state_{timestamp}.png"
                
                pixmap = widget.grab()
                success = pixmap.save(filename)
                
                if success:
                    print(f"DEBUG SCREENSHOT: Saved {filename}")
                    print(f"DEBUG: Window size: {widget.size()}")
                    print(f"DEBUG: Screenshot saved to current directory")
                else:
                    print("DEBUG SCREENSHOT: Failed to save")
                
                # Print debug info about current state
                if hasattr(widget, 'final_chart') and widget.final_chart:
                    print(f"DEBUG: Chart viewport: {widget.final_chart.viewport_start}-{widget.final_chart.viewport_end}")
                    print(f"DEBUG: Data length: {widget.final_chart.data_length}")
                    if hasattr(widget.final_chart, 'datetime_data') and widget.final_chart.datetime_data is not None:
                        print(f"DEBUG: Has datetime data: {len(widget.final_chart.datetime_data)} entries")
                    else:
                        print("DEBUG: No datetime data available")
                
                return filename
    return None

def main():
    """Debug main function"""
    print("="*80)
    print("DEBUG: Starting systematic step-by-step debugging")
    print("="*80)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    # Create test data
    ohlcv_data, trades_csv_path = create_ultimate_test_data()
    
    # Create dashboard
    dashboard = FinalTradingDashboard()
    dashboard.setWindowTitle("DEBUG: Step-by-Step Dashboard")
    dashboard.resize(1920, 1200)
    dashboard.show()
    
    # Load data
    success = dashboard.load_ultimate_dataset(ohlcv_data, trades_csv_path)
    
    print(f"DEBUG: Dataset loaded: {'SUCCESS' if success else 'FAILED'}")
    
    # Schedule screenshot after 3 seconds
    QTimer.singleShot(3000, take_screenshot_after_delay)
    
    # Schedule app exit after 10 seconds  
    QTimer.singleShot(10000, app.quit)
    
    print("DEBUG: Dashboard will take screenshot in 3 seconds, then exit in 10 seconds")
    app.exec_()
    
    print("DEBUG: Complete")

if __name__ == "__main__":
    main()