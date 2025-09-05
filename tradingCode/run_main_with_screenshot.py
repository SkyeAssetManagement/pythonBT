#!/usr/bin/env python3
"""Run main.py and automatically capture screenshot after dashboard loads"""

import sys
import os
import time
from datetime import datetime
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication

# Monkey patch to add screenshot capability
original_show = QMainWindow.show
dashboards = []

def patched_show(self):
    """Add auto-screenshot when dashboard shows"""
    dashboards.append(self)
    print(f"[DASHBOARD] {self.windowTitle() if hasattr(self, 'windowTitle') else 'Window'} is showing")
    
    def capture_screenshots():
        """Capture screenshot of dashboard"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder = "C:/Users/skyeAM/SkyeAM Dropbox/SAMresearch/ABtoPython/2025-08-08_2"
        
        for i, dashboard in enumerate(dashboards):
            try:
                filename = f"{folder}/main_py_verification_{timestamp}_{i}.png"
                
                if hasattr(dashboard, 'export_dashboard_image'):
                    dashboard.export_dashboard_image(filename)
                    print(f"[SUCCESS] Screenshot saved: main_py_verification_{timestamp}_{i}.png")
                else:
                    pixmap = dashboard.grab()
                    pixmap.save(filename)
                    print(f"[SUCCESS] Screenshot saved (grab): main_py_verification_{timestamp}_{i}.png")
                    
            except Exception as e:
                print(f"[ERROR] Screenshot {i} failed: {e}")
        
        # Close after screenshots
        QTimer.singleShot(3000, lambda: QApplication.quit())
    
    # Schedule screenshot after 10 seconds to ensure data loads
    QTimer.singleShot(10000, capture_screenshots)
    
    return original_show(self)

# Apply patch
QMainWindow.show = patched_show

# Now import and run main
from main import main

print("=" * 70)
print("RUNNING MAIN.PY WITH PRODUCTION DATA")
print("=" * 70)
print("Command: python main.py ES simpleSMA")
print("Screenshot will be captured after 10 seconds")
print()

try:
    # Run main with production parameters
    main(
        symbol='ES',
        strategy_name='simpleSMA',
        config_path='config.yaml',
        start_date=None,
        end_date=None,
        launch_viz=True,
        use_defaults=False,
        intraday_performance=False
    )
    print("\n[COMPLETE] main.py finished")
    
except SystemExit:
    print("\n[COMPLETE] Dashboard closed normally")
    
except Exception as e:
    print(f"\n[ERROR] main.py failed: {e}")
    import traceback
    traceback.print_exc()