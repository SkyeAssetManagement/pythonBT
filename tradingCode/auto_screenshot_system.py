#!/usr/bin/env python3
"""Automated screenshot system for dashboard verification"""

import sys
import os
import time
import threading
from datetime import datetime
from pathlib import Path

def setup_auto_screenshot():
    """Setup automatic screenshot capture for the dashboard"""
    
    # Import PyQt5 components
    from PyQt5.QtWidgets import QApplication, QWidget
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QPixmap
    
    screenshot_dir = Path("C:/Users/skyeAM/SkyeAM Dropbox/SAMresearch/ABtoPython/2025-08-08_2")
    screenshot_dir.mkdir(exist_ok=True)
    
    def capture_dashboard():
        """Capture screenshot of the dashboard window"""
        app = QApplication.instance()
        if not app:
            return None
            
        # Find the dashboard window
        dashboard = None
        for widget in app.topLevelWidgets():
            if isinstance(widget, QWidget) and widget.isVisible():
                if 'Dashboard' in widget.windowTitle() or widget.width() > 1000:
                    dashboard = widget
                    break
        
        if dashboard:
            timestamp = datetime.now().strftime('%H%M%S')
            filename = screenshot_dir / f"auto_capture_{timestamp}.png"
            
            # Capture the widget
            pixmap = dashboard.grab()
            pixmap.save(str(filename))
            
            print(f"\n[SCREENSHOT CAPTURED] {filename.name}")
            print(f"  Size: {pixmap.width()}x{pixmap.height()}")
            print(f"  Location: {filename}")
            
            # Also save a copy with fixed name for easy access
            latest_file = screenshot_dir / "latest_dashboard.png"
            pixmap.save(str(latest_file))
            print(f"  Also saved as: latest_dashboard.png")
            
            return filename
        else:
            print("\n[SCREENSHOT FAILED] Dashboard window not found")
            return None
    
    # Schedule multiple captures
    def schedule_captures():
        """Schedule screenshot captures at different intervals"""
        app = QApplication.instance()
        if app:
            # Capture after 3 seconds (initial load)
            QTimer.singleShot(3000, capture_dashboard)
            # Capture after 6 seconds (fully rendered)
            QTimer.singleShot(6000, capture_dashboard)
            # Capture after 9 seconds (final state)
            QTimer.singleShot(9000, capture_dashboard)
            # Auto-close after 12 seconds
            QTimer.singleShot(12000, lambda: app.quit())
    
    return schedule_captures

# Monkey-patch into step6_complete_final
def patch_dashboard():
    """Patch the dashboard to add auto-screenshot capability"""
    import step6_complete_final
    
    original_init = step6_complete_final.FinalTradingDashboard.__init__
    
    def patched_init(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        
        # Add screenshot timer
        from PyQt5.QtCore import QTimer
        
        def take_screenshots():
            timestamp = datetime.now().strftime('%H%M%S')
            screenshot_dir = Path("C:/Users/skyeAM/SkyeAM Dropbox/SAMresearch/ABtoPython/2025-08-08_2")
            filename = screenshot_dir / f"dashboard_{timestamp}.png"
            
            pixmap = self.grab()
            pixmap.save(str(filename))
            print(f"\n[AUTO-SCREENSHOT] Saved: {filename.name}")
        
        # Schedule screenshots
        QTimer.singleShot(5000, take_screenshots)
        QTimer.singleShot(8000, take_screenshots)
        QTimer.singleShot(11000, lambda: self.close())
        
        return result
    
    step6_complete_final.FinalTradingDashboard.__init__ = patched_init
    
if __name__ == "__main__":
    print("Auto-screenshot system loaded")
    print("Import this module before running main.py")