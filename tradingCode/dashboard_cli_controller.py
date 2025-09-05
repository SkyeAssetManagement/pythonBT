#!/usr/bin/env python3
"""
Dashboard CLI Controller - Enables programmatic control of the dashboard
Allows auto-screenshot, auto-close, and headless operation
"""

import sys
import os
import time
from datetime import datetime
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow
import argparse

class DashboardController(QObject):
    """Controller for automated dashboard operations"""
    
    screenshot_taken = pyqtSignal(str)
    ready_to_close = pyqtSignal()
    
    def __init__(self, dashboard, args):
        super().__init__()
        self.dashboard = dashboard
        self.args = args
        self.screenshots = []
        
    def setup_automation(self):
        """Set up automated actions based on CLI arguments"""
        
        if self.args.screenshot_delay:
            # Schedule screenshot
            delay_ms = int(self.args.screenshot_delay * 1000)
            QTimer.singleShot(delay_ms, self.take_screenshot)
            print(f"[AUTOMATION] Screenshot scheduled in {self.args.screenshot_delay} seconds")
            
        if self.args.auto_close:
            # Schedule auto-close
            close_delay_ms = int(self.args.auto_close * 1000)
            QTimer.singleShot(close_delay_ms, self.close_dashboard)
            print(f"[AUTOMATION] Auto-close scheduled in {self.args.auto_close} seconds")
            
        if self.args.screenshot_interval:
            # Set up repeated screenshots
            self.screenshot_timer = QTimer()
            self.screenshot_timer.timeout.connect(self.take_screenshot)
            interval_ms = int(self.args.screenshot_interval * 1000)
            self.screenshot_timer.start(interval_ms)
            print(f"[AUTOMATION] Screenshot interval: every {self.args.screenshot_interval} seconds")
    
    def take_screenshot(self):
        """Capture screenshot of the dashboard"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            if self.args.screenshot_dir:
                os.makedirs(self.args.screenshot_dir, exist_ok=True)
                filename = os.path.join(self.args.screenshot_dir, f"{self.args.screenshot_prefix}_{timestamp}.png")
            else:
                filename = f"{self.args.screenshot_prefix}_{timestamp}.png"
            
            # Try different screenshot methods
            if hasattr(self.dashboard, 'export_dashboard_image'):
                self.dashboard.export_dashboard_image(filename)
                print(f"[SCREENSHOT] Saved: {filename} (export method)")
            else:
                pixmap = self.dashboard.grab()
                pixmap.save(filename)
                print(f"[SCREENSHOT] Saved: {filename} (grab method)")
            
            self.screenshots.append(filename)
            self.screenshot_taken.emit(filename)
            
            # If in headless mode, close after first screenshot
            if self.args.headless and not self.args.auto_close:
                QTimer.singleShot(1000, self.close_dashboard)
                
        except Exception as e:
            print(f"[ERROR] Screenshot failed: {e}")
    
    def close_dashboard(self):
        """Close the dashboard"""
        print(f"[AUTOMATION] Closing dashboard")
        if self.screenshots:
            print(f"[AUTOMATION] Screenshots taken: {len(self.screenshots)}")
            for screenshot in self.screenshots:
                print(f"  - {screenshot}")
        self.ready_to_close.emit()
        QApplication.quit()

def add_cli_controls(parser=None):
    """Add CLI control arguments to parser"""
    if parser is None:
        parser = argparse.ArgumentParser(description='Dashboard CLI Controls')
    
    cli_group = parser.add_argument_group('Dashboard Automation')
    
    cli_group.add_argument(
        '--screenshot-delay',
        type=float,
        default=0,
        help='Take screenshot after N seconds (0=disabled)'
    )
    
    cli_group.add_argument(
        '--screenshot-interval',
        type=float,
        default=0,
        help='Take screenshots every N seconds (0=disabled)'
    )
    
    cli_group.add_argument(
        '--screenshot-dir',
        type=str,
        default='C:/Users/skyeAM/SkyeAM Dropbox/SAMresearch/ABtoPython/2025-08-08_2',
        help='Directory to save screenshots'
    )
    
    cli_group.add_argument(
        '--screenshot-prefix',
        type=str,
        default='dashboard',
        help='Prefix for screenshot filenames'
    )
    
    cli_group.add_argument(
        '--auto-close',
        type=float,
        default=0,
        help='Auto-close dashboard after N seconds (0=disabled)'
    )
    
    cli_group.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (take screenshot and close)'
    )
    
    cli_group.add_argument(
        '--no-show',
        action='store_true',
        help='Do not show dashboard window (for testing)'
    )
    
    return parser

def apply_cli_controls(dashboard, args):
    """Apply CLI controls to a dashboard instance"""
    
    # Create controller
    controller = DashboardController(dashboard, args)
    
    # Store controller reference
    dashboard._cli_controller = controller
    
    # Set up automation
    controller.setup_automation()
    
    # Handle headless mode
    if args.headless:
        print("[AUTOMATION] Running in headless mode")
        if not args.screenshot_delay:
            # Default to 5 seconds for headless screenshot
            args.screenshot_delay = 5
            QTimer.singleShot(5000, controller.take_screenshot)
        if not args.auto_close:
            # Default to close after 10 seconds in headless
            args.auto_close = 10
            QTimer.singleShot(10000, controller.close_dashboard)
    
    # Handle no-show mode
    if args.no_show:
        print("[AUTOMATION] Dashboard window will not be shown")
        # Still take screenshots if requested
        if args.screenshot_delay:
            QTimer.singleShot(int(args.screenshot_delay * 1000), controller.take_screenshot)
    
    return controller

def patch_main_for_cli():
    """Patch main.py to support CLI controls"""
    
    # Monkey patch QMainWindow.show to add CLI controls
    original_show = QMainWindow.show
    
    def patched_show(self):
        """Enhanced show with CLI controls"""
        
        # Check if we have CLI args
        if hasattr(sys, 'argv'):
            # Parse CLI args for dashboard controls
            parser = argparse.ArgumentParser(add_help=False)
            parser = add_cli_controls(parser)
            args, unknown = parser.parse_known_args()
            
            # Apply controls if any are specified
            if any([args.screenshot_delay, args.auto_close, args.headless, args.screenshot_interval]):
                print("[AUTOMATION] CLI controls detected, applying...")
                apply_cli_controls(self, args)
            
            # Handle no-show
            if args.no_show:
                print("[AUTOMATION] Skipping window show")
                return
        
        # Call original show
        return original_show(self)
    
    QMainWindow.show = patched_show
    print("[AUTOMATION] Dashboard CLI controls patched")

# Auto-patch when imported
patch_main_for_cli()

if __name__ == "__main__":
    print("Dashboard CLI Controller Module")
    print("=" * 50)
    print("This module adds CLI controls to the dashboard:")
    print("  --screenshot-delay N   : Take screenshot after N seconds")
    print("  --screenshot-interval N: Take screenshots every N seconds")
    print("  --screenshot-dir PATH  : Directory for screenshots")
    print("  --auto-close N         : Close dashboard after N seconds")
    print("  --headless             : Run headless (screenshot and close)")
    print("  --no-show              : Don't show window")
    print()
    print("Import this module before running main.py to enable CLI controls")
    print("Example:")
    print("  python -c \"import dashboard_cli_controller; import main; main.main('ES', 'simpleSMA')\" --screenshot-delay 5 --auto-close 10")