#!/usr/bin/env python3
r"""
gui_environment_check.py

GUI Environment Detection and Dashboard Fix
Test GUI availability and implement fallback mechanisms
"""

import sys
import os
import threading
from pathlib import Path

def check_gui_environment():
    """
    Check if GUI environment is available for PyQt5
    Returns: (gui_available, error_message)
    """
    
    # Check 1: Main thread requirement
    if threading.current_thread() is not threading.main_thread():
        return False, "Dashboard must run in main thread"
    
    # Check 2: Display environment
    if os.name == 'nt':  # Windows
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Hide the window
            root.destroy()
            print("Windows GUI environment: AVAILABLE")
        except Exception as e:
            return False, f"Windows GUI not available: {e}"
    else:  # Unix/Linux
        display = os.environ.get('DISPLAY')
        if display is None:
            return False, "DISPLAY environment variable not set (headless environment)"
        print(f"Unix DISPLAY environment: {display}")
    
    # Check 3: PyQt5 availability
    try:
        from PyQt5 import QtWidgets, QtCore
        import pyqtgraph as pg
        print("PyQt5 modules: AVAILABLE")
    except ImportError as e:
        return False, f"PyQt5 not available: {e}"
    
    # Check 4: Qt Application test
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            # Try to create a test application
            test_app = QtWidgets.QApplication([])
            test_widget = QtWidgets.QWidget()
            test_widget.show()
            test_widget.hide()
            test_widget.close()
            test_app.quit()
            print("Qt Application test: PASSED")
        else:
            print("Qt Application already exists: OK")
    except Exception as e:
        return False, f"Qt Application test failed: {e}"
    
    return True, "GUI environment fully available"

def test_dashboard_launch_minimal():
    """Test minimal dashboard launch without hanging"""
    
    print("Testing minimal dashboard launch...")
    
    gui_available, message = check_gui_environment()
    print(f"GUI Environment Check: {message}")
    
    if not gui_available:
        print("SOLUTION: Use --no-viz flag to disable dashboard")
        print("COMMAND: python main.py ES time_window_strategy_vectorized --useDefaults --start \"2020-01-01\" --no-viz")
        return False
    
    # Test actual dashboard components
    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from src.dashboard.dashboard_manager import get_dashboard_manager
        from PyQt5 import QtWidgets
        
        # Create minimal test
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        
        print("Creating dashboard manager...")
        dashboard = get_dashboard_manager()
        
        print("Testing Qt initialization...")
        if dashboard.initialize_qt_app():
            print("Qt initialization: SUCCESS")
        else:
            raise Exception("Qt initialization failed")
        
        print("Testing window creation...")
        dashboard.create_main_window()
        print("Window creation: SUCCESS")
        
        # Don't show - just test creation
        print("Dashboard components test: PASSED")
        
        # Clean up
        if hasattr(dashboard, 'main_window') and dashboard.main_window:
            dashboard.main_window.close()
        
        app.quit()
        return True
        
    except Exception as e:
        print(f"Dashboard test failed: {e}")
        print("SOLUTION: Use --no-viz flag to disable dashboard")
        print("COMMAND: python main.py ES time_window_strategy_vectorized --useDefaults --start \"2020-01-01\" --no-viz")
        return False

if __name__ == "__main__":
    print("GUI Environment and Dashboard Test")
    print("=" * 50)
    
    success = test_dashboard_launch_minimal()
    
    if success:
        print("\nRESULT: Dashboard should work properly")
        print("COMMAND: python main.py ES time_window_strategy_vectorized --useDefaults --start \"2020-01-01\"")
    else:
        print("\nRESULT: Dashboard not available")
        print("RECOMMENDED COMMAND: python main.py ES time_window_strategy_vectorized --useDefaults --start \"2020-01-01\" --no-viz")