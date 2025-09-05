#!/usr/bin/env python3
"""
Debug trade navigation specifically with main.py data loading
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Import main.py components 
from main import main, VALID_SYMBOLS, VALID_STRATEGIES

def debug_main_trade_navigation():
    """Debug trade navigation with main.py data"""
    print("="*60)
    print("DEBUGGING TRADE NAVIGATION WITH MAIN.PY")
    print("="*60)
    
    # Set up command line arguments for main.py
    sys.argv = [
        "main.py", "AD", "time_window_strategy_vectorized", 
        "--useDefaults", "--start_date", "2024-10-01", "--end_date", "2024-10-02"
    ]
    
    print("Loading main.py with AD symbol and time_window_strategy_vectorized...")
    print("Date range: 2024-10-01 to 2024-10-02")
    
    # Create QApplication if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Call main function which will create and show dashboard
        dashboard = main()
        
        if dashboard:
            print("Dashboard loaded successfully!")
            print(f"Number of trades: {len(dashboard.trades_data) if hasattr(dashboard, 'trades_data') and dashboard.trades_data else 'Unknown'}")
            
            # Set up debug timers
            def take_initial_screenshot():
                print("\n=== INITIAL STATE ===")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"main_debug_initial_{timestamp}.png"
                pixmap = dashboard.grab()
                pixmap.save(filename)
                print(f"Screenshot saved: {filename}")
                
                # Show current viewport info
                if hasattr(dashboard, 'final_chart'):
                    print(f"Initial viewport: {dashboard.final_chart.viewport_start}-{dashboard.final_chart.viewport_end}")
                
                print("\nNow manually click on different trades in the trade list.")
                print("Watch console for debug messages.")
                print("Check if chart actually moves visually.")
            
            # Take initial screenshot after 3 seconds
            QTimer.singleShot(3000, take_initial_screenshot)
            
            # Run the application
            app.exec_()
            
        else:
            print("ERROR: Dashboard creation failed!")
            
    except Exception as e:
        print(f"ERROR: Failed to run main.py debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_main_trade_navigation()