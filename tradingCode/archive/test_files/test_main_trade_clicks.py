#!/usr/bin/env python3
"""
Test main.py trade clicking with enhanced debugging
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_main_trade_clicking():
    """Test trade clicking with main.py data and debug output"""
    print("="*80)
    print("TESTING MAIN.PY TRADE CLICKING WITH DEBUG")
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
    
    print("Loading main.py with:")
    print("  Symbol: AD")
    print("  Strategy: time_window_strategy_vectorized") 
    print("  Date range: 2024-10-01 to 2024-10-02")
    print()
    
    # Create QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Call main() with proper arguments which returns the dashboard
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
        if hasattr(dashboard, 'trades_data'):
            print(f"Found {len(dashboard.trades_data)} trades")
            
            # Show first few trades for debugging
            for i, trade in enumerate(dashboard.trades_data[:5]):
                print(f"  Trade {i+1}: ID={trade.trade_id}, EntryTime={trade.entry_time}, Side={trade.side}")
        else:
            print("WARNING: No trades data found")
            
        print()
        print("TESTING INSTRUCTIONS:")
        print("1. Wait for dashboard to fully load")
        print("2. Click on FIRST trade in the trade list")
        print("3. Watch console output for debug messages")
        print("4. Note the viewport change in console")
        print("5. Click on SECOND trade in the trade list")
        print("6. Watch console - viewport should change again")
        print("7. Click on THIRD trade")
        print("8. Check if chart visually moves each time")
        print()
        print("Look for these debug messages:")
        print("- 'TRADE CLICK DEBUG: Row X, Column Y clicked'")
        print("- 'ULTIMATE SYNC DEBUG: Handler called...'") 
        print("- 'TRADE NAV: Setting viewport X-Y'")
        print("- 'TRADE NAV: Chart regenerated - vertices: A -> B'")
        print()
        print("If viewport changes but chart doesn't move visually, that's the bug!")
        print()
        
        def take_viewport_snapshot():
            """Take a snapshot of current viewport for debugging"""
            if hasattr(dashboard, 'final_chart'):
                chart = dashboard.final_chart
                print("="*50)
                print("VIEWPORT SNAPSHOT:")
                print(f"  Current viewport: {chart.viewport_start} - {chart.viewport_end}")
                print(f"  Data length: {chart.data_length}")
                print(f"  Vertex count: {getattr(chart, 'candlestick_vertex_count', 'Unknown')}")
                if hasattr(chart, 'ohlcv_data') and chart.ohlcv_data:
                    start_price = chart.ohlcv_data['close'][chart.viewport_start] if chart.viewport_start < len(chart.ohlcv_data['close']) else 'N/A'
                    end_price = chart.ohlcv_data['close'][chart.viewport_end-1] if chart.viewport_end <= len(chart.ohlcv_data['close']) else 'N/A'
                    print(f"  Price range: {start_price:.5f} - {end_price:.5f}")
                print("="*50)
                print()
        
        # Take initial snapshot after 3 seconds
        QTimer.singleShot(3000, take_viewport_snapshot)
        
        # Run the application
        app.exec_()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_main_trade_clicking()