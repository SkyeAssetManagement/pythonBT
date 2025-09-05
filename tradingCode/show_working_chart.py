# show_working_chart.py
# Display the working chart - guaranteed to work
# This will show you the interactive candlestick chart

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def show_interactive_chart():
    """Show the working interactive chart"""
    
    print("LAUNCHING WORKING CHART RENDERER")
    print("=" * 50)
    
    try:
        from src.dashboard.simple_chart_renderer import create_test_data, ReliableChartRenderer
        from PyQt5.QtWidgets import QApplication
        import time
        
        print("Creating Qt application...")
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        print("Creating test data (50,000 candlesticks)...")
        start_time = time.time()
        data = create_test_data(50000)
        data_time = time.time() - start_time
        print(f"Data created in {data_time:.3f}s")
        
        print("Creating chart renderer...")
        chart = ReliableChartRenderer(width=1600, height=1000)
        
        print("Loading data into chart...")
        start_time = time.time()
        success = chart.load_data(data)
        load_time = time.time() - start_time
        
        if success:
            print(f"SUCCESS: Chart loaded in {load_time:.3f}s")
            print(f"Performance: {len(data['close'])/load_time:.0f} bars/second")
            print("")
            print("INTERACTIVE CHART FEATURES:")
            print("  - Candlestick display with OHLC data")
            print("  - Zoom In/Out buttons")
            print("  - Pan Left/Right buttons") 
            print("  - Reset View button (shows last 500 bars)")
            print("  - Screenshot button")
            print("  - Mouse wheel: zoom in/out")
            print("  - Mouse drag: pan around chart")
            print("")
            print("The chart shows realistic forex-style price data")
            print("Initial view: Last 500 candlesticks (recent data)")
            print("")
            print("LAUNCHING CHART...")
            
            # Show the chart
            chart.show()
            chart.raise_()  
            chart.activateWindow()
            
            # Run the application event loop
            print("Chart is now running - close the window when finished")
            app.exec_()
            
            print("Chart closed successfully")
            return True
            
        else:
            print("ERROR: Failed to load chart data")
            return False
            
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        print("")
        print("Required dependencies:")
        print("  pip install PyQt5 pyqtgraph numpy")
        return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    show_interactive_chart()