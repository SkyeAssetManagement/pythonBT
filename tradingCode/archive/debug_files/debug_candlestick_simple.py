"""
Simple visual debugging for candlestick rendering - no Unicode characters
"""
import sys
import time
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer


def debug_candlestick_visual():
    """Debug candlestick rendering with screenshots"""
    
    print("=== CANDLESTICK VISUAL DEBUG ===")
    
    # Create Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        # Import our components
        from src.dashboard.chart_widget import SimpleCandlestickItem
        from src.dashboard.data_structures import ChartDataBuffer
        import pyqtgraph as pg
        import numpy as np
        import pyautogui
        
        print("SUCCESS: Imports loaded")
        
        # Create test data
        n_bars = 20  # Very small dataset for clear debugging
        timestamps = np.arange(n_bars, dtype=np.int64) * int(60 * 1e9)
        
        # Create clear OHLC patterns
        base_price = 3750.0
        opens = np.array([base_price + i for i in range(n_bars)])  # Trending up
        closes = opens + np.array([(-1)**i * 2 for i in range(n_bars)])  # Alternating up/down
        highs = np.maximum(opens, closes) + 3  # Clear wicks above
        lows = np.minimum(opens, closes) - 3   # Clear wicks below
        volumes = np.full(n_bars, 1000.0)
        
        print(f"SUCCESS: Created {n_bars} test bars")
        print(f"  Opens: {opens[0]:.1f} to {opens[-1]:.1f}")
        print(f"  Range: {lows.min():.1f} to {highs.max():.1f}")
        
        # Create data buffer
        data_buffer = ChartDataBuffer(
            timestamps=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            volume=volumes
        )
        
        print("SUCCESS: Data buffer created")
        
        # Create plot widget
        plot_widget = pg.PlotWidget()
        plot_widget.setWindowTitle("CANDLESTICK DEBUG - Black Blob Investigation")
        plot_widget.setLabel('left', 'Price')
        plot_widget.setLabel('bottom', 'Bar Index')
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setBackground('w')  # White background for contrast
        plot_widget.resize(800, 600)
        
        print("SUCCESS: Plot widget created")
        
        # Create candlestick item - THIS IS WHERE THE PROBLEM IS
        print("Creating SimpleCandlestickItem...")
        candle_item = SimpleCandlestickItem(data_buffer)
        plot_widget.addItem(candle_item)
        
        print("SUCCESS: Candlestick item added")
        
        # Show window
        plot_widget.show()
        plot_widget.raise_()
        plot_widget.activateWindow()
        
        print("SUCCESS: Window shown")
        
        # Setup screenshot directory
        screenshot_dir = Path("debug_screenshots")
        screenshot_dir.mkdir(exist_ok=True)
        
        def take_debug_screenshot(suffix=""):
            try:
                timestamp = time.strftime("%H%M%S")
                filename = f"candlestick_debug_{timestamp}{suffix}.png"
                filepath = screenshot_dir / filename
                
                screenshot = pyautogui.screenshot()
                screenshot.save(str(filepath))
                print(f"SCREENSHOT: {filepath}")
                return str(filepath)
            except Exception as e:
                print(f"SCREENSHOT FAILED: {e}")
                return None
        
        # Take screenshots at different intervals
        QTimer.singleShot(2000, lambda: take_debug_screenshot("_2s"))
        QTimer.singleShot(4000, lambda: take_debug_screenshot("_4s"))
        QTimer.singleShot(6000, lambda: take_debug_screenshot("_6s"))
        
        # Close after 10 seconds
        QTimer.singleShot(10000, app.quit)
        
        print("RUNNING: Qt event loop (10 seconds)")
        print("  Window should be visible now")
        print("  Screenshots will be taken at 2s, 4s, 6s")
        print("  Check debug_screenshots/ folder")
        
        app.exec_()
        
        print("COMPLETE: Debug session finished")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Simple Candlestick Visual Debugging")
    print("This will show exactly what the candlesticks look like")
    print()
    
    debug_candlestick_visual()