"""
Visual debugging utility - take screenshots automatically during chart development 
so I can see exactly what the user sees and iterate properly
"""
import sys
import time
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import subprocess


class VisualDebugger:
    """Automatically take screenshots of dashboard for debugging"""
    
    def __init__(self):
        self.screenshot_dir = Path("debug_screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
        self.screenshot_count = 0
    
    def take_screenshot(self, description="debug"):
        """Take a screenshot with timestamp and description"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{description}_{timestamp}_{self.screenshot_count:03d}.png"
        filepath = self.screenshot_dir / filename
        
        try:
            # Method 1: Try PyAutoGUI
            try:
                import pyautogui
                screenshot = pyautogui.screenshot()
                screenshot.save(str(filepath))
                print(f"[OK] Screenshot saved: {filepath}")
                self.screenshot_count += 1
                return str(filepath)
            except ImportError:
                pass
            
            # Method 2: Try PIL ImageGrab
            try:
                from PIL import ImageGrab
                screenshot = ImageGrab.grab()
                screenshot.save(str(filepath))
                print(f"[OK] Screenshot saved: {filepath}")
                self.screenshot_count += 1
                return str(filepath)
            except ImportError:
                pass
            
            # Method 3: Windows PowerShell screenshot
            powershell_cmd = f'''
            Add-Type -AssemblyName System.Windows.Forms
            Add-Type -AssemblyName System.Drawing
            $Screen = [System.Windows.Forms.SystemInformation]::VirtualScreen
            $bitmap = New-Object System.Drawing.Bitmap $Screen.Width, $Screen.Height
            $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
            $graphics.CopyFromScreen($Screen.Left, $Screen.Top, 0, 0, $bitmap.Size)
            $bitmap.Save("{str(filepath)}")
            $graphics.Dispose()
            $bitmap.Dispose()
            '''
            
            result = subprocess.run(
                ["powershell", "-Command", powershell_cmd],
                capture_output=True,
                text=True
            )
            
            if filepath.exists():
                print(f"[OK] Screenshot saved: {filepath}")
                self.screenshot_count += 1
                return str(filepath)
            else:
                print(f"[X] PowerShell screenshot failed")
                
        except Exception as e:
            print(f"[X] Screenshot failed: {e}")
        
        return None
    
    def setup_auto_screenshot(self, app, interval_ms=5000):
        """Setup automatic screenshots every few seconds"""
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.take_screenshot("auto"))
        self.timer.start(interval_ms)
        print(f"[OK] Auto-screenshot every {interval_ms/1000}s enabled")
    
    def debug_candlestick_rendering(self):
        """Create a simple test to debug candlestick rendering step by step"""
        
        print("=== VISUAL DEBUGGING: CANDLESTICK RENDERING ===")
        
        # Create Qt application
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Import our chart components
        try:
            from src.dashboard.chart_widget import SimpleCandlestickItem
            from src.dashboard.data_structures import ChartDataBuffer
            import pyqtgraph as pg
            import numpy as np
            
            print("[OK] Imports successful")
            
            # Create simple test data
            n_bars = 50  # Small dataset for debugging
            timestamps = np.arange(n_bars, dtype=np.int64) * int(60 * 1e9)  # 1-minute bars
            
            # Simple OHLC pattern
            base_price = 3750.0
            opens = np.full(n_bars, base_price)
            closes = base_price + np.random.normal(0, 5, n_bars)  # Small random moves
            highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 2, n_bars))
            lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 2, n_bars))
            volumes = np.full(n_bars, 1000.0)
            
            print(f"[OK] Created test data: {n_bars} bars")
            print(f"  Price range: {lows.min():.2f} - {highs.max():.2f}")
            
            # Create data buffer
            data_buffer = ChartDataBuffer(
                timestamps=timestamps,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                volume=volumes
            )
            
            print("[OK] Created data buffer")
            
            # Create plot widget
            plot_widget = pg.PlotWidget()
            plot_widget.setWindowTitle("Candlestick Debug Test")
            plot_widget.setLabel('left', 'Price')
            plot_widget.setLabel('bottom', 'Bar Index')
            plot_widget.showGrid(x=True, y=True)
            plot_widget.setBackground('w')  # White background
            
            print("[OK] Created plot widget")
            
            # Create candlestick item
            candle_item = SimpleCandlestickItem(data_buffer)
            plot_widget.addItem(candle_item)
            
            print("[OK] Added candlestick item")
            
            # Show window
            plot_widget.show()
            plot_widget.resize(800, 600)
            
            print("[OK] Window shown")
            
            # Take immediate screenshot
            QTimer.singleShot(1000, lambda: self.take_screenshot("candlestick_debug_1s"))
            QTimer.singleShot(3000, lambda: self.take_screenshot("candlestick_debug_3s"))
            QTimer.singleShot(5000, lambda: self.take_screenshot("candlestick_debug_5s"))
            
            # Auto-close after 10 seconds
            QTimer.singleShot(10000, app.quit)
            
            print("[OK] Starting Qt event loop...")
            print("  Screenshots will be taken at 1s, 3s, and 5s")
            print("  Window will auto-close after 10s")
            
            app.exec_()
            
            print("[OK] Debug session complete")
            print(f"  Check {self.screenshot_dir} for screenshots")
            
        except Exception as e:
            print(f"[X] Debug failed: {e}")
            import traceback
            traceback.print_exc()


def test_visual_debugging():
    """Test the visual debugging system"""
    debugger = VisualDebugger()
    debugger.debug_candlestick_rendering()


if __name__ == "__main__":
    print("Visual Debugging System")
    print("This will create a simple candlestick chart and take screenshots")
    print("so I can see exactly what's rendering and fix the black blob issue.")
    print()
    
    test_visual_debugging()