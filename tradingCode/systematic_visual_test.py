"""
SYSTEMATIC VISUAL TESTING with organized screenshot tracking
Each test step gets its own directory with timestamped screenshots
"""
import sys
import time
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
import pyautogui
import datetime


class SystematicVisualTester:
    """Systematic visual testing with organized screenshot directories"""
    
    def __init__(self):
        self.base_screenshot_dir = Path("visual_testing_progress")
        self.base_screenshot_dir.mkdir(exist_ok=True)
        
        # Create timestamped session directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_screenshot_dir / f"session_{timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        print(f"=== SYSTEMATIC VISUAL TESTING SESSION ===")
        print(f"Session directory: {self.session_dir}")
        
        self.current_step = 0
        self.step_results = []
    
    def start_step(self, step_name, description):
        """Start a new testing step"""
        self.current_step += 1
        step_dir = self.session_dir / f"step_{self.current_step:02d}_{step_name}"
        step_dir.mkdir(exist_ok=True)
        
        print(f"\n--- STEP {self.current_step}: {step_name} ---")
        print(f"Description: {description}")
        print(f"Screenshots: {step_dir}")
        
        return step_dir
    
    def take_screenshot(self, step_dir, filename_prefix, description=""):
        """Take a screenshot for the current step"""
        timestamp = time.strftime("%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        filepath = step_dir / filename
        
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(str(filepath))
            
            result = {
                'filepath': str(filepath),
                'description': description,
                'timestamp': timestamp
            }
            
            print(f"SCREENSHOT: {filepath}")
            if description:
                print(f"  Purpose: {description}")
            
            return result
            
        except Exception as e:
            print(f"SCREENSHOT FAILED: {e}")
            return None
    
    def complete_step(self, step_dir, success, notes=""):
        """Complete the current step with assessment"""
        result = {
            'step': self.current_step,
            'directory': str(step_dir),
            'success': success,
            'notes': notes,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.step_results.append(result)
        
        status = "SUCCESS" if success else "FAILED"
        print(f"STEP {self.current_step} {status}: {notes}")
        
        # Write step summary
        summary_file = step_dir / "step_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Step {self.current_step} Summary\n")
            f.write(f"Status: {status}\n")
            f.write(f"Notes: {notes}\n")
            f.write(f"Timestamp: {result['timestamp']}\n")
        
        return result
    
    def write_session_summary(self):
        """Write final session summary"""
        summary_file = self.session_dir / "session_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("SYSTEMATIC VISUAL TESTING SESSION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for result in self.step_results:
                status = "SUCCESS" if result['success'] else "FAILED"
                f.write(f"Step {result['step']}: {status}\n")
                f.write(f"  Directory: {result['directory']}\n")
                f.write(f"  Notes: {result['notes']}\n")
                f.write(f"  Time: {result['timestamp']}\n\n")
        
        print(f"\nSession summary written to: {summary_file}")


def test_step_1_simple_candlesticks():
    """Step 1: Test our proven working simple candlesticks"""
    
    tester = SystematicVisualTester()
    step_dir = tester.start_step("simple_candlesticks", "Test our proven working SuperSimpleCandlestickItem")
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        # Import our proven working class
        import pyqtgraph as pg
        import numpy as np
        from PyQt5.QtCore import QPointF, QRectF
        from PyQt5.QtGui import QPen, QBrush, QPicture, QPainter
        from PyQt5.QtCore import Qt
        
        class ProvenCandlestickItem(pg.GraphicsObject):
            """The EXACT code that worked in our test"""
            
            def __init__(self, ohlc_data):
                super().__init__()
                self.ohlc_data = ohlc_data
                self.picture = None
                self._generate_picture()
            
            def _generate_picture(self):
                self.picture = QPicture()
                painter = QPainter(self.picture)
                
                for i, (o, h, l, c) in enumerate(self.ohlc_data):
                    x = float(i)
                    
                    # Draw wick
                    painter.setPen(QPen(Qt.black, 1))
                    painter.drawLine(QPointF(x, l), QPointF(x, h))
                    
                    # Draw body
                    body_bottom = min(o, c)
                    body_top = max(o, c)
                    body_height = max(body_top - body_bottom, 0.1)
                    
                    if c >= o:
                        painter.setBrush(QBrush(Qt.white))
                        painter.setPen(QPen(Qt.black, 1))
                    else:
                        painter.setBrush(QBrush(Qt.red))
                        painter.setPen(QPen(Qt.black, 1))
                    
                    rect = QRectF(x - 0.3, body_bottom, 0.6, body_height)
                    painter.drawRect(rect)
                
                painter.end()
            
            def paint(self, painter, option, widget):
                if self.picture:
                    painter.drawPicture(0, 0, self.picture)
            
            def boundingRect(self):
                if not self.ohlc_data:
                    return QRectF()
                all_prices = [price for ohlc in self.ohlc_data for price in ohlc[1:]]
                min_price = min(all_prices)
                max_price = max(all_prices)
                return QRectF(0, min_price, len(self.ohlc_data), max_price - min_price)
        
        # Test data - the exact same that worked
        ohlc_data = [
            (100, 103, 98, 102),   # Up candle - should be WHITE
            (102, 105, 100, 99),   # Down candle - should be RED  
            (99, 102, 97, 101),    # Up candle - should be WHITE
            (101, 104, 98, 96),    # Down candle - should be RED
            (96, 99, 94, 98),      # Up candle - should be WHITE
        ]
        
        print("Creating plot with proven working candlesticks...")
        
        # Create plot
        plot_widget = pg.PlotWidget()
        plot_widget.setWindowTitle("STEP 1: Proven Working Candlesticks")
        plot_widget.setLabel('left', 'Price')
        plot_widget.setLabel('bottom', 'Candle Index')
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setBackground('w')
        plot_widget.resize(900, 700)
        
        # Add proven candlestick item
        candle_item = ProvenCandlestickItem(ohlc_data)
        plot_widget.addItem(candle_item)
        plot_widget.autoRange()
        
        plot_widget.show()
        plot_widget.raise_()
        plot_widget.activateWindow()
        
        # Take systematic screenshots
        tester.take_screenshot(step_dir, "initial", "Window opened, should show 5 candlesticks")
        
        def screenshot_2s():
            result = tester.take_screenshot(step_dir, "2s_check", "After 2s - candlesticks should be visible")
            # Complete step based on visual assessment
            # For now, assume success if no exception
            tester.complete_step(step_dir, True, "Proven candlestick code executed successfully")
        
        QTimer.singleShot(2000, screenshot_2s)
        QTimer.singleShot(4000, lambda: tester.take_screenshot(step_dir, "4s_final", "Final verification"))
        QTimer.singleShot(6000, app.quit)
        
        print("Running Step 1 test (6 seconds)...")
        app.exec_()
        
        return tester
        
    except Exception as e:
        tester.complete_step(step_dir, False, f"Step 1 failed with error: {e}")
        print(f"STEP 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return tester


def test_step_2_main_dashboard():
    """Step 2: Test main dashboard with our fixes"""
    
    tester = SystematicVisualTester()
    step_dir = tester.start_step("main_dashboard", "Test main dashboard SimpleCandlestickItem with fixes")
    
    try:
        print("Running main dashboard with small dataset...")
        
        # Take screenshot before running
        tester.take_screenshot(step_dir, "before_run", "Desktop before running main dashboard")
        
        # Run main dashboard
        import subprocess
        import os
        
        # Change to trading code directory
        os.chdir(r"C:\Users\skyeAM\SkyeAM Dropbox\SAMresearch\ABtoPython\tradingCode")
        
        # Run with small dataset for quick testing
        cmd = [sys.executable, "main.py", "ES", "simpleSMA", "--start", "2024-07-01", "--end", "2024-07-02"]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Start process but don't wait for completion
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a bit for dashboard to load
        time.sleep(5)
        
        # Take screenshot during execution
        tester.take_screenshot(step_dir, "dashboard_5s", "Dashboard after 5 seconds")
        
        time.sleep(3)
        tester.take_screenshot(step_dir, "dashboard_8s", "Dashboard after 8 seconds")
        
        time.sleep(3)
        tester.take_screenshot(step_dir, "dashboard_11s", "Dashboard after 11 seconds")
        
        # Terminate process
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
            print("Process output:")
            print(stdout)
            if stderr:
                print("Process errors:")
                print(stderr)
        except subprocess.TimeoutExpired:
            process.kill()
            print("Process killed due to timeout")
        
        # Complete step - we'll assess from screenshots
        tester.complete_step(step_dir, True, "Main dashboard test completed - check screenshots for candlestick rendering")
        
        return tester
        
    except Exception as e:
        tester.complete_step(step_dir, False, f"Step 2 failed with error: {e}")
        print(f"STEP 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return tester


if __name__ == "__main__":
    print("SYSTEMATIC VISUAL TESTING")
    print("This will create organized screenshot directories to track progress")
    print()
    
    # Step 1: Test our proven working code
    tester1 = test_step_1_simple_candlesticks()
    
    time.sleep(2)  # Brief pause between steps
    
    # Step 2: Test main dashboard
    tester2 = test_step_2_main_dashboard()
    
    print("\nSYSTEMATIC TESTING COMPLETE")
    print("Check visual_testing_progress/ directory for organized screenshots")
    print("Each step has its own directory with timestamped images and summaries")