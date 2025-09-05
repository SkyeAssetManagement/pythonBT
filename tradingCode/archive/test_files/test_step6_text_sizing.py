# test_step6_text_sizing.py
# Test Step 6: Verify text sizing consistency in crosshair/hover data windows
# All text should match x and y axis values size (8pt Courier New)

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add src path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QSplitter
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QFont, QFontMetrics

from dashboard.crosshair_widget import CrosshairOverlay, CrosshairInfoWidget
from dashboard.hover_info_widget import HoverInfoWidget
from step6_complete_final import create_ultimate_test_data

class TextSizingVerificationWidget(QMainWindow):
    """Widget to verify Step 6 text sizing consistency"""
    
    def __init__(self):
        super().__init__()
        
        # Test data
        self.ohlcv_data = None
        self.reference_font_size = 8  # 8pt like axis values
        self.reference_font_family = "Courier New"
        
        # Components
        self.crosshair_overlay = None
        self.crosshair_info = None
        self.hover_info = None
        
        # Test results
        self.test_results = {}
        
        self.setWindowTitle("Step 6 Text Sizing Verification - All Text = X/Y Axis Size")
        self.setGeometry(100, 100, 1200, 800)
        
        self._setup_ui()
        self._load_test_data()
        
        print("Step 6 Text Sizing Verification initialized")
    
    def _setup_ui(self):
        """Setup verification UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title and status
        title_label = QLabel("STEP 6 TEXT SIZE VERIFICATION")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: #404040;
                font-weight: bold;
                font-size: 12pt;
                padding: 8px;
                border: 2px solid #606060;
                border-radius: 4px;
                margin: 5px;
            }
        """)
        layout.addWidget(title_label)
        
        # Reference font info
        ref_info = QLabel(f"REFERENCE: X/Y Axis Values = {self.reference_font_size}pt {self.reference_font_family}")
        ref_info.setAlignment(Qt.AlignCenter)
        ref_info.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10pt; padding: 5px;")
        layout.addWidget(ref_info)
        
        # Test area
        test_frame = QWidget()
        test_frame.setMinimumHeight(400)
        test_frame.setStyleSheet("background-color: #1e1e1e; border: 2px solid #444444; border-radius: 4px;")
        
        # Create crosshair overlay on test frame
        self.crosshair_overlay = CrosshairOverlay(test_frame)
        self.crosshair_overlay.setGeometry(0, 0, 800, 400)
        
        layout.addWidget(test_frame)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.test_crosshair_btn = QPushButton("Test Crosshair Info Widget")
        self.test_crosshair_btn.clicked.connect(self._test_crosshair_info)
        controls_layout.addWidget(self.test_crosshair_btn)
        
        self.test_hover_btn = QPushButton("Test Hover Info Widget") 
        self.test_hover_btn.clicked.connect(self._test_hover_info)
        controls_layout.addWidget(self.test_hover_btn)
        
        self.verify_sizing_btn = QPushButton("Verify All Text Sizes")
        self.verify_sizing_btn.clicked.connect(self._verify_all_text_sizes)
        controls_layout.addWidget(self.verify_sizing_btn)
        
        self.screenshot_btn = QPushButton("Screenshot Results")
        self.screenshot_btn.clicked.connect(self._take_verification_screenshot)
        controls_layout.addWidget(self.screenshot_btn)
        
        layout.addLayout(controls_layout)
        
        # Results display
        self.results_label = QLabel("Click 'Verify All Text Sizes' to check consistency")
        self.results_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: #333333;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                padding: 10px;
                border: 1px solid #555555;
                border-radius: 3px;
            }
        """)
        self.results_label.setWordWrap(True)
        self.results_label.setMinimumHeight(150)
        layout.addWidget(self.results_label)
        
        # Create info widgets
        self._create_info_widgets()
        
        # Style main window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #606060;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #606060;
            }
        """)
    
    def _create_info_widgets(self):
        """Create crosshair and hover info widgets"""
        # Crosshair info widget
        self.crosshair_info = CrosshairInfoWidget(self)
        
        # Hover info widget  
        self.hover_info = HoverInfoWidget(self)
        
        print("Info widgets created for text sizing verification")
    
    def _load_test_data(self):
        """Load test OHLCV data"""
        try:
            self.ohlcv_data, _ = create_ultimate_test_data()
            
            # Load into widgets
            if self.crosshair_info:
                self.crosshair_info.load_ohlcv_data(self.ohlcv_data)
            
            if self.hover_info:
                self.hover_info.load_ohlcv_data(self.ohlcv_data)
            
            # Setup crosshair overlay
            if self.crosshair_overlay:
                self.crosshair_overlay.set_chart_bounds(0, 499, 
                    float(self.ohlcv_data['low'].min()), 
                    float(self.ohlcv_data['high'].max()))
                self.crosshair_overlay.set_widget_size(800, 400)
                
                # Connect to crosshair info
                self.crosshair_overlay.position_changed.connect(self._on_crosshair_moved)
            
            print(f"Test data loaded: {len(self.ohlcv_data['close'])} bars")
            
        except Exception as e:
            print(f"ERROR: Failed to load test data: {e}")
    
    def _on_crosshair_moved(self, x_val, y_val):
        """Handle crosshair position changes"""
        if self.crosshair_info:
            self.crosshair_info.update_position(x_val, y_val)
    
    def _test_crosshair_info(self):
        """Test crosshair info widget display"""
        try:
            if self.crosshair_info and self.ohlcv_data:
                # Position crosshair at a test location
                test_x = 250.0
                test_y = float(np.mean([self.ohlcv_data['low'].min(), self.ohlcv_data['high'].max()]))
                
                self.crosshair_overlay.set_position(test_x, test_y)
                self.crosshair_overlay.lock_crosshair(True)
                
                # Show crosshair info
                global_pos = self.mapToGlobal(self.rect().center())
                self.crosshair_info.show_at_position(global_pos)
                
                self.results_label.setText("SUCCESS: Crosshair Info Widget displayed - check text sizes")
                print("Crosshair info widget test completed")
                
        except Exception as e:
            self.results_label.setText(f"FAIL: Crosshair test failed: {e}")
            print(f"ERROR: Crosshair test failed: {e}")
    
    def _test_hover_info(self):
        """Test hover info widget display"""
        try:
            if self.hover_info and self.ohlcv_data:
                # Show hover info for a test bar
                test_bar = 300
                global_pos = self.mapToGlobal(self.rect().center())
                global_pos.setX(global_pos.x() + 200)
                
                self.hover_info.show_at_position(global_pos, test_bar)
                
                self.results_label.setText("SUCCESS: Hover Info Widget displayed - check text sizes")
                print("Hover info widget test completed")
                
        except Exception as e:
            self.results_label.setText(f"FAIL: Hover test failed: {e}")
            print(f"ERROR: Hover test failed: {e}")
    
    def _verify_all_text_sizes(self):
        """Verify all text sizes match the reference (8pt)"""
        print("\\n" + "="*60)
        print("STEP 6 TEXT SIZE VERIFICATION")
        print("="*60)
        
        results = []
        results.append(f"REFERENCE: X/Y Axis Values = {self.reference_font_size}pt {self.reference_font_family}")
        results.append("")
        
        # Test crosshair info widget text sizes
        results.append("CROSSHAIR INFO WIDGET:")
        if self.crosshair_info:
            crosshair_checks = self._check_crosshair_text_sizes()
            for check in crosshair_checks:
                results.append(f"  {check}")
        else:
            results.append("  FAIL: Crosshair info widget not available")
        
        results.append("")
        
        # Test hover info widget text sizes  
        results.append("HOVER INFO WIDGET:")
        if self.hover_info:
            hover_checks = self._check_hover_text_sizes() 
            for check in hover_checks:
                results.append(f"  {check}")
        else:
            results.append("  FAIL: Hover info widget not available")
        
        results.append("")
        
        # Overall result
        all_passed = all("SUCCESS" in line for line in results if line.strip() and not line.startswith("REFERENCE") and not line.endswith("WIDGET:"))
        
        if all_passed:
            results.append("SUCCESS: STEP 6 TEXT SIZING CONSISTENT!")
            results.append("SUCCESS: All crosshair/hover text matches X/Y axis size")
            results.append("SUCCESS: Professional consistency achieved")
        else:
            results.append("FAIL: STEP 6 TEXT SIZING Inconsistent")
            results.append("Some text does not match X/Y axis size")
        
        # Display results
        results_text = "\\n".join(results)
        self.results_label.setText(results_text)
        
        # Print to console
        for line in results:
            print(line)
        
        print("="*60)
        
        return all_passed
    
    def _check_crosshair_text_sizes(self):
        """Check crosshair widget text sizes"""
        checks = []
        
        try:
            # Check position labels (should be 8pt)
            for key, label in self.crosshair_info.position_labels.items():
                style = label.styleSheet()
                if "font-size: 8pt" in style and "Courier New" in style:
                    checks.append(f"SUCCESS Position {key}: 8pt Courier New")
                else:
                    checks.append(f"FAIL Position {key}: Wrong font size/family")
            
            # Check bar labels (should be 8pt)
            for key, label in self.crosshair_info.bar_labels.items():
                style = label.styleSheet()
                if "font-size: 8pt" in style and "Courier New" in style:
                    checks.append(f"SUCCESS Bar {key}: 8pt Courier New")
                else:
                    checks.append(f"FAIL Bar {key}: Wrong font size/family")
            
            # Check stats labels (should be 8pt)
            for key, label in self.crosshair_info.stats_labels.items():
                style = label.styleSheet()
                if "font-size: 8pt" in style and "Courier New" in style:
                    checks.append(f"SUCCESS Stats {key}: 8pt Courier New")
                else:
                    checks.append(f"FAIL Stats {key}: Wrong font size/family")
                    
        except Exception as e:
            checks.append(f"FAIL: Crosshair check error: {e}")
        
        return checks
    
    def _check_hover_text_sizes(self):
        """Check hover widget text sizes"""
        checks = []
        
        try:
            # Check price labels (should be 8pt) 
            for key, label in self.hover_info.price_labels.items():
                style = label.styleSheet()
                if "font-size: 8pt" in style and "Courier New" in style:
                    checks.append(f"SUCCESS Price {key}: 8pt Courier New")
                else:
                    checks.append(f"FAIL Price {key}: Wrong font size/family")
            
            # Check indicator labels (should be 8pt now, was 7pt)
            for key, label in self.hover_info.indicator_labels.items():
                style = label.styleSheet()
                if "font-size: 8pt" in style and "Courier New" in style:
                    checks.append(f"SUCCESS Indicator {key}: 8pt Courier New")
                else:
                    checks.append(f"FAIL Indicator {key}: Wrong font size/family")
                    
        except Exception as e:
            checks.append(f"FAIL: Hover check error: {e}")
        
        return checks
    
    def _take_verification_screenshot(self):
        """Take screenshot of verification results"""
        try:
            timestamp = int(time.time())
            screenshot_path = f"C:\\Users\\skyeAM\\SkyeAM Dropbox\\SAMresearch\\ABtoPython\\2025-08-08_1\\step6_text_sizing_verification_{timestamp}.png"
            
            # Ensure both widgets are visible for screenshot
            self._test_crosshair_info()
            self._test_hover_info()
            
            # Wait a moment for widgets to appear
            QTimer.singleShot(500, lambda: self._capture_screenshot(screenshot_path))
            
        except Exception as e:
            print(f"ERROR: Screenshot failed: {e}")
    
    def _capture_screenshot(self, path):
        """Capture the screenshot"""
        try:
            pixmap = self.grab()
            pixmap.save(path)
            print(f"Screenshot Step 6 text sizing verification saved: {path}")
            
            self.results_label.setText(self.results_label.text() + f"\\n\\nScreenshot saved: {path}")
            
        except Exception as e:
            print(f"ERROR: Screenshot capture failed: {e}")


def test_step6_text_sizing():
    """Test Step 6 text sizing consistency"""
    print("TESTING STEP 6 TEXT SIZING CONSISTENCY")
    print("="*50)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Create verification widget
        verifier = TextSizingVerificationWidget()
        verifier.show()
        
        print("\\nStep 6 Text Sizing Verification Features:")
        print("• Compare all text to X/Y axis reference (8pt Courier New)")
        print("• Test crosshair info widget text consistency")
        print("• Test hover info widget text consistency")
        print("• Visual verification with side-by-side display")
        print("• Automated font size checking")
        print("• Screenshot capture for documentation")
        
        print("\\nTEST PROCEDURE:")
        print("1. Click 'Test Crosshair Info Widget' - shows crosshair data")
        print("2. Click 'Test Hover Info Widget' - shows hover data")
        print("3. Click 'Verify All Text Sizes' - checks font consistency")
        print("4. Click 'Screenshot Results' - captures verification")
        
        print("\\nEXPECTED RESULTS:")
        print("SUCCESS: All text should be 8pt Courier New (matching axis values)")
        print("SUCCESS: Crosshair coordinates, bar data, statistics: 8pt")
        print("SUCCESS: Hover OHLCV data, technical indicators: 8pt")
        print("SUCCESS: No 7pt or other inconsistent sizes")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Text sizing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_step6_text_sizing()
    
    if success:
        print("\\n" + "="*50)
        print("STEP 6 TEXT SIZING TEST LAUNCHED")
        print("="*50)
        print("OBJECTIVE: All crosshair/hover text = X/Y axis size")
        print("REFERENCE: 8pt Courier New monospace font")
        print("SCOPE: CrosshairInfoWidget + HoverInfoWidget")
        print("\\nVerify professional consistency across dashboard!")
    else:
        print("Step 6 text sizing test failed")