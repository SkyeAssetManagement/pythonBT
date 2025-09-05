# tradingCode/test_indicators_functionality.py
# Test indicator addition functionality - Step 5
# Debug and fix the VBT Pro indicators panel

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add src path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QTextEdit, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# Import components
from dashboard.indicators_panel import VBTIndicatorsPanel
from step6_complete_final import create_ultimate_test_data

class IndicatorsTester(QMainWindow):
    """Test widget for indicator functionality debugging"""
    
    def __init__(self):
        super().__init__()
        self.indicators_panel = None
        self.ohlcv_data = None
        
        self.setWindowTitle("Indicators Functionality Test - Step 5 Debug")
        self.setGeometry(100, 100, 1000, 700)
        
        self._setup_ui()
        self._load_test_data()
        self._setup_auto_test()
    
    def _setup_ui(self):
        """Setup the test UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # Left side - indicators panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        left_layout.addWidget(QLabel("VBT Pro Indicators Panel:"))
        
        self.indicators_panel = VBTIndicatorsPanel()
        left_layout.addWidget(self.indicators_panel)
        
        # Test controls
        controls_layout = QHBoxLayout()
        
        self.test_add_btn = QPushButton("Test Add SMA")
        self.test_add_btn.clicked.connect(self._test_add_sma)
        controls_layout.addWidget(self.test_add_btn)
        
        self.test_add_ema_btn = QPushButton("Test Add EMA")
        self.test_add_ema_btn.clicked.connect(self._test_add_ema)
        controls_layout.addWidget(self.test_add_ema_btn)
        
        self.test_bb_btn = QPushButton("Test Add BB")
        self.test_bb_btn.clicked.connect(self._test_add_bb)
        controls_layout.addWidget(self.test_bb_btn)
        
        self.status_btn = QPushButton("Check Status")
        self.status_btn.clicked.connect(self._check_status)
        controls_layout.addWidget(self.status_btn)
        
        left_layout.addLayout(controls_layout)
        
        # Right side - debug log
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        right_layout.addWidget(QLabel("Debug Log:"))
        
        self.debug_log = QTextEdit()
        self.debug_log.setFont(QFont("Consolas", 9))
        self.debug_log.setStyleSheet("background-color: #1e1e1e; color: #ffffff; border: 1px solid #555;")
        right_layout.addWidget(self.debug_log)
        
        # Auto test controls
        auto_layout = QHBoxLayout()
        
        self.auto_test_btn = QPushButton("Run Auto Test")
        self.auto_test_btn.clicked.connect(self._run_auto_test)
        auto_layout.addWidget(self.auto_test_btn)
        
        self.screenshot_btn = QPushButton("Screenshot")
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        auto_layout.addWidget(self.screenshot_btn)
        
        right_layout.addLayout(auto_layout)
        
        # Add to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        print("Indicators tester UI setup complete")
    
    def _load_test_data(self):
        """Load test OHLCV data"""
        try:
            self._log("Loading test OHLCV data...")
            
            # Create test data
            self.ohlcv_data, _ = create_ultimate_test_data()
            self._log(f"Generated OHLCV data: {len(self.ohlcv_data['close'])} bars")
            
            # Load into indicators panel
            if self.indicators_panel:
                success = self.indicators_panel.load_ohlcv_data(self.ohlcv_data)
                if success:
                    self._log("✅ OHLCV data loaded into indicators panel")
                    self._log(f"  OHLC data shapes: O={len(self.ohlcv_data['open'])}, H={len(self.ohlcv_data['high'])}, L={len(self.ohlcv_data['low'])}, C={len(self.ohlcv_data['close'])}")
                else:
                    self._log("❌ Failed to load OHLCV data into indicators panel")
            
        except Exception as e:
            self._log(f"❌ Error loading test data: {e}")
            import traceback
            self._log(traceback.format_exc())
    
    def _setup_auto_test(self):
        """Setup auto-test timer"""
        self.auto_test_timer = QTimer()
        self.auto_test_timer.timeout.connect(self._auto_test_step)
        self.auto_test_step_index = 0
        self.auto_test_steps = [
            ("Add SMA(20)", self._test_add_sma),
            ("Add EMA(12)", self._test_add_ema),
            ("Add Bollinger Bands", self._test_add_bb),
            ("Check Status", self._check_status)
        ]
    
    def _log(self, message):
        """Add message to debug log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.debug_log.append(log_entry)
        
        # Auto-scroll to bottom
        scrollbar = self.debug_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _test_add_sma(self):
        """Test adding SMA indicator via code"""
        try:
            self._log("🧪 Testing SMA addition...")
            
            # Check if panel has data
            if not hasattr(self.indicators_panel, 'ohlcv_data') or self.indicators_panel.ohlcv_data is None:
                self._log("❌ No OHLCV data in indicators panel")
                return
            
            self._log("✅ OHLCV data available")
            
            # Check available indicators
            available = list(self.indicators_panel.available_indicators.keys())
            self._log(f"Available indicators: {available}")
            
            if 'SMA' in available:
                # Set the combo box to SMA
                for i in range(self.indicators_panel.indicator_combo.count()):
                    if self.indicators_panel.indicator_combo.itemText(i) == 'SMA':
                        self.indicators_panel.indicator_combo.setCurrentIndex(i)
                        break
                
                self._log("Set combo box to SMA")
                
                # Trigger the add indicator method
                self.indicators_panel._add_indicator()
                
                # Check if it was added
                indicators_count = len(self.indicators_panel.indicators)
                self._log(f"✅ SMA addition attempt completed. Indicators count: {indicators_count}")
            else:
                self._log("❌ SMA not found in available indicators")
            
        except Exception as e:
            self._log(f"❌ Error testing SMA addition: {e}")
            import traceback
            self._log(traceback.format_exc())
    
    def _test_add_ema(self):
        """Test adding EMA indicator"""
        try:
            self._log("🧪 Testing EMA addition...")
            
            if 'EMA' in self.indicators_panel.available_indicators:
                # Set combo box to EMA
                for i in range(self.indicators_panel.indicator_combo.count()):
                    if self.indicators_panel.indicator_combo.itemText(i) == 'EMA':
                        self.indicators_panel.indicator_combo.setCurrentIndex(i)
                        break
                
                self.indicators_panel._add_indicator()
                
                indicators_count = len(self.indicators_panel.indicators)
                self._log(f"✅ EMA addition completed. Indicators count: {indicators_count}")
            else:
                self._log("❌ EMA not available")
                
        except Exception as e:
            self._log(f"❌ Error testing EMA addition: {e}")
    
    def _test_add_bb(self):
        """Test adding Bollinger Bands"""
        try:
            self._log("🧪 Testing Bollinger Bands addition...")
            
            if 'Bollinger Bands' in self.indicators_panel.available_indicators:
                # Set combo box to Bollinger Bands
                for i in range(self.indicators_panel.indicator_combo.count()):
                    if self.indicators_panel.indicator_combo.itemText(i) == 'Bollinger Bands':
                        self.indicators_panel.indicator_combo.setCurrentIndex(i)
                        break
                
                self.indicators_panel._add_indicator()
                
                indicators_count = len(self.indicators_panel.indicators)
                self._log(f"✅ Bollinger Bands addition completed. Indicators count: {indicators_count}")
            else:
                self._log("❌ Bollinger Bands not available")
                
        except Exception as e:
            self._log(f"❌ Error testing Bollinger Bands addition: {e}")
    
    def _check_status(self):
        """Check current status of indicators panel"""
        try:
            self._log("📊 Checking indicators panel status...")
            
            # Check data availability
            has_data = hasattr(self.indicators_panel, 'ohlcv_data') and self.indicators_panel.ohlcv_data is not None
            self._log(f"Has OHLCV data: {has_data}")
            
            if has_data:
                data_length = len(self.indicators_panel.ohlcv_data['close'])
                self._log(f"Data length: {data_length} bars")
            
            # Check indicators
            indicators_count = len(self.indicators_panel.indicators)
            self._log(f"Active indicators: {indicators_count}")
            
            if indicators_count > 0:
                for indicator_id, config in self.indicators_panel.indicators.items():
                    self._log(f"  {indicator_id}: {config.name} (enabled: {config.enabled})")
            
            # Check indicator data
            data_count = len(self.indicators_panel.indicator_data)
            self._log(f"Calculated indicator data: {data_count}")
            
            # Check available indicators
            available_count = len(self.indicators_panel.available_indicators)
            self._log(f"Available indicator types: {available_count}")
            self._log(f"Available: {list(self.indicators_panel.available_indicators.keys())}")
            
            # Check combo box
            combo_count = self.indicators_panel.indicator_combo.count()
            current_text = self.indicators_panel.indicator_combo.currentText()
            self._log(f"Combo box items: {combo_count}, Current: '{current_text}'")
            
        except Exception as e:
            self._log(f"❌ Error checking status: {e}")
            import traceback
            self._log(traceback.format_exc())
    
    def _run_auto_test(self):
        """Run automated test sequence"""
        self._log("🚀 Starting automated test sequence...")
        self.auto_test_step_index = 0
        self.auto_test_timer.start(2000)  # 2 second intervals
        self.auto_test_btn.setEnabled(False)
    
    def _auto_test_step(self):
        """Execute one auto test step"""
        if self.auto_test_step_index >= len(self.auto_test_steps):
            self.auto_test_timer.stop()
            self.auto_test_btn.setEnabled(True)
            self._log("✅ Auto test sequence completed!")
            self._take_screenshot()
            return
        
        step_name, step_func = self.auto_test_steps[self.auto_test_step_index]
        self._log(f"🧪 Auto test step {self.auto_test_step_index + 1}: {step_name}")
        
        try:
            step_func()
        except Exception as e:
            self._log(f"❌ Auto test step failed: {e}")
        
        self.auto_test_step_index += 1
    
    def _take_screenshot(self):
        """Take screenshot"""
        try:
            screenshot_path = f"C:\\Users\\skyeAM\\SkyeAM Dropbox\\SAMresearch\\ABtoPython\\2025-08-08_1\\step5_indicators_test_{int(time.time())}.png"
            pixmap = self.grab()
            pixmap.save(screenshot_path)
            self._log(f"📸 Screenshot saved: {screenshot_path}")
            
        except Exception as e:
            self._log(f"❌ Screenshot error: {e}")

def test_indicators_functionality():
    """Main test function for indicators functionality"""
    print("TESTING INDICATORS FUNCTIONALITY - STEP 5")
    print("=" * 60)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Create indicators tester
        tester = IndicatorsTester()
        tester.show()
        
        print("\nSUCCESS: Indicators functionality test launched")
        print("\nTEST FEATURES:")
        print("• Manual testing buttons for SMA, EMA, Bollinger Bands")
        print("• Status checker to verify data and configuration")
        print("• Auto test sequence with screenshots")
        print("• Debug log showing all operations")
        
        print("\nEXPECTED BEHAVIOR:")
        print("• OHLCV data should load successfully")
        print("• Add Indicator dropdown should be populated")
        print("• Clicking Add buttons should create indicators")
        print("• Status should show active indicators")
        print("• Debug log should show detailed information")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Indicators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_indicators_functionality()
    
    if success:
        print("\n" + "=" * 60)
        print("STEP 5 INDICATORS TEST LAUNCHED")
        print("=" * 60)
        print("Please verify the following:")
        print("1. OHLCV data loads successfully")
        print("2. Add Indicator buttons work")
        print("3. Indicators appear in the panel")
        print("4. Status check shows correct information")
        print("5. No errors in debug log")
        print("\nIf indicators work properly, Step 5 is COMPLETE!")
    else:
        print("Step 5 test failed - needs investigation")