# tradingCode/test_indicators_fix_verification.py  
# Verify Step 5 fix: Indicator addition functionality working
# Test that indicators can be added and dashboard receives signals

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add src path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt5.QtCore import pyqtSlot

from dashboard.indicators_panel import VBTIndicatorsPanel
from step6_complete_final import create_ultimate_test_data

class IndicatorFixVerification(QMainWindow):
    """Verify the indicators fix is working"""
    
    def __init__(self):
        super().__init__()
        self.indicators_panel = None
        self.signal_received_count = 0
        self.last_indicator_data = None
        
        self.setWindowTitle("Step 5 Fix Verification - Indicators Functionality")
        self.setGeometry(200, 200, 800, 600)
        
        self._setup_ui()
        self._load_test_data()
    
    def _setup_ui(self):
        """Setup verification UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Status display
        self.status_label = QLabel("Initializing indicators test...")
        self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 10px; background-color: #2b2b2b; color: white; border: 1px solid #555;")
        layout.addWidget(self.status_label)
        
        # Signals received counter
        self.signal_label = QLabel("Signals received: 0")
        self.signal_label.setStyleSheet("font-size: 10pt; padding: 5px; color: green; font-weight: bold;")
        layout.addWidget(self.signal_label)
        
        # Test buttons
        test_layout = QVBoxLayout()
        
        self.test1_btn = QPushButton("Test 1: Add SMA(20)")
        self.test1_btn.clicked.connect(self._test_add_sma)
        test_layout.addWidget(self.test1_btn)
        
        self.test2_btn = QPushButton("Test 2: Add EMA(12)")  
        self.test2_btn.clicked.connect(self._test_add_ema)
        test_layout.addWidget(self.test2_btn)
        
        self.test3_btn = QPushButton("Test 3: Add RSI(14)")
        self.test3_btn.clicked.connect(self._test_add_rsi)
        test_layout.addWidget(self.test3_btn)
        
        self.verify_btn = QPushButton("Verify Fix Status")
        self.verify_btn.clicked.connect(self._verify_fix_status)
        test_layout.addWidget(self.verify_btn)
        
        layout.addLayout(test_layout)
        
        # Indicators panel
        layout.addWidget(QLabel("VBT Pro Indicators Panel:"))
        self.indicators_panel = VBTIndicatorsPanel()
        
        # Connect to signal to verify it's working
        self.indicators_panel.indicators_updated.connect(self._on_indicators_updated)
        
        layout.addWidget(self.indicators_panel)
        
        print("Step 5 fix verification UI setup complete")
    
    def _load_test_data(self):
        """Load test data"""
        try:
            print("Loading test OHLCV data for indicators...")
            
            # Create test data
            ohlcv_data, _ = create_ultimate_test_data()
            
            # Load into indicators panel
            success = self.indicators_panel.load_ohlcv_data(ohlcv_data)
            
            if success:
                self.status_label.setText("✅ Ready: OHLCV data loaded successfully")
                self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 10px; background-color: #2d5a27; color: white; border: 1px solid #4a934a;")
                print(f"OHLCV data loaded: {len(ohlcv_data['close'])} bars")
            else:
                self.status_label.setText("❌ Error: Failed to load OHLCV data")
                self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 10px; background-color: #5a2727; color: white; border: 1px solid #934a4a;")
            
        except Exception as e:
            print(f"Error loading test data: {e}")
            self.status_label.setText(f"❌ Error: {e}")
            self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 10px; background-color: #5a2727; color: white; border: 1px solid #934a4a;")
    
    @pyqtSlot(dict)
    def _on_indicators_updated(self, indicator_data):
        """Handle indicators updated signal"""
        self.signal_received_count += 1
        self.last_indicator_data = indicator_data
        
        self.signal_label.setText(f"✅ Signals received: {self.signal_received_count} | Active indicators: {len(indicator_data)}")
        self.signal_label.setStyleSheet("font-size: 10pt; padding: 5px; color: green; font-weight: bold;")
        
        print(f"SIGNAL RECEIVED: indicators_updated with {len(indicator_data)} indicators")
        for indicator_id, data in indicator_data.items():
            print(f"  {indicator_id}: {list(data.keys())}")
    
    def _test_add_sma(self):
        """Test adding SMA indicator"""
        try:
            print("Testing SMA addition...")
            
            # Find and select SMA in combo box
            combo = self.indicators_panel.indicator_combo
            for i in range(combo.count()):
                if combo.itemText(i) == 'SMA':
                    combo.setCurrentIndex(i)
                    break
            
            # Trigger add indicator
            self.indicators_panel._add_indicator()
            
            # Update status
            indicators_count = len(self.indicators_panel.indicators)
            self.status_label.setText(f"SMA test completed - Total indicators: {indicators_count}")
            
        except Exception as e:
            print(f"Error in SMA test: {e}")
            self.status_label.setText(f"❌ SMA test failed: {e}")
            self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 10px; background-color: #5a2727; color: white; border: 1px solid #934a4a;")
    
    def _test_add_ema(self):
        """Test adding EMA indicator"""
        try:
            print("Testing EMA addition...")
            
            # Find and select EMA in combo box
            combo = self.indicators_panel.indicator_combo
            for i in range(combo.count()):
                if combo.itemText(i) == 'EMA':
                    combo.setCurrentIndex(i)
                    break
            
            # Trigger add indicator
            self.indicators_panel._add_indicator()
            
            # Update status
            indicators_count = len(self.indicators_panel.indicators)
            self.status_label.setText(f"EMA test completed - Total indicators: {indicators_count}")
            
        except Exception as e:
            print(f"Error in EMA test: {e}")
            self.status_label.setText(f"❌ EMA test failed: {e}")
    
    def _test_add_rsi(self):
        """Test adding RSI indicator"""
        try:
            print("Testing RSI addition...")
            
            # Find and select RSI in combo box
            combo = self.indicators_panel.indicator_combo
            for i in range(combo.count()):
                if combo.itemText(i) == 'RSI':
                    combo.setCurrentIndex(i)
                    break
            
            # Trigger add indicator
            self.indicators_panel._add_indicator()
            
            # Update status
            indicators_count = len(self.indicators_panel.indicators)
            self.status_label.setText(f"RSI test completed - Total indicators: {indicators_count}")
            
        except Exception as e:
            print(f"Error in RSI test: {e}")
            self.status_label.setText(f"❌ RSI test failed: {e}")
    
    def _verify_fix_status(self):
        """Verify the fix is working"""
        print("\n" + "="*60)
        print("STEP 5 FIX VERIFICATION")
        print("="*60)
        
        # Check signal reception
        if self.signal_received_count > 0:
            print(f"✅ SIGNALS: {self.signal_received_count} indicators_updated signals received")
        else:
            print("❌ SIGNALS: No indicators_updated signals received")
        
        # Check indicators count
        indicators_count = len(self.indicators_panel.indicators)
        if indicators_count > 0:
            print(f"✅ INDICATORS: {indicators_count} indicators successfully added")
            for indicator_id, config in self.indicators_panel.indicators.items():
                print(f"    {indicator_id}: {config.name} ({config.parameters})")
        else:
            print("❌ INDICATORS: No indicators added")
        
        # Check indicator data
        data_count = len(self.indicators_panel.indicator_data)
        if data_count > 0:
            print(f"✅ DATA: {data_count} indicators calculated with data")
        else:
            print("❌ DATA: No indicator data calculated")
        
        # Overall status
        if self.signal_received_count > 0 and indicators_count > 0 and data_count > 0:
            print("\n🎉 STEP 5 FIX STATUS: WORKING!")
            print("✅ Indicator addition functionality is now working correctly")
            print("✅ Dashboard receives indicators_updated signals")
            print("✅ Indicators are calculated and stored properly")
            
            self.status_label.setText("🎉 STEP 5 FIX VERIFIED: Indicators functionality WORKING!")
            self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 10px; background-color: #2d5a27; color: white; border: 1px solid #4a934a;")
            
            # Take screenshot
            self._take_screenshot()
            
        else:
            print("\n❌ STEP 5 FIX STATUS: Still needs work")
            print("Some aspects of indicator functionality are not working")
            
            self.status_label.setText("❌ STEP 5 FIX: Still needs work - check console for details")
            self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 10px; background-color: #5a2727; color: white; border: 1px solid #934a4a;")
        
        print("="*60)
    
    def _take_screenshot(self):
        """Take verification screenshot"""
        try:
            screenshot_path = f"C:\\Users\\skyeAM\\SkyeAM Dropbox\\SAMresearch\\ABtoPython\\2025-08-08_1\\step5_indicators_fix_verified.png"
            pixmap = self.grab()
            pixmap.save(screenshot_path)
            print(f"📸 Verification screenshot saved: {screenshot_path}")
            
        except Exception as e:
            print(f"Screenshot error: {e}")

def test_indicators_fix():
    """Test the indicators fix"""
    print("STEP 5 FIX VERIFICATION TEST")
    print("=" * 50)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Create verification test
        test_widget = IndicatorFixVerification()
        test_widget.show()
        
        print("\nSUCCESS: Step 5 fix verification test launched")
        print("\nTEST PROCEDURE:")
        print("1. Click 'Test 1: Add SMA(20)' - should add SMA indicator")
        print("2. Click 'Test 2: Add EMA(12)' - should add EMA indicator") 
        print("3. Click 'Test 3: Add RSI(14)' - should add RSI indicator")
        print("4. Click 'Verify Fix Status' - should confirm everything works")
        print("\nEXPECTED RESULTS:")
        print("• 'Signals received' counter should increase after each test")
        print("• Status should show indicators being added")
        print("• Verify Fix Status should show 'WORKING!' message")
        print("• Screenshot should be taken automatically on success")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Fix verification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_indicators_fix()
    
    if success:
        print("\n" + "="*50)
        print("STEP 5 FIX VERIFICATION TEST LAUNCHED")
        print("="*50)
        print("The fix adds indicators_updated.emit() to:")
        print("• _add_indicator() method")
        print("• _on_indicator_config_changed() method")  
        print("• _recalculate_all_indicators() method")
        print("\nThis ensures the dashboard receives notifications")
        print("when indicators are added, modified, or recalculated.")
    else:
        print("Step 5 fix verification failed")