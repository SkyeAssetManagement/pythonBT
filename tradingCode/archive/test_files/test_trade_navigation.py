# tradingCode/test_trade_navigation.py
# Test trade jump/navigation functionality - Step 3 
# Create input box to test trade navigation before hooking up click functionality

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add src path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLineEdit, QPushButton, QLabel, QMessageBox, 
                            QSplitter)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt

# Import dashboard components
from step6_complete_final import FinalTradingDashboard, create_ultimate_test_data

class TradeNavigationTester(QMainWindow):
    """Test widget for trade navigation functionality"""
    
    def __init__(self):
        super().__init__()
        self.dashboard = None
        self.trades_data = []
        self.ohlcv_data = None
        
        self.setWindowTitle("Trade Navigation Test - Step 3")
        self.setGeometry(100, 100, 1200, 800)
        
        self._setup_ui()
        self._load_test_data()
    
    def _setup_ui(self):
        """Setup the test UI with input box and dashboard"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Test controls at top
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Trade Jump Test:"))
        
        self.trade_input = QLineEdit()
        self.trade_input.setPlaceholderText("Enter Trade ID (e.g., T001, T21801, etc.)")
        self.trade_input.setMaximumWidth(200)
        controls_layout.addWidget(self.trade_input)
        
        self.jump_button = QPushButton("Jump to Trade")
        self.jump_button.clicked.connect(self._on_jump_to_trade)
        controls_layout.addWidget(self.jump_button)
        
        self.status_label = QLabel("Ready - Enter a trade ID to test navigation")
        self.status_label.setStyleSheet("color: green;")
        controls_layout.addWidget(self.status_label)
        
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        
        # Dashboard area
        self.dashboard = FinalTradingDashboard()
        main_layout.addWidget(self.dashboard)
        
        # Connect Enter key in input
        self.trade_input.returnPressed.connect(self._on_jump_to_trade)
        
        print("Trade navigation tester UI setup complete")
    
    def _load_test_data(self):
        """Load test data into dashboard"""
        try:
            print("Loading test data for trade navigation test...")
            
            # Create test data
            self.ohlcv_data, trades_csv_path = create_ultimate_test_data()
            
            # Load trades from CSV to get realistic trade IDs
            if os.path.exists(trades_csv_path):
                trades_df = pd.read_csv(trades_csv_path)
                print(f"Loaded {len(trades_df)} trades from CSV")
                print(f"Available trade columns: {list(trades_df.columns)}")
                
                # Convert to list of dicts for easier processing
                self.trades_data = trades_df.to_dict('records')
                
                # Print some example trade IDs for testing
                print("\nExample Trade IDs for testing:")
                for i, trade in enumerate(self.trades_data[:10]):
                    trade_id = trade.get('Trade ID', f'T{i+1:03d}')
                    entry_time = trade.get('entry_time', trade.get('Entry Index', i))
                    print(f"  {trade_id} -> Entry Index: {entry_time}")
                
                if len(self.trades_data) > 10:
                    print(f"  ... and {len(self.trades_data) - 10} more trades")
            
            # Load data into dashboard
            success = self.dashboard.load_ultimate_dataset(self.ohlcv_data, trades_csv_path)
            if success:
                print("Test data loaded successfully into dashboard")
                self.status_label.setText(f"Ready - {len(self.trades_data)} trades loaded. Try trade IDs like T001, T002, etc.")
                self.status_label.setStyleSheet("color: green;")
            else:
                print("Failed to load test data into dashboard")
                self.status_label.setText("Error: Failed to load test data")
                self.status_label.setStyleSheet("color: red;")
            
        except Exception as e:
            print(f"Error loading test data: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: red;")
    
    def _on_jump_to_trade(self):
        """Handle jump to trade button click"""
        try:
            trade_id = self.trade_input.text().strip().upper()
            if not trade_id:
                self.status_label.setText("Please enter a trade ID")
                self.status_label.setStyleSheet("color: orange;")
                return
            
            print(f"Attempting to navigate to trade: {trade_id}")
            
            # Find the trade in our data
            found_trade = None
            for trade in self.trades_data:
                # Check various possible trade ID formats
                current_trade_id = str(trade.get('Trade ID', '')).upper()
                if current_trade_id == trade_id:
                    found_trade = trade
                    break
                    
                # Also check if user entered just a number and we can find T###
                if trade_id.isdigit():
                    if current_trade_id == f"T{int(trade_id):03d}":
                        found_trade = trade
                        break
            
            if not found_trade:
                # Show available trade IDs for reference
                available_ids = [str(trade.get('Trade ID', '')).upper() for trade in self.trades_data[:10]]
                msg = f"Trade '{trade_id}' not found. Available IDs include: {', '.join(available_ids[:5])}"
                if len(available_ids) > 5:
                    msg += f" ... (and {len(self.trades_data) - 5} more)"
                    
                self.status_label.setText(msg)
                self.status_label.setStyleSheet("color: red;")
                print(msg)
                return
            
            # Navigate to the trade
            success = self._navigate_to_trade_data(found_trade)
            
            if success:
                self.status_label.setText(f"SUCCESS: Navigated to trade {trade_id}")
                self.status_label.setStyleSheet("color: green;")
                print(f"Successfully navigated to trade {trade_id}")
            else:
                self.status_label.setText(f"Failed to navigate to trade {trade_id}")
                self.status_label.setStyleSheet("color: red;")
                print(f"Failed to navigate to trade {trade_id}")
                
        except Exception as e:
            error_msg = f"Error: {e}"
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("color: red;")
            print(f"Error in jump to trade: {e}")
            import traceback
            traceback.print_exc()
    
    def _navigate_to_trade_data(self, trade_data):
        """Navigate to specific trade using trade data"""
        try:
            # Extract entry time/index from trade data
            entry_time = None
            
            # Try different possible column names for entry time
            for col in ['entry_time', 'Entry Index', 'EntryTime', 'entry_index']:
                if col in trade_data and trade_data[col] is not None:
                    entry_time = int(trade_data[col])
                    break
            
            if entry_time is None:
                print(f"Could not find entry time in trade data. Available keys: {list(trade_data.keys())}")
                return False
            
            # Get chart data length to validate
            data_length = len(self.ohlcv_data['close']) if self.ohlcv_data else 0
            
            print(f"Navigating to entry time: {entry_time}, data length: {data_length}")
            
            # Ensure entry time is within bounds
            if entry_time < 0 or entry_time >= data_length:
                print(f"Entry time {entry_time} is out of bounds (0 to {data_length})")
                return False
            
            # Calculate viewport to center on trade
            viewport_size = 200  # Show 200 bars around the trade
            start_idx = max(0, entry_time - viewport_size // 2)
            end_idx = min(data_length, entry_time + viewport_size // 2)
            
            # Update chart viewport
            if hasattr(self.dashboard, 'final_chart'):
                print(f"Setting viewport: {start_idx} to {end_idx} (centering on {entry_time})")
                self.dashboard.final_chart.viewport_start = start_idx
                self.dashboard.final_chart.viewport_end = end_idx
                
                # Force refresh chart geometry
                self.dashboard.final_chart._generate_candlestick_geometry()
                self.dashboard.final_chart._generate_trade_markers()
                
                if self.dashboard.final_chart.canvas:
                    self.dashboard.final_chart.canvas.update()
                
                # Also update equity curve if available
                if hasattr(self.dashboard, 'equity_widget'):
                    self.dashboard.equity_widget.sync_viewport(start_idx, end_idx)
                
                return True
            else:
                print("No final_chart found in dashboard")
                return False
                
        except Exception as e:
            print(f"Error navigating to trade: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_trade_navigation():
    """Test trade navigation functionality"""
    print("TESTING TRADE JUMP FUNCTIONALITY - STEP 3")
    print("=" * 60)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Create navigation tester
        tester = TradeNavigationTester()
        tester.show()
        
        print("\nSUCCESS: Trade navigation tester launched")
        print("\nTEST INSTRUCTIONS:")
        print("1. Enter a trade ID in the input box (e.g., T001, T002, T021801, etc.)")
        print("2. Press Enter or click 'Jump to Trade' button")
        print("3. Chart should center on the trade location")
        print("4. Verify trade marker triangles are visible at trade location")
        print("5. Status message will confirm success/failure")
        
        print("\nEXPECTED BEHAVIOR:")
        print("• Chart viewport updates to center on trade")
        print("• Trade marker triangles visible")
        print("• Status shows success message")
        print("• Invalid trade IDs show helpful error messages")
        
        # Take screenshot after brief delay
        import time
        time.sleep(3)
        screenshot_path = "C:\\Users\\skyeAM\\SkyeAM Dropbox\\SAMresearch\\ABtoPython\\2025-08-08_1\\step3_trade_navigation_test.png"
        pixmap = tester.grab()
        pixmap.save(screenshot_path)
        print(f"\nScreenshot saved: {screenshot_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Trade navigation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trade_navigation()
    
    if success:
        print("\n" + "=" * 60)
        print("STEP 3A COMPLETED - TRADE NAVIGATION INPUT BOX TEST")
        print("=" * 60)
        print("VERIFICATION REQUIRED:")
        print("1. Test entering trade IDs in the input box")
        print("2. Verify chart centers on trade location")
        print("3. Confirm trade markers are visible")
        print("4. Check status messages are helpful")
        print("\nOnce input box navigation works, proceed to click functionality!")
    else:
        print("Step 3A test failed - needs additional work")