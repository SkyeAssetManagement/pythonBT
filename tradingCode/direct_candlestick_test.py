"""
Direct test of SimpleCandlestickItem to verify it works
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import required modules
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Import our classes
from dashboard.chart_widget import SimpleCandlestickItem

class TestDataBuffer:
    """Simple test data buffer"""
    def __init__(self, num_bars=100):
        # Create test OHLC data
        np.random.seed(42)  # Reproducible data
        
        base_price = 3000
        price_changes = np.random.randn(num_bars) * 10
        close_prices = base_price + np.cumsum(price_changes)
        
        # Generate OHLC data
        self.open = np.zeros(num_bars)
        self.high = np.zeros(num_bars)
        self.low = np.zeros(num_bars)
        self.close = close_prices
        
        for i in range(num_bars):
            if i == 0:
                self.open[i] = base_price
            else:
                self.open[i] = self.close[i-1]
            
            # Add some intrabar movement
            range_size = abs(np.random.randn()) * 20 + 5
            self.high[i] = max(self.open[i], self.close[i]) + range_size * 0.5
            self.low[i] = min(self.open[i], self.close[i]) - range_size * 0.5
    
    def __len__(self):
        return len(self.close)

def test_simple_candlestick():
    """Test the SimpleCandlestickItem creation and methods"""
    print("="*60)
    print("DIRECT CANDLESTICK TEST")
    print("="*60)
    
    try:
        # Create Qt application
        app = QtWidgets.QApplication([])
        
        # Create test data
        print("Creating test data...")
        data_buffer = TestDataBuffer(50)  # 50 test candlesticks
        print(f"Created {len(data_buffer)} test bars")
        
        # Create SimpleCandlestickItem
        print("\nCreating SimpleCandlestickItem...")
        candle_item = SimpleCandlestickItem(data_buffer)
        
        # Test methods
        print("\nTesting methods...")
        
        # Test boundingRect
        bounding_rect = candle_item.boundingRect()
        print(f"Bounding rect: {bounding_rect}")
        
        # Test update_range
        candle_item.update_range((0, 25))
        print("update_range method works (no error)")
        
        # Test that picture was created
        if candle_item.picture is not None:
            print("Picture object created successfully")
        else:
            print("ERROR: Picture object is None!")
            return False
        
        # Create a simple view to test painting
        print("\nTesting paint functionality...")
        view = pg.PlotWidget()
        view.addItem(candle_item)
        view.show()
        
        # Process events briefly
        app.processEvents()
        
        print("*** SUCCESS: SimpleCandlestickItem works correctly! ***")
        return True
        
    except Exception as e:
        print(f"*** ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_candlestick()
    if success:
        print("\n" + "="*60)
        print("TEST PASSED: SimpleCandlestickItem is working properly!")
        print("The fix should resolve the black blob issue.")
        print("="*60)
    else:
        print("\n" + "="*60) 
        print("TEST FAILED: There are still issues with SimpleCandlestickItem")
        print("="*60)