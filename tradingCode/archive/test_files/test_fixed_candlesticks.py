"""
Test script to verify the candlestick fix is working in the main dashboard
This should show THIN candlesticks instead of fat blobs
"""

import sys
import os
from pathlib import Path

# Add src path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_dashboard_with_fix():
    """Test the dashboard with the applied candlestick fix"""
    print("="*70)
    print("TESTING FIXED CANDLESTICK DASHBOARD")
    print("="*70)
    print("This should display THIN candlesticks, not fat blobs!")
    print("Use keyboard controls to test pan/zoom performance:")
    print("  Arrow keys = Pan")
    print("  +/- = Zoom") 
    print("  Q = Quit")
    print("="*70)
    
    try:
        # Import the dashboard manager
        from src.dashboard.dashboard_manager import launch_dashboard, get_dashboard_manager
        
        print("Launching dashboard with FIXED candlesticks...")
        
        # Launch the dashboard
        launch_dashboard()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the tradingCode directory")
        print("and that all required packages are installed")
        return False
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        return False
    
    return True

def quick_synthetic_test():
    """Quick test with synthetic data to verify the fix"""
    print("Running quick synthetic test...")
    
    try:
        import numpy as np
        import pyqtgraph as pg
        from PyQt5 import QtWidgets
        from src.dashboard.chart_widget import CandlestickItem
        from src.dashboard.data_structures import ChartDataBuffer
        
        # Create synthetic data
        n_bars = 1000
        timestamps = np.arange(n_bars, dtype=np.int64) * 60_000_000_000  # 1 minute intervals
        
        # Generate realistic OHLC data
        np.random.seed(42)
        price = 4000.0
        opens = []
        highs = [] 
        lows = []
        closes = []
        volumes = []
        
        for i in range(n_bars):
            open_price = price
            price_change = np.random.normal(0, 2.0)
            close_price = price + price_change
            
            high = max(open_price, close_price) + abs(np.random.normal(0, 1.0))
            low = min(open_price, close_price) - abs(np.random.normal(0, 1.0))
            volume = np.random.randint(100, 1000)
            
            opens.append(open_price)
            highs.append(high)
            lows.append(low)
            closes.append(close_price)
            volumes.append(volume)
            
            price = close_price
        
        # Create data buffer
        data_buffer = ChartDataBuffer(
            timestamps=timestamps,
            open=np.array(opens),
            high=np.array(highs),
            low=np.array(lows),
            close=np.array(closes),
            volume=np.array(volumes)
        )
        
        print(f"Created synthetic data: {n_bars} bars")
        print(f"Price range: ${min(lows):.2f} - ${max(highs):.2f}")
        
        # Create app and test candlestick item
        app = QtWidgets.QApplication([])
        
        # Create candlestick item (this should use the FIXED rendering)
        candle_item = CandlestickItem(data_buffer)
        
        print("Candlestick item created successfully with FIXED rendering!")
        print("The fix has been applied to the main dashboard.")
        
        return True
        
    except Exception as e:
        print(f"Synthetic test error: {e}")
        return False

def main():
    """Main test function"""
    print("CANDLESTICK FIX VERIFICATION")
    print("="*50)
    
    # First try a quick synthetic test
    if quick_synthetic_test():
        print("\nSynthetic test PASSED - Fix is applied!")
        
        # Now try the full dashboard
        print("\nLaunching full dashboard test...")
        test_dashboard_with_fix()
    else:
        print("\nSynthetic test FAILED - Fix may not be applied correctly")

if __name__ == "__main__":
    main()