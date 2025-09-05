"""
Test candlestick fix with automatic screenshot capture
This bypasses GUI issues and creates a direct screenshot
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time

# Add src path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_and_screenshot():
    """Test candlesticks and take automatic screenshot"""
    print("="*70)
    print("CANDLESTICK TEST WITH AUTO-SCREENSHOT")
    print("="*70)
    
    try:
        from PyQt5 import QtWidgets, QtCore, QtGui
        from dashboard.chart_widget import TradingChart
        from dashboard.data_structures import ChartDataBuffer
        
        # Create application (non-interactive)
        app = QtWidgets.QApplication([])
        
        # Load ES data
        es_file = Path(__file__).parent.parent / "dataRaw" / "1m" / "ES" / "Current" / "ES-NONE-1m-EST-NoPad.csv"
        
        if es_file.exists():
            print(f"Loading ES data...")
            df = pd.read_csv(es_file)
            df_test = df.head(200)  # Use 200 bars for clear visibility
            
            # Create data buffer
            timestamps = np.arange(len(df_test), dtype=np.int64) * 60_000_000_000
            data_buffer = ChartDataBuffer(
                timestamps=timestamps,
                open=df_test['Open'].values.astype(np.float64),
                high=df_test['High'].values.astype(np.float64),
                low=df_test['Low'].values.astype(np.float64),
                close=df_test['Close'].values.astype(np.float64),
                volume=df_test['Volume'].values.astype(np.float64)
            )
            
            print(f"Data loaded: {len(data_buffer)} bars")
            print(f"Price range: ${data_buffer.low.min():.2f} - ${data_buffer.high.max():.2f}")
        
        else:
            print("Creating synthetic data...")
            # Synthetic data
            n_bars = 200
            np.random.seed(42)
            
            timestamps = np.arange(n_bars, dtype=np.int64) * 60_000_000_000
            price = 4000.0
            opens, highs, lows, closes, volumes = [], [], [], [], []
            
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
            
            data_buffer = ChartDataBuffer(
                timestamps=timestamps,
                open=np.array(opens),
                high=np.array(highs),
                low=np.array(lows),
                close=np.array(closes),
                volume=np.array(volumes)
            )
            
            print(f"Synthetic data created: {len(data_buffer)} bars")
        
        # Create chart widget
        chart = TradingChart()
        chart.setWindowTitle("CANDLESTICK FIX TEST - Auto Screenshot")
        chart.resize(1400, 900)
        
        # Set data (triggers our fixed candlestick rendering)
        print("Rendering candlesticks with FIXED code...")
        chart.set_data(data_buffer)
        
        # Show and process events
        chart.show()
        app.processEvents()
        
        # Wait for rendering to complete
        time.sleep(2)
        app.processEvents()
        
        # Take screenshot
        print("Taking screenshot...")
        pixmap = chart.grab()
        
        screenshot_file = Path(__file__).parent / "CANDLESTICK_FIX_TEST_RESULT.png"
        success = pixmap.save(str(screenshot_file))
        
        if success:
            print(f"SUCCESS: Screenshot saved to {screenshot_file}")
            print("="*70)
            print("VERIFICATION INSTRUCTIONS:")
            print("="*70)
            print(f"1. Open: {screenshot_file}")
            print("2. Look at the candlesticks:")
            print("   - Should be THIN, not fat blobs")
            print("   - Should have thin black outlines")
            print("   - White bodies for up candles, red for down")
            print("   - Thin vertical wicks")
            print("3. Compare to reference screenshot")
            print("="*70)
            
            # Also test zoomed view
            print("Testing zoomed view...")
            chart.setXRange(150, 200, padding=0)  # Zoom to last 50 bars
            app.processEvents()
            time.sleep(1)
            
            zoomed_pixmap = chart.grab()
            zoomed_file = Path(__file__).parent / "CANDLESTICK_FIX_ZOOMED_TEST.png"
            zoomed_success = zoomed_pixmap.save(str(zoomed_file))
            
            if zoomed_success:
                print(f"ZOOMED screenshot saved: {zoomed_file}")
                
            return True
        else:
            print("ERROR: Failed to save screenshot")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = test_and_screenshot()
    if success:
        print("\\nTest completed successfully - check the screenshot files!")
    else:
        print("\\nTest failed - check error messages above")

if __name__ == "__main__":
    main()