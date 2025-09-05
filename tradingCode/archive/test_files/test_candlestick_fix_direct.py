"""
Direct test of the candlestick fix - bypasses strategy issues
This tests the dashboard directly with candlestick rendering
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add src path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_candlestick_rendering_directly():
    """Test candlestick rendering directly without strategy complications"""
    print("="*70)
    print("DIRECT CANDLESTICK RENDERING TEST")
    print("="*70)
    
    try:
        from PyQt5 import QtWidgets, QtCore
        from dashboard.chart_widget import TradingChart, CandlestickItem
        from dashboard.data_structures import ChartDataBuffer
        
        # Create application
        app = QtWidgets.QApplication(sys.argv)
        
        # Load real ES data directly
        es_file = Path(__file__).parent.parent / "dataRaw" / "1m" / "ES" / "Current" / "ES-NONE-1m-EST-NoPad.csv"
        
        if es_file.exists():
            print(f"Loading ES data from: {es_file}")
            df = pd.read_csv(es_file)
            
            # Use first 500 bars for testing
            df_test = df.head(500)
            print(f"Using {len(df_test)} bars for testing")
            
            # Create timestamps
            timestamps = np.arange(len(df_test), dtype=np.int64) * 60_000_000_000  # 1-minute intervals
            
            # Create data buffer
            data_buffer = ChartDataBuffer(
                timestamps=timestamps,
                open=df_test['Open'].values.astype(np.float64),
                high=df_test['High'].values.astype(np.float64),
                low=df_test['Low'].values.astype(np.float64),
                close=df_test['Close'].values.astype(np.float64),
                volume=df_test['Volume'].values.astype(np.float64)
            )
            
            print(f"Created data buffer with {len(data_buffer)} bars")
            print(f"Price range: ${data_buffer.low.min():.2f} - ${data_buffer.high.max():.2f}")
            
            # Create chart widget (this uses our FIXED candlestick code)
            chart = TradingChart()
            chart.setWindowTitle("DIRECT CANDLESTICK TEST - Should Show THIN Candlesticks")
            chart.resize(1400, 900)
            
            # Set data (this will trigger our fixed candlestick rendering)
            print("Setting data and rendering candlesticks...")
            chart.set_data(data_buffer)
            
            # Show chart
            chart.show()
            
            print("="*70)
            print("DASHBOARD LAUNCHED SUCCESSFULLY!")
            print("="*70)
            print("Instructions:")
            print("1. Look at the candlesticks - they should be THIN, not fat blobs")
            print("2. Use keyboard controls to test:")
            print("   - Arrow Keys: Pan left/right, zoom in/out")
            print("   - R: Reset view")
            print("   - Q: Quit")
            print("3. Take a screenshot and compare to reference")
            print("="*70)
            
            # Enable keyboard focus
            chart.setFocus()
            chart.setFocusPolicy(QtCore.Qt.StrongFocus)
            
            # Run the application
            return app.exec_()
        
        else:
            print(f"ES data file not found: {es_file}")
            
            # Create synthetic data as fallback
            print("Creating synthetic data for testing...")
            n_bars = 200
            np.random.seed(42)
            
            timestamps = np.arange(n_bars, dtype=np.int64) * 60_000_000_000
            
            # Generate realistic OHLC data
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
            
            # Create data buffer
            data_buffer = ChartDataBuffer(
                timestamps=timestamps,
                open=np.array(opens),
                high=np.array(highs),
                low=np.array(lows),
                close=np.array(closes),
                volume=np.array(volumes)
            )
            
            print(f"Created synthetic data: {len(data_buffer)} bars")
            
            # Create and show chart
            chart = TradingChart()
            chart.setWindowTitle("SYNTHETIC DATA TEST - Thin Candlesticks")
            chart.resize(1400, 900)
            chart.set_data(data_buffer)
            chart.show()
            chart.setFocus()
            
            print("Synthetic data dashboard launched - test candlestick appearance!")
            
            return app.exec_()
    
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = test_candlestick_rendering_directly()
    print(f"\\nTest completed with result: {success}")

if __name__ == "__main__":
    main()