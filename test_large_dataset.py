"""
Test script to debug rendering past 500 bars
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_large_dataset(num_bars=10000):
    """Create a large dataset for testing"""
    print(f"Creating dataset with {num_bars} bars...")

    # Generate timestamps
    timestamps = []
    base_time = pd.Timestamp('2020-01-01 09:00:00')
    for i in range(num_bars):
        timestamps.append(base_time + timedelta(minutes=i*5))  # 5-minute bars

    # Generate OHLCV data
    np.random.seed(42)
    close = 4000  # Starting price
    closes = []
    opens = []
    highs = []
    lows = []
    volumes = []

    for i in range(num_bars):
        # Random walk for close
        change = np.random.randn() * 5  # $5 volatility
        close = close + change
        closes.append(close)

        # Generate OHLC from close
        open_price = close + np.random.uniform(-2, 2)
        high = max(open_price, close) + abs(np.random.normal(0, 3))
        low = min(open_price, close) - abs(np.random.normal(0, 3))

        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        volumes.append(np.random.uniform(100, 1000))

    # Create DataFrame
    df = pd.DataFrame({
        'DateTime': timestamps,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })

    # Save as CSV
    filename = f'test_data_{num_bars}_bars.csv'
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

    # Also save as parquet
    parquet_filename = f'test_data_{num_bars}_bars.parquet'
    df.to_parquet(parquet_filename, index=False)
    print(f"Saved to {parquet_filename}")

    return filename, parquet_filename

def test_rendering():
    """Instructions for testing rendering"""
    print("\n" + "="*60)
    print("TEST INSTRUCTIONS")
    print("="*60)
    print("\n1. Launch the integrated trading system:")
    print("   python integrated_trading_launcher.py")
    print("\n2. Click 'Data Visualization'")
    print("\n3. Select one of the test data files created above")
    print("\n4. After chart loads, try to pan beyond bar 500")
    print("\n5. Watch the console for debug output")
    print("\n6. Expected behavior:")
    print("   - Should see [RENDER_RANGE] logs when panning")
    print("   - Should see [ON_RANGE_CHANGED] logs")
    print("   - Bars should continue to render past 500")
    print("\n7. Report any issues:")
    print("   - Does rendering stop at bar 500?")
    print("   - Are the logs showing correct ranges?")
    print("   - Any error messages?")

if __name__ == "__main__":
    # Create test datasets
    csv_file, parquet_file = create_large_dataset(10000)

    print(f"\nTest files created:")
    print(f"  CSV: {csv_file}")
    print(f"  Parquet: {parquet_file}")

    test_rendering()