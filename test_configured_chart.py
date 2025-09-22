#!/usr/bin/env python3
"""
Test ConfiguredChart data loading to verify timestamps
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')
import pandas as pd

# Simulate what happens when loading the CSV
def test_csv_loading():
    print("=" * 80)
    print("Testing CSV Loading with Date and Time Columns")
    print("=" * 80)

    # Load the actual CSV file
    csv_file = r"dataRaw\range-ATR30x0.05\ES\diffAdjusted\ES-DIFF-range-ATR30x0.05-dailyATR.csv"

    print(f"\n1. Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Shape: {df.shape}")
    print(f"\n2. First 3 rows:")
    print(df[['Date', 'Time', 'Open', 'Close']].head(3))

    # Combine Date and Time
    print(f"\n3. Combining Date and Time columns:")
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    print(f"   First 5 DateTime values:")
    for i in range(min(5, len(df))):
        dt = df['DateTime'].iloc[i]
        print(f"      [{i}]: {dt} (type: {type(dt).__name__})")

    # Check if time is preserved
    print(f"\n4. Checking time preservation:")
    timestamps = df['DateTime'].values
    for i in range(min(5, len(timestamps))):
        ts = pd.Timestamp(timestamps[i])
        print(f"      [{i}]: Date={ts.date()}, Time={ts.time()}")

    print("\n" + "=" * 80)
    print("Test Complete!")
    return df

if __name__ == "__main__":
    test_csv_loading()