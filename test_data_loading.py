#!/usr/bin/env python3
"""
Minimal test to trace timestamp issue in data loading
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')

import pandas as pd
import numpy as np
import time

# Simulate what ConfiguredChart does
def test_configured_chart_loading():
    print("=" * 80)
    print("TESTING CONFIGURED CHART DATA LOADING")
    print("=" * 80)

    csv_file = r"dataRaw\range-ATR30x0.05\ES\diffAdjusted\ES-DIFF-range-ATR30x0.05-dailyATR.csv"
    print(f"\n1. Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file, nrows=10)

    # Check original data
    print(f"\n2. Original Date and Time columns:")
    print(df[['Date', 'Time']].head(3))

    # Combine Date and Time exactly as ConfiguredChart does
    print(f"\n3. Combining Date and Time columns...")
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    print(f"   First 3 DateTime values:")
    for i in range(3):
        dt = df['DateTime'].iloc[i]
        print(f"   [{i}] {dt} -> hour={dt.hour}, minute={dt.minute}, second={dt.second}")

    # Convert to numpy array as ConfiguredChart does
    print(f"\n4. Converting to numpy array (as ConfiguredChart does):")
    timestamps = pd.to_datetime(df['DateTime'])
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.values

    print(f"   Type after conversion: {type(timestamps)}")
    print(f"   First 3 values in array:")
    for i in range(3):
        ts_val = timestamps[i]
        ts = pd.Timestamp(ts_val)
        print(f"   [{i}] {ts_val} (type: {type(ts_val).__name__}) -> hour={ts.hour}, minute={ts.minute}, second={ts.second}")

    # Store in dict as ConfiguredChart does
    print(f"\n5. Storing in full_data dict:")
    full_data = {
        'timestamp': timestamps,
        'open': df['Open'].values.astype(np.float32),
        'high': df['High'].values.astype(np.float32),
        'low': df['Low'].values.astype(np.float32),
        'close': df['Close'].values.astype(np.float32)
    }

    print(f"   Type of full_data['timestamp']: {type(full_data['timestamp'])}")
    print(f"   First 3 values from full_data:")
    for i in range(3):
        ts_val = full_data['timestamp'][i]
        ts = pd.Timestamp(ts_val)
        print(f"   [{i}] {ts_val} -> hour={ts.hour}, minute={ts.minute}, second={ts.second}")

    # Test slicing as format_time_axis does
    print(f"\n6. Testing slice operation (as format_time_axis does):")
    start_idx, end_idx = 0, 5
    timestamps_slice = full_data['timestamp'][start_idx:end_idx]
    timestamps_array = timestamps_slice.values if hasattr(timestamps_slice, 'values') else timestamps_slice

    print(f"   Type after slicing: {type(timestamps_array)}")
    print(f"   First 3 values from slice:")
    for i in range(min(3, len(timestamps_array))):
        ts_val = timestamps_array[i]
        ts = pd.Timestamp(ts_val)
        print(f"   [{i}] {ts_val} -> hour={ts.hour}, minute={ts.minute}, second={ts.second}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    return full_data

if __name__ == "__main__":
    full_data = test_configured_chart_loading()
    print(f"\nFinal check - timestamps preserved: {pd.Timestamp(full_data['timestamp'][0]).hour != 0}")