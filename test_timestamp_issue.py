#!/usr/bin/env python3
"""
Test to diagnose why timestamps show only dates without times
"""

import pandas as pd
import numpy as np
import sys
import os

def test_csv_timestamp_loading():
    """Test how CSV data is being loaded and timestamps are processed"""
    print("=" * 80)
    print("TIMESTAMP DIAGNOSTIC TEST")
    print("=" * 80)

    # Load the actual CSV file
    csv_file = r"dataRaw\range-ATR30x0.05\ES\diffAdjusted\ES-DIFF-range-ATR30x0.05-dailyATR.csv"

    print(f"\n1. Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file, nrows=10)  # Just first 10 rows

    print(f"\n2. Raw CSV data:")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"\n   First 5 rows of Date and Time columns:")
    print(df[['Date', 'Time', 'Open', 'Close']].head())

    print(f"\n3. Testing Date+Time combination methods:")

    # Method 1: String concatenation
    print(f"\n   Method 1: String concatenation")
    df['DateTime1'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    print(f"   Result: {df['DateTime1'].iloc[0]}")
    print(f"   Type: {type(df['DateTime1'].iloc[0])}")
    ts1 = df['DateTime1'].iloc[0]
    print(f"   Hour: {ts1.hour}, Minute: {ts1.minute}, Second: {ts1.second}")

    # Method 2: Format specification
    print(f"\n   Method 2: With format specification")
    df['DateTime2'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y/%m/%d %H:%M:%S')
    print(f"   Result: {df['DateTime2'].iloc[0]}")
    ts2 = df['DateTime2'].iloc[0]
    print(f"   Hour: {ts2.hour}, Minute: {ts2.minute}, Second: {ts2.second}")

    # Method 3: Separate columns
    print(f"\n   Method 3: Parse separately and combine")
    dates = pd.to_datetime(df['Date'], format='%Y/%m/%d')
    times = pd.to_timedelta(df['Time'])
    df['DateTime3'] = dates + times
    print(f"   Result: {df['DateTime3'].iloc[0]}")
    ts3 = df['DateTime3'].iloc[0]
    print(f"   Hour: {ts3.hour}, Minute: {ts3.minute}, Second: {ts3.second}")

    print(f"\n4. Testing numpy array conversion:")
    timestamps = df['DateTime1'].values
    print(f"   Array type: {type(timestamps)}")
    print(f"   First element: {timestamps[0]}")
    print(f"   First element type: {type(timestamps[0])}")

    # Test pandas Timestamp conversion
    ts_pd = pd.Timestamp(timestamps[0])
    print(f"   Pandas Timestamp: {ts_pd}")
    print(f"   Hour: {ts_pd.hour}, Minute: {ts_pd.minute}, Second: {ts_pd.second}")

    print(f"\n5. Testing strftime formatting:")
    for i in range(3):
        ts = df['DateTime1'].iloc[i]
        print(f"   [{i}] Original: {ts}")
        print(f"        strftime('%H:%M:%S'): {ts.strftime('%H:%M:%S')}")
        print(f"        strftime('%Y-%m-%d %H:%M:%S'): {ts.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return df

if __name__ == "__main__":
    test_csv_timestamp_loading()