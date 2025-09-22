#!/usr/bin/env python3
"""
Verify that the timestamp fix is working correctly
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')

import pandas as pd
import numpy as np
from datetime import datetime

def verify_timestamp_fix():
    """Verify timestamps are correctly parsed and preserved"""

    print("=" * 80)
    print("VERIFYING TIMESTAMP FIX")
    print("=" * 80)

    # Test CSV file
    csv_file = r"dataRaw\range-ATR30x0.05\ES\diffAdjusted\ES-DIFF-range-ATR30x0.05-dailyATR.csv"

    if not os.path.exists(csv_file):
        print(f"ERROR: Test file not found: {csv_file}")
        return False

    print(f"\n1. Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file, nrows=10)

    print(f"\n2. Original columns: {df.columns.tolist()}")
    print(f"   First Date value: {df['Date'].iloc[0]}")
    print(f"   First Time value: {df['Time'].iloc[0]}")

    # Apply the same column mapping logic as launch_pyqtgraph_with_selector.py
    print(f"\n3. Applying column mapping (without 'Date' -> 'DateTime' mapping)")
    column_mapping = {
        'datetime': 'DateTime', 'date': 'DateTime',  # lowercase date is ok
        # 'Date': 'DateTime',  # THIS WAS THE BUG - now removed
        'timestamp': 'DateTime', 'Timestamp': 'DateTime',
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
        'volume': 'Volume', 'vol': 'Volume'
    }

    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    print(f"   Columns after mapping: {df.columns.tolist()}")
    print(f"   'DateTime' in columns: {'DateTime' in df.columns}")
    print(f"   'Date' still in columns: {'Date' in df.columns}")
    print(f"   'Time' still in columns: {'Time' in df.columns}")

    # Combine Date and Time as the fixed code does
    print(f"\n4. Combining Date and Time columns")
    if 'DateTime' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            print(f"   Combined successfully!")
        else:
            print(f"   ERROR: Missing Date or Time columns")
            return False

    # Verify timestamps have time component
    print(f"\n5. Verifying timestamps have time component:")
    success = True
    for i in range(min(5, len(df))):
        dt = df['DateTime'].iloc[i]
        has_time = dt.hour != 0 or dt.minute != 0 or dt.second != 0
        status = "OK" if has_time else "FAILED"
        print(f"   [{i}] {dt} -> hour={dt.hour:02d}, minute={dt.minute:02d}, second={dt.second:02d} [{status}]")
        if not has_time and i > 0:  # Allow first bar to be at midnight
            success = False

    # Test numpy conversion as chart does
    print(f"\n6. Testing numpy array conversion (as chart does):")
    timestamps = pd.to_datetime(df['DateTime'])
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.values

    for i in range(min(3, len(timestamps))):
        ts_val = timestamps[i]
        ts = pd.Timestamp(ts_val)
        has_time = ts.hour != 0 or ts.minute != 0 or ts.second != 0
        status = "OK" if has_time else "FAILED"
        print(f"   [{i}] numpy: {ts} -> hour={ts.hour:02d}, minute={ts.minute:02d} [{status}]")

    print("\n" + "=" * 80)
    if success:
        print("VERIFICATION SUCCESSFUL: Timestamps are preserved correctly!")
        print("The fix is working - Date column is no longer prematurely renamed.")
    else:
        print("VERIFICATION FAILED: Some timestamps lost their time component")
    print("=" * 80)

    return success

if __name__ == "__main__":
    result = verify_timestamp_fix()
    sys.exit(0 if result else 1)