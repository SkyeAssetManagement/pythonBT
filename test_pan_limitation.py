#!/usr/bin/env python3
"""
Test to find why we can't pan past June 2023
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')

import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg

def test_pan_limitation():
    """Test rendering at different bar ranges"""

    csv_file = r"dataRaw\range-ATR30x0.05\ES\diffAdjusted\ES-DIFF-range-ATR30x0.05-dailyATR.csv"

    print("Loading CSV file...")
    df = pd.read_csv(csv_file)
    print(f"Total rows: {len(df)}")

    # Combine Date and Time
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Find key dates
    dates_to_check = [
        ('2023-06-14', None),
        ('2023-06-15', None),
        ('2023-06-16', None),
        ('2024-01-01', None),
        ('2024-06-01', None),
        ('2025-01-01', None)
    ]

    print("\nFinding bar indices for key dates:")
    for i, (date_str, _) in enumerate(dates_to_check):
        mask = df['DateTime'] >= date_str
        if mask.any():
            idx = df[mask].index[0]
            dates_to_check[i] = (date_str, idx)
            print(f"  {date_str}: bar index {idx}")
        else:
            print(f"  {date_str}: NOT FOUND")

    # Test data access at different ranges
    print("\nTesting data access at different ranges:")
    opens = df['Open'].values.astype(np.float32)

    # Test ranges
    test_ranges = [
        (0, 500),
        (100000, 100500),
        (199000, 199500),
        (199628, 200128),  # June 15, 2023
        (250000, 250500),
        (300000, 300500),
        (370000, 370500)
    ]

    for start, end in test_ranges:
        if end <= len(opens):
            slice_data = opens[start:end]
            if len(slice_data) > 0:
                date_at_start = df['DateTime'].iloc[start]
                print(f"  Range [{start:6}:{end:6}]: OK - {len(slice_data)} bars, starts at {date_at_start}")
            else:
                print(f"  Range [{start:6}:{end:6}]: EMPTY SLICE!")
        else:
            print(f"  Range [{start:6}:{end:6}]: Out of bounds (max: {len(opens)})")

    # Check if there's something special about bar 199628
    print(f"\nChecking bars around June 15, 2023 (index 199628):")
    for offset in [-2, -1, 0, 1, 2]:
        idx = 199628 + offset
        if idx >= 0 and idx < len(df):
            row = df.iloc[idx]
            print(f"  Bar {idx}: {row['Date']} {row['Time']}, Open={row['Open']}, Close={row['Close']}")

    # Check for any data gaps or NaN values
    print(f"\nChecking for data issues around June 2023:")
    june_start = 199600
    june_end = 199650
    june_data = df.iloc[june_start:june_end]

    print(f"  NaN in Open: {june_data['Open'].isna().sum()}")
    print(f"  NaN in Close: {june_data['Close'].isna().sum()}")
    print(f"  Infinite values: {np.isinf(june_data['Open'].values).sum()}")

    # Check data conversion
    print(f"\nTesting data conversion as done in chart:")
    timestamps = pd.to_datetime(df['DateTime'])
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.values

    print(f"  Timestamp array type: {type(timestamps)}")
    print(f"  Timestamp dtype: {timestamps.dtype}")
    print(f"  Length: {len(timestamps)}")

    # Check slicing behavior
    print(f"\nTesting slicing at boundary:")
    test_idx = 199628
    for end_idx in [199629, 199630, 199640, 200000, 250000]:
        if end_idx <= len(opens):
            test_slice = opens[test_idx:end_idx]
            ts_slice = timestamps[test_idx:end_idx]
            print(f"  Slice [{test_idx}:{end_idx}]: {len(test_slice)} values, {len(ts_slice)} timestamps")

    return len(df)

if __name__ == "__main__":
    total_bars = test_pan_limitation()
    print(f"\nTotal bars in dataset: {total_bars}")
    print("If all tests pass, the issue is likely in the chart rendering logic, not data access.")