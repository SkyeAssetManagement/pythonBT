#!/usr/bin/env python3
"""
Test rendering at large bar indices (after June 2023)
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')

import pandas as pd
import numpy as np

def test_large_bar_index():
    """Test if we can access bars after June 2023"""

    csv_file = r"dataRaw\range-ATR30x0.05\ES\diffAdjusted\ES-DIFF-range-ATR30x0.05-dailyATR.csv"

    print("Loading CSV file...")
    df = pd.read_csv(csv_file)
    print(f"Total rows loaded: {len(df)}")

    # Combine Date and Time
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Find June 2023
    june_2023_mask = df['DateTime'] >= '2023-06-15'
    june_2023_idx = df[june_2023_mask].index[0] if june_2023_mask.any() else None

    print(f"\nBar index at 2023-06-15: {june_2023_idx}")

    # Test accessing data at high indices
    test_indices = [100000, 150000, 199628, 250000, 300000, 370000]

    for idx in test_indices:
        if idx < len(df):
            row = df.iloc[idx]
            print(f"\nBar {idx}:")
            print(f"  Date: {row['Date']}")
            print(f"  Time: {row['Time']}")
            print(f"  Open: {row['Open']}")
            print(f"  Close: {row['Close']}")
        else:
            print(f"\nBar {idx}: Out of range (max: {len(df)-1})")

    # Test numpy array slicing at high indices
    print("\nTesting numpy array slicing...")
    opens = df['Open'].values.astype(np.float32)

    # Test slice around June 2023
    if june_2023_idx:
        start = june_2023_idx - 250
        end = june_2023_idx + 250
        slice_data = opens[start:end]
        print(f"Slice [{start}:{end}]: {len(slice_data)} values")
        print(f"  First value: {slice_data[0] if len(slice_data) > 0 else 'EMPTY'}")
        print(f"  Last value: {slice_data[-1] if len(slice_data) > 0 else 'EMPTY'}")

    # Check if there's any NaN or inf values causing issues
    print("\nChecking for data issues...")
    print(f"NaN values in Open: {df['Open'].isna().sum()}")
    print(f"NaN values in Close: {df['Close'].isna().sum()}")
    print(f"Infinite values in Open: {np.isinf(df['Open']).sum()}")
    print(f"Infinite values in Close: {np.isinf(df['Close']).sum()}")

    return june_2023_idx

if __name__ == "__main__":
    june_idx = test_large_bar_index()
    print(f"\nTest complete. June 2023 starts at bar index: {june_idx}")