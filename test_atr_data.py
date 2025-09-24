#!/usr/bin/env python3
"""
Test script to debug ATR data issue
"""

import pandas as pd
import os
import glob

def check_data_files():
    """Find and check all data files"""
    print("\n=== Checking for Data Files ===")

    # Find all parquet files
    parquet_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))

    if parquet_files:
        print(f"Found {len(parquet_files)} parquet file(s):")
        for f in parquet_files[:5]:  # Show first 5
            print(f"  - {f}")
    else:
        print("No parquet files found")

    # Find CSV files
    csv_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.csv') and 'data' in root.lower():
                csv_files.append(os.path.join(root, file))

    if csv_files:
        print(f"\nFound {len(csv_files)} CSV file(s) in data folders:")
        for f in csv_files[:5]:  # Show first 5
            print(f"  - {f}")

    # Check specific locations
    print("\n=== Checking Specific Locations ===")
    specific_paths = [
        'Data/SP500_15min_OHLC_2006-2024.parquet',
        'data/SP500_15min_OHLC_2006-2024.parquet',
        'testData.parquet',
        'Data/testData.parquet'
    ]

    for path in specific_paths:
        if os.path.exists(path):
            print(f"FOUND: {path}")
            return path

    # Check most recent parquet file
    if parquet_files:
        return parquet_files[0]

    return None

def check_atr_columns(file_path):
    """Check for ATR-related columns in the data file"""
    print(f"\n=== Checking ATR columns in: {file_path} ===")

    try:
        # Read the file
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        print(f"Data shape: {df.shape}")
        print(f"\nAll columns: {df.columns.tolist()}")

        # Check for ATR-related columns
        atr_cols = [col for col in df.columns if 'atr' in col.lower() or 'aux' in col.lower()]

        if atr_cols:
            print(f"\nATR/AUX columns found: {atr_cols}")
            print("\nFirst 10 values from each ATR column:")
            for col in atr_cols:
                print(f"\n{col}:")
                print(df[col].head(10).values)

                # Check for non-zero values
                non_zero = df[col][df[col] != 0].head(10)
                if len(non_zero) > 0:
                    print(f"First non-zero values in {col}:")
                    print(non_zero.values)
                else:
                    print(f"WARNING: No non-zero values found in {col}")
        else:
            print("\nNO ATR or AUX columns found!")
            print("This means ATR data needs to be calculated and added to the data file")

            # Check if we have necessary columns to calculate ATR
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                print("\nHowever, High/Low/Close columns exist - ATR can be calculated")

                # Calculate ATR
                print("\nCalculating ATR (14-period)...")
                calculate_and_add_atr(df, file_path)
            elif all(col in df.columns for col in ['high', 'low', 'close']):
                print("\nHowever, high/low/close columns exist - ATR can be calculated")

                # Calculate ATR
                print("\nCalculating ATR (14-period)...")
                df['High'] = df['high']
                df['Low'] = df['low']
                df['Close'] = df['close']
                calculate_and_add_atr(df, file_path)

    except Exception as e:
        print(f"Error reading file: {e}")

def calculate_and_add_atr(df, file_path, period=14):
    """Calculate ATR and add it to the dataframe"""
    print(f"\nCalculating {period}-period ATR...")

    # Calculate True Range
    df['prev_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['prev_close'])
    df['tr3'] = abs(df['Low'] - df['prev_close'])
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    # Calculate ATR (Exponential Moving Average of True Range)
    df['ATR'] = df['TR'].ewm(span=period, adjust=False).mean()

    # Also add a multiplier column for AUX2
    df['AUX1'] = df['ATR']
    df['AUX2'] = 0.1  # Default multiplier

    # Clean up temporary columns
    df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'TR'], axis=1, inplace=True)

    print(f"ATR calculated. First 10 values:")
    print(df['ATR'].head(10).values)

    # Show some statistics
    print(f"\nATR Statistics:")
    print(f"  Mean: {df['ATR'].mean():.2f}")
    print(f"  Std: {df['ATR'].std():.2f}")
    print(f"  Min: {df['ATR'].min():.2f}")
    print(f"  Max: {df['ATR'].max():.2f}")
    print(f"  Non-zero count: {(df['ATR'] != 0).sum()}")

    # Save updated file
    save_path = file_path.replace('.parquet', '_with_atr.parquet').replace('.csv', '_with_atr.csv')

    if save_path.endswith('.parquet'):
        df.to_parquet(save_path, index=False)
    else:
        df.to_csv(save_path, index=False)

    print(f"\nSaved file with ATR to: {save_path}")
    print("Use this file in the data selector dialog to see ATR values")

    return df

if __name__ == "__main__":
    # Find data file
    data_file = check_data_files()

    if data_file:
        # Check for ATR columns
        check_atr_columns(data_file)
    else:
        print("\nNo data files found. Please ensure you have data files in the Data/ directory")