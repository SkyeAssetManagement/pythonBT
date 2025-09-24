#!/usr/bin/env python3
"""
Test script to check what columns are in the actual data files
"""

import pandas as pd
import os
from pathlib import Path

def check_data_columns():
    """Check columns in various data files"""
    print("=" * 60)
    print("Checking Data File Columns")
    print("=" * 60)

    # Look for parquet and CSV files
    data_dirs = [
        "C:\\code\\PythonBT\\data",
        "C:\\code\\PythonBT\\tradingCode\\data",
        "C:\\code\\PythonBT"
    ]

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"\nChecking {data_dir}:")
            # Find parquet files
            for file_path in Path(data_dir).glob("*.parquet"):
                print(f"\n  {file_path.name}:")
                try:
                    df = pd.read_parquet(file_path)
                    print(f"    Shape: {df.shape}")
                    print(f"    Columns: {df.columns.tolist()}")
                    # Check for aux columns
                    aux_cols = [c for c in df.columns if 'aux' in c.lower() or 'AUX' in c]
                    if aux_cols:
                        print(f"    AUX columns found: {aux_cols}")
                        for col in aux_cols:
                            sample = df[col].iloc[0:5].values
                            print(f"      {col} sample: {sample}")
                    else:
                        print("    No AUX columns found")
                except Exception as e:
                    print(f"    Error reading file: {e}")

            # Find CSV files
            for file_path in Path(data_dir).glob("*.csv"):
                if file_path.stat().st_size < 100_000_000:  # Skip huge files
                    print(f"\n  {file_path.name}:")
                    try:
                        df = pd.read_csv(file_path, nrows=5)
                        print(f"    Columns: {df.columns.tolist()}")
                    except Exception as e:
                        print(f"    Error reading file: {e}")

if __name__ == "__main__":
    check_data_columns()