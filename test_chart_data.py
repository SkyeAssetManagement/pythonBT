#!/usr/bin/env python3
"""
Test Chart Data Loading
=======================
Debug chart data loading issues
"""

import sys
import os
import pandas as pd

sys.path.insert(0, 'src')

def test_sample_data_loading():
    """Test loading sample data files"""
    print("=" * 60)
    print("TESTING SAMPLE DATA FILES")
    print("=" * 60)

    data_files = [
        'data/sample_trading_data.csv',
        'data/sample_trading_data_small.csv'
    ]

    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"\n[TEST] Loading {data_file}")
            try:
                df = pd.read_csv(data_file)
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")

                # Check for required OHLCV columns
                required_cols = ['Open', 'High', 'Low', 'Close']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    print(f"  [WARNING] Missing required columns: {missing_cols}")

                    # Check for lowercase versions
                    lowercase_cols = [col for col in required_cols if col.lower() in df.columns]
                    if lowercase_cols:
                        print(f"  [INFO] Found lowercase versions: {[col.lower() for col in lowercase_cols]}")
                else:
                    print("  [OK] All required OHLCV columns present")

                # Check datetime columns
                datetime_cols = [col for col in ['DateTime', 'Date', 'Time', 'timestamp'] if col in df.columns]
                print(f"  DateTime columns: {datetime_cols}")

                # Show sample data
                print(f"  First row: {df.iloc[0].to_dict()}")

            except Exception as e:
                print(f"  [ERROR] Failed to load: {e}")
        else:
            print(f"\n[SKIP] {data_file} not found")

def test_chart_data_conversion():
    """Test converting sample data to chart format"""
    print("\n" + "=" * 60)
    print("TESTING CHART DATA CONVERSION")
    print("=" * 60)

    data_file = 'data/sample_trading_data_small.csv'
    if not os.path.exists(data_file):
        print("[SKIP] Sample data file not found")
        return

    try:
        print(f"\n[TEST] Converting {data_file} to chart format...")
        df = pd.read_csv(data_file)

        # Apply the same logic as headless launcher
        if 'Date' in df.columns and 'Time' in df.columns:
            print("  Converting Date+Time columns...")
            df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        elif 'timestamp' in df.columns:
            print("  Using existing timestamp column...")
            df['DateTime'] = pd.to_datetime(df['timestamp'])
        elif 'DateTime' not in df.columns:
            print("  Creating synthetic datetime...")
            df['DateTime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='5min')

        # Column mapping
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Create chart data structure
        chart_data = {
            'timestamp': df['DateTime'].values,
            'open': df['Open'].values.astype(float),
            'high': df['High'].values.astype(float),
            'low': df['Low'].values.astype(float),
            'close': df['Close'].values.astype(float),
            'volume': df['Volume'].values.astype(float) if 'Volume' in df.columns else None
        }

        print(f"  [SUCCESS] Created chart data with {len(chart_data['timestamp'])} bars")
        print(f"  Columns: {list(chart_data.keys())}")
        print(f"  First timestamp: {chart_data['timestamp'][0]}")
        print(f"  Price range: ${chart_data['close'].min():.2f} - ${chart_data['close'].max():.2f}")

        # Check for None/null values
        for key, values in chart_data.items():
            if values is not None:
                if hasattr(values, '__len__'):
                    null_count = sum(1 for v in values if v is None or (hasattr(v, '__iter__') and pd.isna(v)))
                    print(f"  {key}: {null_count} null values out of {len(values)}")

        return chart_data

    except Exception as e:
        print(f"  [ERROR] Chart data conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_parquet_files():
    """Test loading parquet files if available"""
    print("\n" + "=" * 60)
    print("TESTING PARQUET FILES")
    print("=" * 60)

    # Look for parquet files
    parquet_dirs = ['parquetData', 'data']
    parquet_files = []

    for dir_name in parquet_dirs:
        if os.path.exists(dir_name):
            for file in os.listdir(dir_name):
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(dir_name, file))

    if not parquet_files:
        print("[SKIP] No parquet files found")
        return

    # Test first parquet file
    parquet_file = parquet_files[0]
    print(f"\n[TEST] Loading {parquet_file}")

    try:
        df = pd.read_parquet(parquet_file)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")

        # Check for OHLCV data
        ohlcv_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume',
                                     'open', 'high', 'low', 'close', 'volume'] if col in df.columns]
        print(f"  OHLCV columns: {ohlcv_cols}")

        if len(ohlcv_cols) >= 4:
            print("  [OK] Sufficient OHLCV data for charting")
        else:
            print("  [WARNING] Insufficient OHLCV data")

    except Exception as e:
        print(f"  [ERROR] Failed to load parquet: {e}")

def main():
    """Run all chart data tests"""
    print("CHART DATA LOADING TEST")
    print("Debug chart data issues in headless system")

    # Test 1: Sample data files
    test_sample_data_loading()

    # Test 2: Chart data conversion
    chart_data = test_chart_data_conversion()

    # Test 3: Parquet files
    test_parquet_files()

    # Summary
    print("\n" + "=" * 60)
    print("CHART DATA TEST SUMMARY")
    print("=" * 60)

    if chart_data:
        print("[SUCCESS] Chart data conversion working")
        print("The headless launcher should be able to load and display chart data")
        print("\nTry running: python launch_headless_system.py")
        print("And select one of the sample data files when prompted")
    else:
        print("[ERROR] Chart data conversion failed")
        print("Chart data loading issues detected - needs investigation")

if __name__ == '__main__':
    main()