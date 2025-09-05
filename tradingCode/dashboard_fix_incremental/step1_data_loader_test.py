"""
Step 1B: Data loading test - Load real 1m data and measure performance
Focus: Test data loading speed and memory usage before building dashboard
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.parent  # Go up to ABtoPython folder
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "tradingCode" / "src"))

def test_csv_data_loading():
    """Test loading CSV data from the dataRaw folder"""
    print("\n=== CSV DATA LOADING TEST ===")
    
    # Look for CSV data files
    data_dir = parent_dir / "dataRaw" / "1m"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return None
        
    # Find first available symbol
    for symbol_dir in data_dir.iterdir():
        if symbol_dir.is_dir():
            current_dir = symbol_dir / "Current"
            if current_dir.exists():
                csv_files = list(current_dir.glob("*.csv"))
                if csv_files:
                    csv_file = csv_files[0]
                    print(f"Found data file: {csv_file}")
                    
                    # Load and time it
                    start_time = time.time()
                    df = pd.read_csv(csv_file)
                    load_time = time.time() - start_time
                    
                    print(f"Loaded {len(df):,} rows in {load_time:.3f}s")
                    print(f"Columns: {list(df.columns)}")
                    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                    print(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
                    
                    return df
    
    print("No CSV files found")
    return None

def test_parquet_data_loading():
    """Test loading Parquet data"""
    print("\n=== PARQUET DATA LOADING TEST ===")
    
    # Look for Parquet data files
    data_dir = parent_dir / "parquetData" / "1m"
    
    if not data_dir.exists():
        print(f"Parquet directory not found: {data_dir}")
        return None
        
    # Find first available symbol
    for symbol_dir in data_dir.iterdir():
        if symbol_dir.is_dir():
            current_dir = symbol_dir / "Current"
            if current_dir.exists():
                parquet_files = list(current_dir.glob("*.parquet"))
                if parquet_files:
                    parquet_file = parquet_files[0]
                    print(f"Found parquet file: {parquet_file}")
                    
                    # Load and time it
                    start_time = time.time()
                    df = pd.read_parquet(parquet_file)
                    load_time = time.time() - start_time
                    
                    print(f"Loaded {len(df):,} rows in {load_time:.3f}s")
                    print(f"Columns: {list(df.columns)}")
                    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                    if len(df) > 0:
                        print(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
                    
                    return df
    
    print("No Parquet files found")
    return None

def test_ohlc_conversion(df):
    """Test converting dataframe to OHLC format for candlesticks"""
    if df is None:
        return None
        
    print(f"\n=== OHLC CONVERSION TEST ===")
    start_time = time.time()
    
    # Assume standard OHLC columns - adapt based on actual data
    columns = df.columns.tolist()
    print(f"Available columns: {columns}")
    
    # Try to identify OHLC columns
    ohlc_cols = {}
    for col in columns:
        col_lower = col.lower()
        if 'date' in col_lower or 'time' in col_lower:
            ohlc_cols['datetime'] = col
        elif col_lower in ['open', 'o']:
            ohlc_cols['open'] = col
        elif col_lower in ['high', 'h']:
            ohlc_cols['high'] = col
        elif col_lower in ['low', 'l']:
            ohlc_cols['low'] = col
        elif col_lower in ['close', 'c']:
            ohlc_cols['close'] = col
        elif col_lower in ['volume', 'v', 'vol']:
            ohlc_cols['volume'] = col
    
    print(f"Identified OHLC columns: {ohlc_cols}")
    
    if len(ohlc_cols) < 4:  # Need at least datetime + OHLC
        print("Cannot identify OHLC columns - showing first few rows:")
        print(df.head())
        return None
    
    # Convert to numpy arrays for speed
    ohlc_data = []
    for i in range(len(df)):
        x = i  # Use index as X coordinate
        open_val = df.iloc[i][ohlc_cols['open']]
        high_val = df.iloc[i][ohlc_cols['high']]
        low_val = df.iloc[i][ohlc_cols['low']]
        close_val = df.iloc[i][ohlc_cols['close']]
        
        ohlc_data.append([x, open_val, high_val, low_val, close_val])
    
    conversion_time = time.time() - start_time
    print(f"Converted {len(ohlc_data):,} bars to OHLC format in {conversion_time:.3f}s")
    
    # Show sample data
    print("Sample OHLC data:")
    for i in range(min(5, len(ohlc_data))):
        x, o, h, l, c = ohlc_data[i]
        print(f"  Bar {x}: O={o:.4f} H={h:.4f} L={l:.4f} C={c:.4f}")
    
    return ohlc_data

def measure_memory_usage():
    """Measure current memory usage"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024**2
    return memory_mb

def main():
    """Main test function"""
    print("="*60)
    print("STEP 1B: DATA LOADING PERFORMANCE TEST")
    print("="*60)
    print("Objective: Test loading 1m data quickly into memory")
    print("="*60)
    
    initial_memory = measure_memory_usage()
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Test CSV loading
    csv_data = test_csv_data_loading()
    if csv_data is not None:
        csv_memory = measure_memory_usage()
        print(f"Memory after CSV load: {csv_memory:.1f} MB (+{csv_memory-initial_memory:.1f} MB)")
        ohlc_data = test_ohlc_conversion(csv_data)
        if ohlc_data:
            final_memory = measure_memory_usage()
            print(f"Memory after OHLC conversion: {final_memory:.1f} MB (+{final_memory-csv_memory:.1f} MB)")
        
        print(f"\nReady for dashboard with {len(ohlc_data):,} candlesticks") if ohlc_data else None
        return ohlc_data
    
    # Test Parquet loading if CSV failed
    parquet_data = test_parquet_data_loading()
    if parquet_data is not None:
        parquet_memory = measure_memory_usage()
        print(f"Memory after Parquet load: {parquet_memory:.1f} MB (+{parquet_memory-initial_memory:.1f} MB)")
        ohlc_data = test_ohlc_conversion(parquet_data)
        if ohlc_data:
            final_memory = measure_memory_usage()
            print(f"Memory after OHLC conversion: {final_memory:.1f} MB (+{final_memory-parquet_memory:.1f} MB)")
        
        print(f"\nReady for dashboard with {len(ohlc_data):,} candlesticks") if ohlc_data else None
        return ohlc_data
    
    print("\nNo data files found - will use synthetic data")
    return None

if __name__ == "__main__":
    main()