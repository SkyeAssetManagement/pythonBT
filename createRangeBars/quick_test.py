"""
Quick Performance Test - Windows Compatible
Tests just the CSV to Parquet conversion with basic range bar generation
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

def test_performance():
    csv_path = r"C:\Users\skyeAM\SkyeAM Dropbox\SAMresearch\ABtoPython\dataRaw\tick\ES-DIFF-Tick1-21toT.tick"
    
    print("QUICK PERFORMANCE TEST")
    print("="*50)
    
    # Check file size
    file_size = Path(csv_path).stat().st_size / (1024**3)  # GB
    print(f"Input file size: {file_size:.2f} GB")
    
    # Test CSV reading speed (sample)
    print("\nTesting CSV read speed with sample...")
    
    start_time = time.time()
    
    # Read just first 1M rows to estimate performance
    try:
        sample_data = pd.read_csv(csv_path, nrows=1_000_000)
        sample_time = time.time() - start_time
        
        print(f"Sample read (1M rows): {sample_time:.1f}s")
        print(f"Estimated full read time: {sample_time * (file_size / 0.037):.0f}s")
        print(f"Columns: {list(sample_data.columns)}")
        print(f"Data shape: {sample_data.shape}")
        print(f"Memory usage: {sample_data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        
        # Show data sample
        print("\nFirst 5 rows:")
        print(sample_data.head())
        
        # Test parquet conversion on sample
        print("\nTesting parquet conversion on sample...")
        parquet_start = time.time()
        
        # Convert datetime
        if 'Date' in sample_data.columns and 'Time' in sample_data.columns:
            datetime_str = sample_data['Date'].astype(str) + ' ' + sample_data['Time'].astype(str)
            sample_data['datetime'] = pd.to_datetime(datetime_str, format='%Y/%m/%d %H:%M:%S.%f')
            sample_data = sample_data.drop(['Date', 'Time'], axis=1)
        
        # Optimize dtypes
        for col in sample_data.columns:
            if col == 'datetime':
                continue
            if col in ['Open', 'High', 'Low', 'Close']:
                sample_data[col] = sample_data[col].astype('float32')
            elif col == 'Volume':
                sample_data[col] = sample_data[col].astype('int32')
            elif col in ['Up Ticks', 'Down Ticks', 'Same Ticks']:
                sample_data[col] = sample_data[col].astype('int16')
        
        # Save sample parquet
        sample_data.to_parquet('sample_test.parquet', compression='snappy')
        parquet_time = time.time() - parquet_start
        
        # Check parquet file size
        parquet_size = Path('sample_test.parquet').stat().st_size / (1024**2)  # MB
        original_size = sample_data.memory_usage(deep=True).sum() / (1024**2)
        
        print(f"Parquet conversion (1M rows): {parquet_time:.1f}s")
        print(f"Parquet size: {parquet_size:.1f} MB")
        print(f"Compression ratio: {original_size/parquet_size:.1f}:1")
        
        # Estimate full conversion time
        total_rows_estimate = int(file_size * 1e9 / (sample_data.memory_usage(deep=True).sum() / len(sample_data)))
        full_conversion_time = parquet_time * (total_rows_estimate / 1_000_000)
        
        print(f"\nESTIMATED FULL FILE PERFORMANCE:")
        print(f"Estimated total rows: {total_rows_estimate:,}")
        print(f"Estimated CSV->Parquet time: {full_conversion_time:.0f}s ({full_conversion_time/60:.1f} min)")
        
        # Test simple range bar creation on sample
        print(f"\nTesting range bar creation on sample...")
        
        range_start = time.time()
        
        # Simple range bar logic (0.22 points)
        prices = sample_data['Close'].values
        range_size = 0.22
        
        bars = []
        current_open = prices[0]
        current_high = prices[0]
        current_low = prices[0]
        range_top = current_open + range_size
        range_bottom = current_open - range_size
        
        for price in prices[1:]:
            if price >= range_top or price <= range_bottom:
                # Close bar
                bars.append({
                    'Open': current_open,
                    'High': current_high,
                    'Low': current_low,
                    'Close': prices[len(bars)],  # Previous price
                })
                # Start new bar
                current_open = prices[len(bars)]
                current_high = max(current_open, price)
                current_low = min(current_open, price)
                range_top = current_open + range_size
                range_bottom = current_open - range_size
            else:
                current_high = max(current_high, price)
                current_low = min(current_low, price)
        
        range_time = time.time() - range_start
        n_bars = len(bars)
        
        print(f"Range bars created: {n_bars:,} bars from {len(sample_data):,} ticks")
        print(f"Compression: {len(sample_data)/n_bars:.0f}:1")
        print(f"Range bar time: {range_time:.2f}s")
        print(f"Range bar rate: {n_bars/range_time:,.0f} bars/sec")
        
        # Estimate frequency
        bars_per_day = n_bars / (len(sample_data) / 50000)  # Assume 50k ticks/day
        minutes_per_bar = (17 * 60) / bars_per_day if bars_per_day > 0 else 0
        
        print(f"Estimated frequency: {minutes_per_bar:.1f} min/bar ({bars_per_day:.0f} bars/day)")
        
        return {
            'file_size_gb': file_size,
            'estimated_rows': total_rows_estimate,
            'estimated_conversion_time': full_conversion_time,
            'sample_bars': n_bars,
            'estimated_minutes_per_bar': minutes_per_bar
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    results = test_performance()
    
    if results:
        print(f"\nQUICK TEST SUMMARY:")
        print(f"File Size: {results['file_size_gb']:.2f} GB")
        print(f"Est. Rows: {results['estimated_rows']:,}")
        print(f"Est. Conversion: {results['estimated_conversion_time']:.0f}s")
        print(f"Range Bar Frequency: {results['estimated_minutes_per_bar']:.1f} min/bar")