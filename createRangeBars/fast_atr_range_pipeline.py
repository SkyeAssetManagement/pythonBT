#!/usr/bin/env python3
"""
Fast ATR and Range Bar Pipeline
================================

Efficient single-pass pipeline to create daily bars with ATR and range bars.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import time
from numba import njit

# Configuration
ATR_PERIOD = 30
ATR_MULTIPLIER = 0.1

# Paths
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_CSV = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
OUTPUT_DIR = Path("../parquetData/range")

print("=" * 80)
print("FAST ATR AND RANGE BAR PIPELINE")
print("=" * 80)

# Step 1: Create daily bars quickly
print("\n1. Creating daily bars from ticks...")
start_time = time.time()

# Read tick data in chunks and aggregate immediately
parquet_file = pq.ParquetFile(TICK_FILE)
print(f"   Total ticks: {parquet_file.metadata.num_rows:,}")

daily_data = {}
for rg_idx in range(parquet_file.num_row_groups):
    if rg_idx % 100 == 0:
        print(f"   Processing row group {rg_idx}/{parquet_file.num_row_groups}")
    
    table = parquet_file.read_row_group(rg_idx)
    df = table.to_pandas()
    
    # Quick datetime parse
    df['date'] = pd.to_datetime(df['Date'].astype(str), format='%Y/%m/%d').dt.date
    
    # Aggregate by date
    for date, group in df.groupby('date'):
        closes = group['Close'].values
        if date not in daily_data:
            daily_data[date] = {
                'open': closes[0],
                'high': closes.max(),
                'low': closes.min(),
                'close': closes[-1],
                'volume': len(closes)
            }
        else:
            daily_data[date]['high'] = max(daily_data[date]['high'], closes.max())
            daily_data[date]['low'] = min(daily_data[date]['low'], closes.min())
            daily_data[date]['close'] = closes[-1]
            daily_data[date]['volume'] += len(closes)

# Convert to DataFrame
daily_df = pd.DataFrame.from_dict(daily_data, orient='index')
daily_df.index.name = 'date'
daily_df = daily_df.reset_index()
daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df = daily_df.sort_values('date').reset_index(drop=True)

print(f"   Created {len(daily_df)} daily bars")
print(f"   Time: {(time.time() - start_time)/60:.1f} minutes")

# Step 2: Calculate ATR
print("\n2. Calculating ATR-30...")

# Calculate True Range
high = daily_df['high'].values
low = daily_df['low'].values
close = daily_df['close'].values

tr = np.zeros(len(daily_df))
tr[0] = high[0] - low[0]

for i in range(1, len(daily_df)):
    tr[i] = max(high[i] - low[i], 
                abs(high[i] - close[i-1]), 
                abs(low[i] - close[i-1]))

# Calculate ATR with Wilder's smoothing
atr = np.zeros(len(daily_df))
atr[:ATR_PERIOD] = np.cumsum(tr[:ATR_PERIOD]) / np.arange(1, ATR_PERIOD + 1)

for i in range(ATR_PERIOD, len(daily_df)):
    atr[i] = (atr[i-1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD

daily_df['atr'] = atr
daily_df['atr_lagged'] = daily_df['atr'].shift(1).fillna(atr[0])
daily_df['range_size'] = daily_df['atr_lagged'] * ATR_MULTIPLIER

# Save daily data
daily_df.to_csv(DAILY_CSV, index=False)
print(f"   Saved daily data to: {DAILY_CSV}")
print(f"   ATR range: {daily_df['atr'].min():.2f} to {daily_df['atr'].max():.2f}")
print(f"   Range size: {daily_df['range_size'].min():.2f} to {daily_df['range_size'].max():.2f}")

# Step 3: Create range bars
print("\n3. Creating range bars with daily ATR...")

@njit
def find_bars(timestamps, prices, range_size):
    """Fast bar detection with Numba."""
    n = len(timestamps)
    boundaries = []
    
    bar_start = 0
    bar_high = prices[0]
    bar_low = prices[0]
    
    for i in range(1, n):
        bar_high = max(bar_high, prices[i])
        bar_low = min(bar_low, prices[i])
        
        if (bar_high - bar_low) >= range_size:
            boundaries.append(i)
            if i < n - 1:
                bar_start = i
                bar_high = prices[i]
                bar_low = prices[i]
    
    return boundaries

# Create date lookup
date_to_range = {row['date'].date(): row['range_size'] 
                 for _, row in daily_df.iterrows()}

# Process ticks again to create range bars
all_bars = []
print("   Processing ticks to create range bars...")

for rg_idx in range(parquet_file.num_row_groups):
    if rg_idx % 100 == 0 and all_bars:
        print(f"   Row group {rg_idx}/{parquet_file.num_row_groups}, bars created: {len(all_bars):,}")
    
    table = parquet_file.read_row_group(rg_idx)
    df = table.to_pandas()
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
        format='%Y/%m/%d %H:%M:%S.%f'
    )
    df['date'] = df['datetime'].dt.date
    
    # Process each day
    for date, day_group in df.groupby('date'):
        if date not in date_to_range:
            continue
            
        range_size = date_to_range[date]
        if pd.isna(range_size) or range_size <= 0:
            continue
        
        day_group = day_group.sort_values('datetime')
        timestamps = day_group['datetime'].values.astype(np.int64)
        prices = day_group['Close'].values
        
        boundaries = find_bars(timestamps, prices, range_size)
        
        prev_idx = 0
        for boundary in boundaries:
            if boundary > prev_idx:
                bar_data = day_group.iloc[prev_idx:boundary + 1]
                all_bars.append({
                    'timestamp': bar_data.iloc[-1]['datetime'],
                    'open': bar_data.iloc[0]['Close'],
                    'high': bar_data['Close'].max(),
                    'low': bar_data['Close'].min(),
                    'close': bar_data.iloc[-1]['Close'],
                    'volume': bar_data['Volume'].sum(),
                    'ticks': len(bar_data),
                    'AUX1': daily_df[daily_df['date'].dt.date == date]['atr_lagged'].iloc[0],
                    'AUX2': ATR_MULTIPLIER
                })
                prev_idx = boundary + 1

# Convert to DataFrame and save
if all_bars:
    final_df = pd.DataFrame(all_bars)
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate gaps
    final_df['gap'] = 0.0
    if len(final_df) > 1:
        final_df.loc[1:, 'gap'] = final_df['open'].iloc[1:].values - final_df['close'].iloc[:-1].values
    
    # Save
    output_dir = OUTPUT_DIR / f"ATR{ATR_PERIOD}x{ATR_MULTIPLIER}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    diff_path = output_dir / f"ES-DIFF-range-ATR{ATR_PERIOD}x{ATR_MULTIPLIER}-amibroker.parquet"
    final_df.to_parquet(diff_path, compression='snappy')
    
    none_df = final_df.copy()
    none_df[['open', 'high', 'low', 'close']] -= 50
    none_path = output_dir / f"ES-NONE-range-ATR{ATR_PERIOD}x{ATR_MULTIPLIER}-amibroker.parquet"
    none_df.to_parquet(none_path, compression='snappy')
    
    print(f"\n   Created {len(final_df):,} range bars")
    print(f"   Unique ATR values used: {final_df['AUX1'].nunique()}")
    print(f"   Saved to: {diff_path}")

total_time = (time.time() - start_time) / 60
print(f"\n{'=' * 80}")
print(f"PIPELINE COMPLETE in {total_time:.1f} minutes")
print(f"{'=' * 80}")