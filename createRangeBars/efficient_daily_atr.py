#!/usr/bin/env python3
"""
Efficient Daily ATR Creation
=============================

Creates daily bars and ATR efficiently without loading all ticks at once.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import time

# Configuration
ATR_PERIOD = 30
ATR_MULTIPLIER = 0.1

# Paths
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_CSV = Path("../parquetData/ES-DIFF-daily-with-atr.csv")

print("=" * 80)
print("EFFICIENT DAILY ATR CREATION")
print("=" * 80)
print(f"ATR Period: {ATR_PERIOD}")
print(f"ATR Multiplier: {ATR_MULTIPLIER}")
print("=" * 80)

# Step 1: Create daily bars efficiently
print("\nStep 1: Creating daily bars from tick data...")
start_time = time.time()

parquet_file = pq.ParquetFile(TICK_FILE)
total_rows = parquet_file.metadata.num_rows
num_row_groups = parquet_file.num_row_groups
print(f"Total ticks: {total_rows:,}")
print(f"Row groups: {num_row_groups}")

# Process in batches of row groups
daily_data = {}
batch_size = 10  # Process 10 row groups at a time

for batch_start in range(0, num_row_groups, batch_size):
    batch_end = min(batch_start + batch_size, num_row_groups)
    
    if batch_start % 50 == 0:
        elapsed = (time.time() - start_time) / 60
        pct = (batch_start / num_row_groups) * 100
        print(f"  Processing row groups {batch_start}-{batch_end} ({pct:.1f}%) - {elapsed:.1f} min elapsed")
    
    # Read batch of row groups
    batch_dfs = []
    for rg_idx in range(batch_start, batch_end):
        table = parquet_file.read_row_group(rg_idx)
        batch_dfs.append(table.to_pandas())
    
    # Combine batch
    batch_df = pd.concat(batch_dfs, ignore_index=True)
    
    # Parse date only (faster than full datetime)
    batch_df['date'] = pd.to_datetime(batch_df['Date'], format='%Y/%m/%d')
    
    # Group by date and aggregate
    daily_agg = batch_df.groupby('date')['Close'].agg(['first', 'max', 'min', 'last', 'count'])
    daily_agg.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Merge with existing daily data
    for date, row in daily_agg.iterrows():
        if date not in daily_data:
            daily_data[date] = row.to_dict()
        else:
            # Update high/low if needed
            daily_data[date]['high'] = max(daily_data[date]['high'], row['high'])
            daily_data[date]['low'] = min(daily_data[date]['low'], row['low'])
            daily_data[date]['close'] = row['close']  # Last close wins
            daily_data[date]['volume'] += row['volume']
    
    # Clean up
    del batch_dfs, batch_df, daily_agg

# Convert to DataFrame
daily_df = pd.DataFrame.from_dict(daily_data, orient='index')
daily_df.index.name = 'date'
daily_df = daily_df.reset_index()
daily_df = daily_df.sort_values('date').reset_index(drop=True)

print(f"\nCreated {len(daily_df)} daily bars")
print(f"Date range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")
print(f"Time taken: {(time.time() - start_time)/60:.1f} minutes")

# Step 2: Calculate ATR
print("\nStep 2: Calculating ATR with Wilder's smoothing...")

# Calculate True Range
high = daily_df['high'].values
low = daily_df['low'].values
close = daily_df['close'].values

# TR calculation
tr = np.zeros(len(daily_df))
tr[0] = high[0] - low[0]

for i in range(1, len(daily_df)):
    hl = high[i] - low[i]
    hc = abs(high[i] - close[i-1])
    lc = abs(low[i] - close[i-1])
    tr[i] = max(hl, hc, lc)

# ATR calculation using Wilder's smoothing
atr = np.zeros(len(daily_df))

# Initialize first ATR period
if len(tr) >= ATR_PERIOD:
    atr[ATR_PERIOD-1] = np.mean(tr[:ATR_PERIOD])
    
    # Calculate rest using Wilder's smoothing
    for i in range(ATR_PERIOD, len(daily_df)):
        atr[i] = (atr[i-1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD
else:
    # Not enough data, use cumulative average
    for i in range(len(daily_df)):
        atr[i] = np.mean(tr[:i+1])

# Fill in early ATR values
for i in range(min(ATR_PERIOD-1, len(daily_df))):
    if atr[i] == 0:
        atr[i] = atr[ATR_PERIOD-1] if ATR_PERIOD-1 < len(daily_df) else tr[i]

daily_df['atr'] = atr
daily_df['atr_lagged'] = daily_df['atr'].shift(1)
daily_df.loc[0, 'atr_lagged'] = daily_df.loc[0, 'atr']  # First day uses its own ATR
daily_df['range_size'] = daily_df['atr_lagged'] * ATR_MULTIPLIER

# Step 3: Save results
print("\nStep 3: Saving daily data with ATR...")

# Save to CSV
daily_df.to_csv(DAILY_CSV, index=False)
print(f"Saved to: {DAILY_CSV}")

# Display statistics
print("\nATR Statistics:")
print(f"  Unique ATR values: {daily_df['atr'].nunique()}")
print(f"  ATR range: {daily_df['atr'].min():.2f} to {daily_df['atr'].max():.2f}")
print(f"  Mean ATR: {daily_df['atr'].mean():.2f}")
print(f"  Range size: {daily_df['range_size'].min():.2f} to {daily_df['range_size'].max():.2f}")

# Show sample of ATR progression
print("\nSample of ATR progression (showing how it varies):")
sample_indices = np.linspace(0, len(daily_df)-1, min(10, len(daily_df)), dtype=int)
for idx in sample_indices:
    row = daily_df.iloc[idx]
    print(f"  {row['date'].date()}: ATR={row['atr']:.2f}, Lagged={row['atr_lagged']:.2f}, Range={row['range_size']:.2f}")

print("\n" + "=" * 80)
print("DAILY DATA WITH VARYING ATR COMPLETE!")
print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
print("Ready to create range bars with daily-varying ATR")
print("=" * 80)