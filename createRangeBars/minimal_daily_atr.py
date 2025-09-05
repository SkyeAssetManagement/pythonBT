#!/usr/bin/env python3
"""
Minimal Daily ATR Pipeline
===========================

Bare minimum approach to create daily bars with ATR.
"""

import pyarrow.parquet as pq
import numpy as np
import csv
from datetime import datetime
from collections import defaultdict

print("MINIMAL DAILY ATR PIPELINE")
print("=" * 60)

# Open parquet file
pf = pq.ParquetFile("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
print(f"Total row groups: {pf.num_row_groups}")

# Process each row group
daily_data = defaultdict(lambda: {'open': None, 'high': -float('inf'), 'low': float('inf'), 'close': None, 'count': 0})

for i in range(pf.num_row_groups):
    if i % 50 == 0:
        print(f"Processing row group {i}/{pf.num_row_groups}")
    
    # Read row group
    rg = pf.read_row_group(i)
    
    # Get columns we need
    dates = rg.column('Date').to_pylist()
    closes = rg.column('Close').to_pylist()
    
    # Process each tick
    for date_str, close in zip(dates, closes):
        date = date_str  # Keep as string for simplicity
        
        if daily_data[date]['open'] is None:
            daily_data[date]['open'] = close
        
        daily_data[date]['high'] = max(daily_data[date]['high'], close)
        daily_data[date]['low'] = min(daily_data[date]['low'], close)
        daily_data[date]['close'] = close
        daily_data[date]['count'] += 1

print(f"\nCreated {len(daily_data)} daily bars")

# Convert to sorted list
daily_list = []
for date in sorted(daily_data.keys()):
    daily_list.append({
        'date': date,
        'open': daily_data[date]['open'],
        'high': daily_data[date]['high'],
        'low': daily_data[date]['low'],
        'close': daily_data[date]['close'],
        'volume': daily_data[date]['count']
    })

# Calculate ATR
print("\nCalculating ATR-30...")
atr_period = 30

# Calculate True Range
tr_values = []
for i, day in enumerate(daily_list):
    if i == 0:
        tr = day['high'] - day['low']
    else:
        prev_close = daily_list[i-1]['close']
        tr = max(
            day['high'] - day['low'],
            abs(day['high'] - prev_close),
            abs(day['low'] - prev_close)
        )
    tr_values.append(tr)

# Calculate ATR with Wilder's smoothing
atr_values = []
for i in range(len(tr_values)):
    if i < atr_period:
        # Simple average for first period
        atr = sum(tr_values[:i+1]) / (i + 1)
    elif i == atr_period:
        # First real ATR
        atr = sum(tr_values[:atr_period]) / atr_period
    else:
        # Wilder's smoothing
        prev_atr = atr_values[-1]
        atr = (prev_atr * (atr_period - 1) + tr_values[i]) / atr_period
    atr_values.append(atr)

# Add ATR to daily data
for i, day in enumerate(daily_list):
    day['atr'] = atr_values[i]
    day['atr_lagged'] = atr_values[i-1] if i > 0 else atr_values[0]
    day['range_size'] = day['atr_lagged'] * 0.1

# Save to CSV
output_file = "../parquetData/ES-DIFF-daily-with-atr.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['date', 'open', 'high', 'low', 'close', 'volume', 'atr', 'atr_lagged', 'range_size'])
    writer.writeheader()
    writer.writerows(daily_list)

print(f"\nSaved to: {output_file}")
print(f"ATR range: {min(atr_values):.2f} to {max(atr_values):.2f}")
print(f"Unique ATR values: {len(set(atr_values))}")

# Show sample
print("\nSample ATR progression:")
for i in [0, len(daily_list)//4, len(daily_list)//2, 3*len(daily_list)//4, -1]:
    if 0 <= i < len(daily_list) or i == -1:
        day = daily_list[i]
        print(f"  {day['date']}: ATR={day['atr']:.2f}, Range={day['range_size']:.2f}")

print("\nDONE! Daily bars with varying ATR created.")