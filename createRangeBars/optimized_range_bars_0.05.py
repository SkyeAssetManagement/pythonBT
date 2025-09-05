#!/usr/bin/env python3
"""
Optimized Range Bar Creation - 0.05 Multiplier
===============================================

Creates range bars with 0.05 × daily ATR with optimized memory management.
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
import logging
import gc
from numba import njit
import sys
import platform

# Configuration
ATR_MULTIPLIER = 0.05  # 0.05 multiplier for smaller bars
MIN_BAR_DURATION_MINUTES = 1
BATCH_SIZE = 5  # Even smaller batches for memory efficiency

# Paths
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_ATR_FILE = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
OUTPUT_DIR = Path("../parquetData/range")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# System diagnostics
logger.info("=" * 80)
logger.info("SYSTEM DIAGNOSTICS")
logger.info("=" * 80)
logger.info(f"Python version: {sys.version}")
logger.info(f"Platform: {platform.platform()}")
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"Pandas version: {pd.__version__}")
logger.info("=" * 80)

@njit(cache=True, fastmath=True, nogil=True)
def create_bars_for_day_numba(timestamps, prices, volumes, range_size, min_duration_ns):
    """
    Fast Numba function optimized for memory efficiency.
    """
    n = len(timestamps)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    
    # Pre-allocate conservatively
    max_bars = min(n // 50 + 10, 1000)  # Limit max bars per day
    boundaries = np.empty(max_bars, dtype=np.int64)
    boundary_count = 0
    
    bar_start_time = timestamps[0]
    bar_high = prices[0]
    bar_low = prices[0]
    
    for i in range(1, n):
        # Update high/low
        bar_high = max(bar_high, prices[i])
        bar_low = min(bar_low, prices[i])
        
        # Check bar completion
        if (bar_high - bar_low) >= range_size:
            if (timestamps[i] - bar_start_time) >= min_duration_ns:
                if boundary_count < max_bars:
                    boundaries[boundary_count] = i
                    boundary_count += 1
                    
                    # Reset for next bar
                    if i < n - 1:
                        bar_start_time = timestamps[i]
                        bar_high = prices[i]
                        bar_low = prices[i]
                else:
                    break  # Prevent overflow
    
    return boundaries[:boundary_count]

def process_range_bars():
    """Main processing function with optimized memory management."""
    
    logger.info("=" * 80)
    logger.info(f"RANGE BAR CREATION - {ATR_MULTIPLIER} MULTIPLIER")
    logger.info("=" * 80)
    
    # Force garbage collection before starting
    gc.collect()
    
    # Load daily ATR data
    logger.info("Loading daily ATR data...")
    daily_df = pd.read_csv(DAILY_ATR_FILE)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    logger.info(f"Loaded {len(daily_df)} days of ATR data")
    
    # Recalculate range sizes for 0.05 multiplier
    daily_df['range_size_0.05'] = daily_df['atr_lagged'] * ATR_MULTIPLIER
    logger.info(f"Range size (0.05x): {daily_df['range_size_0.05'].min():.2f} to {daily_df['range_size_0.05'].max():.2f}")
    
    # Create date lookup
    date_to_range = {}
    for _, row in daily_df.iterrows():
        date_to_range[row['date'].date()] = {
            'range_size': row['range_size_0.05'],
            'atr': row['atr_lagged']
        }
    
    # Create output directory
    output_subdir = OUTPUT_DIR / f"ATR30x{ATR_MULTIPLIER}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Open tick file
    logger.info("Opening tick data...")
    parquet_file = pq.ParquetFile(TICK_FILE)
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Total ticks: {total_rows:,}")
    
    all_bars = []
    bars_created = 0
    min_duration_ns = int(MIN_BAR_DURATION_MINUTES * 60 * 1e9)
    
    logger.info("Processing ticks in optimized batches...")
    start_time = time.time()
    
    # Process in very small batches
    for batch_start in range(0, parquet_file.num_row_groups, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, parquet_file.num_row_groups)
        
        # Progress update
        pct_complete = (batch_start / parquet_file.num_row_groups) * 100
        elapsed = (time.time() - start_time) / 60
        if batch_start % 20 == 0:
            logger.info(f"  Batch {batch_start}-{batch_end}/{parquet_file.num_row_groups} "
                       f"({pct_complete:.1f}%), Bars: {bars_created:,}, "
                       f"Time: {elapsed:.1f} min")
        
        # Process batch
        for rg_idx in range(batch_start, batch_end):
            try:
                # Read row group with minimal memory footprint
                table = parquet_file.read_row_group(rg_idx)
                
                # Convert to pandas with minimal columns
                df = table.select(['Date', 'Time', 'Close', 'Volume']).to_pandas()
                
                # Parse datetime efficiently
                df['datetime'] = pd.to_datetime(
                    df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                    format='%Y/%m/%d %H:%M:%S.%f',
                    errors='coerce'
                )
                
                # Filter invalid rows
                df = df[df['datetime'].notna()]
                
                if len(df) == 0:
                    continue
                
                df['date'] = df['datetime'].dt.date
                
                # Process each date
                for date, day_group in df.groupby('date'):
                    if date not in date_to_range:
                        continue
                    
                    range_info = date_to_range[date]
                    range_size = range_info['range_size']
                    atr_value = range_info['atr']
                    
                    if pd.isna(range_size) or range_size <= 0:
                        continue
                    
                    # Sort and convert to numpy
                    day_group = day_group.sort_values('datetime')
                    timestamps = day_group['datetime'].values.astype(np.int64)
                    prices = day_group['Close'].values.astype(np.float64)
                    volumes = day_group['Volume'].values.astype(np.int64)
                    
                    # Find boundaries
                    boundaries = create_bars_for_day_numba(
                        timestamps, prices, volumes, range_size, min_duration_ns
                    )
                    
                    # Create bars
                    prev_idx = 0
                    for boundary_idx in boundaries:
                        if boundary_idx > prev_idx:
                            bar_data = day_group.iloc[prev_idx:boundary_idx + 1]
                            
                            all_bars.append({
                                'timestamp': bar_data.iloc[-1]['datetime'],
                                'open': float(bar_data.iloc[0]['Close']),
                                'high': float(bar_data['Close'].max()),
                                'low': float(bar_data['Close'].min()),
                                'close': float(bar_data.iloc[-1]['Close']),
                                'volume': int(bar_data['Volume'].sum()),
                                'ticks': len(bar_data),
                                'AUX1': float(atr_value),
                                'AUX2': float(ATR_MULTIPLIER)
                            })
                            bars_created += 1
                            prev_idx = boundary_idx + 1
                
                # Aggressive cleanup
                del df, table, day_group
                
            except Exception as e:
                logger.warning(f"Error processing row group {rg_idx}: {e}")
                continue
        
        # Force garbage collection after each batch
        gc.collect()
        
        # Save intermediate results every 100 row groups
        if batch_end % 100 == 0 and all_bars:
            logger.info(f"  Saving intermediate results ({len(all_bars)} bars)...")
            temp_df = pd.DataFrame(all_bars)
            temp_path = output_subdir / f"temp_0.05_bars_{batch_end}.parquet"
            temp_df.to_parquet(temp_path, compression='snappy')
            logger.info(f"  Saved to {temp_path.name}")
    
    # Final processing
    logger.info(f"\nCreated {len(all_bars):,} range bars")
    
    if all_bars:
        final_df = pd.DataFrame(all_bars)
        final_df = final_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate gaps
        final_df['gap'] = 0.0
        if len(final_df) > 1:
            final_df.loc[1:, 'gap'] = final_df['open'].iloc[1:].values - final_df['close'].iloc[:-1].values
        
        # Save DIFF bars
        diff_path = output_subdir / f"ES-DIFF-range-ATR30x{ATR_MULTIPLIER}-amibroker.parquet"
        final_df.to_parquet(diff_path, compression='snappy')
        logger.info(f"Saved DIFF bars: {diff_path}")
        
        # Create NONE bars
        none_df = final_df.copy()
        none_df[['open', 'high', 'low', 'close']] -= 50
        none_path = output_subdir / f"ES-NONE-range-ATR30x{ATR_MULTIPLIER}-amibroker.parquet"
        none_df.to_parquet(none_path, compression='snappy')
        logger.info(f"Saved NONE bars: {none_path}")
        
        # Export to CSV
        logger.info("\nExporting to CSV...")
        csv_dir = Path("../dataRAW/range-ATR30x0.05")
        
        # DIFF CSV
        diff_csv_dir = csv_dir / "ES/diffAdjusted"
        diff_csv_dir.mkdir(parents=True, exist_ok=True)
        
        csv_df = final_df.copy()
        csv_df['Date'] = csv_df['timestamp'].dt.strftime('%Y/%m/%d')
        csv_df['Time'] = csv_df['timestamp'].dt.strftime('%H:%M:%S')
        csv_df['OpenInt'] = csv_df['ticks']
        
        csv_export = csv_df[['Date', 'Time', 'open', 'high', 'low', 'close', 
                            'volume', 'OpenInt', 'AUX1', 'AUX2']].copy()
        csv_export.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 
                              'Volume', 'OpenInt', 'AUX1', 'AUX2']
        
        diff_csv_path = diff_csv_dir / f"ES-DIFF-range-ATR30x{ATR_MULTIPLIER}-dailyATR.csv"
        csv_export.to_csv(diff_csv_path, index=False)
        logger.info(f"Saved DIFF CSV: {diff_csv_path}")
        
        # NONE CSV
        none_csv_dir = csv_dir / "ES/noneAdjusted"
        none_csv_dir.mkdir(parents=True, exist_ok=True)
        
        none_csv = csv_export.copy()
        none_csv[['Open', 'High', 'Low', 'Close']] -= 50
        
        none_csv_path = none_csv_dir / f"ES-NONE-range-ATR30x{ATR_MULTIPLIER}-dailyATR.csv"
        none_csv.to_csv(none_csv_path, index=False)
        logger.info(f"Saved NONE CSV: {none_csv_path}")
        
        # Statistics
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 80)
        
        time_diffs = final_df['timestamp'].diff().dt.total_seconds() / 60
        time_diffs = time_diffs.dropna()
        
        logger.info(f"Total bars: {len(final_df):,}")
        logger.info(f"Average minutes/bar: {time_diffs.mean():.1f}")
        logger.info(f"Median minutes/bar: {time_diffs.median():.1f}")
        logger.info(f"Unique ATR values: {final_df['AUX1'].nunique()}")
        logger.info(f"Processing time: {(time.time() - start_time) / 60:.1f} minutes")
        
        return final_df
    
    return None

if __name__ == "__main__":
    try:
        process_range_bars()
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE - 0.05 MULTIPLIER")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)