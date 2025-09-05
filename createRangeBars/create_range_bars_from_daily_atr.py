#!/usr/bin/env python3
"""
Create Range Bars Using Daily ATR Values
=========================================

Uses the daily ATR data (created by vbt_proper_pipeline.py) to build range bars
where each day's bars use the previous day's ATR value.
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
from datetime import datetime, timedelta

# Configuration
ATR_MULTIPLIER = 0.1
MIN_BAR_DURATION_MINUTES = 1

# Paths
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_ATR_FILE = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
OUTPUT_DIR = Path("../parquetData/range")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@njit(cache=True, fastmath=True)
def create_bars_for_day_numba(timestamps, prices, volumes, range_size, min_duration_ns):
    """
    Fast Numba function to create range bars for a single day.
    
    Returns:
        boundaries: Array of bar end indices
    """
    n = len(timestamps)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    
    max_bars = n // 10 + 1
    boundaries = np.empty(max_bars, dtype=np.int64)
    boundary_count = 0
    
    bar_start_time = timestamps[0]
    bar_high = prices[0]
    bar_low = prices[0]
    bar_start_idx = 0
    
    for i in range(1, n):
        # Update high/low
        if prices[i] > bar_high:
            bar_high = prices[i]
        if prices[i] < bar_low:
            bar_low = prices[i]
        
        # Check bar completion
        if (bar_high - bar_low) >= range_size:
            if (timestamps[i] - bar_start_time) >= min_duration_ns:
                boundaries[boundary_count] = i
                boundary_count += 1
                
                # Reset for next bar
                if i < n - 1:
                    bar_start_time = timestamps[i]
                    bar_high = prices[i]
                    bar_low = prices[i]
                    bar_start_idx = i
    
    return boundaries[:boundary_count]

def create_range_bars_with_daily_atr():
    """Create range bars using daily ATR values."""
    
    logger.info("=" * 80)
    logger.info("CREATE RANGE BARS USING DAILY ATR")
    logger.info("=" * 80)
    
    # Load daily ATR data
    logger.info("Loading daily ATR data...")
    daily_df = pd.read_csv(DAILY_ATR_FILE)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    logger.info(f"Loaded {len(daily_df)} days of ATR data")
    logger.info(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    logger.info(f"Range size: {daily_df['range_size'].min():.2f} to {daily_df['range_size'].max():.2f}")
    
    # Create date lookup for ATR values
    date_to_range = {}
    for _, row in daily_df.iterrows():
        date_to_range[row['date'].date()] = {
            'range_size': row['range_size'],
            'atr': row['atr_lagged']
        }
    
    # Create output directory
    atr_period = 30  # We know it's ATR-30
    output_subdir = OUTPUT_DIR / f"ATR{atr_period}x{ATR_MULTIPLIER}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Output paths
    diff_path = output_subdir / f"ES-DIFF-range-ATR{atr_period}x{ATR_MULTIPLIER}-amibroker.parquet"
    none_path = output_subdir / f"ES-NONE-range-ATR{atr_period}x{ATR_MULTIPLIER}-amibroker.parquet"
    
    logger.info(f"\nProcessing tick data to create range bars...")
    logger.info(f"Output: {diff_path.name}")
    
    # Process tick data day by day
    parquet_file = pq.ParquetFile(TICK_FILE)
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Total ticks: {total_rows:,}")
    
    all_bars = []
    min_duration_ns = int(MIN_BAR_DURATION_MINUTES * 60 * 1e9)
    
    # Process tick data in chunks
    logger.info("Processing ticks by day...")
    current_day_data = []
    current_date = None
    bars_created = 0
    
    # Process in smaller batches to avoid memory issues
    BATCH_SIZE = 10  # Process 10 row groups at a time to avoid segfaults
    
    for batch_start in range(0, parquet_file.num_row_groups, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, parquet_file.num_row_groups)
        
        logger.info(f"  Processing row groups {batch_start}-{batch_end}/{parquet_file.num_row_groups}, bars created: {bars_created:,}")
        
        # Process this batch
        for rg_idx in range(batch_start, batch_end):
            # Read row group
            table = parquet_file.read_row_group(rg_idx)
            df = table.to_pandas()
            
            # Parse datetime
            df['datetime'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                format='%Y/%m/%d %H:%M:%S.%f',
                errors='coerce'
            )
            df = df[df['datetime'].notna()]
            
            if len(df) == 0:
                continue
            
            df['date'] = df['datetime'].dt.date
            
            # Process each date in this chunk
            for date, day_group in df.groupby('date'):
                # Check if we have ATR for this date
                if date not in date_to_range:
                    continue
                
                range_info = date_to_range[date]
                range_size = range_info['range_size']
                atr_value = range_info['atr']
                
                if pd.isna(range_size) or range_size <= 0:
                    continue
                
                # Sort by datetime
                day_group = day_group.sort_values('datetime')
                
                # Convert to numpy arrays for Numba
                timestamps = day_group['datetime'].values.astype(np.int64)
                prices = day_group['Close'].values.astype(np.float64)
                volumes = day_group['Volume'].values.astype(np.int64)
                
                # Find bar boundaries using Numba
                boundaries = create_bars_for_day_numba(
                    timestamps, prices, volumes, range_size, min_duration_ns
                )
                
                # Create bars from boundaries
                prev_idx = 0
                for boundary_idx in boundaries:
                    if boundary_idx > prev_idx:
                        bar_data = day_group.iloc[prev_idx:boundary_idx + 1]
                        
                        bar = {
                            'timestamp': bar_data.iloc[-1]['datetime'],
                            'open': float(bar_data.iloc[0]['Close']),
                            'high': float(bar_data['Close'].max()),
                            'low': float(bar_data['Close'].min()),
                            'close': float(bar_data.iloc[-1]['Close']),
                            'volume': int(bar_data['Volume'].sum()),
                            'ticks': len(bar_data),
                            'AUX1': float(atr_value),
                            'AUX2': float(ATR_MULTIPLIER)
                        }
                        all_bars.append(bar)
                        bars_created += 1
                        prev_idx = boundary_idx + 1
            
            # Clean up after each row group
            del df, table
            gc.collect()
        
        # Extra cleanup after each batch
        gc.collect()
        
        # Save progress every 100 row groups
        if batch_end % 100 == 0 and all_bars:
            logger.info(f"  Saving intermediate results ({len(all_bars)} bars)...")
            temp_df = pd.DataFrame(all_bars)
            temp_df = temp_df.sort_values('timestamp').reset_index(drop=True)
            temp_path = output_subdir / f"temp_bars_{batch_end}.parquet"
            temp_df.to_parquet(temp_path, compression='snappy')
            logger.info(f"  Saved to {temp_path.name}")
    
    logger.info(f"\nCreated {len(all_bars):,} range bars")
    
    # Convert to DataFrame
    if all_bars:
        final_df = pd.DataFrame(all_bars)
        final_df = final_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate gaps
        final_df['gap'] = 0.0
        if len(final_df) > 1:
            gaps = final_df['open'].iloc[1:].values - final_df['close'].iloc[:-1].values
            final_df.loc[1:, 'gap'] = gaps
        
        # Save DIFF bars
        final_df.to_parquet(diff_path, compression='snappy')
        logger.info(f"Saved DIFF bars to: {diff_path}")
        
        # Create and save NONE bars
        none_df = final_df.copy()
        none_df[['open', 'high', 'low', 'close']] -= 50
        none_df.to_parquet(none_path, compression='snappy')
        logger.info(f"Saved NONE bars to: {none_path}")
        
        # Statistics
        atr_stats = final_df['AUX1'].describe()
        logger.info("\nATR Distribution in Range Bars:")
        logger.info(f"  Unique ATR values: {final_df['AUX1'].nunique()}")
        logger.info(f"  Min: {atr_stats['min']:.2f}")
        logger.info(f"  25%: {atr_stats['25%']:.2f}")
        logger.info(f"  50%: {atr_stats['50%']:.2f}")
        logger.info(f"  75%: {atr_stats['75%']:.2f}")
        logger.info(f"  Max: {atr_stats['max']:.2f}")
        
        # Time statistics
        time_diffs = final_df['timestamp'].diff().dt.total_seconds() / 60
        time_diffs = time_diffs.dropna()
        logger.info(f"\nTiming Statistics:")
        logger.info(f"  Avg minutes per bar: {time_diffs.mean():.1f}")
        logger.info(f"  Median minutes per bar: {time_diffs.median():.1f}")
        
        return final_df
    
    return pd.DataFrame()

def main():
    """Main execution."""
    
    logger.info("=" * 100)
    logger.info("RANGE BAR CREATION WITH DAILY ATR")
    logger.info("=" * 100)
    logger.info(f"Configuration:")
    logger.info(f"  ATR Multiplier: {ATR_MULTIPLIER}")
    logger.info(f"  Min Bar Duration: {MIN_BAR_DURATION_MINUTES} minutes")
    logger.info("  Each day uses previous day's ATR (lagged)")
    logger.info("=" * 100)
    
    start_time = time.time()
    
    try:
        # Check if daily ATR data exists
        if not DAILY_ATR_FILE.exists():
            logger.error(f"Daily ATR file not found: {DAILY_ATR_FILE}")
            logger.error("Please run vbt_proper_pipeline.py first to create daily ATR data")
            return
        
        # Create range bars
        range_bars = create_range_bars_with_daily_atr()
        
        elapsed = (time.time() - start_time) / 60
        
        logger.info("\n" + "=" * 100)
        logger.info("RANGE BAR CREATION COMPLETE!")
        logger.info("=" * 100)
        logger.info(f"Processing time: {elapsed:.1f} minutes")
        logger.info(f"Created {len(range_bars):,} range bars with daily-varying ATR")
        logger.info("=" * 100)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()