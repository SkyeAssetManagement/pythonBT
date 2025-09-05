#!/usr/bin/env python3
"""
VectorBT Pro Range Bar Pipeline with Daily-Updating ATR
========================================================

Production pipeline for creating AmiBroker-style range bars from ES futures tick data
using VectorBT Pro for ATR calculation. 

KEY FEATURE: ATR is calculated daily and applied to the next day's bars.
- Each day uses the ATR value from the previous trading day
- Range size = Previous Day's ATR × Multiplier (e.g., 0.1)
- AUX1 field stores the actual ATR value used for each bar
- AUX2 field stores the multiplier

This creates dynamic range sizing that adapts to market volatility.
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
from datetime import datetime, timedelta
import gc
import logging
from numba import njit
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parameters - Default to ATR-30 with 0.1 multiplier
ATR_PERIODS = [30]  # Default to 30-day ATR only
ATR_MULTIPLIER = 0.1  # Default 0.1 multiplier

# Paths
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_FILE = Path("../parquetData/ES-DIFF-daily.parquet")  # Store in parquetData root
OUTPUT_DIR = Path("../parquetData/range")

# Processing parameters
TICK_BATCH_SIZE = 5_000_000  # Process ticks in 5M batches
SAVE_EVERY_N_BARS = 10_000  # Save to disk every 10k bars
MIN_BAR_DURATION_MINUTES = 1  # Minimum bar duration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vbt_pipeline_daily_atr.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: CREATE OR UPDATE DAILY DATA
# ============================================================================

def create_or_update_daily_data():
    """
    Create or update daily OHLCV data from tick data.
    Only processes new data if daily file exists.
    
    Returns:
        pd.DataFrame: Daily OHLCV data with datetime index
    """
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: CREATE/UPDATE DAILY DATA")
    logger.info("=" * 80)
    
    # Check for existing daily data
    if DAILY_FILE.exists():
        logger.info(f"Loading existing daily data from {DAILY_FILE}")
        daily_df = pd.read_parquet(DAILY_FILE)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.set_index('date')
        last_date = daily_df.index[-1]
        logger.info(f"  Last date in daily data: {last_date}")
    else:
        logger.info("No existing daily data found. Creating from scratch.")
        daily_df = pd.DataFrame()
        last_date = None
    
    # Open tick file to check for new data
    parquet_file = pq.ParquetFile(TICK_FILE)
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Total tick rows: {total_rows:,}")
    
    # Process ticks to create daily bars
    daily_records = []
    row_group_size = 1048576
    
    for rg_idx in range(0, parquet_file.num_row_groups, 10):
        # Read 10 row groups at a time
        end_rg = min(rg_idx + 10, parquet_file.num_row_groups)
        
        tables = []
        for i in range(rg_idx, end_rg):
            tables.append(parquet_file.read_row_group(i))
        
        if not tables:
            continue
            
        combined_table = pa.concat_tables(tables)
        df = combined_table.to_pandas()
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format='%Y/%m/%d %H:%M:%S.%f',
            errors='coerce'
        )
        df = df[df['datetime'].notna()]
        
        # Skip if all data is before last_date
        if last_date is not None and df['datetime'].max() <= last_date:
            continue
        
        # Filter to only new data
        if last_date is not None:
            df = df[df['datetime'] > last_date]
        
        if len(df) == 0:
            continue
        
        # Create daily bars
        df['date'] = df['datetime'].dt.date
        batch = df[df['Close'].notna()]
        
        daily_group = batch.groupby('date').agg({
            'Close': ['first', 'max', 'min', 'last', 'count']
        }).reset_index()
        
        daily_group.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        daily_records.append(daily_group)
        
        # Clean up
        del df, batch
        gc.collect()
    
    # Combine all daily records
    if daily_records:
        new_daily = pd.concat(daily_records, ignore_index=True)
        
        # Group by date (in case same date appears in multiple row groups)
        new_daily = new_daily.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        # Append to existing or create new
        if not daily_df.empty:
            daily_df = pd.concat([daily_df, new_daily.set_index('date')], axis=0)
        else:
            daily_df = new_daily.set_index('date')
        
        # Save updated daily data
        daily_df.reset_index().to_parquet(DAILY_FILE)
        logger.info(f"  Saved daily data: {len(daily_df)} days")
    
    return daily_df

# ============================================================================
# STEP 2: CALCULATE DAILY ATR VALUES
# ============================================================================

def calculate_daily_atr_values(daily_df, atr_period=30):
    """
    Calculate ATR values for each day in the dataset.
    
    Args:
        daily_df: DataFrame with daily OHLC data
        atr_period: Period for ATR calculation (default 30)
        
    Returns:
        pd.Series: Daily ATR values indexed by date
    """
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CALCULATE DAILY ATR VALUES")
    logger.info("=" * 80)
    
    logger.info(f"Calculating ATR-{atr_period} for each day...")
    logger.info(f"Daily data shape: {len(daily_df)} days")
    logger.info(f"Date range: {daily_df.index[0]} to {daily_df.index[-1]}")
    
    # Calculate True Range
    high = daily_df['high'].values
    low = daily_df['low'].values
    close = daily_df['close'].values
    
    # Calculate TR components (need previous close)
    tr_hl = high - low  # High - Low
    tr_hc = np.abs(high - np.roll(close, 1))  # |High - Previous Close|
    tr_lc = np.abs(low - np.roll(close, 1))   # |Low - Previous Close|
    
    # True Range is the maximum of the three
    true_range = np.maximum(tr_hl, np.maximum(tr_hc, tr_lc))
    
    # Set first value to high-low (no previous close)
    true_range[0] = high[0] - low[0]
    
    # Calculate ATR using Wilder's smoothing (exponential moving average)
    atr_values = np.zeros(len(daily_df))
    
    # Initialize with simple average of first n periods
    if len(true_range) >= atr_period:
        atr_values[atr_period-1] = np.mean(true_range[:atr_period])
        
        # Calculate ATR using Wilder's smoothing
        for i in range(atr_period, len(daily_df)):
            atr_values[i] = (atr_values[i-1] * (atr_period - 1) + true_range[i]) / atr_period
    else:
        # If not enough data, use simple moving average
        for i in range(len(daily_df)):
            if i == 0:
                atr_values[i] = true_range[i]
            else:
                window_start = max(0, i - atr_period + 1)
                atr_values[i] = np.mean(true_range[window_start:i+1])
    
    # Create Series with date index
    atr_series = pd.Series(atr_values, index=daily_df.index, name=f'ATR_{atr_period}')
    
    # Log statistics
    logger.info(f"  ATR range: {atr_series.min():.3f} to {atr_series.max():.3f}")
    logger.info(f"  Mean ATR: {atr_series.mean():.3f}")
    logger.info(f"  Latest ATR: {atr_series.iloc[-1]:.3f}")
    
    # Show how ATR changes over time (sample)
    sample_dates = [0, len(atr_series)//4, len(atr_series)//2, 3*len(atr_series)//4, -1]
    logger.info("  Sample ATR values:")
    for idx in sample_dates:
        if 0 <= idx < len(atr_series) or idx == -1:
            date = atr_series.index[idx]
            value = atr_series.iloc[idx]
            logger.info(f"    {date}: {value:.3f}")
    
    return atr_series

# ============================================================================
# STEP 3: NUMBA FUNCTIONS FOR RANGE BAR CREATION
# ============================================================================

@njit(cache=True, fastmath=True)
def find_bar_boundaries_with_daily_atr(timestamps, prices, daily_atr_values, daily_boundaries, 
                                       atr_multiplier, min_duration_ns):
    """
    Fast boundary detection with daily-changing ATR values.
    
    Args:
        timestamps: Array of tick timestamps (int64)
        prices: Array of tick prices (float64)
        daily_atr_values: Array of ATR values for each day
        daily_boundaries: Indices where each day starts in the tick data
        atr_multiplier: Multiplier for ATR (e.g., 0.1)
        min_duration_ns: Minimum bar duration in nanoseconds
        
    Returns:
        boundaries: Array of bar boundary indices
        atr_used: Array of ATR values used for each bar
    """
    n = len(timestamps)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    
    max_bars = n // 10
    boundaries = np.empty(max_bars, dtype=np.int64)
    atr_used = np.empty(max_bars, dtype=np.float64)
    boundary_count = 0
    
    # Find which day we're in
    current_day_idx = 0
    for i in range(len(daily_boundaries) - 1):
        if daily_boundaries[i+1] > 0:
            current_day_idx = i
            break
    
    # Current ATR and range size
    current_atr = daily_atr_values[current_day_idx] if current_day_idx < len(daily_atr_values) else daily_atr_values[-1]
    range_size = current_atr * atr_multiplier
    
    bar_start_time = timestamps[0]
    bar_high = prices[0]
    bar_low = prices[0]
    
    for i in range(1, n):
        # Check if we've crossed into a new day
        for day_idx in range(current_day_idx + 1, len(daily_boundaries)):
            if i >= daily_boundaries[day_idx] and daily_boundaries[day_idx] > 0:
                current_day_idx = day_idx
                # Update ATR and range size for new day
                if current_day_idx < len(daily_atr_values):
                    current_atr = daily_atr_values[current_day_idx]
                    range_size = current_atr * atr_multiplier
                break
        
        # Update high/low
        if prices[i] > bar_high:
            bar_high = prices[i]
        if prices[i] < bar_low:
            bar_low = prices[i]
        
        # Check bar completion
        if (bar_high - bar_low) >= range_size:
            if (timestamps[i] - bar_start_time) >= min_duration_ns:
                boundaries[boundary_count] = i
                atr_used[boundary_count] = current_atr
                boundary_count += 1
                
                # Reset
                bar_start_time = timestamps[i]
                bar_high = prices[i]
                bar_low = prices[i]
    
    return boundaries[:boundary_count], atr_used[:boundary_count]

# ============================================================================
# STEP 4: CREATE RANGE BARS WITH DAILY ATR
# ============================================================================

def create_range_bars_with_daily_atr(atr_period, daily_atr_series):
    """
    Create range bars using daily-updating ATR values.
    
    Args:
        atr_period: ATR period (e.g., 30)
        daily_atr_series: Series of daily ATR values
        
    Returns:
        dict: Statistics about the created bars
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Creating range bars for ATR-{atr_period} with daily updates")
    logger.info(f"{'='*80}")
    
    # Create output directory
    output_subdir = OUTPUT_DIR / f"ATR{atr_period}x{ATR_MULTIPLIER}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Output paths
    diff_path = output_subdir / f"ES-DIFF-range-ATR{atr_period}x{ATR_MULTIPLIER}-amibroker.parquet"
    none_path = output_subdir / f"ES-NONE-range-ATR{atr_period}x{ATR_MULTIPLIER}-amibroker.parquet"
    
    logger.info(f"Output: {diff_path.name}")
    
    # Open tick file
    parquet_file = pq.ParquetFile(TICK_FILE)
    total_rows = parquet_file.metadata.num_rows
    
    # First pass: identify day boundaries in tick data
    logger.info("Identifying day boundaries in tick data...")
    day_starts = {}  # date -> tick index
    tick_idx = 0
    
    for rg_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(rg_idx)
        df = table.to_pandas()
        
        df['datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format='%Y/%m/%d %H:%M:%S.%f',
            errors='coerce'
        )
        df = df[df['datetime'].notna()]
        
        if len(df) == 0:
            continue
        
        # Identify unique dates in this batch
        df['date'] = df['datetime'].dt.date
        unique_dates = df['date'].unique()
        
        for date in unique_dates:
            if date not in day_starts:
                # Find first tick of this date
                first_idx = df[df['date'] == date].index[0]
                day_starts[date] = tick_idx + first_idx
        
        tick_idx += len(df)
        del df, table
        gc.collect()
    
    # Convert to arrays for Numba
    sorted_dates = sorted(day_starts.keys())
    daily_boundaries = np.array([day_starts[date] for date in sorted_dates], dtype=np.int64)
    
    # Get ATR values for each day (using previous day's ATR)
    daily_atr_values = []
    for date in sorted_dates:
        # Use previous trading day's ATR
        date_pd = pd.Timestamp(date)
        if date_pd in daily_atr_series.index:
            # Find previous trading day
            idx = daily_atr_series.index.get_loc(date_pd)
            if idx > 0:
                prev_atr = daily_atr_series.iloc[idx - 1]
            else:
                prev_atr = daily_atr_series.iloc[0]
        else:
            # Use last known ATR
            prev_atr = daily_atr_series.iloc[-1]
        daily_atr_values.append(prev_atr)
    
    daily_atr_values = np.array(daily_atr_values, dtype=np.float64)
    
    logger.info(f"Found {len(sorted_dates)} trading days in tick data")
    logger.info(f"ATR will update daily, range from {daily_atr_values.min():.3f} to {daily_atr_values.max():.3f}")
    
    # Process ticks in batches
    min_duration_ns = int(MIN_BAR_DURATION_MINUTES * 60 * 1e9)
    
    # Create saver for incremental saves
    class IncrementalBarSaver:
        def __init__(self, output_path):
            self.output_path = output_path
            self.parts = []
            self.current_batch = []
            self.total_bars = 0
            self.part_count = 0
        
        def add_bars(self, bars_df):
            self.current_batch.append(bars_df)
            batch_size = sum(len(b) for b in self.current_batch)
            
            if batch_size >= SAVE_EVERY_N_BARS:
                self._save_batch()
        
        def _save_batch(self):
            if not self.current_batch:
                return
            
            combined = pd.concat(self.current_batch, ignore_index=True)
            self.current_batch = []
            
            part_file = self.output_path.parent / f"temp_{self.output_path.stem}_part{self.part_count}.parquet"
            combined.to_parquet(part_file, compression='snappy')
            
            self.parts.append(part_file)
            self.part_count += 1
            self.total_bars += len(combined)
            
            logger.info(f"    Saved part {self.part_count}: {len(combined):,} bars (total: {self.total_bars:,})")
            
            del combined
            gc.collect()
        
        def finalize(self):
            if self.current_batch:
                self._save_batch()
            
            if not self.parts:
                return pd.DataFrame()
            
            # Combine all parts
            logger.info("  Combining all parts...")
            all_bars = []
            for part_file in self.parts:
                all_bars.append(pd.read_parquet(part_file))
                part_file.unlink()
            
            final_df = pd.concat(all_bars, ignore_index=True)
            
            # Calculate gaps
            final_df['gap'] = 0.0
            if len(final_df) > 1:
                gaps = final_df['open'].iloc[1:].values - final_df['close'].iloc[:-1].values
                final_df.loc[1:, 'gap'] = gaps
            
            # Save final file
            final_df.to_parquet(self.output_path, compression='snappy')
            logger.info(f"  Saved final file: {len(final_df):,} bars")
            
            return final_df
    
    saver = IncrementalBarSaver(diff_path)
    
    # Process all tick data
    logger.info("Processing ticks to create range bars...")
    start_time = time.time()
    
    # Read and process all ticks at once (or in large batches)
    all_timestamps = []
    all_prices = []
    all_volumes = []
    all_datetimes = []
    
    for rg_idx in range(parquet_file.num_row_groups):
        if rg_idx % 10 == 0:
            logger.info(f"  Reading row group {rg_idx}/{parquet_file.num_row_groups}")
        
        table = parquet_file.read_row_group(rg_idx)
        df = table.to_pandas()
        
        df['datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format='%Y/%m/%d %H:%M:%S.%f',
            errors='coerce'
        )
        df = df[df['datetime'].notna()]
        
        if len(df) > 0:
            all_timestamps.append(df['datetime'].values.astype(np.int64))
            all_prices.append(df['Close'].values.astype(np.float64))
            all_volumes.append(df['Volume'].values.astype(np.int64))
            all_datetimes.append(df['datetime'].values)
        
        del df, table
        gc.collect()
    
    # Concatenate all data
    timestamps = np.concatenate(all_timestamps)
    prices = np.concatenate(all_prices)
    volumes = np.concatenate(all_volumes)
    datetimes = np.concatenate(all_datetimes)
    
    logger.info(f"  Loaded {len(timestamps):,} ticks")
    
    # Find boundaries with daily ATR
    logger.info("  Finding bar boundaries with daily ATR updates...")
    boundaries, atr_used = find_bar_boundaries_with_daily_atr(
        timestamps, prices, daily_atr_values, daily_boundaries,
        ATR_MULTIPLIER, min_duration_ns
    )
    
    logger.info(f"  Found {len(boundaries):,} range bars")
    
    # Create bars DataFrame
    if len(boundaries) > 0:
        bars_list = []
        prev_end = 0
        
        for i, boundary in enumerate(boundaries):
            if boundary > prev_end:
                bar_timestamps = datetimes[prev_end:boundary]
                bar_prices = prices[prev_end:boundary]
                bar_volumes = volumes[prev_end:boundary]
                
                if len(bar_prices) > 0:
                    bars_list.append({
                        'timestamp': bar_timestamps[-1],
                        'open': float(bar_prices[0]),
                        'high': float(bar_prices.max()),
                        'low': float(bar_prices.min()),
                        'close': float(bar_prices[-1]),
                        'volume': int(bar_volumes.sum()),
                        'ticks': len(bar_prices),
                        'AUX1': float(atr_used[i]),  # Actual ATR value used
                        'AUX2': float(ATR_MULTIPLIER)  # Multiplier
                    })
                    prev_end = boundary
        
        if bars_list:
            bars_df = pd.DataFrame(bars_list)
            saver.add_bars(bars_df)
    
    # Finalize
    final_df = saver.finalize()
    
    # Create NONE bars
    none_df = final_df.copy()
    none_df[['open', 'high', 'low', 'close']] -= 50
    none_df.to_parquet(none_path, compression='snappy')
    logger.info(f"  Created NONE bars: {none_path.name}")
    
    elapsed = time.time() - start_time
    
    # Calculate statistics
    if len(final_df) > 0:
        time_diffs = final_df['timestamp'].diff().dt.total_seconds() / 60
        time_diffs = time_diffs.dropna()
        
        # Show ATR distribution in bars
        atr_stats = final_df['AUX1'].describe()
        logger.info("\n  ATR Distribution in Range Bars:")
        logger.info(f"    Min: {atr_stats['min']:.3f}")
        logger.info(f"    25%: {atr_stats['25%']:.3f}")
        logger.info(f"    50%: {atr_stats['50%']:.3f}")
        logger.info(f"    75%: {atr_stats['75%']:.3f}")
        logger.info(f"    Max: {atr_stats['max']:.3f}")
        
        return {
            'period': atr_period,
            'bars': len(final_df),
            'processing_time': elapsed / 60,
            'avg_mins_per_bar': time_diffs.mean(),
            'median_mins_per_bar': time_diffs.median(),
            'atr_min': atr_stats['min'],
            'atr_max': atr_stats['max'],
            'atr_mean': atr_stats['mean']
        }
    
    return {
        'period': atr_period,
        'bars': 0,
        'processing_time': elapsed / 60
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution"""
    
    logger.info("=" * 100)
    logger.info("VECTORBT PRO RANGE BAR PIPELINE WITH DAILY ATR")
    logger.info("=" * 100)
    logger.info(f"ATR periods: {ATR_PERIODS}")
    logger.info(f"Multiplier: {ATR_MULTIPLIER}")
    logger.info("ATR updates daily - each day uses previous day's ATR value")
    logger.info("=" * 100)
    
    # Step 1: Create/update daily data
    daily_df = create_or_update_daily_data()
    
    # Step 2: Calculate daily ATR values for each period
    results = []
    
    for period in ATR_PERIODS:
        try:
            # Calculate daily ATR
            daily_atr = calculate_daily_atr_values(daily_df, period)
            
            # Warm up Numba JIT
            if period == ATR_PERIODS[0]:
                logger.info("\nWarming up Numba JIT...")
                dummy = np.arange(100, dtype=np.int64)
                dummy_atr = np.ones(10, dtype=np.float64) * 10.0
                dummy_boundaries = np.array([0, 50], dtype=np.int64)
                _ = find_bar_boundaries_with_daily_atr(
                    dummy, dummy.astype(np.float64), dummy_atr, 
                    dummy_boundaries, 0.1, 1000000000
                )
                logger.info("JIT ready!")
            
            # Create range bars with daily ATR
            result = create_range_bars_with_daily_atr(period, daily_atr)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed for ATR-{period}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Report results
    logger.info("\n" + "=" * 100)
    logger.info("FINAL RESULTS")
    logger.info("=" * 100)
    
    print(f"\n{'ATR':<10} {'Bars':<12} {'Proc Time':<12} {'Avg min/bar':<12} {'Med min/bar':<12} {'ATR Range'}")
    print("-" * 90)
    
    for result in results:
        if result['bars'] > 0:
            atr_range = f"{result['atr_min']:.1f}-{result['atr_max']:.1f}"
            print(f"ATR-{result['period']:<6} {result['bars']:<12,} "
                  f"{result['processing_time']:<12.1f} {result['avg_mins_per_bar']:<12.1f} "
                  f"{result['median_mins_per_bar']:<12.1f} {atr_range}")
    
    logger.info("\n" + "=" * 100)
    logger.info("PIPELINE COMPLETE - ATR UPDATES DAILY")
    logger.info("=" * 100)

if __name__ == "__main__":
    main()