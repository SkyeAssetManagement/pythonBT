#!/usr/bin/env python3
"""
Production Range Bar Creation Pipeline
=======================================

This script creates AmiBroker-style range bars from ES futures tick data.

Features:
- Creates/updates daily OHLCV data from 600M+ ticks
- Calculates dynamic ATR values for range sizing
- Generates range bars with gaps (no synthetic prices)
- Adds AUX1 (ATR value) and AUX2 (multiplier) fields for analysis
- Processes incrementally to manage memory (50GB target)
- Outputs both DIFF and NONE (offset by -50) versions

Default Configuration:
- ATR Period: 30 days
- Multiplier: 0.1 (creates bars averaging ~18 min)
- Min Duration: 1 second (AmiBroker requirement)

Output Structure:
- parquetData/range/ATR30x0.1/
  - ES-DIFF-range-ATR30x0.1-amibroker.parquet
  - ES-NONE-range-ATR30x0.1-amibroker.parquet

Author: SkyeAM
Date: August 2024
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import time
import logging
import gc
from numba import njit
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_FILE = Path("../parquetData/ES-DIFF-daily.parquet")  # Store in parquetData root
OUTPUT_DIR = Path("../parquetData/range")

# Parameters - Default to ATR-30 with 0.1 multiplier
ATR_PERIODS = [30]  # Default to 30-day ATR only
ATR_MULTIPLIER = 0.1  # Default 0.1 multiplier (avg ~18 min/bar)
BATCH_SIZE = 50_000_000  # Process 50M ticks at a time
SAVE_EVERY_N_BARS = 500_000  # Save every 500k bars to avoid memory issues
MIN_DURATION_SECONDS = 1  # AmiBroker minimum bar duration

# ============================================================================
# STEP 1: CREATE/UPDATE DAILY DATA
# ============================================================================

def create_or_update_daily_data():
    """
    Create or update daily OHLCV from tick data.
    
    Returns:
        pd.DataFrame: Daily OHLCV data with index as date
        
    Note:
        - Checks for existing daily file and only updates if needed
        - Processes 572 row groups (~1M ticks each) to build daily bars
        - Takes ~13 minutes for full rebuild from 600M ticks
    """
    
    logger.info("=" * 80)
    logger.info("STEP 1: CREATE/UPDATE DAILY DATA")
    logger.info("=" * 80)
    
    # Check if daily file exists
    if DAILY_FILE.exists():
        logger.info("Loading existing daily data...")
        daily_df = pd.read_parquet(DAILY_FILE)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df.set_index('date', inplace=True)
        
        last_date = daily_df.index[-1]
        logger.info(f"  Last date in daily data: {last_date.date()}")
        
        # Check tick data for new dates
        parquet_file = pq.ParquetFile(TICK_FILE)
        
        # Read last row group to check latest date
        last_rg = parquet_file.read_row_group(parquet_file.num_row_groups - 1)
        last_df = last_rg.to_pandas()
        last_df['datetime'] = pd.to_datetime(last_df['Date'].astype(str) + ' ' + last_df['Time'].astype(str))
        tick_last_date = last_df['datetime'].max()
        
        logger.info(f"  Last date in tick data: {tick_last_date.date()}")
        
        if tick_last_date.date() <= last_date.date():
            logger.info("  Daily data is up to date!")
            return daily_df
        
        logger.info(f"  Need to update from {last_date.date()} to {tick_last_date.date()}")
        start_from = last_date
        
    else:
        logger.info("No existing daily data, creating from scratch...")
        daily_df = pd.DataFrame()
        start_from = pd.Timestamp('1900-01-01')  # Process all data
    
    # Read tick data and create daily OHLCV
    logger.info("Processing tick data to daily OHLCV...")
    
    parquet_file = pq.ParquetFile(TICK_FILE)
    daily_records = []
    
    for rg_idx in range(parquet_file.num_row_groups):
        if rg_idx % 50 == 0:
            logger.info(f"  Processing row group {rg_idx}/{parquet_file.num_row_groups}")
        
        # Read row group
        batch = parquet_file.read_row_group(rg_idx)
        df = batch.to_pandas()
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format='%Y/%m/%d %H:%M:%S.%f',
            errors='coerce'
        )
        
        # Filter for dates after last_date
        df = df[df['datetime'] > start_from]
        
        if len(df) == 0:
            continue
        
        # Create daily OHLCV
        df['date'] = df['datetime'].dt.date
        
        daily_group = df.groupby('date').agg({
            'Close': ['first', 'max', 'min', 'last'],
            'Volume': 'sum'
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
# STEP 2: CALCULATE ATR VALUES USING VECTORBT PRO
# ============================================================================

def calculate_atr_values(daily_df):
    """
    Calculate ATR (Average True Range) values from daily OHLCV data.
    
    Args:
        daily_df: DataFrame with daily OHLC data
        
    Returns:
        dict: ATR values for each period with 'last', 'avg_20d', and 'use_value'
        
    Note:
        - Uses Wilder's smoothing method (exponential moving average)
        - Returns 20-day average as 'use_value' for stability
    """
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CALCULATE ATR VALUES WITH VECTORBT PRO")
    logger.info("=" * 80)
    
    logger.info(f"Daily data shape: {len(daily_df)} days")
    logger.info(f"Date range: {daily_df.index[0]} to {daily_df.index[-1]}")
    
    # Calculate ATR for each period
    atr_values = {}
    
    for period in ATR_PERIODS:
        # Calculate ATR using pandas/numpy directly
        high = daily_df['high'].values
        low = daily_df['low'].values
        close = daily_df['close'].values
        
        # Calculate True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Initialize ATR with first TR value
        atr = np.zeros(len(daily_df))
        atr[period] = np.mean(tr[:period])
        
        # Calculate ATR using exponential moving average
        for i in range(period + 1, len(daily_df)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
        
        # Get the last (most recent) ATR value
        last_atr = atr[-1]
        
        # Also get average of last 20 days for stability
        avg_atr = np.mean(atr[-20:])
        
        atr_values[period] = {
            'last': float(last_atr),
            'avg_20d': float(avg_atr),
            'use_value': float(avg_atr)  # Use 20-day average for stability
        }
        
        logger.info(f"ATR-{period}:")
        logger.info(f"  Last value: {last_atr:.3f}")
        logger.info(f"  20-day avg: {avg_atr:.3f}")
        logger.info(f"  Using: {avg_atr:.3f}")
    
    return atr_values

# ============================================================================
# STEP 3: NUMBA FUNCTIONS FOR RANGE BAR CREATION
# ============================================================================

@njit(cache=True, fastmath=True)
def find_bar_boundaries_fast(timestamps, prices, range_size, min_duration_ns):
    """
    Fast boundary detection with Numba JIT compilation.
    
    AmiBroker Logic:
    - Bar completes when high-low >= range_size AND duration >= min_duration
    - Next bar starts at completion point (creates gaps)
    - No synthetic prices between bars
    """
    n = len(timestamps)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    
    max_bars = n // 10
    boundaries = np.empty(max_bars, dtype=np.int64)
    boundary_count = 0
    
    bar_start_time = timestamps[0]
    bar_high = prices[0]
    bar_low = prices[0]
    
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
                
                # Reset
                bar_start_time = timestamps[i]
                bar_high = prices[i]
                bar_low = prices[i]
    
    return boundaries[:boundary_count]

# ============================================================================
# STEP 4: CREATE RANGE BARS WITH INCREMENTAL SAVES
# ============================================================================

class IncrementalBarSaver:
    """Save bars incrementally to avoid memory issues"""
    
    def __init__(self, output_path):
        self.output_path = output_path
        self.parts = []
        self.current_batch = []
        self.total_bars = 0
        self.part_count = 0
    
    def add_bars(self, bars_df):
        """Add bars and save if threshold reached"""
        self.current_batch.append(bars_df)
        batch_size = sum(len(b) for b in self.current_batch)
        
        if batch_size >= SAVE_EVERY_N_BARS:
            self._save_batch()
    
    def _save_batch(self):
        """Save current batch to disk"""
        if not self.current_batch:
            return
        
        combined = pd.concat(self.current_batch, ignore_index=True)
        self.current_batch = []
        
        # Save part
        part_file = self.output_path.parent / f"temp_{self.output_path.stem}_part{self.part_count}.parquet"
        combined.to_parquet(part_file, compression='snappy')
        
        self.parts.append(part_file)
        self.part_count += 1
        self.total_bars += len(combined)
        
        logger.info(f"    Saved part {self.part_count}: {len(combined):,} bars (total: {self.total_bars:,})")
        
        del combined
        gc.collect()
    
    def finalize(self):
        """Combine all parts into final file"""
        # Save any remaining
        if self.current_batch:
            self._save_batch()
        
        if not self.parts:
            return pd.DataFrame()
        
        # Combine parts
        logger.info(f"  Combining {len(self.parts)} parts...")
        all_parts = []
        
        for part_file in self.parts:
            all_parts.append(pd.read_parquet(part_file))
            part_file.unlink()  # Delete temp file
        
        final_df = pd.concat(all_parts, ignore_index=True)
        
        # Add gaps
        final_df['gap'] = final_df['open'].diff().abs().fillna(0)
        
        # Save final
        final_df.to_parquet(self.output_path, compression='snappy')
        logger.info(f"  Final save: {len(final_df):,} bars")
        
        return final_df

def create_range_bars_for_atr(atr_period, atr_value):
    """Create range bars for a specific ATR period"""
    
    range_size = atr_value * ATR_MULTIPLIER
    
    logger.info(f"\n" + "-" * 60)
    logger.info(f"Processing ATR-{atr_period}:")
    logger.info(f"  ATR value: {atr_value:.3f}")
    logger.info(f"  Multiplier: {ATR_MULTIPLIER}")
    logger.info(f"  Range size: {range_size:.3f}")
    
    start_time = time.time()
    
    # Setup output path
    output_path = OUTPUT_DIR / f"ATR{atr_period}x{ATR_MULTIPLIER}" / f"ES-DIFF-range-ATR{atr_period}x{ATR_MULTIPLIER}-amibroker.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    saver = IncrementalBarSaver(output_path)
    
    # Process tick data in batches
    parquet_file = pq.ParquetFile(TICK_FILE)
    total_rows = parquet_file.metadata.num_rows
    num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    
    min_duration_ns = int(MIN_DURATION_SECONDS * 1e9)
    processed = 0
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min((batch_idx + 1) * BATCH_SIZE, total_rows)
        
        # Read batch
        start_rg = batch_start // 1048576
        end_rg = min((batch_end - 1) // 1048576 + 1, parquet_file.num_row_groups)
        
        tables = []
        for rg_idx in range(start_rg, end_rg):
            tables.append(parquet_file.read_row_group(rg_idx))
        
        combined_table = pa.concat_tables(tables)
        df = combined_table.to_pandas()
        
        # Slice to exact batch
        start_offset = batch_start - (start_rg * 1048576)
        end_offset = start_offset + (batch_end - batch_start)
        df = df.iloc[start_offset:end_offset].copy()
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format='%Y/%m/%d %H:%M:%S.%f',
            errors='coerce'
        )
        df = df[df['datetime'].notna()]
        
        if len(df) == 0:
            continue
        
        # Convert to numpy
        timestamps = df['datetime'].values.astype(np.int64)
        prices = df['Close'].values.astype(np.float64)
        volumes = df['Volume'].values.astype(np.int64)
        
        # Find boundaries with Numba
        boundaries = find_bar_boundaries_fast(timestamps, prices, range_size, min_duration_ns)
        
        if len(boundaries) > 0:
            # Create bars
            bars_list = []
            prev_end = 0
            
            for boundary in boundaries:
                if boundary > prev_end:
                    bar_slice = df.iloc[prev_end:boundary]
                    if len(bar_slice) > 0:
                        bars_list.append({
                            'timestamp': bar_slice['datetime'].iloc[-1],
                            'open': float(bar_slice['Close'].iloc[0]),
                            'high': float(bar_slice['Close'].max()),
                            'low': float(bar_slice['Close'].min()),
                            'close': float(bar_slice['Close'].iloc[-1]),
                            'volume': int(bar_slice['Volume'].sum()),
                            'ticks': len(bar_slice)
                        })
                        prev_end = boundary
            
            if bars_list:
                bars_df = pd.DataFrame(bars_list)
                # Add AUX1 (daily ATR value) and AUX2 (ATR multiplier)
                bars_df['AUX1'] = atr_value
                bars_df['AUX2'] = ATR_MULTIPLIER
                saver.add_bars(bars_df)
        
        processed += len(df)
        progress = (processed / total_rows) * 100
        
        if batch_idx % 2 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{num_batches}: {progress:.1f}% complete")
        
        # Clean up
        del df, tables, combined_table
        gc.collect()
    
    # Finalize
    final_df = saver.finalize()
    
    # Create NONE bars (keep AUX1 and AUX2 fields)
    none_df = final_df.copy()
    none_df[['open', 'high', 'low', 'close']] -= 50
    
    none_path = OUTPUT_DIR / f"ATR{atr_period}x{ATR_MULTIPLIER}" / f"ES-NONE-range-ATR{atr_period}x{ATR_MULTIPLIER}-amibroker.parquet"
    none_df.to_parquet(none_path, compression='snappy')
    
    elapsed = time.time() - start_time
    
    # Calculate statistics
    time_diffs = final_df['timestamp'].diff().dt.total_seconds() / 60
    time_diffs = time_diffs.dropna()
    
    return {
        'period': atr_period,
        'atr_value': atr_value,
        'range_size': range_size,
        'bars': len(final_df),
        'processing_time': elapsed / 60,
        'avg_mins_per_bar': time_diffs.mean(),
        'median_mins_per_bar': time_diffs.median()
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution"""
    
    logger.info("=" * 100)
    logger.info("VECTORBT PRO RANGE BAR PIPELINE")
    logger.info("=" * 100)
    logger.info(f"ATR periods: {ATR_PERIODS}")
    logger.info(f"Multiplier: {ATR_MULTIPLIER} (for all periods)")
    logger.info("=" * 100)
    
    # Step 1: Create/update daily data
    daily_df = create_or_update_daily_data()
    
    # Step 2: Calculate ATR values
    atr_values = calculate_atr_values(daily_df)
    
    # Step 3: Warm up Numba JIT
    logger.info("\nWarming up Numba JIT...")
    dummy = np.arange(100, dtype=np.int64)
    _ = find_bar_boundaries_fast(dummy, dummy.astype(np.float64), 1.0, 1000000000)
    logger.info("JIT ready!")
    
    # Step 4: Create range bars for each ATR period
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CREATE RANGE BARS")
    logger.info("=" * 80)
    
    results = []
    
    for period in ATR_PERIODS:
        try:
            result = create_range_bars_for_atr(period, atr_values[period]['use_value'])
            results.append(result)
        except Exception as e:
            logger.error(f"Failed for ATR-{period}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Step 5: Report results
    logger.info("\n" + "=" * 100)
    logger.info("FINAL RESULTS")
    logger.info("=" * 100)
    
    print(f"\n{'ATR':<10} {'Range':<10} {'Bars':<12} {'Proc Time':<12} {'Avg min/bar':<12} {'Median min/bar'}")
    print("-" * 80)
    
    for result in results:
        print(f"ATR-{result['period']:<6} {result['range_size']:<10.3f} {result['bars']:<12,} "
              f"{result['processing_time']:<12.1f} {result['avg_mins_per_bar']:<12.2f} "
              f"{result['median_mins_per_bar']:.2f}")
    
    logger.info("\n" + "=" * 100)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 100)

if __name__ == "__main__":
    main()