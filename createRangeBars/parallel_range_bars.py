#!/usr/bin/env python3
"""
Parallel Range Bar Creation with Pre-mapped Memory Chunks
==========================================================

Creates range bars using daily ATR values with optimized parallel processing.
Pre-allocates memory chunks and processes them in parallel using multiprocessing.
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
import logging
import gc
from numba import njit, prange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import platform
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
@dataclass
class Config:
    atr_multiplier: float = 0.1
    min_bar_duration_minutes: int = 1
    chunk_size_mb: int = 50  # Size of each memory chunk in MB
    n_workers: int = min(4, mp.cpu_count() - 1)  # Leave one CPU free
    max_bars_per_chunk: int = 10000  # Max bars per processing chunk
    
    # Paths
    tick_file: Path = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
    daily_atr_file: Path = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
    output_dir: Path = Path("../parquetData/range")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def create_bars_parallel(timestamps, prices, volumes, range_size, min_duration_ns):
    """
    Parallel Numba function to create range bars.
    Uses prange for automatic parallelization within Numba.
    """
    n = len(timestamps)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    
    # Pre-allocate with reasonable size
    max_bars = min(n // 20 + 100, 5000)
    boundaries = np.zeros(max_bars, dtype=np.int64)
    boundary_count = 0
    
    bar_start_time = timestamps[0]
    bar_high = prices[0]
    bar_low = prices[0]
    
    # Sequential processing (can't parallelize bar creation due to dependencies)
    for i in range(1, n):
        # Update high/low
        if prices[i] > bar_high:
            bar_high = prices[i]
        if prices[i] < bar_low:
            bar_low = prices[i]
        
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
                    break
    
    return boundaries[:boundary_count]

def process_day_chunk(args: Tuple) -> List[Dict]:
    """
    Process a single day's data - designed for parallel execution.
    Returns list of bar dictionaries.
    """
    date, day_df, range_info, min_duration_ns = args
    
    if len(day_df) == 0:
        return []
    
    range_size = range_info['range_size']
    atr_value = range_info['atr']
    
    if pd.isna(range_size) or range_size <= 0:
        return []
    
    # Sort by datetime
    day_df = day_df.sort_values('datetime')
    
    # Convert to numpy arrays
    timestamps = day_df['datetime'].values.astype(np.int64)
    prices = day_df['Close'].values.astype(np.float64)
    volumes = day_df['Volume'].values.astype(np.int64)
    
    # Find bar boundaries
    boundaries = create_bars_parallel(timestamps, prices, volumes, range_size, min_duration_ns)
    
    # Create bars from boundaries
    bars = []
    prev_idx = 0
    for boundary_idx in boundaries:
        if boundary_idx > prev_idx:
            bar_data = day_df.iloc[prev_idx:boundary_idx + 1]
            
            bars.append({
                'timestamp': bar_data.iloc[-1]['datetime'],
                'open': float(bar_data.iloc[0]['Close']),
                'high': float(bar_data['Close'].max()),
                'low': float(bar_data['Close'].min()),
                'close': float(bar_data.iloc[-1]['Close']),
                'volume': int(bar_data['Volume'].sum()),
                'ticks': len(bar_data),
                'AUX1': float(atr_value),
                'AUX2': float(range_info['multiplier'])
            })
            prev_idx = boundary_idx + 1
    
    return bars

def process_row_group_batch(rg_indices: List[int], parquet_path: Path, 
                           date_to_range: Dict, config: Config) -> List[Dict]:
    """
    Process a batch of row groups - designed for parallel execution.
    """
    all_bars = []
    min_duration_ns = int(config.min_bar_duration_minutes * 60 * 1e9)
    
    # Open parquet file in this process
    parquet_file = pq.ParquetFile(parquet_path)
    
    for rg_idx in rg_indices:
        try:
            # Read row group
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
                
                range_info = date_to_range[date].copy()
                range_info['multiplier'] = config.atr_multiplier
                
                # Process this day's data
                day_bars = process_day_chunk((date, day_group, range_info, min_duration_ns))
                all_bars.extend(day_bars)
            
            # Clean up
            del df, table
            
        except Exception as e:
            logger.warning(f"Error processing row group {rg_idx}: {e}")
            continue
    
    gc.collect()
    return all_bars

def create_range_bars_parallel(config: Config = Config()):
    """Main processing function with parallel execution."""
    
    logger.info("=" * 80)
    logger.info(f"PARALLEL RANGE BAR CREATION - {config.atr_multiplier} MULTIPLIER")
    logger.info("=" * 80)
    logger.info(f"System: {platform.platform()}")
    logger.info(f"CPUs: {mp.cpu_count()}, Workers: {config.n_workers}")
    logger.info(f"Chunk size: {config.chunk_size_mb}MB, Max bars/chunk: {config.max_bars_per_chunk}")
    logger.info("=" * 80)
    
    # Load daily ATR data
    logger.info("Loading daily ATR data...")
    daily_df = pd.read_csv(config.daily_atr_file)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Calculate range sizes with multiplier
    daily_df[f'range_size_{config.atr_multiplier}'] = daily_df['atr_lagged'] * config.atr_multiplier
    
    logger.info(f"Loaded {len(daily_df)} days of ATR data")
    logger.info(f"Range size ({config.atr_multiplier}x): "
                f"{daily_df[f'range_size_{config.atr_multiplier}'].min():.2f} to "
                f"{daily_df[f'range_size_{config.atr_multiplier}'].max():.2f}")
    
    # Create date lookup
    date_to_range = {}
    for _, row in daily_df.iterrows():
        date_to_range[row['date'].date()] = {
            'range_size': row[f'range_size_{config.atr_multiplier}'],
            'atr': row['atr_lagged']
        }
    
    # Create output directory
    output_subdir = config.output_dir / f"ATR30x{config.atr_multiplier}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Get row group information
    logger.info("Analyzing tick data structure...")
    parquet_file = pq.ParquetFile(config.tick_file)
    total_row_groups = parquet_file.num_row_groups
    total_rows = parquet_file.metadata.num_rows
    
    logger.info(f"Total ticks: {total_rows:,}")
    logger.info(f"Total row groups: {total_row_groups}")
    
    # Determine optimal batch size based on memory
    rows_per_group = total_rows // total_row_groups if total_row_groups > 0 else 1
    bytes_per_row = 50  # Estimated bytes per row
    bytes_per_group = rows_per_group * bytes_per_row
    groups_per_batch = max(1, (config.chunk_size_mb * 1024 * 1024) // bytes_per_group)
    groups_per_batch = min(groups_per_batch, 10)  # Cap at 10 for safety
    
    logger.info(f"Processing strategy: {groups_per_batch} row groups per batch")
    
    # Create batches for parallel processing
    batches = []
    for i in range(0, total_row_groups, groups_per_batch):
        batch = list(range(i, min(i + groups_per_batch, total_row_groups)))
        batches.append(batch)
    
    logger.info(f"Created {len(batches)} batches for parallel processing")
    
    # Process batches in parallel
    all_bars = []
    start_time = time.time()
    
    logger.info(f"Starting parallel processing with {config.n_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        # Submit all batch jobs
        future_to_batch = {
            executor.submit(process_row_group_batch, batch, config.tick_file, 
                          date_to_range, config): i 
            for i, batch in enumerate(batches)
        }
        
        # Process completed batches
        completed = 0
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_bars = future.result()
                all_bars.extend(batch_bars)
                completed += 1
                
                # Progress update
                pct_complete = (completed / len(batches)) * 100
                elapsed = (time.time() - start_time) / 60
                logger.info(f"  Completed batch {completed}/{len(batches)} "
                           f"({pct_complete:.1f}%), Bars: {len(all_bars):,}, "
                           f"Time: {elapsed:.1f} min")
                
                # Save intermediate results periodically
                if completed % 50 == 0 and all_bars:
                    logger.info(f"  Saving intermediate results ({len(all_bars)} bars)...")
                    temp_df = pd.DataFrame(all_bars)
                    temp_df = temp_df.sort_values('timestamp').reset_index(drop=True)
                    temp_path = output_subdir / f"temp_parallel_{config.atr_multiplier}_{completed}.parquet"
                    temp_df.to_parquet(temp_path, compression='snappy')
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
    
    logger.info(f"\nCreated {len(all_bars):,} range bars")
    
    if all_bars:
        # Final processing
        final_df = pd.DataFrame(all_bars)
        final_df = final_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate gaps
        final_df['gap'] = 0.0
        if len(final_df) > 1:
            final_df.loc[1:, 'gap'] = final_df['open'].iloc[1:].values - final_df['close'].iloc[:-1].values
        
        # Save DIFF bars
        diff_path = output_subdir / f"ES-DIFF-range-ATR30x{config.atr_multiplier}-amibroker.parquet"
        final_df.to_parquet(diff_path, compression='snappy')
        logger.info(f"Saved DIFF bars: {diff_path}")
        
        # Create NONE bars
        none_df = final_df.copy()
        none_df[['open', 'high', 'low', 'close']] -= 50
        none_path = output_subdir / f"ES-NONE-range-ATR30x{config.atr_multiplier}-amibroker.parquet"
        none_df.to_parquet(none_path, compression='snappy')
        logger.info(f"Saved NONE bars: {none_path}")
        
        # Export to CSV
        logger.info("\nExporting to CSV...")
        csv_dir = Path(f"../dataRAW/range-ATR30x{config.atr_multiplier}")
        
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
        
        diff_csv_path = diff_csv_dir / f"ES-DIFF-range-ATR30x{config.atr_multiplier}-dailyATR.csv"
        csv_export.to_csv(diff_csv_path, index=False)
        logger.info(f"Saved DIFF CSV: {diff_csv_path}")
        
        # NONE CSV
        none_csv_dir = csv_dir / "ES/noneAdjusted"
        none_csv_dir.mkdir(parents=True, exist_ok=True)
        
        none_csv = csv_export.copy()
        none_csv[['Open', 'High', 'Low', 'Close']] -= 50
        
        none_csv_path = none_csv_dir / f"ES-NONE-range-ATR30x{config.atr_multiplier}-dailyATR.csv"
        none_csv.to_csv(none_csv_path, index=False)
        logger.info(f"Saved NONE CSV: {none_csv_path}")
        
        # Clean up temp files
        for temp_file in output_subdir.glob("temp_parallel_*.parquet"):
            temp_file.unlink()
        
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

def main():
    """Main execution with configurable parameters."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Create range bars with parallel processing')
    parser.add_argument('--multiplier', type=float, default=0.1, 
                       help='ATR multiplier (default: 0.1)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--chunk-mb', type=int, default=50,
                       help='Memory chunk size in MB (default: 50)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        atr_multiplier=args.multiplier,
        n_workers=args.workers if args.workers else min(4, mp.cpu_count() - 1),
        chunk_size_mb=args.chunk_mb
    )
    
    logger.info("=" * 100)
    logger.info("PARALLEL RANGE BAR CREATION PIPELINE")
    logger.info("=" * 100)
    logger.info(f"Configuration:")
    logger.info(f"  ATR Multiplier: {config.atr_multiplier}")
    logger.info(f"  Workers: {config.n_workers}")
    logger.info(f"  Chunk Size: {config.chunk_size_mb}MB")
    logger.info(f"  Min Bar Duration: {config.min_bar_duration_minutes} minutes")
    logger.info("=" * 100)
    
    try:
        # Check if daily ATR data exists
        if not config.daily_atr_file.exists():
            logger.error(f"Daily ATR file not found: {config.daily_atr_file}")
            logger.error("Please run vbt_proper_pipeline.py first to create daily ATR data")
            return 1
        
        # Create range bars
        range_bars = create_range_bars_parallel(config)
        
        if range_bars is not None:
            logger.info("\n" + "=" * 100)
            logger.info("PARALLEL PIPELINE COMPLETE!")
            logger.info("=" * 100)
            logger.info(f"Created {len(range_bars):,} range bars with daily-varying ATR")
            logger.info("=" * 100)
            return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 1

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    sys.exit(main())