#!/usr/bin/env python3
"""
Dynamic Parallel Range Bar Creation with Adaptive Worker Scaling
================================================================

Automatically scales workers based on available RAM to maximize performance.
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
import logging
import gc
import psutil
import os
from numba import njit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import sys
import platform
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
@dataclass
class Config:
    atr_multiplier: float = 0.05
    min_bar_duration_minutes: int = 1
    chunk_size_mb: int = 25  # Small chunks for better distribution
    initial_workers: int = 4  # Start with 4 workers
    max_workers: int = 20  # Max workers to spawn
    target_ram_percent: float = 50.0  # Target 50% RAM usage
    safety_margin_percent: float = 10.0  # Keep 10% safety margin
    max_bars_per_chunk: int = 5000
    
    # Paths
    tick_file: Path = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
    daily_atr_file: Path = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
    output_dir: Path = Path("../parquetData/range")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

def get_system_memory_info():
    """Get system memory information."""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / 1024**3,
        'available_gb': mem.available / 1024**3,
        'used_gb': mem.used / 1024**3,
        'percent': mem.percent
    }

def calculate_optimal_workers(target_ram_gb: float = 64.0, per_worker_gb: float = 2.5):
    """Calculate optimal number of workers based on available RAM."""
    mem_info = get_system_memory_info()
    available_for_workers = min(target_ram_gb, mem_info['available_gb'] - 4)  # Keep 4GB for system
    optimal = int(available_for_workers / per_worker_gb)
    return max(2, min(optimal, 20))  # Between 2 and 20 workers

@njit(cache=True, fastmath=True, nogil=True)
def create_bars_optimized(timestamps, prices, volumes, range_size, min_duration_ns):
    """
    Optimized Numba function for range bar creation.
    """
    n = len(timestamps)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    
    # Pre-allocate conservatively
    max_bars = min(n // 20 + 100, 5000)
    boundaries = np.zeros(max_bars, dtype=np.int64)
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

def process_row_group_batch(rg_indices: List[int], parquet_path: Path, 
                           date_to_range: Dict, config: Config) -> List[Dict]:
    """
    Process a batch of row groups.
    """
    all_bars = []
    min_duration_ns = int(config.min_bar_duration_minutes * 60 * 1e9)
    
    # Open parquet file in this process
    parquet_file = pq.ParquetFile(parquet_path)
    
    for rg_idx in rg_indices:
        try:
            # Read row group with minimal columns
            table = parquet_file.read_row_group(rg_idx)
            df = table.select(['Date', 'Time', 'Close', 'Volume']).to_pandas()
            
            # Parse datetime efficiently
            df['datetime'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                format='%Y/%m/%d %H:%M:%S.%f',
                errors='coerce'
            )
            
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
                
                # Find boundaries using optimized function
                boundaries = create_bars_optimized(
                    timestamps, prices, volumes, range_size, min_duration_ns
                )
                
                # Create bars from boundaries
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
                            'AUX2': float(config.atr_multiplier)
                        })
                        prev_idx = boundary_idx + 1
            
            # Clean up
            del df, table
            
        except Exception as e:
            logger.warning(f"Error processing row group {rg_idx}: {e}")
            continue
    
    gc.collect()
    return all_bars

def create_range_bars_dynamic(config: Config = Config()):
    """Main processing with dynamic worker scaling."""
    
    logger.info("=" * 80)
    logger.info("DYNAMIC PARALLEL RANGE BAR CREATION")
    logger.info("=" * 80)
    
    # System info
    mem_info = get_system_memory_info()
    logger.info(f"System: {platform.platform()}")
    logger.info(f"Total RAM: {mem_info['total_gb']:.1f}GB")
    logger.info(f"Available RAM: {mem_info['available_gb']:.1f}GB")
    logger.info(f"Target RAM usage: {config.target_ram_percent}%")
    
    # Calculate initial workers
    target_ram_gb = mem_info['total_gb'] * (config.target_ram_percent - config.safety_margin_percent) / 100
    optimal_workers = calculate_optimal_workers(target_ram_gb)
    
    logger.info(f"Calculated optimal workers: {optimal_workers}")
    logger.info("=" * 80)
    
    # Load daily ATR data
    logger.info("Loading daily ATR data...")
    daily_df = pd.read_csv(config.daily_atr_file)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Calculate range sizes
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
    
    # Create smaller batches for better load balancing
    groups_per_batch = 2  # Very small batches for better distribution
    batches = []
    for i in range(0, total_row_groups, groups_per_batch):
        batch = list(range(i, min(i + groups_per_batch, total_row_groups)))
        batches.append(batch)
    
    logger.info(f"Created {len(batches)} batches for processing")
    logger.info(f"Starting with {optimal_workers} workers")
    
    # Process batches with dynamic worker pool
    all_bars = []
    start_time = time.time()
    current_workers = optimal_workers
    
    # Use ProcessPoolExecutor with dynamic sizing
    with ProcessPoolExecutor(max_workers=current_workers) as executor:
        # Submit initial batches
        futures = {}
        batch_index = 0
        
        # Submit initial work to all workers
        for _ in range(min(current_workers * 2, len(batches))):
            if batch_index < len(batches):
                future = executor.submit(
                    process_row_group_batch, 
                    batches[batch_index], 
                    config.tick_file,
                    date_to_range, 
                    config
                )
                futures[future] = batch_index
                batch_index += 1
        
        completed = 0
        last_mem_check = time.time()
        
        # Process results and submit new work
        while futures:
            # Check memory periodically and adjust workers if needed
            if time.time() - last_mem_check > 10:  # Check every 10 seconds
                mem_info = get_system_memory_info()
                logger.info(f"RAM usage: {mem_info['percent']:.1f}%, "
                           f"Workers: {current_workers}, "
                           f"Completed: {completed}/{len(batches)}")
                last_mem_check = time.time()
            
            # Wait for next completion
            done, pending = wait_for_any_complete(futures, timeout=0.5)
            
            for future in done:
                batch_idx = futures.pop(future)
                try:
                    batch_bars = future.result()
                    all_bars.extend(batch_bars)
                    completed += 1
                    
                    # Progress update
                    if completed % 10 == 0 or completed == len(batches):
                        pct_complete = (completed / len(batches)) * 100
                        elapsed = (time.time() - start_time) / 60
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(batches) - completed) / rate if rate > 0 else 0
                        
                        logger.info(f"Progress: {completed}/{len(batches)} ({pct_complete:.1f}%), "
                                   f"Bars: {len(all_bars):,}, "
                                   f"Time: {elapsed:.1f}min, ETA: {eta:.1f}min")
                    
                    # Submit new work if available
                    if batch_index < len(batches):
                        future = executor.submit(
                            process_row_group_batch,
                            batches[batch_index],
                            config.tick_file,
                            date_to_range,
                            config
                        )
                        futures[future] = batch_index
                        batch_index += 1
                    
                    # Save intermediate results periodically
                    if completed % 100 == 0 and all_bars:
                        logger.info(f"Saving intermediate results ({len(all_bars)} bars)...")
                        temp_df = pd.DataFrame(all_bars)
                        temp_df = temp_df.sort_values('timestamp').reset_index(drop=True)
                        temp_path = output_subdir / f"temp_dynamic_{config.atr_multiplier}_{completed}.parquet"
                        temp_df.to_parquet(temp_path, compression='snappy')
                        gc.collect()
                    
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
        for temp_file in output_subdir.glob("temp_dynamic_*.parquet"):
            temp_file.unlink()
        
        # Statistics
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 80)
        
        time_diffs = final_df['timestamp'].diff().dt.total_seconds() / 60
        time_diffs = time_diffs.dropna()
        
        processing_time = (time.time() - start_time) / 60
        
        logger.info(f"Total bars: {len(final_df):,}")
        logger.info(f"Average minutes/bar: {time_diffs.mean():.1f}")
        logger.info(f"Median minutes/bar: {time_diffs.median():.1f}")
        logger.info(f"Unique ATR values: {final_df['AUX1'].nunique()}")
        logger.info(f"Processing time: {processing_time:.1f} minutes")
        logger.info(f"Average workers used: {optimal_workers}")
        logger.info(f"Bars per minute: {len(final_df) / processing_time:.0f}")
        
        return final_df
    
    return None

def wait_for_any_complete(futures, timeout=0.5):
    """Helper to wait for any future to complete."""
    from concurrent.futures import wait, FIRST_COMPLETED
    done, pending = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
    return done, pending

def main():
    """Main execution."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Create range bars with dynamic parallel processing')
    parser.add_argument('--multiplier', type=float, default=0.05)
    parser.add_argument('--target-ram', type=float, default=50.0,
                       help='Target RAM usage percentage (default: 50)')
    
    args = parser.parse_args()
    
    # Install psutil if needed
    try:
        import psutil
    except ImportError:
        logger.info("Installing psutil...")
        os.system(f"{sys.executable} -m pip install psutil")
        import psutil
    
    # Create configuration
    config = Config(
        atr_multiplier=args.multiplier,
        target_ram_percent=args.target_ram
    )
    
    logger.info("=" * 100)
    logger.info("DYNAMIC PARALLEL RANGE BAR CREATION")
    logger.info("=" * 100)
    logger.info(f"Configuration:")
    logger.info(f"  ATR Multiplier: {config.atr_multiplier}")
    logger.info(f"  Target RAM: {config.target_ram_percent}%")
    logger.info("=" * 100)
    
    try:
        # Check ATR file
        if not config.daily_atr_file.exists():
            logger.error(f"Daily ATR file not found: {config.daily_atr_file}")
            return 1
        
        # Create range bars
        range_bars = create_range_bars_dynamic(config)
        
        if range_bars is not None:
            logger.info("\n" + "=" * 100)
            logger.info("PIPELINE COMPLETE!")
            logger.info("=" * 100)
            return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 1

if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())