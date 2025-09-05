#!/usr/bin/env python3
"""
Parallel Range Bar Creation with Memory Monitoring
==================================================

Creates range bars with parallel processing and real-time memory tracking.
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
    chunk_size_mb: int = 25  # Smaller chunks for better memory control
    n_workers: int = min(4, mp.cpu_count() - 1)
    max_bars_per_chunk: int = 5000  # Reduced for memory efficiency
    memory_limit_gb: float = 4.0  # Max memory per process
    
    # Paths
    tick_file: Path = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
    daily_atr_file: Path = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
    output_dir: Path = Path("../parquetData/range")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,
        'vms_mb': mem_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }

def log_memory(prefix="Memory"):
    """Log current memory usage."""
    mem = get_memory_usage()
    logger.info(f"{prefix}: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, {mem['percent']:.1f}%")

@njit(cache=True, fastmath=True, nogil=True)
def create_bars_chunked(timestamps, prices, volumes, range_size, min_duration_ns, chunk_size=10000):
    """
    Create range bars with chunked processing to minimize memory allocation.
    Processes data in smaller chunks to avoid large memory allocations.
    """
    n = len(timestamps)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    
    # Process in chunks
    all_boundaries = []
    chunk_start = 0
    
    # Carry over state between chunks
    bar_start_time = timestamps[0]
    bar_high = prices[0]
    bar_low = prices[0]
    
    while chunk_start < n:
        chunk_end = min(chunk_start + chunk_size, n)
        
        # Pre-allocate for this chunk only
        max_bars_chunk = min((chunk_end - chunk_start) // 20 + 10, 1000)
        boundaries_chunk = np.zeros(max_bars_chunk, dtype=np.int64)
        boundary_count = 0
        
        # Process chunk
        for i in range(chunk_start, chunk_end):
            if i == 0:
                continue
                
            # Update high/low
            if prices[i] > bar_high:
                bar_high = prices[i]
            if prices[i] < bar_low:
                bar_low = prices[i]
            
            # Check bar completion
            if (bar_high - bar_low) >= range_size:
                if (timestamps[i] - bar_start_time) >= min_duration_ns:
                    if boundary_count < max_bars_chunk:
                        boundaries_chunk[boundary_count] = i
                        boundary_count += 1
                        
                        # Reset for next bar
                        if i < n - 1:
                            bar_start_time = timestamps[i]
                            bar_high = prices[i]
                            bar_low = prices[i]
        
        # Collect boundaries from this chunk
        if boundary_count > 0:
            all_boundaries.append(boundaries_chunk[:boundary_count])
        
        chunk_start = chunk_end
    
    # Combine all boundaries
    if len(all_boundaries) == 0:
        return np.empty(0, dtype=np.int64)
    elif len(all_boundaries) == 1:
        return all_boundaries[0]
    else:
        total_boundaries = sum(len(b) for b in all_boundaries)
        combined = np.zeros(total_boundaries, dtype=np.int64)
        pos = 0
        for boundaries in all_boundaries:
            combined[pos:pos+len(boundaries)] = boundaries
            pos += len(boundaries)
        return combined

def process_row_group_batch_monitored(rg_indices: List[int], parquet_path: Path, 
                                      date_to_range: Dict, config: Config) -> Tuple[List[Dict], Dict]:
    """
    Process a batch with memory monitoring.
    Returns bars and memory statistics.
    """
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    peak_memory = initial_memory
    
    all_bars = []
    min_duration_ns = int(config.min_bar_duration_minutes * 60 * 1e9)
    
    # Open parquet file
    parquet_file = pq.ParquetFile(parquet_path)
    
    for rg_idx in rg_indices:
        try:
            # Check memory before processing
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            
            # Skip if approaching memory limit
            if current_memory > config.memory_limit_gb * 1024:
                logger.warning(f"Memory limit approached: {current_memory:.1f}MB, skipping row group {rg_idx}")
                gc.collect()
                continue
            
            # Read row group with minimal columns
            table = parquet_file.read_row_group(rg_idx)
            df = table.select(['Date', 'Time', 'Close', 'Volume']).to_pandas()
            
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
                
                # Use chunked processing
                boundaries = create_bars_chunked(
                    timestamps, prices, volumes, range_size, min_duration_ns, 
                    chunk_size=config.max_bars_per_chunk
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
            
            # Clean up aggressively
            del df, table, day_group
            
        except Exception as e:
            logger.warning(f"Error processing row group {rg_idx}: {e}")
            continue
    
    # Force garbage collection
    gc.collect()
    
    # Return results with memory stats
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_stats = {
        'initial_mb': initial_memory,
        'peak_mb': peak_memory,
        'final_mb': final_memory,
        'delta_mb': final_memory - initial_memory
    }
    
    return all_bars, memory_stats

def create_range_bars_monitored(config: Config = Config()):
    """Main processing with memory monitoring."""
    
    logger.info("=" * 80)
    logger.info(f"PARALLEL RANGE BAR CREATION WITH MEMORY MONITORING")
    logger.info("=" * 80)
    logger.info(f"System: {platform.platform()}")
    logger.info(f"CPUs: {mp.cpu_count()}, Workers: {config.n_workers}")
    logger.info(f"Memory limit per process: {config.memory_limit_gb}GB")
    logger.info(f"Chunk size: {config.chunk_size_mb}MB")
    
    # Log initial memory
    log_memory("Initial memory")
    
    # Load daily ATR data
    logger.info("Loading daily ATR data...")
    daily_df = pd.read_csv(config.daily_atr_file)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Calculate range sizes
    daily_df[f'range_size_{config.atr_multiplier}'] = daily_df['atr_lagged'] * config.atr_multiplier
    
    logger.info(f"Loaded {len(daily_df)} days of ATR data")
    log_memory("After loading ATR")
    
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
    
    # Analyze parquet structure
    logger.info("Analyzing tick data...")
    parquet_file = pq.ParquetFile(config.tick_file)
    total_row_groups = parquet_file.num_row_groups
    total_rows = parquet_file.metadata.num_rows
    
    logger.info(f"Total ticks: {total_rows:,}")
    logger.info(f"Total row groups: {total_row_groups}")
    
    # Create smaller batches for better memory control
    rows_per_group = total_rows // total_row_groups if total_row_groups > 0 else 1
    bytes_per_row = 50
    bytes_per_group = rows_per_group * bytes_per_row
    groups_per_batch = max(1, (config.chunk_size_mb * 1024 * 1024) // bytes_per_group)
    groups_per_batch = min(groups_per_batch, 5)  # Even smaller batches
    
    logger.info(f"Processing strategy: {groups_per_batch} row groups per batch")
    
    # Create batches
    batches = []
    for i in range(0, total_row_groups, groups_per_batch):
        batch = list(range(i, min(i + groups_per_batch, total_row_groups)))
        batches.append(batch)
    
    logger.info(f"Created {len(batches)} batches for processing")
    log_memory("Before parallel processing")
    
    # Process with memory monitoring
    all_bars = []
    total_memory_stats = {
        'peak_mb': 0,
        'avg_delta_mb': []
    }
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        # Submit jobs
        future_to_batch = {
            executor.submit(process_row_group_batch_monitored, batch, config.tick_file, 
                          date_to_range, config): i 
            for i, batch in enumerate(batches)
        }
        
        # Process results
        completed = 0
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_bars, memory_stats = future.result()
                all_bars.extend(batch_bars)
                completed += 1
                
                # Track memory statistics
                total_memory_stats['peak_mb'] = max(total_memory_stats['peak_mb'], memory_stats['peak_mb'])
                total_memory_stats['avg_delta_mb'].append(memory_stats['delta_mb'])
                
                # Progress update with memory info
                pct_complete = (completed / len(batches)) * 100
                elapsed = (time.time() - start_time) / 60
                avg_delta = np.mean(total_memory_stats['avg_delta_mb']) if total_memory_stats['avg_delta_mb'] else 0
                
                logger.info(f"Batch {completed}/{len(batches)} ({pct_complete:.1f}%), "
                           f"Bars: {len(all_bars):,}, Time: {elapsed:.1f}min, "
                           f"Peak mem: {memory_stats['peak_mb']:.0f}MB, Avg delta: {avg_delta:.0f}MB")
                
                # Log main process memory periodically
                if completed % 10 == 0:
                    log_memory(f"Main process after {completed} batches")
                
                # Save intermediate results
                if completed % 50 == 0 and all_bars:
                    logger.info(f"Saving intermediate results ({len(all_bars)} bars)...")
                    temp_df = pd.DataFrame(all_bars)
                    temp_df = temp_df.sort_values('timestamp').reset_index(drop=True)
                    temp_path = output_subdir / f"temp_monitored_{config.atr_multiplier}_{completed}.parquet"
                    temp_df.to_parquet(temp_path, compression='snappy')
                    
                    # Force garbage collection after saving
                    gc.collect()
                    log_memory("After saving intermediate")
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
    
    logger.info(f"\nCreated {len(all_bars):,} range bars")
    log_memory("After parallel processing")
    
    if all_bars:
        # Final processing
        final_df = pd.DataFrame(all_bars)
        final_df = final_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate gaps
        final_df['gap'] = 0.0
        if len(final_df) > 1:
            final_df.loc[1:, 'gap'] = final_df['open'].iloc[1:].values - final_df['close'].iloc[:-1].values
        
        # Save files
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
        for temp_file in output_subdir.glob("temp_monitored_*.parquet"):
            temp_file.unlink()
        
        # Final statistics
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
        
        # Memory statistics
        logger.info("\nMEMORY STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Peak memory in workers: {total_memory_stats['peak_mb']:.0f}MB")
        logger.info(f"Average memory delta: {np.mean(total_memory_stats['avg_delta_mb']):.0f}MB")
        log_memory("Final memory")
        
        return final_df
    
    return None

def main():
    """Main execution."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Create range bars with memory monitoring')
    parser.add_argument('--multiplier', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--chunk-mb', type=int, default=25)
    parser.add_argument('--memory-limit-gb', type=float, default=4.0)
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        atr_multiplier=args.multiplier,
        n_workers=args.workers if args.workers else min(4, mp.cpu_count() - 1),
        chunk_size_mb=args.chunk_mb,
        memory_limit_gb=args.memory_limit_gb
    )
    
    logger.info("=" * 100)
    logger.info("MEMORY-MONITORED PARALLEL RANGE BAR CREATION")
    logger.info("=" * 100)
    logger.info(f"Configuration:")
    logger.info(f"  ATR Multiplier: {config.atr_multiplier}")
    logger.info(f"  Workers: {config.n_workers}")
    logger.info(f"  Chunk Size: {config.chunk_size_mb}MB")
    logger.info(f"  Memory Limit: {config.memory_limit_gb}GB per process")
    logger.info("=" * 100)
    
    try:
        # Check dependencies
        try:
            import psutil
        except ImportError:
            logger.error("psutil not installed. Installing...")
            os.system(f"{sys.executable} -m pip install psutil")
            import psutil
        
        # Check ATR file
        if not config.daily_atr_file.exists():
            logger.error(f"Daily ATR file not found: {config.daily_atr_file}")
            return 1
        
        # Create range bars
        range_bars = create_range_bars_monitored(config)
        
        if range_bars is not None:
            logger.info("\n" + "=" * 100)
            logger.info("PIPELINE COMPLETE WITH MEMORY MONITORING!")
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