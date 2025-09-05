"""
Ultra-Fast Raw Price Range Bars Implementation

This module implements raw price range bars using hyper-optimized algorithms
with Numba JIT compilation and vectorized NumPy operations for maximum performance
on large tick datasets (37GB+).

Range bars are formed when the price moves a specified amount (range) from the 
previous range bar's close price.
"""

import numpy as np
import pandas as pd
import numba as nb
from numba import jit, prange
from typing import Union, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

from performance import measure_time, benchmark_func

def monitor_memory(operation=""):
    """Monitor memory usage and trigger GC if needed"""
    memory_info = psutil.virtual_memory()
    memory_gb = memory_info.used / (1024 ** 3)
    memory_percent = memory_info.percent
    
    if memory_percent > 80:  # High memory usage
        print(f"⚠️  High memory: {memory_percent:.1f}% ({memory_gb:.1f} GB) - {operation}")
        gc.collect()  # Force garbage collection
        
    if memory_gb > 6.0:  # 6GB limit
        print(f"❌ Memory limit exceeded: {memory_gb:.1f} GB > 6.0 GB")
        raise MemoryError(f"Memory limit of 6.0 GB exceeded")
        
    return memory_gb

def fix_duplicate_timestamps_fast(timestamps):
    """Add microseconds to duplicate timestamps using vectorized operations"""
    if len(timestamps) == 0:
        return timestamps
        
    # Convert to datetime if not already
    if not isinstance(timestamps.iloc[0], pd.Timestamp):
        timestamps = pd.to_datetime(timestamps)
    
    # Find duplicates
    duplicated_mask = timestamps.duplicated(keep=False)
    n_duplicates = duplicated_mask.sum()
    
    if n_duplicates == 0:
        return timestamps
        
    print(f"🔧 Fixing {n_duplicates} duplicate timestamps...")
    
    # Create a copy
    fixed_timestamps = timestamps.copy()
    
    # Group by timestamp and add incremental microseconds
    for timestamp, group in timestamps[duplicated_mask].groupby(timestamps[duplicated_mask]):
        indices = group.index
        # Add 1, 2, 3... microseconds to duplicates
        microsecond_additions = pd.to_timedelta(range(len(indices)), unit='us')
        fixed_timestamps.iloc[indices] = timestamp + microsecond_additions
        
    print(f"✅ Fixed {n_duplicates} duplicate timestamps")
    return fixed_timestamps

@dataclass
class RangeBarConfig:
    """Configuration for raw price range bars"""
    range_size: float
    use_bid_ask: bool = False      # Use bid/ask if available
    min_volume: int = 0            # Minimum volume threshold
    allow_gaps: bool = True        # Allow price gaps to create new bars
    tick_volume_aggregation: str = 'sum'  # How to aggregate tick volumes
    chunk_size: int = 5_000_000    # Process data in chunks to manage memory
    memory_limit_gb: float = 6.0   # Memory limit in GB
    fix_duplicate_timestamps: bool = True  # Add microseconds to duplicate timestamps

@jit(nopython=True, cache=True, parallel=False)
def build_range_bars_raw_core(prices: np.ndarray, 
                              volumes: np.ndarray,
                              timestamps: np.ndarray,
                              range_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                        np.ndarray, np.ndarray, np.ndarray,
                                                        np.ndarray]:
    """
    Ultra-fast core range bar building algorithm using Numba JIT.
    
    This is the performance-critical inner loop optimized for maximum speed.
    
    Args:
        prices: Array of tick prices (Close prices)
        volumes: Array of tick volumes  
        timestamps: Array of timestamp indices
        range_size: Fixed range size for bars
        
    Returns:
        Tuple of arrays: (opens, highs, lows, closes, volumes, start_times, end_times)
    """
    n_ticks = len(prices)
    if n_ticks == 0:
        # Return empty arrays with correct types
        empty_float = np.array([], dtype=np.float32)
        empty_int = np.array([], dtype=np.int64)
        return empty_float, empty_float, empty_float, empty_float, empty_int, empty_int, empty_int
    
    # Pre-allocate arrays for maximum performance
    # Estimate max possible bars (conservative upper bound)
    max_bars = min(n_ticks, 1000000)  # Cap to prevent memory issues
    
    bar_opens = np.empty(max_bars, dtype=np.float32)
    bar_highs = np.empty(max_bars, dtype=np.float32) 
    bar_lows = np.empty(max_bars, dtype=np.float32)
    bar_closes = np.empty(max_bars, dtype=np.float32)
    bar_volumes = np.empty(max_bars, dtype=np.int64)
    bar_start_times = np.empty(max_bars, dtype=np.int64)
    bar_end_times = np.empty(max_bars, dtype=np.int64)
    
    # Initialize first bar
    bar_count = 0
    current_open = prices[0]
    current_high = prices[0]
    current_low = prices[0]
    current_volume = volumes[0]
    current_start_time = timestamps[0]
    range_top = current_open + range_size
    range_bottom = current_open - range_size
    
    # Process each tick
    for i in range(1, n_ticks):
        current_price = prices[i]
        current_vol = volumes[i]
        
        # Check if price breaches range
        if current_price >= range_top or current_price <= range_bottom:
            # Close current bar
            if bar_count < max_bars:
                bar_opens[bar_count] = current_open
                bar_highs[bar_count] = current_high  
                bar_lows[bar_count] = current_low
                bar_closes[bar_count] = prices[i-1]  # Previous tick is close
                bar_volumes[bar_count] = current_volume
                bar_start_times[bar_count] = current_start_time
                bar_end_times[bar_count] = timestamps[i-1]
                bar_count += 1
            
            # Start new bar
            current_open = prices[i-1]  # Previous close becomes new open
            current_high = max(current_open, current_price)
            current_low = min(current_open, current_price)
            current_volume = current_vol
            current_start_time = timestamps[i-1]
            
            # Set new range boundaries
            range_top = current_open + range_size
            range_bottom = current_open - range_size
        else:
            # Update current bar
            current_high = max(current_high, current_price)
            current_low = min(current_low, current_price)
            current_volume += current_vol
    
    # Close final bar
    if bar_count < max_bars:
        bar_opens[bar_count] = current_open
        bar_highs[bar_count] = current_high
        bar_lows[bar_count] = current_low
        bar_closes[bar_count] = prices[-1]
        bar_volumes[bar_count] = current_volume
        bar_start_times[bar_count] = current_start_time
        bar_end_times[bar_count] = timestamps[-1]
        bar_count += 1
    
    # Return only the filled portion of arrays
    return (bar_opens[:bar_count].copy(),
            bar_highs[:bar_count].copy(), 
            bar_lows[:bar_count].copy(),
            bar_closes[:bar_count].copy(),
            bar_volumes[:bar_count].copy(),
            bar_start_times[:bar_count].copy(),
            bar_end_times[:bar_count].copy())

class RawPriceRangeBars:
    """
    Ultra-fast raw price range bar generator optimized for massive datasets.
    
    Features:
    - Numba JIT compilation for core algorithms  
    - Vectorized NumPy operations
    - Chunked processing for memory efficiency
    - Parallel processing support
    - Comprehensive performance monitoring
    """
    
    def __init__(self, config: RangeBarConfig):
        """
        Initialize range bar generator.
        
        Args:
            config: Range bar configuration
        """
        self.config = config
        self.stats = {
            'total_ticks_processed': 0,
            'total_bars_created': 0,
            'processing_time': 0.0,
            'ticks_per_second': 0.0
        }
        
        # Warm up Numba JIT compilation
        self._warmup_jit()
    
    def _warmup_jit(self):
        """Warm up Numba JIT compilation with small dataset"""
        with measure_time("numba_jit_warmup"):
            dummy_prices = np.array([100.0, 100.5, 101.0, 100.5, 100.0], dtype=np.float32)
            dummy_volumes = np.array([10, 20, 15, 25, 30], dtype=np.int32)
            dummy_timestamps = np.arange(5, dtype=np.int64)
            
            # This will trigger JIT compilation
            build_range_bars_raw_core(dummy_prices, dummy_volumes, dummy_timestamps, 0.25)
            print("🔥 Numba JIT compilation completed")
    
    @benchmark_func("create_range_bars_raw")
    def create_range_bars(self, 
                         tick_data: pd.DataFrame,
                         price_column: str = 'Close',
                         volume_column: str = 'Volume',
                         datetime_column: str = 'datetime') -> pd.DataFrame:
        """
        Create raw price range bars from tick data.
        
        Args:
            tick_data: DataFrame with tick data
            price_column: Name of price column to use
            volume_column: Name of volume column
            datetime_column: Name of datetime column
            
        Returns:
            DataFrame with range bar OHLCV data
        """
        
        print(f"🎯 Creating raw price range bars (range: {self.config.range_size})")
        print(f"📊 Input: {len(tick_data):,} ticks")
        
        # Validate input data
        required_columns = [price_column, volume_column, datetime_column]
        missing_columns = [col for col in required_columns if col not in tick_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by datetime if not already sorted
        if not tick_data[datetime_column].is_monotonic_increasing:
            print("📈 Sorting data by datetime...")
            tick_data = tick_data.sort_values(datetime_column)
        
        # Extract arrays for maximum performance
        with measure_time("extract_arrays"):
            prices = tick_data[price_column].astype(np.float32).values
            volumes = tick_data[volume_column].astype(np.int64).values
            
            # Convert datetime to numeric for performance (nanoseconds since epoch)
            timestamps = tick_data[datetime_column].astype(np.int64).values
        
        # Apply minimum volume filter if specified
        if self.config.min_volume > 0:
            with measure_time("volume_filter"):
                mask = volumes >= self.config.min_volume
                prices = prices[mask]
                volumes = volumes[mask]
                timestamps = timestamps[mask]
                print(f"📊 After volume filter: {len(prices):,} ticks")
        
        # Build range bars using optimized core algorithm
        with measure_time("build_range_bars", 
                         ticks_processed=len(prices), 
                         range_size=self.config.range_size):
            
            (bar_opens, bar_highs, bar_lows, bar_closes, 
             bar_volumes, bar_start_times, bar_end_times) = build_range_bars_raw_core(
                prices, volumes, timestamps, self.config.range_size
            )
        
        # Convert back to DataFrame with memory management
        with measure_time("create_dataframe", bars_created=len(bar_opens)):
            monitor_memory("Before DataFrame creation")
            
            # Create datetime columns (bar_start_times represents START of bar)
            start_timestamps = pd.to_datetime(bar_start_times)
            end_timestamps = pd.to_datetime(bar_end_times)
            
            # Fix duplicate timestamps if requested
            if self.config.fix_duplicate_timestamps:
                start_timestamps = fix_duplicate_timestamps_fast(pd.Series(start_timestamps))
            
            range_bars = pd.DataFrame({
                'Open': bar_opens.astype(np.float32),
                'High': bar_highs.astype(np.float32),
                'Low': bar_lows.astype(np.float32), 
                'Close': bar_closes.astype(np.float32),
                'Volume': bar_volumes.astype(np.int32),
                'timestamp': start_timestamps,  # Bar START time
                'EndDateTime': end_timestamps,
                'RangeSize': self.config.range_size,
                'TickCount': 1  # Will be calculated properly in post-processing
            })
            
            # Clean up intermediate arrays
            del bar_opens, bar_highs, bar_lows, bar_closes, bar_volumes
            del bar_start_times, bar_end_times, start_timestamps, end_timestamps
            gc.collect()
            
            monitor_memory("After DataFrame creation")
        
        # Calculate additional metrics
        with measure_time("calculate_metrics"):
            range_bars['Range'] = range_bars['High'] - range_bars['Low']
            range_bars['MidPrice'] = (range_bars['High'] + range_bars['Low']) / 2.0
            range_bars['TypicalPrice'] = (range_bars['High'] + range_bars['Low'] + range_bars['Close']) / 3.0
            range_bars['Duration'] = (range_bars['EndDateTime'] - range_bars['timestamp']).dt.total_seconds()
            
            # VWAP calculation  
            range_bars['VWAP'] = (
                (range_bars['High'] + range_bars['Low'] + range_bars['Close']) / 3.0 * 
                range_bars['Volume']
            ).cumsum() / range_bars['Volume'].cumsum()
        
        # Update statistics
        self.stats.update({
            'total_ticks_processed': len(tick_data),
            'total_bars_created': len(range_bars),
            'bars_per_tick_ratio': len(range_bars) / len(tick_data),
            'avg_ticks_per_bar': len(tick_data) / len(range_bars) if len(range_bars) > 0 else 0,
            'avg_volume_per_bar': range_bars['Volume'].mean() if len(range_bars) > 0 else 0
        })
        
        print(f"✅ Range bars created: {len(range_bars):,}")
        print(f"📈 Compression ratio: {len(tick_data)/len(range_bars):.1f}:1")
        print(f"📊 Avg volume per bar: {self.stats['avg_volume_per_bar']:,.0f}")
        
        monitor_memory("Completed range bar creation")
        gc.collect()
        
        return range_bars
    
    def create_multiple_ranges(self, 
                              tick_data: pd.DataFrame,
                              range_sizes: List[float],
                              parallel: bool = True) -> Dict[float, pd.DataFrame]:
        """
        Create range bars for multiple range sizes efficiently.
        
        Args:
            tick_data: DataFrame with tick data
            range_sizes: List of range sizes to process
            parallel: Use parallel processing
            
        Returns:
            Dictionary mapping range_size to DataFrame
        """
        print(f"🎯 Creating range bars for {len(range_sizes)} different ranges")
        print(f"📊 Range sizes: {range_sizes}")
        
        results = {}
        
        if parallel and len(range_sizes) > 1:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp
            
            with ProcessPoolExecutor(max_workers=min(len(range_sizes), mp.cpu_count())) as executor:
                futures = {}
                
                for range_size in range_sizes:
                    config = RangeBarConfig(range_size=range_size, 
                                          min_volume=self.config.min_volume)
                    generator = RawPriceRangeBars(config)
                    future = executor.submit(generator.create_range_bars, tick_data.copy())
                    futures[range_size] = future
                
                # Collect results
                for range_size, future in futures.items():
                    results[range_size] = future.result()
        else:
            # Sequential processing
            for range_size in range_sizes:
                config = RangeBarConfig(range_size=range_size,
                                      min_volume=self.config.min_volume)
                generator = RawPriceRangeBars(config)
                results[range_size] = generator.create_range_bars(tick_data)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()

# High-level convenience functions
def create_raw_range_bars(tick_data: pd.DataFrame, 
                         range_size: float,
                         min_volume: int = 0,
                         price_column: str = 'Close') -> pd.DataFrame:
    """
    Convenience function to create raw price range bars.
    
    Args:
        tick_data: DataFrame with tick data
        range_size: Fixed range size for bars
        min_volume: Minimum volume threshold  
        price_column: Price column to use
        
    Returns:
        DataFrame with range bar data
    """
    config = RangeBarConfig(range_size=range_size, min_volume=min_volume)
    generator = RawPriceRangeBars(config)
    return generator.create_range_bars(tick_data, price_column=price_column)

def batch_create_range_bars(tick_data: pd.DataFrame,
                           range_sizes: List[float],
                           output_dir: Optional[str] = None) -> Dict[float, pd.DataFrame]:
    """
    Create multiple range bar datasets and optionally save to files.
    
    Args:
        tick_data: DataFrame with tick data  
        range_sizes: List of range sizes
        output_dir: Directory to save parquet files (optional)
        
    Returns:
        Dictionary of range bar DataFrames
    """
    config = RangeBarConfig(range_size=1.0)  # Will be overridden
    generator = RawPriceRangeBars(config)
    
    results = generator.create_multiple_ranges(tick_data, range_sizes, parallel=True)
    
    if output_dir:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for range_size, df in results.items():
            filename = f"range_bars_{range_size}.parquet"
            filepath = output_path / filename
            df.to_parquet(filepath, compression='snappy')
            print(f"💾 Saved: {filepath}")
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic data
    print("🧪 Testing Raw Price Range Bars...")
    
    # Create synthetic tick data
    np.random.seed(42)
    n_ticks = 1_000_000
    
    base_price = 4300.0
    price_changes = np.random.normal(0, 0.25, n_ticks).cumsum()
    prices = base_price + price_changes
    volumes = np.random.randint(1, 1000, n_ticks)
    datetimes = pd.date_range('2021-01-01 09:30:00', periods=n_ticks, freq='1s')
    
    tick_data = pd.DataFrame({
        'datetime': datetimes,
        'Close': prices,
        'Volume': volumes
    })
    
    print(f"📊 Created synthetic data: {len(tick_data):,} ticks")
    
    # Test single range
    range_bars = create_raw_range_bars(tick_data, range_size=1.0, min_volume=10)
    print(f"✅ Single range test completed: {len(range_bars)} bars")
    
    # Test multiple ranges
    range_sizes = [0.25, 0.5, 1.0, 2.0]
    multiple_results = batch_create_range_bars(tick_data, range_sizes)
    
    print("📈 Multiple range results:")
    for size, df in multiple_results.items():
        compression = len(tick_data) / len(df)
        print(f"   Range {size}: {len(df):,} bars (compression: {compression:.1f}:1)")
    
    from performance import print_summary
    print_summary()