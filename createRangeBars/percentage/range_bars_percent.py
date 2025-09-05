"""
Ultra-Fast Percentage Range Bars Implementation

This module implements percentage-based range bars where the range size is 
dynamically calculated as a percentage of the current price level. Uses 
hyper-optimized algorithms with Numba JIT compilation for maximum performance.

Range = Current Price × Percentage / 100

Features adaptive range sizing that adjusts to different price levels automatically.
"""

import numpy as np
import pandas as pd
import numba as nb
from numba import jit, prange
from typing import Union, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from performance import measure_time, benchmark_func

@dataclass
class PercentageRangeBarConfig:
    """Configuration for percentage-based range bars"""
    percentage: float              # Percentage for range calculation (e.g., 0.1 = 0.1%)
    min_range_size: float = 0.01   # Minimum absolute range size
    max_range_size: float = 10.0   # Maximum absolute range size
    price_reference: str = 'close' # Price reference for % calculation ('open', 'close', 'mid')
    min_volume: int = 0            # Minimum volume threshold
    allow_gaps: bool = True        # Allow price gaps to create new bars

@jit(nopython=True, cache=True, parallel=False)
def build_percentage_range_bars_core(prices: np.ndarray, 
                                    volumes: np.ndarray,
                                    timestamps: np.ndarray,
                                    percentage: float,
                                    min_range: float,
                                    max_range: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                             np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
    """
    Ultra-fast core percentage range bar building algorithm using Numba JIT.
    
    The range size is calculated dynamically as: range = price * percentage / 100
    
    Args:
        prices: Array of tick prices
        volumes: Array of tick volumes  
        timestamps: Array of timestamp indices
        percentage: Percentage for range calculation
        min_range: Minimum allowed range size
        max_range: Maximum allowed range size
        
    Returns:
        Tuple of arrays: (opens, highs, lows, closes, volumes, start_times, end_times, range_sizes)
    """
    n_ticks = len(prices)
    if n_ticks == 0:
        empty_float = np.array([], dtype=np.float32)
        empty_int = np.array([], dtype=np.int64)
        return empty_float, empty_float, empty_float, empty_float, empty_int, empty_int, empty_int, empty_float
    
    # Pre-allocate arrays for maximum performance
    max_bars = min(n_ticks, 1000000)  # Conservative upper bound
    
    bar_opens = np.empty(max_bars, dtype=np.float32)
    bar_highs = np.empty(max_bars, dtype=np.float32) 
    bar_lows = np.empty(max_bars, dtype=np.float32)
    bar_closes = np.empty(max_bars, dtype=np.float32)
    bar_volumes = np.empty(max_bars, dtype=np.int64)
    bar_start_times = np.empty(max_bars, dtype=np.int64)
    bar_end_times = np.empty(max_bars, dtype=np.int64)
    bar_range_sizes = np.empty(max_bars, dtype=np.float32)
    
    # Initialize first bar
    bar_count = 0
    current_open = prices[0]
    current_high = prices[0]
    current_low = prices[0]
    current_volume = volumes[0]
    current_start_time = timestamps[0]
    
    # Calculate initial range size
    current_range = max(min_range, min(max_range, current_open * percentage / 100.0))
    range_top = current_open + current_range
    range_bottom = current_open - current_range
    
    # Process each tick
    for i in range(1, n_ticks):
        current_price = prices[i]
        current_vol = volumes[i]
        
        # Check if price breaches current range
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
                bar_range_sizes[bar_count] = current_range
                bar_count += 1
            
            # Start new bar with new range calculation
            new_open = prices[i-1]  # Previous close becomes new open
            current_open = new_open
            current_high = max(new_open, current_price)
            current_low = min(new_open, current_price)
            current_volume = current_vol
            current_start_time = timestamps[i-1]
            
            # Calculate new range size based on new open price
            current_range = max(min_range, min(max_range, new_open * percentage / 100.0))
            range_top = new_open + current_range
            range_bottom = new_open - current_range
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
        bar_range_sizes[bar_count] = current_range
        bar_count += 1
    
    # Return only the filled portion of arrays
    return (bar_opens[:bar_count].copy(),
            bar_highs[:bar_count].copy(), 
            bar_lows[:bar_count].copy(),
            bar_closes[:bar_count].copy(),
            bar_volumes[:bar_count].copy(),
            bar_start_times[:bar_count].copy(),
            bar_end_times[:bar_count].copy(),
            bar_range_sizes[:bar_count].copy())

class PercentageRangeBars:
    """
    Ultra-fast percentage range bar generator optimized for massive datasets.
    
    Features:
    - Dynamic range sizing based on price percentage
    - Numba JIT compilation for core algorithms  
    - Vectorized NumPy operations
    - Chunked processing for memory efficiency
    - Comprehensive performance monitoring
    - Adaptive range bounds (min/max limits)
    """
    
    def __init__(self, config: PercentageRangeBarConfig):
        """
        Initialize percentage range bar generator.
        
        Args:
            config: Percentage range bar configuration
        """
        self.config = config
        self.stats = {
            'total_ticks_processed': 0,
            'total_bars_created': 0,
            'processing_time': 0.0,
            'avg_range_size': 0.0,
            'min_range_size': float('inf'),
            'max_range_size': 0.0
        }
        
        # Validate configuration
        if self.config.percentage <= 0:
            raise ValueError("Percentage must be positive")
        if self.config.min_range_size >= self.config.max_range_size:
            raise ValueError("min_range_size must be less than max_range_size")
        
        # Warm up Numba JIT compilation
        self._warmup_jit()
    
    def _warmup_jit(self):
        """Warm up Numba JIT compilation with small dataset"""
        with measure_time("numba_jit_warmup_percentage"):
            dummy_prices = np.array([100.0, 100.5, 101.0, 100.5, 100.0], dtype=np.float32)
            dummy_volumes = np.array([10, 20, 15, 25, 30], dtype=np.int32)
            dummy_timestamps = np.arange(5, dtype=np.int64)
            
            # This will trigger JIT compilation
            build_percentage_range_bars_core(dummy_prices, dummy_volumes, dummy_timestamps, 
                                           0.1, 0.01, 10.0)
            print("🔥 Numba JIT compilation completed (Percentage Range Bars)")
    
    @benchmark_func("create_percentage_range_bars")
    def create_range_bars(self, 
                         tick_data: pd.DataFrame,
                         price_column: str = 'Close',
                         volume_column: str = 'Volume',
                         datetime_column: str = 'datetime') -> pd.DataFrame:
        """
        Create percentage-based range bars from tick data.
        
        Args:
            tick_data: DataFrame with tick data
            price_column: Name of price column to use
            volume_column: Name of volume column
            datetime_column: Name of datetime column
            
        Returns:
            DataFrame with range bar OHLCV data
        """
        
        print(f"🎯 Creating percentage range bars ({self.config.percentage:.2f}%)")
        print(f"📊 Input: {len(tick_data):,} ticks")
        print(f"📏 Range bounds: {self.config.min_range_size} - {self.config.max_range_size}")
        
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
        with measure_time("build_percentage_range_bars", 
                         ticks_processed=len(prices), 
                         percentage=self.config.percentage):
            
            (bar_opens, bar_highs, bar_lows, bar_closes, 
             bar_volumes, bar_start_times, bar_end_times,
             bar_range_sizes) = build_percentage_range_bars_core(
                prices, volumes, timestamps, 
                self.config.percentage,
                self.config.min_range_size,
                self.config.max_range_size
            )
        
        # Convert back to DataFrame
        with measure_time("create_dataframe", bars_created=len(bar_opens)):
            range_bars = pd.DataFrame({
                'Open': bar_opens,
                'High': bar_highs,
                'Low': bar_lows, 
                'Close': bar_closes,
                'Volume': bar_volumes,
                'DateTime': pd.to_datetime(bar_start_times),
                'EndDateTime': pd.to_datetime(bar_end_times),
                'RangeSize': bar_range_sizes,
                'RangePercent': self.config.percentage,
                'TickCount': 1  # Will be calculated properly in post-processing
            })
        
        # Calculate additional metrics
        with measure_time("calculate_metrics"):
            range_bars['ActualRange'] = range_bars['High'] - range_bars['Low']
            range_bars['MidPrice'] = (range_bars['High'] + range_bars['Low']) / 2.0
            range_bars['TypicalPrice'] = (range_bars['High'] + range_bars['Low'] + range_bars['Close']) / 3.0
            range_bars['Duration'] = (range_bars['EndDateTime'] - range_bars['DateTime']).dt.total_seconds()
            
            # Range efficiency (how much of the calculated range was used)
            range_bars['RangeEfficiency'] = (range_bars['ActualRange'] / range_bars['RangeSize']).clip(0, 1)
            
            # VWAP calculation  
            range_bars['VWAP'] = (
                range_bars['TypicalPrice'] * range_bars['Volume']
            ).cumsum() / range_bars['Volume'].cumsum()
            
            # Price level (for analysis of range adaptation)
            range_bars['PriceLevel'] = range_bars['Open']
            range_bars['RangeAsPercent'] = (range_bars['RangeSize'] / range_bars['Open'] * 100)
        
        # Update statistics
        if len(bar_range_sizes) > 0:
            self.stats.update({
                'total_ticks_processed': len(tick_data),
                'total_bars_created': len(range_bars),
                'bars_per_tick_ratio': len(range_bars) / len(tick_data),
                'avg_ticks_per_bar': len(tick_data) / len(range_bars),
                'avg_volume_per_bar': range_bars['Volume'].mean(),
                'avg_range_size': float(np.mean(bar_range_sizes)),
                'min_range_size': float(np.min(bar_range_sizes)),
                'max_range_size': float(np.max(bar_range_sizes)),
                'range_size_std': float(np.std(bar_range_sizes))
            })
        
        print(f"✅ Range bars created: {len(range_bars):,}")
        print(f"📈 Compression ratio: {len(tick_data)/len(range_bars):.1f}:1")
        print(f"📊 Avg range size: {self.stats['avg_range_size']:.4f}")
        print(f"📏 Range size bounds: {self.stats['min_range_size']:.4f} - {self.stats['max_range_size']:.4f}")
        
        return range_bars
    
    def create_multiple_percentages(self, 
                                   tick_data: pd.DataFrame,
                                   percentages: List[float],
                                   parallel: bool = True) -> Dict[float, pd.DataFrame]:
        """
        Create range bars for multiple percentage values efficiently.
        
        Args:
            tick_data: DataFrame with tick data
            percentages: List of percentage values to process
            parallel: Use parallel processing
            
        Returns:
            Dictionary mapping percentage to DataFrame
        """
        print(f"🎯 Creating percentage range bars for {len(percentages)} different percentages")
        print(f"📊 Percentages: {percentages}")
        
        results = {}
        
        if parallel and len(percentages) > 1:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp
            
            with ProcessPoolExecutor(max_workers=min(len(percentages), mp.cpu_count())) as executor:
                futures = {}
                
                for percentage in percentages:
                    config = PercentageRangeBarConfig(
                        percentage=percentage,
                        min_range_size=self.config.min_range_size,
                        max_range_size=self.config.max_range_size,
                        min_volume=self.config.min_volume
                    )
                    generator = PercentageRangeBars(config)
                    future = executor.submit(generator.create_range_bars, tick_data.copy())
                    futures[percentage] = future
                
                # Collect results
                for percentage, future in futures.items():
                    results[percentage] = future.result()
        else:
            # Sequential processing
            for percentage in percentages:
                config = PercentageRangeBarConfig(
                    percentage=percentage,
                    min_range_size=self.config.min_range_size,
                    max_range_size=self.config.max_range_size,
                    min_volume=self.config.min_volume
                )
                generator = PercentageRangeBars(config)
                results[percentage] = generator.create_range_bars(tick_data)
        
        return results
    
    def analyze_range_adaptation(self, range_bars: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze how well the percentage ranges adapted to price levels.
        
        Args:
            range_bars: DataFrame with range bar data
            
        Returns:
            Dictionary with adaptation analysis
        """
        with measure_time("analyze_range_adaptation"):
            analysis = {
                'price_levels': {
                    'min_price': float(range_bars['Open'].min()),
                    'max_price': float(range_bars['Open'].max()),
                    'price_range': float(range_bars['Open'].max() - range_bars['Open'].min()),
                    'avg_price': float(range_bars['Open'].mean())
                },
                'range_adaptation': {
                    'min_range_size': float(range_bars['RangeSize'].min()),
                    'max_range_size': float(range_bars['RangeSize'].max()),
                    'avg_range_size': float(range_bars['RangeSize'].mean()),
                    'range_size_cv': float(range_bars['RangeSize'].std() / range_bars['RangeSize'].mean())
                },
                'efficiency_metrics': {
                    'avg_range_efficiency': float(range_bars['RangeEfficiency'].mean()),
                    'min_range_efficiency': float(range_bars['RangeEfficiency'].min()),
                    'max_range_efficiency': float(range_bars['RangeEfficiency'].max()),
                    'high_efficiency_bars': int((range_bars['RangeEfficiency'] > 0.8).sum())
                }
            }
            
            # Price level correlation
            if len(range_bars) > 10:
                correlation = np.corrcoef(range_bars['Open'], range_bars['RangeSize'])[0, 1]
                analysis['adaptation_correlation'] = float(correlation)
        
        return analysis
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()

# High-level convenience functions
def create_percentage_range_bars(tick_data: pd.DataFrame, 
                                percentage: float,
                                min_range: float = 0.01,
                                max_range: float = 10.0,
                                min_volume: int = 0,
                                price_column: str = 'Close') -> pd.DataFrame:
    """
    Convenience function to create percentage range bars.
    
    Args:
        tick_data: DataFrame with tick data
        percentage: Percentage for range calculation (e.g., 0.1 for 0.1%)
        min_range: Minimum absolute range size
        max_range: Maximum absolute range size
        min_volume: Minimum volume threshold  
        price_column: Price column to use
        
    Returns:
        DataFrame with range bar data
    """
    config = PercentageRangeBarConfig(
        percentage=percentage,
        min_range_size=min_range,
        max_range_size=max_range,
        min_volume=min_volume
    )
    generator = PercentageRangeBars(config)
    return generator.create_range_bars(tick_data, price_column=price_column)

def batch_create_percentage_range_bars(tick_data: pd.DataFrame,
                                      percentages: List[float],
                                      output_dir: Optional[str] = None) -> Dict[float, pd.DataFrame]:
    """
    Create multiple percentage range bar datasets and optionally save to files.
    
    Args:
        tick_data: DataFrame with tick data  
        percentages: List of percentage values
        output_dir: Directory to save parquet files (optional)
        
    Returns:
        Dictionary of range bar DataFrames
    """
    config = PercentageRangeBarConfig(percentage=1.0)  # Will be overridden
    generator = PercentageRangeBars(config)
    
    results = generator.create_multiple_percentages(tick_data, percentages, parallel=True)
    
    if output_dir:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for percentage, df in results.items():
            filename = f"percentage_range_bars_{percentage:.3f}.parquet"
            filepath = output_path / filename
            df.to_parquet(filepath, compression='snappy')
            print(f"💾 Saved: {filepath}")
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic data
    print("🧪 Testing Percentage Range Bars...")
    
    # Create synthetic tick data with varying price levels
    np.random.seed(42)
    n_ticks = 1_000_000
    
    # Create price series that varies in level (trending upward)
    base_price = 4300.0
    trend = np.linspace(0, 200, n_ticks)  # 200 point uptrend
    noise = np.random.normal(0, 0.5, n_ticks).cumsum()
    prices = base_price + trend + noise
    
    volumes = np.random.randint(1, 1000, n_ticks)
    datetimes = pd.date_range('2021-01-01 09:30:00', periods=n_ticks, freq='1s')
    
    tick_data = pd.DataFrame({
        'datetime': datetimes,
        'Close': prices,
        'Volume': volumes
    })
    
    print(f"📊 Created synthetic data: {len(tick_data):,} ticks")
    print(f"📈 Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Test single percentage
    range_bars = create_percentage_range_bars(
        tick_data, 
        percentage=0.1,  # 0.1%
        min_range=0.25,
        max_range=5.0,
        min_volume=10
    )
    print(f"✅ Single percentage test completed: {len(range_bars)} bars")
    
    # Analyze adaptation
    config = PercentageRangeBarConfig(percentage=0.1, min_range_size=0.25, max_range_size=5.0)
    generator = PercentageRangeBars(config)
    analysis = generator.analyze_range_adaptation(range_bars)
    
    print("\n📈 Range Adaptation Analysis:")
    print(f"   Price range: {analysis['price_levels']['min_price']:.2f} - {analysis['price_levels']['max_price']:.2f}")
    print(f"   Range size adaptation: {analysis['range_adaptation']['min_range_size']:.4f} - {analysis['range_adaptation']['max_range_size']:.4f}")
    print(f"   Avg range efficiency: {analysis['efficiency_metrics']['avg_range_efficiency']:.3f}")
    
    # Test multiple percentages
    percentages = [0.05, 0.1, 0.25, 0.5]
    multiple_results = batch_create_percentage_range_bars(tick_data, percentages)
    
    print("\n📈 Multiple percentage results:")
    for pct, df in multiple_results.items():
        compression = len(tick_data) / len(df)
        avg_range = df['RangeSize'].mean()
        print(f"   {pct:.3f}%: {len(df):,} bars (compression: {compression:.1f}:1, avg range: {avg_range:.4f})")
    
    from performance import print_summary
    print_summary()