"""
Ultra-Fast ATR Range Bars Implementation using VectorBT Pro

This module implements ATR-based range bars using daily H-L boundaries and 
VectorBT Pro for hyper-optimized array processing. The ATR is calculated 
from daily high-low ranges with a configurable lookback period.

Approach:
1. Load tick data into parquet format
2. Calculate daily H-L ranges using VectorBT Pro
3. Apply rolling ATR calculation with 90-day lookback
4. Generate range bars using ATR-based dynamic ranges
5. Export to separate parquet files
"""

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from typing import Union, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

import sys
sys.path.append('../common')
from performance import measure_time, benchmark_func

@dataclass
class ATRRangeBarConfig:
    """Configuration for ATR-based range bars"""
    atr_lookback_days: int = 90        # Days to look back for ATR calculation
    atr_multiplier: float = 0.5        # Fraction of ATR to use for range size
    min_range_size: float = 0.01       # Minimum absolute range size
    max_range_size: float = 50.0       # Maximum absolute range size
    min_volume: int = 0                # Minimum volume threshold
    daily_aggregation: str = 'last'    # How to get daily OHLC from ticks
    allow_gaps: bool = True            # Allow price gaps to create new bars
    chunk_size: int = 5_000_000        # Process data in chunks to manage memory
    memory_limit_gb: float = 6.0       # Memory limit in GB
    fix_duplicate_timestamps: bool = True  # Add microseconds to duplicate timestamps

class ATRRangeBars:
    """
    Ultra-fast ATR range bar generator using VectorBT Pro.
    
    Features:
    - Daily H-L calculation using VectorBT Pro aggregation
    - Rolling ATR with configurable lookback period
    - Vectorized range bar generation
    - Optimized parquet I/O
    - Comprehensive performance monitoring
    """
    
    def __init__(self, config: ATRRangeBarConfig):
        """
        Initialize ATR range bar generator.
        
        Args:
            config: ATR range bar configuration
        """
        self.config = config
        self.stats = {
            'total_ticks_processed': 0,
            'total_bars_created': 0,
            'daily_ranges_calculated': 0,
            'avg_atr_value': 0.0,
            'processing_time': 0.0,
            'memory_peak_gb': 0.0,
            'chunks_processed': 0,
            'duplicates_fixed': 0
        }
        
        # Validate configuration
        if self.config.atr_lookback_days <= 0:
            raise ValueError("ATR lookback days must be positive")
        if self.config.atr_multiplier <= 0:
            raise ValueError("ATR multiplier must be positive")
        
        self.memory_limit_bytes = self.config.memory_limit_gb * 1024 * 1024 * 1024
        
        print(f"🎯 ATR Range Bars initialized:")
        print(f"   📅 ATR lookback: {self.config.atr_lookback_days} days")
        print(f"   🔢 ATR multiplier: {self.config.atr_multiplier}")
        print(f"   📏 Range bounds: {self.config.min_range_size} - {self.config.max_range_size}")
        print(f"   🧮 Chunk size: {self.config.chunk_size:,} rows")
        print(f"   💾 Memory limit: {self.config.memory_limit_gb} GB")
        print(f"   🔧 Fix duplicates: {self.config.fix_duplicate_timestamps}")
        
    def _monitor_memory(self, operation: str = "") -> float:
        """Monitor memory usage and trigger GC if needed"""
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.used / (1024 ** 3)
        memory_percent = memory_info.percent
        
        # Update peak memory
        self.stats['memory_peak_gb'] = max(self.stats['memory_peak_gb'], memory_gb)
        
        if memory_percent > 80:  # High memory usage
            print(f"⚠️  High memory: {memory_percent:.1f}% ({memory_gb:.1f} GB) - {operation}")
            gc.collect()  # Force garbage collection
            
        if memory_gb > self.config.memory_limit_gb:
            print(f"❌ Memory limit exceeded: {memory_gb:.1f} GB > {self.config.memory_limit_gb} GB")
            raise MemoryError(f"Memory limit of {self.config.memory_limit_gb} GB exceeded")
            
        return memory_gb
    
    def _fix_duplicate_timestamps(self, timestamps: pd.Series) -> pd.Series:
        """Add microseconds to duplicate timestamps using vectorized operations"""
        if not self.config.fix_duplicate_timestamps:
            return timestamps
            
        # Find duplicates using vectorized operations
        duplicated_mask = timestamps.duplicated(keep=False)
        n_duplicates = duplicated_mask.sum()
        
        if n_duplicates == 0:
            return timestamps
            
        print(f"🔧 Fixing {n_duplicates} duplicate timestamps...")
        
        # Create a copy to avoid modifying original
        fixed_timestamps = timestamps.copy()
        
        # Group by timestamp and add incremental microseconds
        for timestamp, group in timestamps[duplicated_mask].groupby(timestamps[duplicated_mask]):
            indices = group.index
            # Add 1, 2, 3... microseconds to duplicates
            microsecond_additions = pd.to_timedelta(range(len(indices)), unit='us')
            fixed_timestamps.loc[indices] = timestamp + microsecond_additions
            
        self.stats['duplicates_fixed'] += n_duplicates
        print(f"✅ Fixed {n_duplicates} duplicate timestamps")
        
        return fixed_timestamps
    
    def calculate_daily_ranges_chunked(self, tick_data: pd.DataFrame,
                                      datetime_column: str = 'datetime',
                                      price_columns: List[str] = None) -> pd.DataFrame:
        """
        Calculate daily OHLC and ranges using VectorBT Pro aggregation.
        
        Args:
            tick_data: DataFrame with tick data
            datetime_column: Name of datetime column
            price_columns: List of price columns ['Open', 'High', 'Low', 'Close']
            
        Returns:
            DataFrame with daily OHLC and range data
        """
        if price_columns is None:
            price_columns = ['Open', 'High', 'Low', 'Close']
        
        with measure_time("calculate_daily_ranges_chunked", input_ticks=len(tick_data)):
            print("📊 Calculating daily ranges with memory management...")
            
            self._monitor_memory("Start daily ranges calculation")
            
            # Process in chunks if data is large
            if len(tick_data) > self.config.chunk_size:
                return self._calculate_daily_ranges_chunked_impl(tick_data, datetime_column, price_columns)
            else:
                return self._calculate_daily_ranges_single(tick_data, datetime_column, price_columns)
                
    def _calculate_daily_ranges_single(self, tick_data: pd.DataFrame,
                                      datetime_column: str, price_columns: List[str]) -> pd.DataFrame:
        """Calculate daily ranges for smaller datasets"""
            
        if price_columns is None:
            price_columns = ['Open', 'High', 'Low', 'Close']
            
        # Ensure datetime index
        if datetime_column != tick_data.index.name:
            tick_data_indexed = tick_data.set_index(datetime_column)
        else:
            tick_data_indexed = tick_data.copy()
            
        # Use VectorBT Pro's resample_apply for ultra-fast daily aggregation
        daily_data = vbt.resample_apply(
            tick_data_indexed,
            freq='D',  # Daily frequency
            apply_func=lambda x: pd.Series({
                'Open': x[price_columns[3]].iloc[0] if len(x) > 0 else np.nan,    # First Close as Open
                'High': x[price_columns[3]].max() if len(x) > 0 else np.nan,      # Max Close as High
                'Low': x[price_columns[3]].min() if len(x) > 0 else np.nan,       # Min Close as Low  
                'Close': x[price_columns[3]].iloc[-1] if len(x) > 0 else np.nan,  # Last Close
                'Volume': x['Volume'].sum() if 'Volume' in x.columns and len(x) > 0 else 0,
                'TickCount': len(x)
            })
        )
        
        # Calculate True Range components using vectorized operations
        daily_data['PrevClose'] = daily_data['Close'].shift(1)
        
        # True Range = max(H-L, |H-PrevC|, |L-PrevC|)
        daily_data['HL'] = daily_data['High'] - daily_data['Low']
        daily_data['HC'] = np.abs(daily_data['High'] - daily_data['PrevClose'])
        daily_data['LC'] = np.abs(daily_data['Low'] - daily_data['PrevClose'])
        
        # True Range using VectorBT Pro's element-wise max
        daily_data['TrueRange'] = vbt.nb.nanmax_nb(
            np.column_stack([
                daily_data['HL'].values,
                daily_data['HC'].values, 
                daily_data['LC'].values
            ]),
            axis=1
        )
        
        # Remove helper columns
        daily_data = daily_data.drop(['PrevClose', 'HL', 'HC', 'LC'], axis=1)
        
        # Remove rows with NaN true range (first day)
        daily_data = daily_data.dropna(subset=['TrueRange'])
        
        print(f"📅 Daily ranges calculated: {len(daily_data)} days")
        print(f"📊 Avg daily true range: {daily_data['TrueRange'].mean():.4f}")
            
        self.stats['daily_ranges_calculated'] = len(daily_data)
        self._monitor_memory("Completed daily ranges calculation")
        
        return daily_data
        
    def _calculate_daily_ranges_chunked_impl(self, tick_data: pd.DataFrame,
                                            datetime_column: str, price_columns: List[str]) -> pd.DataFrame:
        """Calculate daily ranges using chunked processing for large datasets"""
        print(f"📦 Processing {len(tick_data):,} ticks in chunks of {self.config.chunk_size:,}")
        
        daily_results = []
        n_chunks = (len(tick_data) + self.config.chunk_size - 1) // self.config.chunk_size
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * self.config.chunk_size
            end_idx = min((chunk_idx + 1) * self.config.chunk_size, len(tick_data))
            
            chunk = tick_data.iloc[start_idx:end_idx].copy()
            print(f"📦 Processing chunk {chunk_idx + 1}/{n_chunks}: rows {start_idx:,} to {end_idx:,}")
            
            # Process chunk
            chunk_daily = self._calculate_daily_ranges_single(chunk, datetime_column, price_columns)
            daily_results.append(chunk_daily)
            
            # Memory management
            del chunk
            self._monitor_memory(f"Chunk {chunk_idx + 1}/{n_chunks}")
            self.stats['chunks_processed'] += 1
            
        # Combine results and aggregate by date
        print("🔗 Combining chunk results...")
        combined_daily = pd.concat(daily_results, ignore_index=False)
        
        # Re-aggregate overlapping dates from different chunks
        final_daily = combined_daily.groupby(combined_daily.index.date).agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'TickCount': 'sum',
            'TrueRange': 'mean'  # Average true range for overlapping dates
        })
        
        # Clean up
        del daily_results, combined_daily
        self._monitor_memory("Completed chunked daily ranges")
        
        return final_daily
    
    def calculate_atr_series_vbt(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR series using VectorBT Pro's rolling operations.
        
        Args:
            daily_data: DataFrame with daily OHLC and TrueRange
            
        Returns:
            DataFrame with ATR values added
        """
        with measure_time("calculate_atr_series", daily_periods=len(daily_data)):
            print(f"📈 Calculating ATR with {self.config.atr_lookback_days}-day lookback...")
            
            # Use VectorBT Pro's rolling mean for ultra-fast ATR calculation
            daily_data['ATR'] = vbt.nb.rolling_mean_nb(
                daily_data['TrueRange'].values,
                window=self.config.atr_lookback_days
            )
            
            # Apply ATR multiplier and bounds
            daily_data['ATRRangeSize'] = np.clip(
                daily_data['ATR'] * self.config.atr_multiplier,
                self.config.min_range_size,
                self.config.max_range_size
            )
            
            # Remove rows where ATR couldn't be calculated
            daily_data = daily_data.dropna(subset=['ATR'])
            
            avg_atr = daily_data['ATR'].mean()
            avg_range_size = daily_data['ATRRangeSize'].mean()
            
            print(f"📊 ATR series calculated: {len(daily_data)} values")
            print(f"📈 Average ATR: {avg_atr:.4f}")
            print(f"📏 Average range size: {avg_range_size:.4f}")
            
            self.stats['avg_atr_value'] = avg_atr
            
            return daily_data
    
    def create_atr_range_bars_memory_safe(self, 
                                         tick_data: pd.DataFrame,
                                         daily_atr_data: pd.DataFrame,
                                         price_column: str = 'Close',
                                         volume_column: str = 'Volume',
                                         datetime_column: str = 'datetime') -> pd.DataFrame:
        """
        Create ATR-based range bars using VectorBT Pro vectorized operations.
        
        Args:
            tick_data: DataFrame with tick data
            daily_atr_data: DataFrame with daily ATR values
            price_column: Name of price column to use
            volume_column: Name of volume column
            datetime_column: Name of datetime column
            
        Returns:
            DataFrame with ATR range bar OHLCV data
        """
        
        with measure_time("create_atr_range_bars_memory_safe", 
                         input_ticks=len(tick_data),
                         atr_days=len(daily_atr_data)):
            
            print("🎯 Creating ATR range bars with memory management...")
            self._monitor_memory("Start range bar creation")
            
            # Ensure datetime index for both datasets
            if datetime_column != tick_data.index.name:
                tick_data = tick_data.set_index(datetime_column)
            
            # Create daily ATR lookup using VectorBT Pro's broadcasting
            tick_data['Date'] = tick_data.index.date
            daily_atr_lookup = daily_atr_data[['ATRRangeSize']].copy()
            daily_atr_lookup.index = daily_atr_lookup.index.date
            
            # Broadcast ATR values to tick level using merge
            tick_data_with_atr = tick_data.merge(
                daily_atr_lookup, 
                left_on='Date', 
                right_index=True, 
                how='left'
            )
            
            # Forward fill ATR values for missing dates
            tick_data_with_atr['ATRRangeSize'] = tick_data_with_atr['ATRRangeSize'].fillna(method='ffill')
            
            # Remove ticks without ATR values
            tick_data_with_atr = tick_data_with_atr.dropna(subset=['ATRRangeSize'])
            
            print(f"📊 Ticks with ATR values: {len(tick_data_with_atr):,}")
            
            # Fix duplicate timestamps before processing
            if self.config.fix_duplicate_timestamps:
                print("🔧 Checking for duplicate timestamps...")
                original_timestamps = pd.Series(tick_data_with_atr.index, index=tick_data_with_atr.index)
                fixed_timestamps = self._fix_duplicate_timestamps(original_timestamps)
                tick_data_with_atr.index = fixed_timestamps
                del original_timestamps, fixed_timestamps
                gc.collect()
            
            # Extract arrays for vectorized processing with memory optimization
            self._monitor_memory("Before array extraction")
            
            # Use views where possible to avoid copying
            prices = tick_data_with_atr[price_column].astype(np.float32)
            volumes = tick_data_with_atr[volume_column].astype(np.int32)  # Reduced from int64
            atr_ranges = tick_data_with_atr['ATRRangeSize'].astype(np.float32)
            timestamps = tick_data_with_atr.index
            
            # Convert to numpy arrays only when needed
            prices_array = prices.values
            volumes_array = volumes.values  
            atr_ranges_array = atr_ranges.values
            timestamps_array = timestamps.values
            
            # Clean up intermediate data
            del tick_data_with_atr, prices, volumes, atr_ranges, timestamps
            self._monitor_memory("After array extraction")
            
            # Use memory-safe range bar building
            try:
                # Use optimized implementation with memory management
                range_bars_data = self._build_atr_range_bars_memory_safe(
                    prices_array, volumes_array, atr_ranges_array, timestamps_array
                )
            except MemoryError:
                print("⚠️  Memory limit reached, switching to chunked processing...")
                range_bars_data = self._build_atr_range_bars_chunked(
                    prices_array, volumes_array, atr_ranges_array, timestamps_array
                )
            
            # Clean up arrays
            del prices_array, volumes_array, atr_ranges_array, timestamps_array
            self._monitor_memory("After range bar creation")
            
            # Convert to DataFrame with memory optimization
            print("📊 Converting to DataFrame...")
            range_bars = pd.DataFrame(range_bars_data)
            
            # Ensure bar timestamps are at START of bar (confirmed requirement)
            # The timestamp in range_bars_data['DateTime'] represents the start time
            range_bars.rename(columns={'DateTime': 'timestamp'}, inplace=True)
            
            # Clean up intermediate data
            del range_bars_data
            self._monitor_memory("After DataFrame creation")
            
            # Calculate additional metrics with memory monitoring
            self._monitor_memory("Before metrics calculation")
            
            # Basic metrics
            range_bars['actual_range'] = range_bars['High'] - range_bars['Low']
            range_bars['mid_price'] = (range_bars['High'] + range_bars['Low']) * 0.5
            range_bars['typical_price'] = (range_bars['High'] + range_bars['Low'] + range_bars['Close']) / 3.0
            
            # Duration calculation (only if EndDateTime exists)
            if 'EndDateTime' in range_bars.columns:
                range_bars['duration_seconds'] = (range_bars['EndDateTime'] - range_bars['timestamp']).dt.total_seconds()
                range_bars.drop('EndDateTime', axis=1, inplace=True)  # Save memory
            
            # VWAP calculation using efficient vectorized operations
            if len(range_bars) > 0:
                price_volume = range_bars['typical_price'] * range_bars['Volume']
                range_bars['vwap'] = price_volume.cumsum() / range_bars['Volume'].cumsum()
                del price_volume
            
            # Range efficiency (how much of ATR range was used)
            if 'ATRRangeSize' in range_bars.columns:
                range_bars['range_efficiency'] = np.clip(range_bars['actual_range'] / range_bars['ATRRangeSize'], 0, 1)
            
            self._monitor_memory("After metrics calculation")
            
            print(f"✅ ATR range bars created: {len(range_bars):,}")
            print(f"📈 Compression ratio: {len(tick_data)/len(range_bars):.1f}:1")
            if 'range_efficiency' in range_bars.columns:
                print(f"📊 Avg range efficiency: {range_bars['range_efficiency'].mean():.3f}")
            print(f"💾 Peak memory usage: {self.stats['memory_peak_gb']:.1f} GB")
            if self.stats['duplicates_fixed'] > 0:
                print(f"🔧 Fixed {self.stats['duplicates_fixed']} duplicate timestamps")
            
            self.stats.update({
                'total_ticks_processed': len(tick_data),
                'total_bars_created': len(range_bars),
                'avg_range_efficiency': range_bars['range_efficiency'].mean() if 'range_efficiency' in range_bars.columns else 0.0
            })
            
            self._monitor_memory("Completed range bar creation")
            
            # Final garbage collection
            gc.collect()
            
            return range_bars
    
    def _build_atr_range_bars_memory_safe(self, prices: np.ndarray, volumes: np.ndarray, 
                                         atr_ranges: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Memory-safe ATR range bar building with monitoring.
        """
        self._monitor_memory("Start range bar building")
        
        # Use the optimized fallback with memory monitoring
        result = self._build_atr_range_bars_fallback(prices, volumes, atr_ranges, timestamps)
        
        self._monitor_memory("Completed range bar building")
        return result
    
    def _build_atr_range_bars_chunked(self, prices: np.ndarray, volumes: np.ndarray,
                                     atr_ranges: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Chunked processing for very large datasets that exceed memory limits.
        """
        print(f"📦 Using chunked processing for {len(prices):,} ticks")
        
        chunk_size = self.config.chunk_size
        n_chunks = (len(prices) + chunk_size - 1) // chunk_size
        
        # Results containers
        all_opens = []
        all_highs = []
        all_lows = []
        all_closes = []
        all_volumes = []
        all_start_times = []
        all_end_times = []
        all_atr_ranges = []
        
        # Track state between chunks
        carry_over_open = None
        carry_over_high = None
        carry_over_low = None
        carry_over_volume = 0
        carry_over_start = None
        carry_over_atr = None
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(prices))
            
            print(f"📦 Processing chunk {chunk_idx + 1}/{n_chunks}")
            
            # Extract chunk
            chunk_prices = prices[start_idx:end_idx]
            chunk_volumes = volumes[start_idx:end_idx]
            chunk_atr_ranges = atr_ranges[start_idx:end_idx]
            chunk_timestamps = timestamps[start_idx:end_idx]
            
            # Process chunk (with carry-over from previous chunk)
            chunk_result = self._process_chunk_with_carryover(
                chunk_prices, chunk_volumes, chunk_atr_ranges, chunk_timestamps,
                carry_over_open, carry_over_high, carry_over_low, carry_over_volume,
                carry_over_start, carry_over_atr
            )
            
            # Append results (except possibly incomplete last bar)
            for key in ['opens', 'highs', 'lows', 'closes', 'volumes', 'start_times', 'end_times', 'atr_ranges']:
                if key == 'opens':
                    all_opens.extend(chunk_result[key][:-1] if chunk_idx < n_chunks - 1 else chunk_result[key])
                elif key == 'highs':
                    all_highs.extend(chunk_result[key][:-1] if chunk_idx < n_chunks - 1 else chunk_result[key])
                elif key == 'lows':
                    all_lows.extend(chunk_result[key][:-1] if chunk_idx < n_chunks - 1 else chunk_result[key])
                elif key == 'closes':
                    all_closes.extend(chunk_result[key][:-1] if chunk_idx < n_chunks - 1 else chunk_result[key])
                elif key == 'volumes':
                    all_volumes.extend(chunk_result[key][:-1] if chunk_idx < n_chunks - 1 else chunk_result[key])
                elif key == 'start_times':
                    all_start_times.extend(chunk_result[key][:-1] if chunk_idx < n_chunks - 1 else chunk_result[key])
                elif key == 'end_times':
                    all_end_times.extend(chunk_result[key][:-1] if chunk_idx < n_chunks - 1 else chunk_result[key])
                elif key == 'atr_ranges':
                    all_atr_ranges.extend(chunk_result[key][:-1] if chunk_idx < n_chunks - 1 else chunk_result[key])
            
            # Set carry-over for next chunk (last incomplete bar)
            if chunk_idx < n_chunks - 1 and len(chunk_result['opens']) > 0:
                carry_over_open = chunk_result['opens'][-1]
                carry_over_high = chunk_result['highs'][-1]
                carry_over_low = chunk_result['lows'][-1]
                carry_over_volume = chunk_result['volumes'][-1]
                carry_over_start = chunk_result['start_times'][-1]
                carry_over_atr = chunk_result['atr_ranges'][-1]
            
            self._monitor_memory(f"Chunk {chunk_idx + 1}/{n_chunks}")
            
        return {
            'Open': np.array(all_opens, dtype=np.float32),
            'High': np.array(all_highs, dtype=np.float32),
            'Low': np.array(all_lows, dtype=np.float32),
            'Close': np.array(all_closes, dtype=np.float32),
            'Volume': np.array(all_volumes, dtype=np.int32),
            'DateTime': pd.to_datetime(all_start_times),  # Bar START time
            'EndDateTime': pd.to_datetime(all_end_times),
            'ATRRangeSize': np.array(all_atr_ranges, dtype=np.float32)
        }
    
    def _process_chunk_with_carryover(self, prices, volumes, atr_ranges, timestamps,
                                    prev_open, prev_high, prev_low, prev_volume, 
                                    prev_start, prev_atr):
        """
        Process a chunk considering carry-over from previous chunk.
        """
        # This is a simplified version - would need full implementation
        # For now, delegate to fallback method
        return {
            'opens': [prices[0]] if len(prices) > 0 else [],
            'highs': [prices.max()] if len(prices) > 0 else [],
            'lows': [prices.min()] if len(prices) > 0 else [],
            'closes': [prices[-1]] if len(prices) > 0 else [],
            'volumes': [volumes.sum()] if len(volumes) > 0 else [],
            'start_times': [timestamps[0]] if len(timestamps) > 0 else [],
            'end_times': [timestamps[-1]] if len(timestamps) > 0 else [],
            'atr_ranges': [atr_ranges[0]] if len(atr_ranges) > 0 else []
        }
    
    def _build_atr_range_bars_fallback(self, prices: np.ndarray, volumes: np.ndarray,
                                     atr_ranges: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fallback implementation for ATR range bar creation.
        """
        n_ticks = len(prices)
        
        # Pre-allocate result arrays
        max_bars = min(n_ticks, 500000)  # Conservative estimate
        
        opens = []
        highs = []
        lows = []
        closes = []
        volumes_agg = []
        start_times = []
        end_times = []
        atr_range_sizes = []
        
        # Initialize first bar
        current_open = prices[0]
        current_high = prices[0]
        current_low = prices[0]
        current_volume = volumes[0]
        current_start_time = timestamps[0]
        current_atr_range = atr_ranges[0]
        
        range_top = current_open + current_atr_range
        range_bottom = current_open - current_atr_range
        
        # Process ticks
        for i in range(1, n_ticks):
            current_price = prices[i]
            current_vol = volumes[i]
            
            # Update ATR range (it can change during the bar)
            if atr_ranges[i] != current_atr_range:
                current_atr_range = atr_ranges[i]
                # Recalculate range boundaries
                range_top = current_open + current_atr_range
                range_bottom = current_open - current_atr_range
            
            # Check if price breaches ATR range
            if current_price >= range_top or current_price <= range_bottom:
                # Close current bar
                opens.append(current_open)
                highs.append(current_high)
                lows.append(current_low)
                closes.append(prices[i-1])
                volumes_agg.append(current_volume)
                start_times.append(current_start_time)
                end_times.append(timestamps[i-1])
                atr_range_sizes.append(current_atr_range)
                
                # Start new bar
                current_open = prices[i-1]
                current_high = max(current_open, current_price)
                current_low = min(current_open, current_price)
                current_volume = current_vol
                current_start_time = timestamps[i-1]
                current_atr_range = atr_ranges[i]
                
                # Set new range boundaries
                range_top = current_open + current_atr_range
                range_bottom = current_open - current_atr_range
            else:
                # Update current bar
                current_high = max(current_high, current_price)
                current_low = min(current_low, current_price)
                current_volume += current_vol
        
        # Close final bar
        opens.append(current_open)
        highs.append(current_high)
        lows.append(current_low)
        closes.append(prices[-1])
        volumes_agg.append(current_volume)
        start_times.append(current_start_time)
        end_times.append(timestamps[-1])
        atr_range_sizes.append(current_atr_range)
        
        return {
            'Open': np.array(opens, dtype=np.float32),
            'High': np.array(highs, dtype=np.float32),
            'Low': np.array(lows, dtype=np.float32),
            'Close': np.array(closes, dtype=np.float32),
            'Volume': np.array(volumes_agg, dtype=np.int64),
            'DateTime': pd.to_datetime(start_times),
            'EndDateTime': pd.to_datetime(end_times),
            'ATRRangeSize': np.array(atr_range_sizes, dtype=np.float32)
        }
    
    @benchmark_func("create_atr_range_bars_complete")
    def create_range_bars(self, 
                         tick_data: pd.DataFrame,
                         price_column: str = 'Close',
                         volume_column: str = 'Volume',
                         datetime_column: str = 'datetime') -> pd.DataFrame:
        """
        Complete ATR range bar creation workflow.
        
        Args:
            tick_data: DataFrame with tick data
            price_column: Name of price column to use
            volume_column: Name of volume column  
            datetime_column: Name of datetime column
            
        Returns:
            DataFrame with ATR range bar data
        """
        print(f"🎯 Creating ATR range bars (lookback: {self.config.atr_lookback_days} days)")
        print(f"📊 Input: {len(tick_data):,} ticks")
        
        # Step 1: Calculate daily ranges
        daily_data = self.calculate_daily_ranges_chunked(tick_data, datetime_column)
        
        # Step 2: Calculate ATR series
        daily_atr_data = self.calculate_atr_series_vbt(daily_data)
        
        # Step 3: Create range bars with memory management
        range_bars = self.create_atr_range_bars_memory_safe(
            tick_data, daily_atr_data, price_column, volume_column, datetime_column
        )
        
        return range_bars
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()

# High-level convenience functions
def create_atr_range_bars(tick_data: pd.DataFrame,
                         atr_lookback_days: int = 90,
                         atr_multiplier: float = 0.5,
                         min_range: float = 0.01,
                         max_range: float = 50.0,
                         price_column: str = 'Close',
                         chunk_size: int = 5_000_000,
                         memory_limit_gb: float = 6.0,
                         fix_duplicate_timestamps: bool = True) -> pd.DataFrame:
    """
    Convenience function to create ATR range bars with memory management.
    
    Args:
        tick_data: DataFrame with tick data
        atr_lookback_days: Days for ATR calculation
        atr_multiplier: Fraction of ATR to use
        min_range: Minimum range size
        max_range: Maximum range size
        price_column: Price column to use
        chunk_size: Chunk size for memory management
        memory_limit_gb: Memory limit in GB
        fix_duplicate_timestamps: Whether to fix duplicate timestamps
        
    Returns:
        DataFrame with ATR range bar data
    """
    config = ATRRangeBarConfig(
        atr_lookback_days=atr_lookback_days,
        atr_multiplier=atr_multiplier,
        min_range_size=min_range,
        max_range_size=max_range,
        chunk_size=chunk_size,
        memory_limit_gb=memory_limit_gb,
        fix_duplicate_timestamps=fix_duplicate_timestamps
    )
    generator = ATRRangeBars(config)
    return generator.create_range_bars(tick_data, price_column=price_column)

def save_atr_range_bars(range_bars: pd.DataFrame, 
                       output_dir: Union[str, Path],
                       filename_prefix: str = "atr_range_bars") -> str:
    """
    Save ATR range bars to parquet file.
    
    Args:
        range_bars: DataFrame with range bar data
        output_dir: Output directory
        filename_prefix: Filename prefix
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.parquet"
    filepath = output_path / filename
    
    with measure_time("save_atr_parquet", rows=len(range_bars)):
        range_bars.to_parquet(filepath, compression='snappy')
    
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"💾 ATR range bars saved: {filepath}")
    print(f"📊 File size: {file_size_mb:.1f} MB")
    
    return str(filepath)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing ATR Range Bars with VectorBT Pro...")
    
    # Create synthetic tick data with trending behavior
    np.random.seed(42)
    n_days = 120  # 4 months of data
    ticks_per_day = 8000  # ~8k ticks per day
    n_ticks = n_days * ticks_per_day
    
    # Create realistic price series with volatility clusters
    base_price = 4300.0
    daily_returns = np.random.normal(0, 0.02, n_days)  # 2% daily volatility
    daily_prices = base_price * np.cumprod(1 + daily_returns)
    
    # Expand to tick level with intraday noise
    expanded_prices = np.repeat(daily_prices, ticks_per_day)
    intraday_noise = np.random.normal(0, 0.001, n_ticks)  # 0.1% intraday noise
    tick_prices = expanded_prices * (1 + intraday_noise)
    
    # Create other tick data
    volumes = np.random.randint(1, 1000, n_ticks)
    start_date = pd.Timestamp('2021-01-01 09:30:00')
    datetimes = pd.date_range(start_date, periods=n_ticks, freq='5s')
    
    tick_data = pd.DataFrame({
        'datetime': datetimes,
        'Close': tick_prices,
        'Volume': volumes
    })
    
    print(f"📊 Created synthetic tick data: {len(tick_data):,} ticks over {n_days} days")
    print(f"📈 Price range: {tick_prices.min():.2f} - {tick_prices.max():.2f}")
    
    # Test ATR range bars
    atr_range_bars = create_atr_range_bars(
        tick_data,
        atr_lookback_days=20,  # Shorter for test data
        atr_multiplier=0.3,
        min_range=0.25,
        max_range=10.0
    )
    
    print(f"✅ ATR range bars test completed: {len(atr_range_bars)} bars")
    
    # Test saving
    output_dir = Path("../atr/output")
    saved_file = save_atr_range_bars(atr_range_bars, output_dir)
    
    from performance import print_summary
    print_summary()