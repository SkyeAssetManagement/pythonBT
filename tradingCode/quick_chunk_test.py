#!/usr/bin/env python3
"""
Quick chunk size test with small parameter set
"""

import sys
import time
import pandas as pd
import numpy as np
import logging
import gc
import vectorbtpro as vbt

sys.path.append('src')
from src.data.parquet_converter import ParquetConverter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test(chunk_size=25):
    """Quick test with small chunk size"""
    
    logger.info(f"=== QUICK TEST: {chunk_size} COMBINATIONS ===")
    
    # Load smaller data sample (last 100k bars)
    logger.info("Loading test data...")
    start_time = time.time()
    
    parquet_converter = ParquetConverter()
    data = parquet_converter.load_or_convert("GC", "1m", "diffAdjusted")
    data = parquet_converter.filter_data_by_date(data, "2024-01-01", "2024-12-31")  # Just 2024 data
    
    close_series = pd.Series(
        data['close'], 
        index=pd.to_datetime(data['datetime_ns']),
        name='close'
    )
    
    # Take last 100k bars for speed
    if len(close_series) > 100000:
        close_series = close_series.tail(100000)
    
    load_time = time.time() - start_time
    logger.info(f"Loaded {len(close_series):,} bars in {load_time:.2f}s")
    
    # Generate test parameter combinations
    ma1_values = np.arange(10, 60, 10)  # [10, 20, 30, 40, 50]
    ma2_values = np.arange(100, 600, 100)  # [100, 200, 300, 400, 500]
    
    param_combinations = []
    for ma1 in ma1_values:
        for ma2 in ma2_values:
            if ma2 > ma1 and len(param_combinations) < chunk_size:
                param_combinations.append([ma1, ma2])
    
    param_combinations = np.array(param_combinations)
    actual_chunk_size = len(param_combinations)
    
    logger.info(f"Testing {actual_chunk_size} combinations")
    
    # Test processing
    test_start = time.time()
    
    try:
        # Generate signals
        entries_dict = {}
        exits_dict = {}
        
        for i, combo in enumerate(param_combinations):
            fast_period = int(combo[0])
            slow_period = int(combo[1])
            
            # Calculate MAs
            fast_ma = vbt.MA.run(close_series, fast_period).ma
            slow_ma = vbt.MA.run(close_series, slow_period).ma
            
            # Generate signals
            entries = fast_ma.vbt.crossed_above(slow_ma)
            exits = fast_ma.vbt.crossed_below(slow_ma)
            
            entries_dict[i] = entries
            exits_dict[i] = exits
        
        # Convert to DataFrames
        entries_df = pd.DataFrame(entries_dict, index=close_series.index)
        exits_df = pd.DataFrame(exits_dict, index=close_series.index)
        
        # Create portfolio
        close_df = pd.concat([close_series] * actual_chunk_size, axis=1, keys=range(actual_chunk_size))
        
        pf = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries_df,
            exits=exits_df,
            size=np.inf,
            init_cash=100000,
            fees=0.001,
            freq='1min'
        )
        
        # Extract metrics
        total_returns = pf.total_return * 100
        sharpe_ratios = pf.sharpe_ratio
        max_drawdowns = pf.max_drawdown * 100
        
        total_time = time.time() - test_start
        rate = actual_chunk_size / total_time
        
        logger.info("=== RESULTS ===")
        logger.info(f"[OK] SUCCESS: {actual_chunk_size} combinations in {total_time:.2f}s")
        logger.info(f"Rate: {rate:.2f} combinations/second")
        
        # Memory cleanup
        del pf, entries_df, exits_df, close_df
        gc.collect()
        
        return True, total_time, actual_chunk_size, rate
        
    except Exception as e:
        logger.error(f"[X] FAILED: {e}")
        gc.collect()
        return False, 0, actual_chunk_size, 0

def main():
    """Main test"""
    
    print("=" * 80)
    print("QUICK CHUNK SIZE TEST")
    print("=" * 80)
    
    # Test different chunk sizes
    test_sizes = [25, 50, 100]
    
    for chunk_size in test_sizes:
        success, time_taken, actual_size, rate = quick_test(chunk_size)
        
        if success:
            print(f"\nChunk {actual_size}: {time_taken:.2f}s, {rate:.2f} comb/s")
            
            # Estimate for full dataset
            full_data_multiplier = 6522902 / 100000  # Full dataset vs test dataset
            adjusted_rate = rate / full_data_multiplier
            
            full_combinations = 9540
            full_time = full_combinations / adjusted_rate
            
            print(f"  Full dataset estimate: {full_time:.0f}s ({full_time/60:.1f} min)")
            
            # Recommend chunk size
            if adjusted_rate > 1:
                recommended = min(500, max(100, int(300 / adjusted_rate) * 100))
            else:
                recommended = 50
            
            print(f"  Recommended chunk for full: {recommended}")
        else:
            print(f"\nChunk {chunk_size}: FAILED")

if __name__ == "__main__":
    main()