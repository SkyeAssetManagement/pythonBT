#!/usr/bin/env python3
"""
Quick speed test: Single default vs 40 combinations since 2000
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data.parquet_converter import ParquetConverter
from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy

def main():
    print("SPEED TEST: Single Default vs 40 Combinations since 2000")
    print("=" * 60)
    
    # Load data since 2000
    print("Loading ES data since 2000-01-01...")
    start_time = time.time()
    
    try:
        parquet_converter = ParquetConverter()
        data = parquet_converter.load_or_convert("ES", "1m", "diffAdjusted")
        data = parquet_converter.filter_data_by_date(data, "2000-01-01", None)
        
        load_time = time.time() - start_time
        n_bars = len(data['close'])
        print(f"SUCCESS: Loaded {n_bars:,} bars in {load_time:.2f}s")
        
    except Exception as e:
        print(f"ERROR: Data loading failed: {e}")
        return
    
    # Setup
    strategy = TimeWindowVectorizedStrategy() 
    config = {'initial_capital': 100000, 'commission': 0.001, 'slippage': 0.0001}
    
    # Test 1: Single Default
    print(f"\nTest 1: Single Default Run (--useDefaults)")
    try:
        start_time = time.time()
        pf_single = strategy.run_vectorized_backtest(data, config, use_defaults_only=True)
        single_time = time.time() - start_time
        single_trades = len(pf_single.trades.records_readable)
        single_columns = pf_single.wrapper.shape[1] if hasattr(pf_single.wrapper, 'shape') else 1
        print(f"SUCCESS: {single_time:.2f}s | {single_trades} trades | {single_columns} columns")
        
    except Exception as e:
        print(f"ERROR: Single run failed: {e}")
        return
    
    # Test 2: Full Optimization  
    print(f"\nTest 2: Full Optimization (40 combinations)")
    try:
        start_time = time.time()
        pf_full = strategy.run_vectorized_backtest(data, config, use_defaults_only=False)
        full_time = time.time() - start_time
        full_trades = len(pf_full.trades.records_readable)
        full_columns = pf_full.wrapper.shape[1] if hasattr(pf_full.wrapper, 'shape') else 1
        print(f"SUCCESS: {full_time:.2f}s | {full_trades} trades | {full_columns} columns")
        
    except Exception as e:
        print(f"ERROR: Full optimization failed: {e}")
        return
    
    # Results Summary
    ratio = full_time / single_time
    print(f"\nSPEED TEST RESULTS:")
    print("=" * 40)
    print(f"Data size: {n_bars:,} bars (since 2000)")
    print(f"Single run: {single_time:.2f}s ({n_bars/single_time:,.0f} bars/sec)")
    print(f"Full optimization: {full_time:.2f}s ({n_bars/full_time:,.0f} bars/sec)")
    print(f"Speed ratio: {ratio:.1f}x slower for 40x more work")
    print(f"Vectorization efficiency: {40/ratio:.1f}x better than linear")
    print(f"Trade scaling: {single_trades} -> {full_trades} ({full_trades/single_trades:.1f}x)")
    print(f"Column scaling: {single_columns} -> {full_columns} ({full_columns/single_columns:.1f}x)")

if __name__ == "__main__":
    main()