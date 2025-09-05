# tradingCode/performance_diagnostic.py
# Performance regression diagnostic tool for TimeWindowVectorizedStrategy
# Analyzes execution bottlenecks to identify performance degradation causes

#!/usr/bin/env python3
"""
Performance diagnostic tool to identify regression in single-run execution.

This tool profiles the current implementation to find bottlenecks that are causing
the significant performance degradation (from ~200K+ bars/sec to ~82K bars/sec).
"""

import cProfile
import pstats
import io
import time
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project paths for imports
sys.path.insert(0, os.path.dirname(__file__))

from src.data.parquet_converter import ParquetConverter
from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy

def create_test_dataset(n_bars=100000):
    """
    Create standardized test dataset for consistent performance measurement.
    
    Args:
        n_bars: Number of bars to generate (default 100K for quick profiling)
    
    Returns:
        dict: Standard OHLCV data structure
    """
    print(f"Creating test dataset with {n_bars:,} bars...")
    
    # Generate realistic timestamp sequence (1-minute bars)
    start_time = pd.Timestamp('2020-01-01 09:30:00')
    timestamps = []
    current_time = start_time
    
    # Create trading day structure (395 minutes per day)
    minutes_per_day = 395
    for bar in range(n_bars):
        # Convert to UTC nanoseconds (subtract 5 hours for EST->UTC)
        utc_time = current_time - pd.Timedelta(hours=5)
        timestamps.append(int(utc_time.value))
        
        # Advance to next minute, skip weekends/gaps
        current_time += pd.Timedelta(minutes=1)
        if (bar + 1) % minutes_per_day == 0:
            current_time += pd.Timedelta(days=1)
    
    # Generate realistic price data with proper statistical properties
    np.random.seed(42)  # Deterministic for consistent benchmarking
    base_price = 4200.0
    
    # Use realistic price movement (mean-reverting with volatility)
    price_changes = np.random.normal(0, 0.5, n_bars)
    close_prices = base_price + np.cumsum(price_changes)
    
    # Generate OHLC from close prices
    high_prices = close_prices + np.abs(np.random.normal(0, 0.25, n_bars))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.25, n_bars))
    open_prices = np.roll(close_prices, 1) + np.random.normal(0, 0.1, n_bars)
    open_prices[0] = close_prices[0]  # Fix first bar
    
    volumes = np.random.randint(100, 1000, n_bars)
    
    return {
        'datetime': np.array(timestamps),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }

def profile_single_run_performance():
    """
    Profile the current single-run implementation to identify bottlenecks.
    
    This function runs cProfile on the strategy execution to pinpoint exactly
    where time is being spent and identify performance regressions.
    """
    print("=" * 60)
    print("PERFORMANCE DIAGNOSTIC: Single Run Profiling")
    print("=" * 60)
    
    # Create test data (moderate size for detailed profiling)
    test_data = create_test_dataset(n_bars=50000)  # 50K bars for quick analysis
    
    # Setup strategy and config
    strategy = TimeWindowVectorizedStrategy()
    config = {'initial_capital': 100000, 'commission': 0.001, 'slippage': 0.0001}
    
    print("Starting performance profiling...")
    
    # Profile the execution
    profiler = cProfile.Profile()
    
    # Run the profiled execution
    start_time = time.time()
    profiler.enable()
    
    # This is the code we're profiling for bottlenecks
    pf = strategy.run_vectorized_backtest(test_data, config, use_defaults_only=True)
    
    profiler.disable()
    execution_time = time.time() - start_time
    
    # Analyze results
    trades = len(pf.trades.records_readable)
    bars_per_sec = len(test_data['close']) / execution_time
    
    print(f"\nExecution Results:")
    print(f"  Dataset size: {len(test_data['close']):,} bars")
    print(f"  Execution time: {execution_time:.3f} seconds")
    print(f"  Throughput: {bars_per_sec:,.0f} bars/second")
    print(f"  Total trades: {trades}")
    
    # Generate profiling report
    print(f"\n" + "=" * 60)
    print("TOP PERFORMANCE BOTTLENECKS")
    print("=" * 60)
    
    # Create string buffer for profile output
    profile_output = io.StringIO()
    stats = pstats.Stats(profiler, stream=profile_output)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions by cumulative time
    
    profile_lines = profile_output.getvalue().split('\n')
    
    # Parse and display key bottlenecks
    print("Top functions by cumulative time spent:")
    print("-" * 60)
    
    # Find the main data section (skip headers)
    data_started = False
    for line in profile_lines:
        if 'cumulative' in line and 'filename:lineno(function)' in line:
            data_started = True
            continue
        
        if data_started and line.strip():
            if len(line.split()) >= 6:  # Valid data line
                print(line)
            
            # Stop after reasonable number of entries
            if 'run_vectorized_backtest' in line or '_generate_gradual_signals' in line:
                print(line)
    
    # Performance analysis
    print(f"\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    expected_throughput = 200000  # Expected ~200K bars/sec based on earlier results
    performance_ratio = bars_per_sec / expected_throughput
    
    print(f"Current throughput: {bars_per_sec:,.0f} bars/sec")
    print(f"Expected throughput: {expected_throughput:,.0f} bars/sec")
    print(f"Performance ratio: {performance_ratio:.2f}x")
    
    if performance_ratio < 0.7:
        print(f"CRITICAL: Performance regression detected!")
        print(f"   Current performance is {(1-performance_ratio)*100:.1f}% slower than expected")
    elif performance_ratio < 0.9:
        print(f"WARNING: Performance degradation detected")
        print(f"   Current performance is {(1-performance_ratio)*100:.1f}% slower than expected")
    else:
        print(f"Performance within acceptable range")
    
    return {
        'execution_time': execution_time,
        'throughput': bars_per_sec,
        'trades': trades,
        'performance_ratio': performance_ratio
    }

def compare_method_performance():
    """
    Compare performance of different strategy execution paths to identify regressions.
    
    This specifically tests if the gradual signal generation is causing overhead
    even in single-run mode.
    """
    print(f"\n" + "=" * 60)
    print("METHOD PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Create smaller test dataset for quick comparison
    test_data = create_test_dataset(n_bars=10000)
    strategy = TimeWindowVectorizedStrategy()
    config = {'initial_capital': 100000, 'commission': 0.001, 'slippage': 0.0001}
    
    print("Testing different execution paths:")
    
    # Test 1: Current single run (use_defaults_only=True)
    print("\n1. Current single run (use_defaults_only=True):")
    start_time = time.time()
    pf1 = strategy.run_vectorized_backtest(test_data, config, use_defaults_only=True)
    time1 = time.time() - start_time
    
    trades1 = len(pf1.trades.records_readable)
    throughput1 = len(test_data['close']) / time1
    print(f"   Time: {time1:.3f}s | Throughput: {throughput1:,.0f} bars/sec | Trades: {trades1}")
    
    # Test 2: Direct gradual backtest method
    print("\n2. Direct gradual backtest method:")
    start_time = time.time()
    pf2 = strategy.run_gradual_backtest(test_data, config, use_defaults_only=True)
    time2 = time.time() - start_time
    
    trades2 = len(pf2.trades.records_readable)
    throughput2 = len(test_data['close']) / time2
    print(f"   Time: {time2:.3f}s | Throughput: {throughput2:,.0f} bars/sec | Trades: {trades2}")
    
    # Analysis
    print(f"\nComparison Analysis:")
    if abs(time1 - time2) < 0.001:
        print("Methods have similar performance (both using same code path)")
    else:
        ratio = time1 / time2 if time2 > 0 else float('inf')
        if ratio > 1.1:
            print(f"WARNING: Current single run is {ratio:.1f}x slower than direct gradual method")
        elif ratio < 0.9:
            print(f"SUCCESS: Current single run is {1/ratio:.1f}x faster than direct gradual method")
        else:
            print(f"SUCCESS: Performance difference minimal ({ratio:.2f}x)")

def main():
    """
    Main diagnostic routine following CLAUDE.md protocols.
    
    Executes comprehensive performance analysis to identify and diagnose
    performance regressions in the TimeWindowVectorizedStrategy implementation.
    """
    try:
        # Step 1: Profile current implementation
        results = profile_single_run_performance()
        
        # Step 2: Compare execution methods
        compare_method_performance()
        
        # Step 3: Generate recommendations
        print(f"\n" + "=" * 60)
        print("DIAGNOSTIC RECOMMENDATIONS")
        print("=" * 60)
        
        if results['performance_ratio'] < 0.7:
            print("CRITICAL PERFORMANCE REGRESSION DETECTED")
            print("\nLikely causes:")
            print("1. Inefficient gradual signal generation in single-run mode")
            print("2. Unnecessary multi-column operations for single combination")
            print("3. Memory allocation overhead from large array operations")
            print("4. VectorBT portfolio creation inefficiencies")
            
            print("\nRecommended fixes:")
            print("1. Create separate optimized single-run path")
            print("2. Bypass multi-column operations when use_defaults_only=True")
            print("3. Use single-column signals for single combination")
            print("4. Implement direct entry/exit signals without gradual spread")
        
        return results
        
    except Exception as e:
        print(f"ERROR: Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()