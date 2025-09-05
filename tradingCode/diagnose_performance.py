#!/usr/bin/env python3
"""
Diagnose why gradual entry/exit is 4x slower than expected
Step-by-step profiling to find the bottleneck
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import cProfile
import pstats
from io import StringIO

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from strategies.time_window_strategy_vectorized import TimeWindowVectorizedSingle

def create_test_data(days=252):
    """Create standardized test data"""
    
    minutes_per_day = 395
    total_bars = days * minutes_per_day
    
    start_time_est = pd.Timestamp('2000-01-01 09:25:00')
    timestamps = []
    
    current_time = start_time_est
    
    for day in range(days):
        for minute in range(minutes_per_day):
            time_est = current_time + pd.Timedelta(minutes=minute)
            utc_time = time_est - pd.Timedelta(hours=5)
            timestamp_ns = int(utc_time.value)
            timestamps.append(timestamp_ns)
        
        current_time += pd.Timedelta(days=1)
    
    timestamps = np.array(timestamps)
    
    np.random.seed(42)
    base_price = 1400.0
    price_changes = np.random.normal(0, 0.5, total_bars)
    close_prices = base_price + np.cumsum(price_changes)
    
    high_prices = close_prices + np.random.uniform(0.5, 2.0, total_bars)
    low_prices = close_prices - np.random.uniform(0.5, 2.0, total_bars)
    open_prices = close_prices + np.random.normal(0, 0.25, total_bars)
    volumes = np.random.randint(100, 1000, total_bars)
    
    return {
        'datetime': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }

def profile_signal_generation(data):
    """Profile the signal generation step"""
    
    print("STEP 1: Signal Generation Performance")
    print("-" * 50)
    
    strategy = TimeWindowVectorizedSingle()
    params = strategy.get_parameter_combinations()[0]
    
    # Time the signal generation
    print("Testing _generate_gradual_signals...")
    start_time = time.time()
    entry_signals_2d, exit_signals_2d, prices_2d = strategy._generate_gradual_signals(data, params)
    signal_time = time.time() - start_time
    
    print(f"Signal generation time: {signal_time:.4f} seconds")
    print(f"Entry signals shape: {entry_signals_2d.shape}")
    print(f"Exit signals shape: {exit_signals_2d.shape}")
    print(f"Prices shape: {prices_2d.shape}")
    print(f"Total entry signals: {np.sum(entry_signals_2d)}")
    print(f"Total exit signals: {np.sum(exit_signals_2d)}")
    
    # Check signal density
    signal_density = np.sum(entry_signals_2d) / entry_signals_2d.size
    print(f"Signal density: {signal_density:.6f} ({signal_density*100:.4f}%)")
    
    return signal_time, entry_signals_2d, exit_signals_2d, prices_2d

def profile_portfolio_creation(data, entry_signals_2d, exit_signals_2d, prices_2d):
    """Profile VectorBT portfolio creation step"""
    
    print("\nSTEP 2: VectorBT Portfolio Creation Performance")
    print("-" * 50)
    
    import vectorbtpro as vbt
    
    close_prices = data['close']
    n_bars = len(close_prices)
    n_entry_bars = entry_signals_2d.shape[1]
    
    print(f"Portfolio dimensions: {n_bars} bars x {n_entry_bars} columns")
    
    # Prepare arrays for VectorBT
    print("Preparing arrays...")
    prep_start = time.time()
    
    # Create long/short signal arrays
    long_entries = entry_signals_2d
    short_entries = np.zeros_like(entry_signals_2d)
    long_exits = exit_signals_2d
    short_exits = np.zeros_like(exit_signals_2d)
    
    prep_time = time.time() - prep_start
    print(f"Array preparation time: {prep_time:.4f} seconds")
    
    # Portfolio configuration
    portfolio_config = {
        'init_cash': 100000,
        'fees': 0.001,
        'slippage': 0.0001,
        'freq': '1min',
        'size': 1.0 / n_entry_bars,  # Fractional size
        'size_type': 'amount'
    }
    
    # Time the VectorBT portfolio creation
    print("Creating VectorBT portfolio...")
    vbt_start = time.time()
    
    pf = vbt.Portfolio.from_signals(
        close=close_prices,
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        price=prices_2d[:, :n_entry_bars] if prices_2d.shape[1] >= n_entry_bars else prices_2d,
        **portfolio_config
    )
    
    vbt_time = time.time() - vbt_start
    print(f"VectorBT portfolio creation time: {vbt_time:.4f} seconds")
    
    # Check portfolio properties
    print(f"Portfolio wrapper shape: {pf.wrapper.shape}")
    trades = pf.trades.records_readable
    print(f"Total trades generated: {len(trades)}")
    
    return prep_time, vbt_time, pf

def compare_array_shapes():
    """Compare single vs multi-column array processing"""
    
    print("\nSTEP 3: Array Shape Comparison")
    print("-" * 50)
    
    n_bars = 99540  # 1 year
    
    # Single column arrays
    single_entries = np.random.random((n_bars,)) < 0.001  # Sparse
    single_exits = np.random.random((n_bars,)) < 0.001
    single_prices = np.random.random((n_bars,)) * 1000
    
    # Multi-column arrays (5 columns)
    multi_entries = np.random.random((n_bars, 5)) < 0.0002  # Same density per element
    multi_exits = np.random.random((n_bars, 5)) < 0.0002
    multi_prices = np.random.random((n_bars, 5)) * 1000
    
    print(f"Single column arrays:")
    print(f"  Entries: {single_entries.shape}, {single_entries.nbytes:,} bytes")
    print(f"  Exits: {single_exits.shape}, {single_exits.nbytes:,} bytes")
    print(f"  Prices: {single_prices.shape}, {single_prices.nbytes:,} bytes")
    
    print(f"Multi-column arrays:")
    print(f"  Entries: {multi_entries.shape}, {multi_entries.nbytes:,} bytes")
    print(f"  Exits: {multi_exits.shape}, {multi_exits.nbytes:,} bytes")
    print(f"  Prices: {multi_prices.shape}, {multi_prices.nbytes:,} bytes")
    
    # Test array operations speed
    print("\nArray operation speed tests:")
    
    # Single column operations
    start = time.time()
    result1 = np.sum(single_entries)
    result2 = np.mean(single_prices)
    result3 = single_entries & single_exits
    single_op_time = time.time() - start
    
    # Multi-column operations
    start = time.time()
    result4 = np.sum(multi_entries)
    result5 = np.mean(multi_prices)
    result6 = multi_entries & multi_exits
    multi_op_time = time.time() - start
    
    print(f"Single column ops: {single_op_time:.6f} seconds")
    print(f"Multi-column ops: {multi_op_time:.6f} seconds")
    print(f"Speed ratio: {multi_op_time/single_op_time:.2f}x")

def profile_full_backtest():
    """Profile the complete gradual backtest with detailed timing"""
    
    print("\nSTEP 4: Full Backtest Profiling")
    print("-" * 50)
    
    data = create_test_data(days=252)
    config = {'initial_capital': 100000, 'commission': 0.001, 'slippage': 0.0001}
    
    strategy = TimeWindowVectorizedSingle()
    
    # Profile the complete run
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    pf = strategy.run_vectorized_backtest(data, config)
    total_time = time.time() - start_time
    
    profiler.disable()
    
    # Analyze profiling results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    profile_output = s.getvalue()
    
    print(f"Total backtest time: {total_time:.4f} seconds")
    print(f"Portfolio shape: {pf.wrapper.shape}")
    
    print("\nTop time-consuming functions:")
    print("=" * 60)
    
    # Parse and display key profiling info
    lines = profile_output.split('\n')
    for line in lines[5:25]:  # Skip header, show top 20
        if line.strip() and 'function calls' not in line:
            print(line)
    
    return total_time, profile_output

def diagnose_performance_bottleneck():
    """Main diagnostic function"""
    
    print("PERFORMANCE BOTTLENECK DIAGNOSIS")
    print("=" * 60)
    print("Analyzing why gradual entry/exit is 4x slower than expected")
    print("Expected: ~1.2-1.5x slower (due to 5x columns)")
    print("Actual: ~4x slower")
    print("=" * 60)
    
    # Create test data
    data = create_test_data(days=252)
    print(f"Test dataset: {len(data['close']):,} bars")
    
    # Step 1: Profile signal generation
    signal_time, entry_2d, exit_2d, prices_2d = profile_signal_generation(data)
    
    # Step 2: Profile portfolio creation
    prep_time, vbt_time, pf = profile_portfolio_creation(data, entry_2d, exit_2d, prices_2d)
    
    # Step 3: Compare array processing
    compare_array_shapes()
    
    # Step 4: Full profiling
    total_time, profile_output = profile_full_backtest()
    
    # Analysis
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    print(f"Time breakdown:")
    print(f"  Signal generation: {signal_time:.4f}s ({signal_time/total_time*100:.1f}%)")
    print(f"  Array preparation: {prep_time:.4f}s ({prep_time/total_time*100:.1f}%)")
    print(f"  VectorBT portfolio: {vbt_time:.4f}s ({vbt_time/total_time*100:.1f}%)")
    print(f"  Other overhead: {total_time - signal_time - prep_time - vbt_time:.4f}s")
    print(f"  Total time: {total_time:.4f}s")
    
    # Identify primary bottleneck
    times = {
        'Signal Generation': signal_time,
        'VectorBT Portfolio': vbt_time,
        'Array Preparation': prep_time
    }
    
    bottleneck = max(times, key=times.get)
    bottleneck_pct = times[bottleneck] / total_time * 100
    
    print(f"\nPRIMARY BOTTLENECK: {bottleneck}")
    print(f"  Time: {times[bottleneck]:.4f}s")
    print(f"  Percentage: {bottleneck_pct:.1f}% of total time")
    
    # Hypothesis about the performance issue
    print(f"\nPERFORMANCE HYPOTHESIS:")
    
    if bottleneck == 'Signal Generation':
        print("  Issue: Signal generation is not fully vectorized")
        print("  Likely cause: Python loops in _generate_gradual_signals")
        print("  Solution: Replace loops with pure NumPy operations")
        
    elif bottleneck == 'VectorBT Portfolio':
        print("  Issue: VectorBT Portfolio.from_signals scaling poorly")
        print("  Likely cause: Memory allocation or internal VectorBT overhead")
        print("  Solution: Optimize VectorBT parameters or array structure")
        
    else:
        print("  Issue: Array preparation or other overhead")
        print("  Solution: Optimize array creation and data structures")
    
    return {
        'signal_time': signal_time,
        'prep_time': prep_time,
        'vbt_time': vbt_time,
        'total_time': total_time,
        'bottleneck': bottleneck
    }

if __name__ == "__main__":
    try:
        results = diagnose_performance_bottleneck()
        print("\nDiagnosis completed successfully!")
        
    except Exception as e:
        print(f"Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()