# tradingCode/scaling_analysis.py
# Deep performance analysis tool to identify non-linear scaling bottlenecks
# Analyzes memory allocation and array operation inefficiencies causing performance degradation

#!/usr/bin/env python3
"""
Scaling analysis tool to identify non-linear performance bottlenecks.

This tool identifies specific operations in the gradual signal generation that
don't scale linearly with data size, causing the 6x performance degradation
observed between small and medium datasets.
"""

import time
import sys
import os
import tracemalloc
import numpy as np
import pandas as pd
from memory_profiler import profile

sys.path.insert(0, os.path.dirname(__file__))

from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy

def create_test_dataset(n_bars):
    """Create test dataset of specified size for scaling analysis"""
    
    # Generate timestamps efficiently
    start_time = pd.Timestamp('2020-01-01 09:30:00')
    timestamps = []
    current_time = start_time
    
    minutes_per_day = 395
    for bar in range(n_bars):
        utc_time = current_time - pd.Timedelta(hours=5)
        timestamps.append(int(utc_time.value))
        current_time += pd.Timedelta(minutes=1)
        if (bar + 1) % minutes_per_day == 0:
            current_time += pd.Timedelta(days=1)
    
    # Generate price data
    np.random.seed(42)
    base_price = 4200.0
    close_prices = base_price + np.cumsum(np.random.normal(0, 0.5, n_bars))
    high_prices = close_prices + np.abs(np.random.normal(0, 0.25, n_bars))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.25, n_bars))
    open_prices = np.roll(close_prices, 1) + np.random.normal(0, 0.1, n_bars)
    volumes = np.random.randint(100, 1000, n_bars)
    
    return {
        'datetime': np.array(timestamps),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }

def time_individual_operations(data, strategy, params):
    """
    Time individual operations in the gradual signal generation to identify bottlenecks.
    
    This breaks down _generate_gradual_signals_vectorized into components to find
    which specific operations are causing non-linear scaling issues.
    """
    print(f"Analyzing individual operations for {len(data['close']):,} bars...")
    
    timestamps = data['datetime']
    n_bars = len(timestamps)
    
    # Step 1: Timestamp conversion
    print("  Step 1: Timestamp conversion...")
    start_time = time.time()
    timestamps_sec = timestamps / 1_000_000_000
    est_timestamps_sec = timestamps_sec + (5 * 3600)
    dt_objects = pd.to_datetime(est_timestamps_sec, unit='s')
    step1_time = time.time() - start_time
    print(f"    Time: {step1_time:.4f}s ({n_bars/step1_time:,.0f} bars/sec)")
    
    # Step 2: Time extraction (FIXED - using optimized method)
    print("  Step 2: Time extraction...")
    start_time = time.time()
    # Use the actual optimized method from the strategy
    try:
        entry_signals_2d, exit_signals_2d, prices_2d = strategy._generate_gradual_signals_vectorized(data, params)
        step2_time = time.time() - start_time
        print(f"    Time: {step2_time:.4f}s ({n_bars/step2_time:,.0f} bars/sec)")
        return {'step1_time': 0, 'step2_time': step2_time, 'step3_time': 0, 'step4_time': 0, 'step5_time': 0, 'total_time': step2_time, 'n_days': len(np.unique(dates_ordinal))}
    except:
        # Fallback to old method for comparison
        minutes_of_day = dt_objects.hour * 60 + dt_objects.minute
        dates_ordinal = dt_objects.map(pd.Timestamp.toordinal)
        step2_time = time.time() - start_time
        print(f"    Time: {step2_time:.4f}s ({n_bars/step2_time:,.0f} bars/sec)")
    
    # Step 3: Entry window detection
    print("  Step 3: Entry window detection...")
    start_time = time.time()
    entry_hour, entry_minute = map(int, params['entry_time'].split(':'))
    entry_minutes = entry_hour * 60 + entry_minute
    n_entry_bars = params.get('entry_spread', 5)
    entry_start = entry_minutes + 1
    
    entry_windows = np.zeros((n_bars, n_entry_bars), dtype=bool)
    for bar_offset in range(n_entry_bars):
        target_minute = entry_start + bar_offset
        entry_windows[:, bar_offset] = (minutes_of_day == target_minute)
    step3_time = time.time() - start_time
    print(f"    Time: {step3_time:.4f}s ({n_bars/step3_time:,.0f} bars/sec)")
    
    # Step 4: Daily signal generation (SUSPECTED BOTTLENECK)
    print("  Step 4: Daily signal generation (CRITICAL)...")
    start_time = time.time()
    unique_dates = np.unique(dates_ordinal)
    n_unique_dates = len(unique_dates)
    print(f"    Processing {n_unique_dates} unique trading days...")
    
    entry_signals_2d = np.zeros((n_bars, n_entry_bars), dtype=bool)
    exit_signals_2d = np.zeros((n_bars, n_entry_bars), dtype=bool)
    
    # This is the suspected bottleneck - O(days * bars) complexity
    for date_ord in unique_dates:
        day_mask = (dates_ordinal == date_ord)
        day_indices = np.where(day_mask)[0]
        
        if len(day_indices) == 0:
            continue
        
        for col in range(n_entry_bars):
            entry_candidates = day_indices[entry_windows[day_indices, col]]
            if len(entry_candidates) > 0:
                entry_bar_idx = entry_candidates[0]
                entry_signals_2d[entry_bar_idx, col] = True
                
                exit_start_idx = entry_bar_idx + params['hold_time']
                for exit_offset in range(n_entry_bars):
                    exit_bar_idx = exit_start_idx + exit_offset
                    if exit_bar_idx < n_bars:
                        exit_signals_2d[exit_bar_idx, col] = True
    
    step4_time = time.time() - start_time
    print(f"    Time: {step4_time:.4f}s ({n_bars/step4_time:,.0f} bars/sec)")
    print(f"    Complexity: O({n_unique_dates} days * {n_bars} bars) = O({n_unique_dates * n_bars:,} operations)")
    
    # Step 5: Price array creation
    print("  Step 5: Price array creation...")
    start_time = time.time()
    signal_prices = (data['high'] + data['low']) / 2.0
    prices_2d = np.broadcast_to(signal_prices.reshape(-1, 1), (n_bars, n_entry_bars)).copy()
    step5_time = time.time() - start_time
    print(f"    Time: {step5_time:.4f}s ({n_bars/step5_time:,.0f} bars/sec)")
    
    total_time = step1_time + step2_time + step3_time + step4_time + step5_time
    print(f"\n  BOTTLENECK ANALYSIS:")
    print(f"    Step 1 (Timestamp): {step1_time/total_time*100:.1f}% of time")
    print(f"    Step 2 (Time extract): {step2_time/total_time*100:.1f}% of time")
    print(f"    Step 3 (Entry windows): {step3_time/total_time*100:.1f}% of time")
    print(f"    Step 4 (Daily signals): {step4_time/total_time*100:.1f}% of time <<< SUSPECTED BOTTLENECK")
    print(f"    Step 5 (Price arrays): {step5_time/total_time*100:.1f}% of time")
    
    return {
        'step1_time': step1_time,
        'step2_time': step2_time,
        'step3_time': step3_time,
        'step4_time': step4_time,
        'step5_time': step5_time,
        'total_time': total_time,
        'n_days': n_unique_dates
    }

def analyze_scaling_performance():
    """
    Analyze performance scaling across different dataset sizes to identify
    non-linear bottlenecks causing the performance degradation.
    """
    
    print("SCALING PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("Analyzing performance across dataset sizes to identify scaling bottlenecks...")
    
    # Test different dataset sizes
    test_sizes = [5000, 10000, 25000, 50000, 100000]  # 5K to 100K bars
    strategy = TimeWindowVectorizedStrategy()
    default_params = {
        'entry_time': '09:30',
        'direction': 'long',
        'hold_time': 60,
        'entry_spread': 5,
        'exit_spread': 5,
        'max_trades_per_day': 1
    }
    
    results = []
    
    for size in test_sizes:
        print(f"\n{'='*20} Dataset Size: {size:,} bars {'='*20}")
        
        # Create test data
        data = create_test_dataset(size)
        
        # Time individual operations
        operation_results = time_individual_operations(data, strategy, default_params)
        
        # Calculate scaling metrics
        bars_per_day = size / operation_results['n_days'] if operation_results['n_days'] > 0 else 0
        
        results.append({
            'size': size,
            'total_time': operation_results['total_time'],
            'step4_time': operation_results['step4_time'],
            'throughput': size / operation_results['total_time'],
            'step4_throughput': size / operation_results['step4_time'],
            'n_days': operation_results['n_days'],
            'bars_per_day': bars_per_day
        })
    
    # Analyze scaling patterns
    print(f"\n{'='*60}")
    print("SCALING ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"{'Size':<8} {'Total(s)':<9} {'Step4(s)':<9} {'Throughput':<11} {'Step4 T/P':<11} {'Days':<6} {'B/Day':<8}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['size']:<8,} {r['total_time']:<9.3f} {r['step4_time']:<9.3f} "
              f"{r['throughput']:<11,.0f} {r['step4_throughput']:<11,.0f} "
              f"{r['n_days']:<6} {r['bars_per_day']:<8.0f}")
    
    # Calculate scaling efficiency
    print(f"\nSCALING EFFICIENCY ANALYSIS:")
    if len(results) >= 2:
        baseline = results[0]  # 5K bars
        
        for r in results[1:]:
            size_ratio = r['size'] / baseline['size']
            time_ratio = r['total_time'] / baseline['total_time']
            step4_ratio = r['step4_time'] / baseline['step4_time']
            
            efficiency = size_ratio / time_ratio  # Should be ~1.0 for linear scaling
            step4_efficiency = size_ratio / step4_ratio
            
            print(f"  {r['size']:,} vs {baseline['size']:,} bars:")
            print(f"    Size ratio: {size_ratio:.1f}x")
            print(f"    Time ratio: {time_ratio:.1f}x")
            print(f"    Overall efficiency: {efficiency:.2f} (1.0 = perfect linear)")
            print(f"    Step4 efficiency: {step4_efficiency:.2f} (1.0 = perfect linear)")
            
            if efficiency < 0.8:
                print(f"    *** NON-LINEAR SCALING DETECTED ***")
            if step4_efficiency < 0.8:
                print(f"    *** STEP 4 IS THE BOTTLENECK ***")
    
    return results

if __name__ == "__main__":
    analyze_scaling_performance()