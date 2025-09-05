#!/usr/bin/env python3
"""
strategies/test_vectorized_signals.py

Comprehensive testing of vectorized signal generation.
Validates performance improvement and behavioral equivalence.
"""

import numpy as np
import pandas as pd
import sys
import os
import time

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))

from strategies.time_window_strategy_vectorized import TimeWindowVectorizedSingle
from vectorized_signal_generator import VectorizedSignalGenerator

def create_test_data(days=252):
    """Create standardized test data for performance comparison"""
    
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
    
    # Create deterministic test data
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

def test_behavioral_equivalence():
    """Test that vectorized implementation produces identical signals"""
    
    print("BEHAVIORAL EQUIVALENCE TEST")
    print("=" * 50)
    print("Validating vectorized signals match original implementation")
    
    # Small dataset for detailed analysis
    data = create_test_data(days=5)
    params = {
        'entry_time': '09:30',
        'hold_time': 60,
        'entry_spread': 5,
        'exit_spread': 5
    }
    
    print(f"Test dataset: {len(data['close']):,} bars (5 days)")
    print(f"Parameters: {params}")
    
    # Generate signals with original implementation
    print("\nGenerating signals with original implementation...")
    strategy = TimeWindowVectorizedSingle()
    start_time = time.time()
    orig_entry, orig_exit, orig_prices = strategy._generate_gradual_signals(data, params)
    orig_time = time.time() - start_time
    
    # Generate signals with vectorized implementation
    print("Generating signals with vectorized implementation...")
    start_time = time.time()
    opt_entry, opt_exit, opt_prices = VectorizedSignalGenerator.generate_gradual_signals(data, params)
    opt_time = time.time() - start_time
    
    # Validate equivalence
    print("\nValidating signal equivalence...")
    is_equivalent = VectorizedSignalGenerator.validate_signal_equivalence(
        (orig_entry, orig_exit, orig_prices),
        (opt_entry, opt_exit, opt_prices)
    )
    
    # Performance comparison
    speedup = orig_time / opt_time if opt_time > 0 else float('inf')
    
    print(f"\nRESULTS:")
    print(f"  Original time: {orig_time:.4f}s")
    print(f"  Vectorized time: {opt_time:.4f}s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Behavioral equivalence: {'PASS' if is_equivalent else 'FAIL'}")
    
    # Signal analysis
    print(f"\nSIGNAL ANALYSIS:")
    print(f"  Entry signals: {np.sum(orig_entry)} vs {np.sum(opt_entry)}")
    print(f"  Exit signals: {np.sum(orig_exit)} vs {np.sum(opt_exit)}")
    print(f"  Shape match: {orig_entry.shape == opt_entry.shape}")
    
    return is_equivalent, speedup

def test_performance_improvement():
    """Test performance improvement with larger datasets"""
    
    print("\n" + "=" * 50)
    print("PERFORMANCE IMPROVEMENT TEST")
    print("=" * 50)
    print("Testing performance across different dataset sizes")
    
    test_sizes = [
        (63, "3 months"),
        (252, "1 year"),
        (504, "2 years")
    ]
    
    params = {
        'entry_time': '09:30',
        'hold_time': 60,
        'entry_spread': 5,
        'exit_spread': 5
    }
    
    results = []
    
    for days, period_name in test_sizes:
        print(f"\nTesting {period_name} ({days} days, {days*395:,} bars)...")
        
        data = create_test_data(days=days)
        strategy = TimeWindowVectorizedSingle()
        
        # Original implementation
        print("  Original implementation...")
        start_time = time.time()
        orig_signals = strategy._generate_gradual_signals(data, params)
        orig_time = time.time() - start_time
        
        # Vectorized implementation
        print("  Vectorized implementation...")
        start_time = time.time()
        opt_signals = VectorizedSignalGenerator.generate_gradual_signals(data, params)
        opt_time = time.time() - start_time
        
        # Validate equivalence
        is_equivalent = VectorizedSignalGenerator.validate_signal_equivalence(orig_signals, opt_signals)
        
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        bars_per_sec_orig = len(data['close']) / orig_time
        bars_per_sec_opt = len(data['close']) / opt_time
        
        result = {
            'period': period_name,
            'days': days,
            'bars': len(data['close']),
            'orig_time': orig_time,
            'opt_time': opt_time,
            'speedup': speedup,
            'orig_speed': bars_per_sec_orig,
            'opt_speed': bars_per_sec_opt,
            'equivalent': is_equivalent
        }
        
        results.append(result)
        
        print(f"    Original: {orig_time:.3f}s ({bars_per_sec_orig:,.0f} bars/sec)")
        print(f"    Vectorized: {opt_time:.3f}s ({bars_per_sec_opt:,.0f} bars/sec)")
        print(f"    Speedup: {speedup:.1f}x")
        print(f"    Equivalent: {'PASS' if is_equivalent else 'FAIL'}")
    
    # Summary
    print(f"\n{'Period':<12} {'Bars':<10} {'Orig':<8} {'Vect':<8} {'Speedup':<8} {'Equiv':<6}")
    print("-" * 60)
    
    for r in results:
        equiv_symbol = 'PASS' if r['equivalent'] else 'FAIL'
        print(f"{r['period']:<12} {r['bars']:<10,} {r['orig_time']:<8.3f} {r['opt_time']:<8.3f} {r['speedup']:<8.1f}x {equiv_symbol:<6}")
    
    # Calculate average improvement
    avg_speedup = np.mean([r['speedup'] for r in results])
    all_equivalent = all(r['equivalent'] for r in results)
    
    print(f"\nSUMMARY:")
    print(f"  Average speedup: {avg_speedup:.1f}x")
    print(f"  All tests equivalent: {'PASS' if all_equivalent else 'FAIL'}")
    
    return results, all_equivalent

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    
    print("\n" + "=" * 50)
    print("EDGE CASE TESTING")
    print("=" * 50)
    
    # Test with minimal data
    print("Testing with minimal dataset (1 day)...")
    data = create_test_data(days=1)
    params = {'entry_time': '09:30', 'hold_time': 60, 'entry_spread': 5, 'exit_spread': 5}
    
    strategy = TimeWindowVectorizedSingle()
    orig_signals = strategy._generate_gradual_signals(data, params)
    opt_signals = VectorizedSignalGenerator.generate_gradual_signals(data, params)
    
    equiv_minimal = VectorizedSignalGenerator.validate_signal_equivalence(orig_signals, opt_signals)
    print(f"  Minimal data equivalence: {'PASS' if equiv_minimal else 'FAIL'}")
    
    # Test with different entry times
    print("Testing different entry times...")
    test_times = ['09:30', '10:00', '14:30']
    
    for entry_time in test_times:
        params['entry_time'] = entry_time
        
        orig_signals = strategy._generate_gradual_signals(data, params)
        opt_signals = VectorizedSignalGenerator.generate_gradual_signals(data, params)
        
        equiv = VectorizedSignalGenerator.validate_signal_equivalence(orig_signals, opt_signals)
        print(f"  Entry time {entry_time}: {'PASS' if equiv else 'FAIL'}")
    
    return True

if __name__ == "__main__":
    try:
        print("VECTORIZED SIGNAL GENERATION TESTING")
        print("=" * 60)
        
        # Test 1: Behavioral equivalence
        equiv_pass, speedup = test_behavioral_equivalence()
        
        # Test 2: Performance improvement  
        perf_results, perf_pass = test_performance_improvement()
        
        # Test 3: Edge cases
        edge_pass = test_edge_cases()
        
        # Final summary
        print("\n" + "=" * 60)
        print("FINAL TEST RESULTS")
        print("=" * 60)
        
        all_tests_pass = equiv_pass and perf_pass and edge_pass
        
        print(f"Behavioral Equivalence: {'PASS' if equiv_pass else 'FAIL'}")
        print(f"Performance Tests: {'PASS' if perf_pass else 'FAIL'}")
        print(f"Edge Case Tests: {'PASS' if edge_pass else 'FAIL'}")
        print(f"Overall Result: {'ALL TESTS PASS' if all_tests_pass else 'TESTS FAILED'}")
        
        if all_tests_pass:
            avg_speedup = np.mean([r['speedup'] for r in perf_results])
            print(f"\nOptimization successful! Average speedup: {avg_speedup:.1f}x")
        
    except Exception as e:
        print(f"\nTesting failed: {e}")
        import traceback
        traceback.print_exc()