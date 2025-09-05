#!/usr/bin/env python3
"""
Final speed test: Single vs 40 combinations - optimized for large datasets
"""

import time
import sys
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from src.data.parquet_converter import ParquetConverter
from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy

def measure_strategy_performance(data, strategy, config, use_defaults_only=False):
    """Measure pure strategy performance excluding data loading"""
    
    print(f"  Measuring {'single default' if use_defaults_only else 'full optimization'} performance...")
    
    # Warm-up run (not timed)
    try:
        _ = strategy.run_vectorized_backtest(data, config, use_defaults_only=use_defaults_only)
    except:
        pass
    
    # Timed run
    start_time = time.time()
    pf = strategy.run_vectorized_backtest(data, config, use_defaults_only=use_defaults_only)
    duration = time.time() - start_time
    
    # Extract metrics
    trades = len(pf.trades.records_readable)
    columns = pf.wrapper.shape[1] if hasattr(pf.wrapper, 'shape') else 1
    
    return {
        'duration': duration,
        'trades': trades,
        'columns': columns,
        'bars_per_sec': len(data['close']) / duration if duration > 0 else 0
    }

def run_speed_test():
    """Run comprehensive speed test"""
    
    print("COMPREHENSIVE SPEED TEST: Single Default vs 40 Combinations")
    print("=" * 70)
    
    # Test different time periods
    test_periods = [
        ("6 months", "2024-07-01"),
        ("1 year", "2024-01-01"),
        ("2 years", "2023-01-01"),
        ("5 years", "2020-01-01"),
        ("Since 2000", "2000-01-01")
    ]
    
    strategy = TimeWindowVectorizedStrategy()
    config = {'initial_capital': 100000, 'commission': 0.001, 'slippage': 0.0001}
    
    results = []
    
    parquet_converter = ParquetConverter()
    
    for period_name, start_date in test_periods:
        print(f"\nTesting {period_name} ({start_date} to present)")
        print("-" * 50)
        
        # Load data with timeout protection
        print("Loading data...")
        load_start = time.time()
        
        try:
            data = parquet_converter.load_or_convert("ES", "1m", "diffAdjusted")
            if data:
                data = parquet_converter.filter_data_by_date(data, start_date, None)
            
            load_time = time.time() - load_start
            
            if not data or len(data['close']) == 0:
                print(f"  ERROR: No data available for {period_name}")
                continue
                
            n_bars = len(data['close'])
            print(f"  SUCCESS: Loaded {n_bars:,} bars in {load_time:.2f}s")
            
            # Skip if data loading took too long (likely to timeout on strategy)
            if load_time > 30:
                print(f"  SKIP: Data loading too slow ({load_time:.1f}s), skipping strategy test")
                continue
                
        except Exception as e:
            print(f"  ERROR: Data loading failed: {e}")
            continue
        
        # Test 1: Single Default
        print("\n  Test 1: Single Default Run")
        try:
            single_result = measure_strategy_performance(data, strategy, config, use_defaults_only=True)
            print(f"    SUCCESS: {single_result['duration']:.2f}s | {single_result['trades']} trades | {single_result['bars_per_sec']:,.0f} bars/sec")
            
        except Exception as e:
            print(f"    ERROR: Single run failed: {e}")
            continue
        
        # Test 2: Full Optimization (with timeout protection)
        print("  Test 2: Full Optimization (40 combinations)")
        try:
            # Skip full optimization for very large datasets to avoid timeout
            if n_bars > 500000:  # More than ~2 years of minute data
                print(f"    SKIP: Dataset too large ({n_bars:,} bars) for full optimization demo")
                full_result = None
            else:
                full_result = measure_strategy_performance(data, strategy, config, use_defaults_only=False)
                print(f"    SUCCESS: {full_result['duration']:.2f}s | {full_result['trades']} trades | {full_result['bars_per_sec']:,.0f} bars/sec")
            
        except Exception as e:
            print(f"    ERROR: Full optimization failed: {e}")
            full_result = None
        
        # Calculate performance metrics
        if single_result and full_result:
            ratio = full_result['duration'] / single_result['duration']
            efficiency = 40 / ratio
            
            print(f"\n  Performance Summary:")
            print(f"    Data size: {n_bars:,} bars")
            print(f"    Speed ratio: {ratio:.1f}x slower for 40x more work")
            print(f"    Vectorization efficiency: {efficiency:.1f}x better than linear")
            print(f"    Trade scaling: {single_result['trades']} -> {full_result['trades']} ({full_result['trades']/single_result['trades']:.1f}x)")
            
            results.append({
                'period': period_name,
                'bars': n_bars,
                'single_time': single_result['duration'],
                'full_time': full_result['duration'],
                'ratio': ratio,
                'efficiency': efficiency
            })
        else:
            print(f"\n  Performance Summary:")
            print(f"    Data size: {n_bars:,} bars")
            print(f"    Single run: {single_result['duration']:.2f}s ({single_result['bars_per_sec']:,.0f} bars/sec)")
            if not full_result:
                print(f"    Full optimization: Skipped (dataset too large)")
    
    # Summary
    if results:
        print(f"\n{'='*70}")
        print("SPEED TEST SUMMARY")
        print(f"{'='*70}")
        print(f"{'Period':<12} {'Bars':<10} {'Single':<8} {'Full':<8} {'Ratio':<8} {'Efficiency':<10}")
        print("-" * 70)
        
        for r in results:
            print(f"{r['period']:<12} {r['bars']:<10,} {r['single_time']:<8.2f} {r['full_time']:<8.2f} {r['ratio']:<8.1f} {r['efficiency']:<10.1f}")
        
        avg_efficiency = sum(r['efficiency'] for r in results) / len(results)
        print(f"\nAverage vectorization efficiency: {avg_efficiency:.1f}x better than linear scaling")

if __name__ == "__main__":
    run_speed_test()