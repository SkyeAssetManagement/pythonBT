#!/usr/bin/env python3
"""
Extended speed test: Multiple timeframes from 1 year to 25 years
"""

import numpy as np
import pandas as pd
import sys
import os
import time

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from strategies.time_window_strategy_vectorized import TimeWindowVectorizedSingle

def create_test_data(days=252):
    """Create test data for specified trading days"""
    
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
    
    # Create price data
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

def benchmark_single_entry(data, config):
    """Benchmark single entry approach"""
    
    import vectorbtpro as vbt
    
    timestamps = data['datetime']
    n_bars = len(timestamps)
    
    timestamps_sec = timestamps / 1_000_000_000
    est_timestamps_sec = timestamps_sec + (5 * 3600)
    dt_objects = pd.to_datetime(est_timestamps_sec, unit='s')
    minutes_of_day = dt_objects.hour * 60 + dt_objects.minute
    dates = dt_objects.date
    
    entry_minute = 9 * 60 + 31
    in_entry_window = minutes_of_day == entry_minute
    
    df = pd.DataFrame({
        'date': dates,
        'in_window': in_entry_window,
        'bar_idx': np.arange(n_bars)
    })
    
    daily_entries = df[df['in_window']].groupby('date')['bar_idx'].first()
    
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    
    if len(daily_entries) > 0:
        entries[daily_entries.values] = True
        exit_indices = daily_entries.values + 60
        valid_exits = exit_indices[exit_indices < n_bars]
        exits[valid_exits] = True
    
    start_time = time.time()
    
    pf = vbt.Portfolio.from_signals(
        close=data['close'],
        long_entries=entries,
        long_exits=exits,
        price=(data['high'] + data['low']) / 2,
        init_cash=config.get('initial_capital', 100000),
        fees=config.get('commission', 0.001),
        freq='1min',
        size=1.0
    )
    
    duration = time.time() - start_time
    trades = pf.trades.records_readable
    
    return duration, len(trades)

def benchmark_gradual_entry(data, config):
    """Benchmark gradual entry approach"""
    
    strategy = TimeWindowVectorizedSingle()
    
    start_time = time.time()
    pf = strategy.run_vectorized_backtest(data, config)
    duration = time.time() - start_time
    
    trades = pf.trades.records_readable
    
    return duration, len(trades)

def run_extended_benchmark():
    """Run benchmark across multiple timeframes"""
    
    print("EXTENDED SPEED BENCHMARK")
    print("=" * 60)
    print("Testing backtest speeds from 1 year to 10+ years")
    print("=" * 60)
    
    # Test different timeframes
    test_periods = [
        (252, "1 year"),
        (504, "2 years"), 
        (1260, "5 years"),
        (2520, "10 years"),
        (3780, "15 years"),
        (5040, "20 years")
    ]
    
    config = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0001
    }
    
    results = []
    
    for days, period_name in test_periods:
        print(f"\nTesting {period_name} ({days:,} days, {days*395:,} bars)...")
        
        # Create data
        data = create_test_data(days)
        bars = len(data['close'])
        
        try:
            # Benchmark single entry
            single_time, single_trades = benchmark_single_entry(data, config)
            single_speed = bars / single_time
            
            # Benchmark gradual entry  
            gradual_time, gradual_trades = benchmark_gradual_entry(data, config)  
            gradual_speed = bars / gradual_time
            
            # Calculate ratios
            speed_ratio = single_time / gradual_time
            throughput_ratio = gradual_speed / single_speed
            
            result = {
                'period': period_name,
                'days': days,
                'bars': bars,
                'single_time': single_time,
                'single_speed': single_speed,
                'single_trades': single_trades,
                'gradual_time': gradual_time,
                'gradual_speed': gradual_speed,
                'gradual_trades': gradual_trades,
                'speed_ratio': speed_ratio,
                'throughput_ratio': throughput_ratio
            }
            
            results.append(result)
            
            print(f"  Single:  {single_time:.2f}s ({single_speed:,.0f} bars/sec, {single_trades} trades)")
            print(f"  Gradual: {gradual_time:.2f}s ({gradual_speed:,.0f} bars/sec, {gradual_trades} trades)")
            
            if speed_ratio > 1:
                print(f"  Result: Gradual {speed_ratio:.2f}x FASTER")
            else:
                print(f"  Result: Gradual {1/speed_ratio:.2f}x SLOWER")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Summary analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"{'Period':<12} {'Bars':<10} {'Single':<8} {'Gradual':<8} {'Ratio':<8}")
    print("-" * 60)
    
    for r in results:
        ratio_str = f"{r['speed_ratio']:.2f}x" if r['speed_ratio'] > 1 else f"{1/r['speed_ratio']:.2f}x"
        print(f"{r['period']:<12} {r['bars']:<10,} {r['single_time']:<8.2f} {r['gradual_time']:<8.2f} {ratio_str:<8}")
    
    # Average performance
    if results:
        avg_single_speed = np.mean([r['single_speed'] for r in results])
        avg_gradual_speed = np.mean([r['gradual_speed'] for r in results])
        avg_ratio = np.mean([r['speed_ratio'] for r in results])
        
        print(f"\nAverage Performance:")
        print(f"  Single Entry: {avg_single_speed:,.0f} bars/sec")
        print(f"  Gradual Entry: {avg_gradual_speed:,.0f} bars/sec")
        
        if avg_ratio > 1:
            print(f"  Overall: Gradual {avg_ratio:.2f}x FASTER on average")
        else:
            print(f"  Overall: Gradual {1/avg_ratio:.2f}x SLOWER on average")
    
    # Extrapolate to 25 years (2000-2025)
    if results:
        # Use 20-year performance to estimate 25-year
        last_result = results[-1]
        
        bars_25_years = 25 * 252 * 395  # 2,488,500 bars
        
        estimated_single_time = bars_25_years / last_result['single_speed']
        estimated_gradual_time = bars_25_years / last_result['gradual_speed']
        
        print(f"\nEstimated 25-year performance (2000-2025):")
        print(f"  Dataset: {bars_25_years:,} bars")
        print(f"  Single Entry: {estimated_single_time:.1f} seconds ({estimated_single_time/60:.1f} minutes)")
        print(f"  Gradual Entry: {estimated_gradual_time:.1f} seconds ({estimated_gradual_time/60:.1f} minutes)")
        
        if estimated_single_time < estimated_gradual_time:
            ratio = estimated_gradual_time / estimated_single_time
            print(f"  Difference: Gradual takes {ratio:.1f}x longer")
        else:
            ratio = estimated_single_time / estimated_gradual_time
            print(f"  Difference: Gradual is {ratio:.1f}x faster")
    
    return results

if __name__ == "__main__":
    try:
        results = run_extended_benchmark()
        print("\nExtended benchmark completed!")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()