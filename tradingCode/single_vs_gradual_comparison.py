#!/usr/bin/env python3
"""
Compare single entry vs optimized gradual entry performance for 25-year dataset
"""

import numpy as np
import pandas as pd
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

def create_25_year_sample():
    """Create 5-year sample to project 25-year performance"""
    
    days = 5 * 252  # 5 years
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

def benchmark_single_entry(data):
    """Benchmark single entry strategy"""
    
    import vectorbtpro as vbt
    
    timestamps = data['datetime']
    n_bars = len(timestamps)
    
    # Convert to EST and find entry times
    timestamps_sec = timestamps / 1_000_000_000
    est_timestamps_sec = timestamps_sec + (5 * 3600)
    dt_objects = pd.to_datetime(est_timestamps_sec, unit='s')
    minutes_of_day = dt_objects.hour * 60 + dt_objects.minute
    dates = dt_objects.date
    
    # Entry at 09:31 (571 minutes)
    entry_minute = 9 * 60 + 31
    in_entry_window = minutes_of_day == entry_minute
    
    # Get first entry per day using pandas groupby
    df = pd.DataFrame({
        'date': dates,
        'in_window': in_entry_window,
        'bar_idx': np.arange(n_bars)
    })
    
    daily_entries = df[df['in_window']].groupby('date')['bar_idx'].first()
    
    # Create single-column signals
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    
    if len(daily_entries) > 0:
        entries[daily_entries.values] = True
        
        # Exits 60 minutes later
        entry_indices = daily_entries.values
        exit_indices = entry_indices + 60
        valid_exits = exit_indices[exit_indices < n_bars]
        exits[valid_exits] = True
    
    print("  Testing single entry backtest...")
    start_time = time.time()
    
    # VectorBT portfolio (single column)
    pf = vbt.Portfolio.from_signals(
        close=data['close'],
        long_entries=entries,
        long_exits=exits,
        price=(data['high'] + data['low']) / 2,
        init_cash=100000,
        fees=0.001,
        slippage=0.0001,
        freq='1min',
        size=1.0,
        size_type='amount'
    )
    
    total_time = time.time() - start_time
    speed = n_bars / total_time
    
    # Get results
    trades = pf.trades.records_readable
    total_return = pf.total_return.iloc[0] if hasattr(pf.total_return, 'iloc') else pf.total_return
    
    print(f"  Single entry: {total_time:.3f}s ({speed:,.0f} bars/sec)")
    print(f"  Trades: {len(trades)}, Return: {total_return*100:.2f}%")
    
    return {
        'time': total_time,
        'speed': speed,
        'trades': len(trades),
        'return': total_return
    }

def benchmark_gradual_entry(data):
    """Benchmark optimized gradual entry strategy"""
    
    from strategies.time_window_strategy_vectorized import TimeWindowVectorizedSingle
    
    strategy = TimeWindowVectorizedSingle()
    config = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0001
    }
    
    n_bars = len(data['close'])
    
    print("  Testing gradual entry backtest...")
    start_time = time.time()
    pf = strategy.run_vectorized_backtest(data, config)
    total_time = time.time() - start_time
    
    speed = n_bars / total_time
    
    # Get results
    trades = pf.trades.records_readable
    if hasattr(pf.total_return, '__len__') and len(pf.total_return) > 1:
        total_return = pf.total_return.sum()
    else:
        total_return = pf.total_return.iloc[0] if hasattr(pf.total_return, 'iloc') else pf.total_return
    
    print(f"  Gradual entry: {total_time:.3f}s ({speed:,.0f} bars/sec)")
    print(f"  Trades: {len(trades)}, Return: {total_return*100:.2f}%")
    print(f"  Columns: {len(trades['Column'].unique()) if len(trades) > 0 else 0}")
    
    return {
        'time': total_time,
        'speed': speed,
        'trades': len(trades),
        'return': total_return,
        'columns': len(trades['Column'].unique()) if len(trades) > 0 else 0
    }

def project_to_25_years(single_results, gradual_results):
    """Project both results to 25-year timeframe"""
    
    bars_25_years = 25 * 252 * 395  # 2,488,500 bars
    
    # Project times
    single_25y = bars_25_years / single_results['speed']
    gradual_25y = bars_25_years / gradual_results['speed']
    
    # Calculate ratios
    speed_ratio = single_results['speed'] / gradual_results['speed']
    time_ratio = gradual_25y / single_25y
    
    print(f"\n25-YEAR PROJECTIONS ({bars_25_years:,} bars):")
    print("=" * 50)
    print(f"Single Entry (25 years):  {single_25y:.1f} seconds")
    print(f"Gradual Entry (25 years): {gradual_25y:.1f} seconds")
    print(f"\nPerformance Comparison:")
    print(f"  Single entry speed: {single_results['speed']:,.0f} bars/sec")
    print(f"  Gradual entry speed: {gradual_results['speed']:,.0f} bars/sec")
    print(f"  Speed ratio: Single is {speed_ratio:.1f}x faster than Gradual")
    print(f"  Time ratio: Gradual takes {time_ratio:.1f}x longer than Single")
    
    # Trading comparison
    single_trades_25y = (single_results['trades'] / 5) * 25  # 5-year sample to 25 years
    gradual_trades_25y = (gradual_results['trades'] / 5) * 25
    
    print(f"\nTrading Volume (25 years):")
    print(f"  Single entry: {single_trades_25y:.0f} trades")
    print(f"  Gradual entry: {gradual_trades_25y:.0f} trades ({gradual_results['columns']} columns)")
    print(f"  Trade ratio: {gradual_trades_25y/single_trades_25y:.0f}x more trades with gradual")
    
    # Value proposition
    additional_time = gradual_25y - single_25y
    print(f"\nValue Analysis:")
    print(f"  Additional time for gradual execution: {additional_time:.1f} seconds")
    print(f"  Benefit: More realistic trading simulation with gradual position building")
    print(f"  Cost: {time_ratio:.1f}x execution time for {gradual_results['columns']}x trading complexity")
    
    return {
        'single_25y': single_25y,
        'gradual_25y': gradual_25y,
        'time_ratio': time_ratio,
        'additional_time': additional_time
    }

if __name__ == "__main__":
    print("SINGLE vs GRADUAL ENTRY PERFORMANCE COMPARISON")
    print("=" * 60)
    print("Comparing optimized implementations for 25-year timeframe")
    print("=" * 60)
    
    # Create test data
    data = create_25_year_sample()
    print(f"Test dataset: {len(data['close']):,} bars (5-year sample)")
    
    print("\nBenchmarking both strategies...")
    
    # Benchmark both approaches
    single_results = benchmark_single_entry(data)
    gradual_results = benchmark_gradual_entry(data)
    
    # Project to 25 years
    projection = project_to_25_years(single_results, gradual_results)
    
    print(f"\n" + "=" * 60)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"For 25-year ES backtest (2000-2025):")
    print(f"  Single Entry:  {projection['single_25y']:.1f} seconds")
    print(f"  Gradual Entry: {projection['gradual_25y']:.1f} seconds")
    print(f"  Difference: +{projection['additional_time']:.1f} seconds for realistic execution")
    print(f"  Performance trade-off: {projection['time_ratio']:.1f}x time for 5x trading realism")