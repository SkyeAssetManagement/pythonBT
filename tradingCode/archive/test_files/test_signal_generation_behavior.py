#!/usr/bin/env python3
"""
strategies/test_signal_generation_behavior.py

Test signal generation behavior before optimization.
Validates current gradual entry/exit logic for preservation during refactoring.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to the path for importing strategies
sys.path.insert(0, os.path.dirname(__file__))

from strategies.time_window_strategy_vectorized import TimeWindowVectorizedSingle

def create_reference_data():
    """
    Create standardized test data for behavior validation.
    
    Purpose: Ensure consistent test results across optimization iterations.
    This creates exactly 3 trading days with known entry/exit patterns.
    """
    
    # Create 3 days of minute data (3 * 395 = 1185 bars)
    # Day 1: 2000-01-01, Day 2: 2000-01-02, Day 3: 2000-01-03
    
    start_time_est = pd.Timestamp('2000-01-01 09:25:00')
    timestamps = []
    
    minutes_per_day = 395  # 9:25 AM to 4:00 PM EST
    
    for day in range(3):  # 3 trading days
        day_start = start_time_est + pd.Timedelta(days=day)
        
        for minute in range(minutes_per_day):
            time_est = day_start + pd.Timedelta(minutes=minute)
            utc_time = time_est - pd.Timedelta(hours=5)  # Convert EST to UTC
            timestamp_ns = int(utc_time.value)
            timestamps.append(timestamp_ns)
    
    timestamps = np.array(timestamps)
    
    # Create predictable price data for testing
    np.random.seed(42)  # Deterministic results
    n_bars = len(timestamps)
    
    base_price = 1400.0
    close_prices = base_price + np.random.normal(0, 1.0, n_bars)
    high_prices = close_prices + np.abs(np.random.normal(0, 0.5, n_bars))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.5, n_bars))
    open_prices = close_prices + np.random.normal(0, 0.25, n_bars)
    volumes = np.random.randint(100, 1000, n_bars)
    
    return {
        'datetime': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }

def validate_current_behavior():
    """
    Validate and document current signal generation behavior.
    
    Purpose: Create reference results to ensure optimization maintains identical logic.
    This captures the exact signal patterns for comparison after optimization.
    """
    
    print("CURRENT BEHAVIOR VALIDATION")
    print("=" * 60)
    print("Testing current _generate_gradual_signals implementation")
    print("Purpose: Capture reference behavior for optimization validation")
    print()
    
    # Create test data
    data = create_reference_data()
    n_bars = len(data['close'])
    
    print(f"Test dataset: {n_bars} bars (3 trading days)")
    
    # Test parameters
    strategy = TimeWindowVectorizedSingle()
    params = {
        'entry_time': '09:30',
        'hold_time': 60,
        'entry_spread': 5,
        'exit_spread': 5
    }
    
    print(f"Test parameters: {params}")
    print()
    
    # Generate signals using current implementation
    print("Generating signals with current implementation...")
    
    entry_signals_2d, exit_signals_2d, prices_2d = strategy._generate_gradual_signals(data, params)
    
    # Validate signal structure
    print("SIGNAL STRUCTURE VALIDATION:")
    print(f"  Entry signals shape: {entry_signals_2d.shape}")
    print(f"  Exit signals shape: {exit_signals_2d.shape}")
    print(f"  Prices shape: {prices_2d.shape}")
    print(f"  Entry signals total: {np.sum(entry_signals_2d)}")
    print(f"  Exit signals total: {np.sum(exit_signals_2d)}")
    print()
    
    # Analyze signal patterns per day
    print("DAILY SIGNAL PATTERN ANALYSIS:")
    
    # Convert timestamps back to EST for analysis
    timestamps_sec = data['datetime'] / 1_000_000_000
    est_timestamps_sec = timestamps_sec + (5 * 3600)
    dt_objects = pd.to_datetime(est_timestamps_sec, unit='s')
    dates = dt_objects.date
    minutes_of_day = dt_objects.hour * 60 + dt_objects.minute
    
    # Track signals by day and time
    signal_log = []
    
    for bar_idx in range(n_bars):
        # Check if this bar has any entry signals
        entry_signals_this_bar = np.sum(entry_signals_2d[bar_idx, :])
        exit_signals_this_bar = np.sum(exit_signals_2d[bar_idx, :])
        
        if entry_signals_this_bar > 0 or exit_signals_this_bar > 0:
            date = dates[bar_idx]
            minute = minutes_of_day[bar_idx]
            hour = minute // 60
            min_part = minute % 60
            
            signal_log.append({
                'bar_idx': bar_idx,
                'date': date,
                'time': f"{hour:02d}:{min_part:02d}",
                'entries': entry_signals_this_bar,
                'exits': exit_signals_this_bar,
                'entry_columns': [i for i in range(5) if entry_signals_2d[bar_idx, i]],
                'exit_columns': [i for i in range(5) if exit_signals_2d[bar_idx, i]]
            })
    
    # Display signal patterns
    for day_num, target_date in enumerate([pd.Timestamp('2000-01-01').date(), 
                                          pd.Timestamp('2000-01-02').date(), 
                                          pd.Timestamp('2000-01-03').date()]):
        
        day_signals = [s for s in signal_log if s['date'] == target_date]
        
        print(f"  Day {day_num + 1} ({target_date}):")
        
        if day_signals:
            # Group by signal type
            entry_signals = [s for s in day_signals if s['entries'] > 0]
            exit_signals = [s for s in day_signals if s['exits'] > 0]
            
            print(f"    Entry signals: {len(entry_signals)} bars")
            for s in entry_signals:
                print(f"      Bar {s['bar_idx']} at {s['time']}: {s['entries']} entries (columns {s['entry_columns']})")
            
            print(f"    Exit signals: {len(exit_signals)} bars")
            for s in exit_signals[:3]:  # Show first 3 exit signals
                print(f"      Bar {s['bar_idx']} at {s['time']}: {s['exits']} exits (columns {s['exit_columns']})")
            
            if len(exit_signals) > 3:
                print(f"      ... and {len(exit_signals) - 3} more exit signals")
        else:
            print("    No signals generated")
        
        print()
    
    # Create reference behavior snapshot
    reference_behavior = {
        'entry_signals_2d': entry_signals_2d.copy(),
        'exit_signals_2d': exit_signals_2d.copy(),
        'prices_2d': prices_2d.copy(),
        'total_entries': np.sum(entry_signals_2d),
        'total_exits': np.sum(exit_signals_2d),
        'signal_log': signal_log,
        'test_params': params,
        'data_shape': n_bars
    }
    
    print("REFERENCE BEHAVIOR CAPTURED:")
    print(f"  Total entry signals: {reference_behavior['total_entries']}")
    print(f"  Total exit signals: {reference_behavior['total_exits']}")
    print(f"  Signal events logged: {len(signal_log)}")
    print(f"  Ready for optimization validation")
    
    return reference_behavior

if __name__ == "__main__":
    try:
        reference = validate_current_behavior()
        print("\nBehavior validation completed successfully!")
        print("Reference data captured for optimization testing.")
        
    except Exception as e:
        print(f"\nValidation failed: {e}")
        import traceback
        traceback.print_exc()