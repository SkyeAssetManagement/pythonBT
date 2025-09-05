#!/usr/bin/env python3
"""
Debug Signal Generation
Check why no trades are happening
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add strategies to path
strategies_path = Path(__file__).parent / "strategies"
if str(strategies_path) not in sys.path:
    sys.path.insert(0, str(strategies_path))

from time_window_strategy_vectorized import TimeWindowVectorizedSingle

def create_debug_data():
    """Create simple test data for debugging"""
    
    # Create 1 day of simple data with clear entry window
    start_time = pd.Timestamp('2024-01-15 09:00:00')
    end_time = pd.Timestamp('2024-01-15 16:00:00')
    timestamps = pd.date_range(start_time, end_time, freq='1min')[:-1]
    
    n_bars = len(timestamps)
    print(f"Created {n_bars} bars of debug data")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
    
    # Convert to nanosecond timestamps
    timestamp_ns = np.array([int(ts.timestamp() * 1_000_000_000) for ts in timestamps])
    
    # Simple price data
    base_price = 0.67500
    close_prices = np.full(n_bars, base_price)
    open_prices = np.full(n_bars, base_price)
    high_prices = np.full(n_bars, base_price + 0.0001)
    low_prices = np.full(n_bars, base_price - 0.0001)
    volume = np.full(n_bars, 1000.0)
    
    return {
        'datetime': timestamp_ns,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }

def debug_signal_generation():
    """Debug the signal generation logic"""
    
    print("=== DEBUGGING SIGNAL GENERATION ===")
    
    # Create debug data
    data = create_debug_data()
    
    # Create strategy
    strategy = TimeWindowVectorizedSingle()
    params = strategy.get_parameter_combinations()[0]
    
    print(f"Strategy parameters: {params}")
    
    # Generate signals
    entries, exits = strategy._generate_signals_for_params(data, params)
    
    print(f"Entries: {np.sum(entries)} signals")
    print(f"Exits: {np.sum(exits)} signals")
    
    if np.sum(entries) > 0:
        entry_indices = np.where(entries)[0]
        print(f"Entry indices: {entry_indices}")
        
        # Show entry times
        timestamps = data['datetime']
        for idx in entry_indices[:5]:  # Show first 5
            timestamp_sec = timestamps[idx] / 1_000_000_000
            dt = pd.to_datetime(timestamp_sec, unit='s')
            print(f"  Entry at bar {idx}: {dt}")
    
    if np.sum(exits) > 0:
        exit_indices = np.where(exits)[0]
        print(f"Exit indices: {exit_indices}")
        
        # Show exit times
        timestamps = data['datetime']
        for idx in exit_indices[:5]:  # Show first 5
            timestamp_sec = timestamps[idx] / 1_000_000_000
            dt = pd.to_datetime(timestamp_sec, unit='s')
            print(f"  Exit at bar {idx}: {dt}")
    
    # Debug the time window logic
    print("\n=== DEBUGGING TIME WINDOW LOGIC ===")
    
    timestamps = data['datetime']
    timestamps_sec = timestamps / 1_000_000_000
    dt_objects = pd.to_datetime(timestamps_sec, unit='s')
    minutes_of_day = dt_objects.hour * 60 + dt_objects.minute
    
    # Entry time: 09:30 -> entry window 09:31-09:35
    entry_minutes = 9 * 60 + 30  # 570 minutes
    entry_start = entry_minutes + 1  # 571 (09:31)
    entry_end = entry_minutes + params['entry_spread']  # 575 (09:35)
    
    print(f"Entry time: {params['entry_time']} (minute {entry_minutes})")
    print(f"Entry window: minutes {entry_start}-{entry_end} (09:31-09:35)")
    
    # Check which bars are in entry window
    in_entry_window = (minutes_of_day >= entry_start) & (minutes_of_day <= entry_end)
    entry_window_count = np.sum(in_entry_window)
    
    print(f"Bars in entry window: {entry_window_count}")
    
    if entry_window_count > 0:
        window_indices = np.where(in_entry_window)[0]
        print(f"Entry window bar indices: {window_indices}")
        
        for idx in window_indices:
            dt = dt_objects[idx]
            print(f"  Bar {idx}: {dt} (minute {minutes_of_day[idx]})")
    
    return entries, exits

def debug_portfolio_creation():
    """Debug portfolio creation"""
    
    print("\n=== DEBUGGING PORTFOLIO CREATION ===")
    
    data = create_debug_data()
    strategy = TimeWindowVectorizedSingle()
    
    config = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0001
    }
    
    # Run backtest
    pf = strategy.run_vectorized_backtest(data, config)
    
    print(f"Portfolio created successfully")
    print(f"Orders count: {pf.orders.count()}")
    print(f"Total return: {pf.total_return}")
    print(f"Final value: {pf.value[-1]}")
    
    # Check if any orders exist
    if pf.orders.count() > 0:
        print("Orders exist - checking details...")
        orders_df = pf.orders.records_readable
        print(f"Orders shape: {orders_df.shape}")
        if len(orders_df) > 0:
            print("First few orders:")
            print(orders_df.head())
    else:
        print("No orders generated!")

if __name__ == "__main__":
    print("SIGNAL GENERATION DEBUG")
    print("=" * 50)
    
    # Debug signal generation
    entries, exits = debug_signal_generation()
    
    # Debug portfolio
    debug_portfolio_creation()
    
    print("\nDEBUG COMPLETE")