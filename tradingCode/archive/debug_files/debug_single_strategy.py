#!/usr/bin/env python3
"""
Debug what the single strategy is actually doing
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy

def debug_strategy():
    """Debug the single strategy execution"""
    
    print("DEBUGGING SINGLE STRATEGY")
    print("=" * 50)
    
    # Create small test data
    days = 3
    minutes_per_day = 395
    total_bars = days * minutes_per_day
    
    start_time_est = pd.Timestamp('2025-01-01 09:25:00')
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
    base_price = 6250.0
    close_prices = base_price + np.random.normal(0, 1.0, total_bars)
    high_prices = close_prices + np.abs(np.random.normal(0, 0.5, total_bars))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.5, total_bars))
    open_prices = close_prices + np.random.normal(0, 0.25, total_bars)
    volumes = np.random.randint(100, 1000, total_bars)
    
    data = {
        'datetime': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }
    
    print(f"Test data: {len(data['close'])} bars ({days} days)")
    
    # Test the strategy
    strategy = TimeWindowVectorizedStrategy()
    print(f"Strategy class: {strategy.__class__.__name__}")
    
    # Check parameter combinations with useDefaults=True
    params = strategy.get_parameter_combinations(use_defaults_only=True)
    print(f"Parameter combinations: {len(params)}")
    print(f"Default params: {params[0]}")
    
    # Test signal generation directly
    print("\nTesting signal generation...")
    entry_signals, exit_signals, prices = strategy._generate_gradual_signals(data, params[0])
    
    print(f"Entry signals shape: {entry_signals.shape}")
    print(f"Exit signals shape: {exit_signals.shape}")
    print(f"Total entry signals: {np.sum(entry_signals)}")
    print(f"Total exit signals: {np.sum(exit_signals)}")
    
    # Check first few entries
    for bar in range(50):
        entry_count = np.sum(entry_signals[bar, :])
        exit_count = np.sum(exit_signals[bar, :])
        if entry_count > 0 or exit_count > 0:
            # Convert to EST time
            utc_time = pd.Timestamp(timestamps[bar])
            est_time = utc_time + pd.Timedelta(hours=5)
            print(f"  Bar {bar} at {est_time.strftime('%H:%M')}: {entry_count} entries, {exit_count} exits")
    
    # Test full backtest
    print("\nTesting full backtest...")
    config = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0001
    }
    
    pf = strategy.run_vectorized_backtest(data, config, use_defaults_only=True)
    print(f"Portfolio shape: {pf.wrapper.shape}")
    
    # Check trades
    trades = pf.trades.records_readable
    print(f"Total trades: {len(trades)}")
    
    if len(trades) > 0:
        print(f"Trade columns: {sorted(trades['Column'].unique())}")
        print(f"Trade sizes: {sorted(trades['Size'].unique())}")
        
        print("\nFirst few trades:")
        for i in range(min(10, len(trades))):
            trade = trades.iloc[i]
            print(f"  Trade {i}: Column {trade['Column']}, Size {trade['Size']}, Entry {trade['Entry Index']}")
    
    return len(trades) > 0 and len(trades['Column'].unique()) > 1

if __name__ == "__main__":
    success = debug_strategy()
    
    print(f"\nDEBUG RESULT: {'SUCCESS - Multiple columns detected' if success else 'ISSUE - Single column only'}")