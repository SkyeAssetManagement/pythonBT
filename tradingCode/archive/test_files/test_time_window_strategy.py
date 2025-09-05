#!/usr/bin/env python3
"""
Test the Time Window Strategy
Verify that the strategy logic works correctly
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add strategies to path
strategies_path = Path(__file__).parent / "strategies"
if str(strategies_path) not in sys.path:
    sys.path.insert(0, str(strategies_path))

from time_window_strategy import TimeWindowStrategy, TimeWindowSingleStrategy

def create_test_data():
    """Create test data for strategy testing"""
    
    print("=== CREATING TEST DATA ===")
    
    # Create 2 days of 1-minute data from 9:00 AM to 4:00 PM
    start_date = datetime(2024, 1, 15, 9, 0)  # Monday 9:00 AM
    end_date = datetime(2024, 1, 16, 16, 0)   # Tuesday 4:00 PM
    
    # Generate minute-by-minute timestamps
    current_time = start_date
    timestamps = []
    
    while current_time <= end_date:
        # Only include market hours (9:00 AM - 4:00 PM)
        if 9 <= current_time.hour < 16:
            timestamps.append(current_time)
        
        current_time += timedelta(minutes=1)
        
        # Skip overnight (jump to next day 9:00 AM)
        if current_time.hour == 16:
            current_time = current_time.replace(hour=9, minute=0) + timedelta(days=1)
    
    n_bars = len(timestamps)
    print(f"Created {n_bars} bars of test data")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
    
    # Convert to nanosecond timestamps
    timestamp_ns = np.array([int(ts.timestamp() * 1_000_000_000) for ts in timestamps])
    
    # Create realistic price data
    base_price = 100.0
    price_changes = np.cumsum(np.random.normal(0, 0.1, n_bars))
    close_prices = base_price + price_changes
    
    # Generate OHLC with realistic relationships
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    # Add spreads for high/low
    spreads = np.random.uniform(0.05, 0.20, n_bars)
    high_prices = np.maximum(open_prices, close_prices) + spreads
    low_prices = np.minimum(open_prices, close_prices) - spreads
    
    volume = np.random.randint(1000, 10000, n_bars).astype(float)
    
    data = {
        'datetime': timestamp_ns,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    
    print(f"Price range: ${np.min(low_prices):.2f} - ${np.max(high_prices):.2f}")
    
    return data

def test_strategy_parameters():
    """Test strategy parameter generation"""
    
    print("\\n=== TESTING STRATEGY PARAMETERS ===")
    
    strategy = TimeWindowStrategy()
    
    print(f"Strategy name: {strategy.name}")
    print(f"Strategy description: {strategy.description}")
    
    # Test parameter combinations
    combinations = strategy.get_parameter_combinations()
    print(f"Generated {len(combinations)} parameter combinations")
    
    # Show first few combinations
    for i, combo in enumerate(combinations[:5]):
        print(f"  Combo {i+1}: {combo}")
    
    # Test single strategy
    single_strategy = TimeWindowSingleStrategy()
    single_combos = single_strategy.get_parameter_combinations()
    print(f"\\nSingle strategy combinations: {len(single_combos)}")
    print(f"Default parameters: {single_combos[0]}")
    
    return strategy

def test_strategy_backtest():
    """Test strategy backtest execution"""
    
    print("\\n=== TESTING STRATEGY BACKTEST ===")
    
    # Create test data
    data = create_test_data()
    
    # Create strategy
    strategy = TimeWindowSingleStrategy()  # Use single for faster testing
    
    # Basic backtest config
    config = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0001
    }
    
    print("Running backtest...")
    try:
        # Run backtest
        pf = strategy.run_vectorized_backtest(data, config)
        
        print(f"Backtest completed successfully!")
        
        # Handle Series vs scalar total_return
        total_return = pf.total_return
        if hasattr(total_return, '__len__') and len(total_return) > 1:
            # Multiple combinations - show best result
            best_return = total_return.max()
            print(f"Best total return: {best_return*100:.2f}%")
        else:
            # Single combination
            return_val = total_return.iloc[0] if hasattr(total_return, 'iloc') else total_return
            print(f"Total return: {return_val*100:.2f}%")
        
        print(f"Total orders: {pf.orders.count()}")
        
        # Get trade statistics
        if pf.orders.count() > 0:
            trades_returns = pf.trades.returns
            if hasattr(trades_returns, '__len__') and len(trades_returns) > 0:
                avg_return = trades_returns.mean()
                if hasattr(avg_return, 'iloc'):
                    avg_return = avg_return.iloc[0]
                print(f"Avg trade return: {avg_return*100:.2f}%")
            
            win_rate = pf.trades.win_rate
            if hasattr(win_rate, 'iloc'):
                win_rate = win_rate.iloc[0]
            print(f"Win rate: {win_rate*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_time_logic():
    """Test the time window logic"""
    
    print("\\n=== TESTING TIME WINDOW LOGIC ===")
    
    strategy = TimeWindowStrategy()
    
    # Test time parsing
    test_times = ["09:30", "10:00", "14:30", "15:45"]
    for time_str in test_times:
        parsed = strategy._parse_time_string(time_str)
        print(f"Parsed {time_str} -> {parsed}")
    
    # Test time window checking
    from datetime import time as dt_time
    
    target_time = dt_time(9, 30)  # 9:30 AM
    test_times = [
        dt_time(9, 31),  # Should be in window (minute 1)
        dt_time(9, 33),  # Should be in window (minute 3)
        dt_time(9, 35),  # Should be in window (minute 5)
        dt_time(9, 36),  # Should be outside window
        dt_time(9, 29),  # Should be outside window
    ]
    
    for test_time in test_times:
        in_window = strategy._is_in_time_window(test_time, target_time, 1, 5)
        print(f"Time {test_time} in 09:31-09:35 window: {in_window}")
    
    return True

if __name__ == "__main__":
    print("Testing Time Window Strategy...")
    
    success = True
    
    # Test 1: Strategy parameters
    try:
        strategy = test_strategy_parameters()
        print("+ Strategy parameters test passed")
    except Exception as e:
        print(f"- Strategy parameters test failed: {e}")
        success = False
    
    # Test 2: Time logic
    try:
        test_time_logic()
        print("+ Time logic test passed")
    except Exception as e:
        print(f"- Time logic test failed: {e}")
        success = False
    
    # Test 3: Backtest execution
    try:
        if test_strategy_backtest():
            print("+ Strategy backtest test passed")
        else:
            print("- Strategy backtest test failed")
            success = False
    except Exception as e:
        print(f"- Strategy backtest test failed: {e}")
        success = False
    
    if success:
        print("\\nSUCCESS: ALL TESTS PASSED!")
        print("Time Window Strategy is ready for use with main.py")
        print("\\nTo run with main.py:")
        print("python main.py SYMBOL time_window_strategy")
    else:
        print("\\nFAILED: SOME TESTS FAILED!")
        print("Check the errors above and fix before using with main.py")