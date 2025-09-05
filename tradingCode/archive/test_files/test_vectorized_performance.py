#!/usr/bin/env python3
"""
Test Vectorized Time Window Strategy Performance
Goal: Achieve <1 second backtest time
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add strategies to path
strategies_path = Path(__file__).parent / "strategies"
if str(strategies_path) not in sys.path:
    sys.path.insert(0, str(strategies_path))

from time_window_strategy_vectorized import TimeWindowVectorizedStrategy, TimeWindowVectorizedSingle

def create_realistic_test_data(n_days=5):
    """Create realistic forex-like test data"""
    
    print(f"Creating {n_days} days of realistic test data...")
    
    # Create minute-by-minute data for market hours (9:00-16:00)
    dates = pd.date_range('2024-01-15', periods=n_days, freq='D')
    all_timestamps = []
    
    for date in dates:
        # Create intraday timestamps (9:00 AM to 4:00 PM)
        day_start = date.replace(hour=9, minute=0)
        day_end = date.replace(hour=16, minute=0)
        day_timestamps = pd.date_range(day_start, day_end, freq='1min')[:-1]  # Exclude 4:00 PM
        all_timestamps.extend(day_timestamps)
    
    n_bars = len(all_timestamps)
    print(f"Generated {n_bars} bars ({n_bars/60/7:.1f} hours of data)")
    
    # Convert to nanosecond timestamps
    timestamp_ns = np.array([int(ts.timestamp() * 1_000_000_000) for ts in all_timestamps])
    
    # Create realistic forex price movement (AD currency pair style)
    base_price = 0.67500  # AUD/USD around 67.5 cents
    
    # Generate realistic price changes with some trending behavior
    returns = np.random.normal(0, 0.0005, n_bars)  # 0.05% std dev
    
    # Add some trend and volatility clustering
    trend = np.sin(np.arange(n_bars) * 2 * np.pi / (60 * 7)) * 0.001  # Weekly cycle
    volatility = 1 + 0.5 * np.sin(np.arange(n_bars) * 2 * np.pi / (60 * 2))  # 2-hour volatility cycle
    
    adjusted_returns = returns * volatility + trend
    price_changes = np.cumsum(adjusted_returns)
    close_prices = base_price + price_changes
    
    # Generate realistic OHLC
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    # Realistic spreads for forex (0.5-2 pips)
    spreads = np.random.uniform(0.00005, 0.0002, n_bars)
    high_prices = np.maximum(open_prices, close_prices) + spreads
    low_prices = np.minimum(open_prices, close_prices) - spreads
    
    # Volume (not used in strategy but needed for completeness)
    volume = np.random.randint(100, 1000, n_bars).astype(float)
    
    data = {
        'datetime': timestamp_ns,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    
    print(f"Price range: ${np.min(low_prices):.5f} - ${np.max(high_prices):.5f}")
    print(f"Avg spread: {np.mean(spreads)*10000:.1f} pips")
    
    return data

def test_single_strategy_speed():
    """Test single strategy version for baseline speed"""
    
    print("\n=== TESTING SINGLE STRATEGY SPEED ===")
    
    # Create test data
    data = create_realistic_test_data(n_days=2)  # 2 days for quick test
    
    # Create single strategy
    strategy = TimeWindowVectorizedSingle()
    
    # Config
    config = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0001
    }
    
    print(f"Testing single strategy (1 combination)...")
    
    # Time the backtest
    start_time = time.perf_counter()
    pf = strategy.run_vectorized_backtest(data, config)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    print(f"Single strategy completed in {duration:.3f} seconds")
    
    # Results
    total_return = pf.total_return
    if hasattr(total_return, 'iloc'):
        return_val = total_return.iloc[0]
    else:
        return_val = total_return
    
    print(f"Return: {return_val*100:.2f}%")
    print(f"Orders: {pf.orders.count()}")
    
    return duration < 0.5  # Single strategy should be reasonably fast

def test_multi_strategy_speed():
    """Test multiple strategy combinations for target performance"""
    
    print("\n=== TESTING MULTI-STRATEGY SPEED ===")
    
    # Create test data (more realistic size)
    data = create_realistic_test_data(n_days=5)  # 5 days = ~2100 bars
    
    # Create multi-strategy
    strategy = TimeWindowVectorizedStrategy()
    
    # Get parameter combinations
    combinations = strategy.get_parameter_combinations()
    print(f"Testing {len(combinations)} parameter combinations")
    
    # Config
    config = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0001
    }
    
    # Time the backtest
    print("Starting vectorized backtest...")
    start_time = time.perf_counter()
    pf = strategy.run_vectorized_backtest(data, config)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    
    print(f"\n=== PERFORMANCE RESULTS ===")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Target: <1.000 seconds")
    print(f"Success: {'YES' if duration < 1.0 else 'NO'}")
    print(f"Speed: {len(combinations)/duration:.0f} combinations/second")
    
    # Trading results
    total_return = pf.total_return
    if hasattr(total_return, '__len__') and len(total_return) > 1:
        returns = total_return.values if hasattr(total_return, 'values') else total_return
        best_return = np.max(returns)
        avg_return = np.mean(returns)
        print(f"Best return: {best_return*100:.2f}%")
        print(f"Avg return: {avg_return*100:.2f}%")
    else:
        return_val = total_return.iloc[0] if hasattr(total_return, 'iloc') else total_return
        print(f"Total return: {return_val*100:.2f}%")
    
    print(f"Total orders: {pf.orders.count()}")
    
    return duration < 1.0

def benchmark_against_original():
    """Compare with original strategy performance"""
    
    print("\n=== BENCHMARKING AGAINST ORIGINAL ===")
    
    # Import original strategy
    try:
        from time_window_strategy import TimeWindowSingleStrategy
        
        # Create smaller test data for comparison
        data = create_realistic_test_data(n_days=1)  # 1 day for fair comparison
        
        config = {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.0001
        }
        
        # Test original strategy
        print("Testing original strategy...")
        original_strategy = TimeWindowSingleStrategy()
        
        start_time = time.perf_counter()
        original_pf = original_strategy.run_vectorized_backtest(data, config)
        original_duration = time.perf_counter() - start_time
        
        # Test vectorized strategy
        print("Testing vectorized strategy...")
        vectorized_strategy = TimeWindowVectorizedSingle()
        
        start_time = time.perf_counter()
        vectorized_pf = vectorized_strategy.run_vectorized_backtest(data, config)
        vectorized_duration = time.perf_counter() - start_time
        
        # Compare
        speedup = original_duration / vectorized_duration
        print(f"\n=== SPEED COMPARISON ===")
        print(f"Original: {original_duration:.3f} seconds")
        print(f"Vectorized: {vectorized_duration:.3f} seconds")
        print(f"Speedup: {speedup:.1f}x faster")
        
        return speedup >= 1.5  # Should be at least 1.5x faster
        
    except ImportError:
        print("Original strategy not available for comparison")
        return True

if __name__ == "__main__":
    print("VECTORIZED TIME WINDOW STRATEGY PERFORMANCE TEST")
    print("=" * 60)
    
    success = True
    
    # Test 1: Single strategy speed
    try:
        if test_single_strategy_speed():
            print("PASS: Single strategy speed test")
        else:
            print("FAIL: Single strategy speed test")
            success = False
    except Exception as e:
        print(f"ERROR: Single strategy speed test - {e}")
        success = False
    
    # Test 2: Multi-strategy speed (main test)
    try:
        if test_multi_strategy_speed():
            print("PASS: Multi-strategy speed test (<1 second)")
        else:
            print("FAIL: Multi-strategy speed test (>1 second)")
            success = False
    except Exception as e:
        print(f"ERROR: Multi-strategy speed test - {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Test 3: Benchmark against original
    try:
        if benchmark_against_original():
            print("PASS: Benchmark test")
        else:
            print("FAIL: Benchmark test")
            success = False
    except Exception as e:
        print(f"ERROR: Benchmark test - {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: ALL TESTS PASSED - VECTORIZED STRATEGY IS READY!")
        print("The strategy achieves <1 second performance target")
        print("Ready for integration with main.py")
    else:
        print("FAILURE: SOME TESTS FAILED - NEEDS OPTIMIZATION")
        print("Review the errors above and optimize further")