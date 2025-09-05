"""Simple test to verify vectorBT works correctly."""

import numpy as np
import vectorbtpro as vbt

# Generate test data
np.random.seed(42)
n_bars = 1000
returns = np.random.normal(0.0001, 0.01, n_bars)
close = 100 * np.exp(np.cumsum(returns))

print(f"Generated {n_bars} bars of test data")

# Test simple MA crossover
fast_ma = vbt.MA.run(close, 20).ma
slow_ma = vbt.MA.run(close, 50).ma

print(f"Fast MA shape: {fast_ma.shape}")
print(f"Slow MA shape: {slow_ma.shape}")

# Generate simple signals
entries = fast_ma > slow_ma
exits = fast_ma < slow_ma

print(f"Entries shape: {entries.shape}")
print(f"Exits shape: {exits.shape}")
print(f"Entry signals: {np.sum(entries)}")

# Test single backtest
try:
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=0.001
    )
    print(f"Single backtest successful!")
    print(f"Total return: {pf.total_return():.2%}")
except Exception as e:
    print(f"Single backtest failed: {e}")

# Test multiple parameter backtest
try:
    # Create multiple MA periods
    fast_periods = [10, 15, 20]
    slow_periods = [30, 40, 50]
    
    # Use vectorBT's parameter optimization
    ma_fast = vbt.MA.run(close, fast_periods, short_name='fast')
    ma_slow = vbt.MA.run(close, slow_periods, short_name='slow')
    
    print(f"Fast MA array shape: {ma_fast.ma.shape}")
    print(f"Slow MA array shape: {ma_slow.ma.shape}")
    
    # This should work for parameter optimization
    entries_multi = ma_fast.ma > ma_slow.ma
    exits_multi = ma_fast.ma < ma_slow.ma
    
    print(f"Multi entries shape: {entries_multi.shape}")
    
    pf_multi = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_multi,
        exits=exits_multi,
        init_cash=100000,
        fees=0.001
    )
    
    print(f"Multi-parameter backtest successful!")
    print(f"Results shape: {pf_multi.total_return().shape}")
    
except Exception as e:
    print(f"Multi-parameter backtest failed: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")