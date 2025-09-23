#!/usr/bin/env python3
"""
Test SMA crossover execution to verify:
1. Trades execute on the bar AFTER crossover detection (with lag=1)
2. Execution prices use the formula from config.yaml
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine
from strategies.simpleSMA import SimpleSMAStrategy


def main():
    """Test SMA crossover with detailed signal and execution analysis."""
    print("="*60)
    print("SMA CROSSOVER EXECUTION TEST")
    print("="*60)

    # Load configuration
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\nConfiguration:")
    print(f"  Signal Lag: {config['backtest']['signal_lag']} bars")
    print(f"  Buy Formula: {config['backtest']['buy_execution_formula']}")
    print(f"  Position Size: ${config['backtest']['position_size']}")

    # Create synthetic data with clear crossover
    n_bars = 200  # Need more bars for 100-period SMA
    timestamps = pd.date_range(start='2024-01-01 09:30:00', periods=n_bars, freq='5min')

    # Create prices that will generate clear crossovers
    # Use a sine wave pattern that will create predictable MA crossovers
    t = np.linspace(0, 4 * np.pi, n_bars)

    # Fast oscillation for price
    prices = 100 + 5 * np.sin(t) + 2 * np.sin(3 * t)

    # Add some trend to ensure crossovers
    prices[:80] = prices[:80] - np.linspace(5, 0, 80)  # Downtrend
    prices[80:120] = prices[80:120] + np.linspace(0, 15, 40)  # Sharp uptrend
    prices[120:160] = prices[120:160] + 10  # Plateau
    prices[160:] = prices[160:] - np.linspace(0, 12, 40)  # Downtrend

    # Create OHLC data
    data = {
        'open': prices * 0.995,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.ones(n_bars) * 1000,
        'timestamp': timestamps.values
    }

    print(f"\nGenerated {n_bars} bars of test data")

    # Initialize strategy and generate signals
    strategy = SimpleSMAStrategy()

    # Calculate the SMAs to debug
    import vectorbtpro as vbt
    close_prices = data['close']
    fast_ma = vbt.MA.run(close_prices, 20).ma.values
    slow_ma = vbt.MA.run(close_prices, 100).ma.values

    print(f"\nSMA values at key points:")
    print(f"  Bar 100: Fast MA={fast_ma[100]:.2f}, Slow MA={slow_ma[100]:.2f}")
    print(f"  Bar 120: Fast MA={fast_ma[120]:.2f}, Slow MA={slow_ma[120]:.2f}")
    print(f"  Bar 140: Fast MA={fast_ma[140]:.2f}, Slow MA={slow_ma[140]:.2f}")

    entries, exits = strategy.generate_signals(data)

    # Find crossover points
    entry_bars = np.where(entries)[0]
    exit_bars = np.where(exits)[0]

    print(f"\nSignals detected:")
    print(f"  Entry signals at bars: {entry_bars}")
    print(f"  Exit signals at bars: {exit_bars}")

    # Run backtest
    engine = VectorBTEngine(config_path)
    pf = engine.run_vectorized_backtest(data, entries, exits, symbol="TEST_SMA")

    # Get trades
    trades = pf.trades.records_readable

    if len(trades) > 0:
        print(f"\nTrades executed: {len(trades)}")
        print("-" * 50)

        for i in range(min(3, len(trades))):
            trade = trades.iloc[i]
            entry_idx = int(trade.get('Entry Index', 0))
            exit_idx = int(trade.get('Exit Index', 0))
            entry_price = trade.get('Avg Entry Price', 0)
            exit_price = trade.get('Avg Exit Price', 0)
            size = trade.get('Size', 0)

            # Calculate expected formula prices
            expected_entry_formula = (data['high'][entry_idx] + data['low'][entry_idx] + data['close'][entry_idx]) / 3
            expected_exit_formula = (data['high'][exit_idx] + data['low'][exit_idx] + data['close'][exit_idx]) / 3

            print(f"\nTrade {i+1}:")
            print(f"  Signal Detection:")
            if len(entry_bars) > i:
                print(f"    Entry signal at bar: {entry_bars[i]}")
                print(f"    Trade entry at bar: {entry_idx}")
                print(f"    Lag applied: {entry_idx - entry_bars[i]} bars")

            print(f"  Execution Prices:")
            print(f"    Entry: ${entry_price:.2f}")
            print(f"    Expected (formula): ${expected_entry_formula:.2f}")
            print(f"    Difference: ${abs(entry_price - expected_entry_formula):.2f}")

            print(f"  Position Size:")
            print(f"    Size: {size:.6f} shares")
            print(f"    Value: ${size * entry_price:.2f}")

        # Verify lag is correctly applied
        if len(entry_bars) > 0 and len(trades) > 0:
            first_signal_bar = entry_bars[0]
            first_trade_bar = int(trades.iloc[0].get('Entry Index', 0))
            actual_lag = first_trade_bar - first_signal_bar

            print("\n" + "="*60)
            print("LAG VERIFICATION")
            print("="*60)
            print(f"Configured lag: {config['backtest']['signal_lag']} bars")
            print(f"Actual lag: {actual_lag} bars")

            if actual_lag == config['backtest']['signal_lag']:
                print("✓ Bar lag correctly applied!")
            else:
                print("✗ Bar lag mismatch!")

    else:
        print("\nNo trades executed")

    return 0


if __name__ == "__main__":
    sys.exit(main())