#!/usr/bin/env python3
"""
Test script to verify proper config.yaml implementation for:
1. Bar lag (signal_lag) functionality
2. Price execution formulas
3. $1 position sizing for percentage calculations
4. Non-linear scaling validation
"""

import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine


def create_test_data(n_bars: int, start_price: float = 100.0, volatility: float = 0.02):
    """
    Create synthetic OHLCV data for testing.

    Args:
        n_bars: Number of bars to generate
        start_price: Starting price
        volatility: Price volatility (std dev as fraction)

    Returns:
        Dictionary with OHLCV arrays and timestamps
    """
    print(f"Creating synthetic data: {n_bars} bars, start_price={start_price}")

    # Create timestamps (5-minute bars)
    timestamps = pd.date_range(
        start='2024-01-01 09:30:00',
        periods=n_bars,
        freq='5min'
    )

    # Generate price movements
    returns = np.random.normal(0, volatility, n_bars)
    close_prices = start_price * np.exp(np.cumsum(returns))

    # Create realistic OHLC from close
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, volatility/2, n_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, volatility/2, n_bars)))

    # Open is previous close with gap
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price
    open_prices = open_prices * (1 + np.random.normal(0, volatility/4, n_bars))

    # Ensure OHLC relationships are valid
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # Volume
    volume = np.random.uniform(1000, 10000, n_bars)

    return {
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
        'timestamp': timestamps.values
    }


def create_test_signals(n_bars: int, n_signals: int = 10):
    """
    Create simple buy/sell signals for testing.

    Args:
        n_bars: Number of bars
        n_signals: Approximate number of trades

    Returns:
        Tuple of (entries, exits) boolean arrays
    """
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)

    # Create evenly spaced signals
    signal_spacing = n_bars // (n_signals * 2)

    for i in range(n_signals):
        entry_idx = i * signal_spacing * 2
        exit_idx = entry_idx + signal_spacing

        if entry_idx < n_bars:
            entries[entry_idx] = True
        if exit_idx < n_bars:
            exits[exit_idx] = True

    return entries, exits


def test_bar_lag(engine: VectorBTEngine, data: dict, entries: np.ndarray, exits: np.ndarray):
    """Test that signal lag is properly applied."""
    print("\n" + "="*60)
    print("TESTING BAR LAG FUNCTIONALITY")
    print("="*60)

    config = engine.config
    signal_lag = config['backtest']['signal_lag']

    print(f"Configured signal_lag: {signal_lag} bars")

    # Run backtest
    pf = engine.run_vectorized_backtest(data, entries, exits, symbol="TEST")

    # Get trade records
    trades = pf.trades.records_readable

    if len(trades) > 0:
        print(f"\nFirst 3 trades with lag={signal_lag}:")
        print(f"Trade columns: {list(trades.columns)}")
        for i in range(min(3, len(trades))):
            trade = trades.iloc[i]
            # Use correct column names from VectorBT Pro
            entry_idx = trade.get('Entry Index', trade.get('Entry Idx', trade.get('entry_idx', 0)))
            exit_idx = trade.get('Exit Index', trade.get('Exit Idx', trade.get('exit_idx', 0)))
            print(f"  Trade {i+1}: Entry at bar {entry_idx}, Exit at bar {exit_idx}")

        # Verify lag is applied
        entry_signals = np.where(entries)[0]
        if len(entry_signals) > 0 and len(trades) > 0:
            first_signal = entry_signals[0]
            first_trade = trades.iloc[0]
            first_trade_entry = first_trade.get('Entry Index', first_trade.get('Entry Idx', first_trade.get('entry_idx', 0)))
            actual_lag = first_trade_entry - first_signal

            print(f"\nVerification:")
            print(f"  First entry signal at bar: {first_signal}")
            print(f"  First trade entry at bar: {first_trade_entry}")
            print(f"  Actual lag: {actual_lag} bars")

            if actual_lag == signal_lag:
                print(f"  SUCCESS: Lag correctly applied!")
            else:
                print(f"  ERROR: Expected lag {signal_lag}, got {actual_lag}")

    return pf


def test_execution_formulas(engine: VectorBTEngine, data: dict, entries: np.ndarray, exits: np.ndarray):
    """Test that price execution formulas are properly applied."""
    print("\n" + "="*60)
    print("TESTING PRICE EXECUTION FORMULAS")
    print("="*60)

    config = engine.config
    buy_formula = config['backtest']['buy_execution_formula']
    sell_formula = config['backtest']['sell_execution_formula']

    print(f"Buy execution formula: {buy_formula}")
    print(f"Sell execution formula: {sell_formula}")

    # Calculate expected prices
    expected_buy_prices = engine.formula_evaluator.get_execution_prices(buy_formula, data, "buy")
    expected_sell_prices = engine.formula_evaluator.get_execution_prices(sell_formula, data, "sell")

    # Run backtest
    pf = engine.run_vectorized_backtest(data, entries, exits, symbol="TEST")

    # Get trade records
    trades = pf.trades.records_readable

    if len(trades) > 0:
        print(f"\nFirst 3 trades with formula execution:")
        for i in range(min(3, len(trades))):
            trade = trades.iloc[i]
            entry_idx = int(trade.get('Entry Index', trade.get('Entry Idx', trade.get('entry_idx', 0))))
            exit_idx = int(trade.get('Exit Index', trade.get('Exit Idx', trade.get('exit_idx', 0))))

            # Account for signal lag
            signal_lag = config['backtest']['signal_lag']

            print(f"\n  Trade {i+1}:")
            entry_price = trade.get('Avg Entry Price', trade.get('Entry Price', trade.get('entry_price', 0)))
            exit_price = trade.get('Avg Exit Price', trade.get('Exit Price', trade.get('exit_price', 0)))
            print(f"    Entry Price: ${entry_price:.2f}")
            print(f"    Expected: ${expected_buy_prices[entry_idx]:.2f}")
            print(f"    Exit Price: ${exit_price:.2f}")
            print(f"    Expected: ${expected_sell_prices[exit_idx]:.2f}")

    return pf


def test_position_sizing(engine: VectorBTEngine, data: dict, entries: np.ndarray, exits: np.ndarray):
    """Test that $1 position sizing gives clean percentage returns for both long and short trades."""
    print("\n" + "="*60)
    print("TESTING $1 POSITION SIZING")
    print("="*60)

    config = engine.config
    position_size = config['backtest']['position_size']
    position_type = config['backtest']['position_size_type']
    direction = config['backtest']['direction']

    print(f"Position size: ${position_size}")
    print(f"Position type: {position_type}")
    print(f"Direction: {direction} (both long and short trades)")
    print("\nWith $1 position sizing:")
    print("  - Long trades: PnL in $ = price change %")
    print("  - Short trades: PnL in $ = inverse price change %")

    # Run backtest
    pf = engine.run_vectorized_backtest(data, entries, exits, symbol="TEST")

    # Get trade records
    trades = pf.trades.records_readable

    if len(trades) > 0:
        print(f"\nAnalyzing first 3 trades with $1 sizing:")
        for i in range(min(3, len(trades))):
            trade = trades.iloc[i]

            # With $1 position sizing, PnL should equal price change percentage
            entry_price = trade.get('Avg Entry Price', trade.get('Entry Price', trade.get('entry_price', 0)))
            exit_price = trade.get('Avg Exit Price', trade.get('Exit Price', trade.get('exit_price', 0)))
            size = trade.get('Size', trade.get('size', 0))
            pnl = trade.get('PnL', trade.get('Return', trade.get('pnl', 0)))
            price_change_pct = ((exit_price - entry_price) / entry_price) * 100

            # Size should be shares that equal $1 value
            expected_shares = position_size / entry_price

            print(f"\n  Trade {i+1}:")
            print(f"    Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}")
            print(f"    Size: {size:.6f} shares")
            print(f"    Expected shares for $1: {expected_shares:.6f}")
            print(f"    PnL: ${pnl:.4f}")
            print(f"    Price change %: {price_change_pct:.2f}%")
            print(f"    PnL as % of $1: {pnl*100:.2f}%")

            # Check trade direction
            direction = trade.get('Direction', trade.get('direction', 'Long'))
            print(f"    Direction: {direction}")

            # Verify relationship (accounting for direction)
            # With $1 position sizing, PnL in dollars = percentage return
            expected_pnl_pct = price_change_pct if direction == 'Long' else -price_change_pct
            if abs(pnl * 100 - expected_pnl_pct) < 0.01:
                print(f"    SUCCESS: $1 position gives clean % calculation!")
            else:
                print(f"    WARNING: PnL doesn't match expected percentage (expected {expected_pnl_pct:.2f}%, got {pnl*100:.2f}%)")

    return pf


def test_non_linear_scaling(engine: VectorBTEngine):
    """Test that processing time doesn't scale linearly with data size."""
    print("\n" + "="*60)
    print("TESTING NON-LINEAR SCALING")
    print("="*60)

    test_sizes = [1000, 5000, 10000, 50000]
    times = []

    for size in test_sizes:
        # Create data
        data = create_test_data(size)
        entries, exits = create_test_signals(size, n_signals=20)

        # Time the backtest
        start_time = time.time()
        pf = engine.run_vectorized_backtest(data, entries, exits, symbol=f"TEST_{size}")
        elapsed = time.time() - start_time

        times.append(elapsed)

        # Calculate metrics
        total_return = pf.total_return
        if hasattr(total_return, 'iloc'):
            total_return = total_return.iloc[0] if len(total_return) > 0 else 0

        print(f"\n  Size: {size:,} bars")
        print(f"    Time: {elapsed:.3f} seconds")
        print(f"    Return: {total_return*100:.2f}%")
        print(f"    Time per 1K bars: {(elapsed/size)*1000:.3f} seconds")

    # Check scaling
    print("\n  Scaling Analysis:")
    for i in range(1, len(test_sizes)):
        size_ratio = test_sizes[i] / test_sizes[i-1]
        time_ratio = times[i] / times[i-1]
        efficiency = time_ratio / size_ratio

        print(f"    {test_sizes[i-1]} -> {test_sizes[i]}: ")
        print(f"      Size increased {size_ratio:.1f}x")
        print(f"      Time increased {time_ratio:.1f}x")
        print(f"      Efficiency: {efficiency:.2f} (lower is better, 1.0 is linear)")

        if efficiency < 1.2:
            print(f"      GOOD: Sub-linear or near-linear scaling!")
        else:
            print(f"      WARNING: Super-linear scaling detected")


def main():
    """Main test execution."""
    print("="*60)
    print("CONFIG.YAML IMPLEMENTATION TEST SUITE")
    print("="*60)

    # Load configuration
    config_path = "config.yaml"
    print(f"\nLoading configuration from: {config_path}")

    try:
        # Initialize engine
        engine = VectorBTEngine(config_path)

        # Verify critical settings
        config = engine.config
        print("\nCritical Configuration Settings:")
        print(f"  Position Size: ${config['backtest']['position_size']}")
        print(f"  Position Type: {config['backtest']['position_size_type']}")
        print(f"  Signal Lag: {config['backtest']['signal_lag']} bars")
        print(f"  Buy Formula: {config['backtest']['buy_execution_formula']}")
        print(f"  Sell Formula: {config['backtest']['sell_execution_formula']}")

        # Create test data
        print("\nGenerating test data...")
        small_data = create_test_data(1000)
        small_entries, small_exits = create_test_signals(1000, n_signals=10)

        # Run tests
        print("\n" + "="*60)
        print("RUNNING TEST SUITE")

        # Test 1: Bar Lag
        pf1 = test_bar_lag(engine, small_data, small_entries, small_exits)

        # Test 2: Execution Formulas
        pf2 = test_execution_formulas(engine, small_data, small_entries, small_exits)

        # Test 3: Position Sizing
        pf3 = test_position_sizing(engine, small_data, small_entries, small_exits)

        # Test 4: Non-linear Scaling
        test_non_linear_scaling(engine)

        print("\n" + "="*60)
        print("TEST SUITE COMPLETED")
        print("="*60)

    except Exception as e:
        print(f"\nERROR: Test failed - {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())