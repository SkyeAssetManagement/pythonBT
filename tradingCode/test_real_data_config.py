#!/usr/bin/env python3
"""
Test config.yaml implementation with real market data to verify:
1. Bar lag functionality
2. Execution price formulas
3. $1 position sizing for clean percentage calculations
"""

import numpy as np
import pandas as pd
import yaml
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.parquet_converter import ParquetConverter
from src.backtest.vbt_engine import VectorBTEngine
from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy


def main():
    """Test with real market data."""
    print("="*60)
    print("REAL DATA CONFIG TEST")
    print("="*60)

    # Load configuration
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\nConfiguration Settings:")
    print(f"  Position Size: ${config['backtest']['position_size']}")
    print(f"  Position Type: {config['backtest']['position_size_type']}")
    print(f"  Signal Lag: {config['backtest']['signal_lag']} bars")
    print(f"  Direction: {config['backtest']['direction']}")
    print(f"  Buy Formula: {config['backtest']['buy_execution_formula']}")
    print(f"  Sell Formula: {config['backtest']['sell_execution_formula']}")

    # Load real data
    symbol = "AAPL"
    print(f"\nLoading {symbol} data...")

    converter = ParquetConverter()
    data = converter.load_or_convert(symbol, "1m", "diffAdjusted")

    # Use a subset for testing (last 10000 bars)
    test_bars = 10000
    if len(data['close']) > test_bars:
        print(f"Using last {test_bars} bars for testing...")
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key][-test_bars:]

    print(f"Data loaded: {len(data['close'])} bars")
    print(f"Date range: {pd.Timestamp(data['timestamp'][0], unit='ns')} to {pd.Timestamp(data['timestamp'][-1], unit='ns')}")

    # Initialize strategy
    print("\nInitializing strategy...")
    strategy = TimeWindowVectorizedStrategy()

    # Get default parameters
    params = strategy.get_parameter_combinations(use_defaults_only=True)[0]
    print(f"Strategy parameters: {params}")

    # Generate signals
    print("\nGenerating signals...")
    entries, exits = strategy.generate_signals(data, **params)

    n_entries = np.sum(entries)
    n_exits = np.sum(exits)
    print(f"Generated {n_entries} entry signals, {n_exits} exit signals")

    # Run backtest with config settings
    print("\nRunning backtest with config settings...")
    engine = VectorBTEngine(config_path)

    start_time = time.time()
    pf = engine.run_vectorized_backtest(data, entries, exits, symbol=symbol)
    elapsed = time.time() - start_time

    print(f"Backtest completed in {elapsed:.2f} seconds")

    # Analyze results
    print("\n" + "="*60)
    print("BACKTEST RESULTS ANALYSIS")
    print("="*60)

    # Get trades
    trades = pf.trades.records_readable
    n_trades = len(trades)

    print(f"\nTotal trades: {n_trades}")

    if n_trades > 0:
        # Analyze first few trades to verify config settings
        print("\nFirst 5 trades analysis:")
        print("-" * 50)

        for i in range(min(5, n_trades)):
            trade = trades.iloc[i]

            entry_idx = int(trade.get('Entry Index', 0))
            exit_idx = int(trade.get('Exit Index', 0))
            entry_price = trade.get('Avg Entry Price', 0)
            exit_price = trade.get('Avg Exit Price', 0)
            size = trade.get('Size', 0)
            pnl = trade.get('PnL', 0)
            direction = trade.get('Direction', 'Long')

            # Calculate price change percentage
            if direction == 'Long':
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100

            # With $1 position sizing, PnL should equal percentage
            position_value = size * entry_price

            print(f"\nTrade {i+1} ({direction}):")
            print(f"  Entry: Bar {entry_idx}, Price ${entry_price:.2f}")
            print(f"  Exit: Bar {exit_idx}, Price ${exit_price:.2f}")
            print(f"  Size: {size:.6f} shares (${position_value:.2f} value)")
            print(f"  PnL: ${pnl:.4f}")
            print(f"  Price Change: {price_change_pct:.2f}%")

            # Verify $1 position sizing
            if abs(position_value - config['backtest']['position_size']) < 0.01:
                print(f"  ✓ Position size correctly set to ${config['backtest']['position_size']}")
            else:
                print(f"  ✗ Position size mismatch: expected ${config['backtest']['position_size']}, got ${position_value:.2f}")

            # Verify PnL matches percentage
            if abs(pnl * 100 - price_change_pct) < 0.1:
                print(f"  ✓ PnL matches percentage (${pnl:.4f} = {pnl*100:.2f}%)")
            else:
                print(f"  ✗ PnL doesn't match percentage")

        # Overall statistics
        print("\n" + "="*60)
        print("OVERALL STATISTICS")
        print("="*60)

        total_return = pf.total_return
        if hasattr(total_return, 'iloc'):
            total_return = total_return.iloc[0]

        win_rate = pf.win_rate
        if hasattr(win_rate, 'iloc'):
            win_rate = win_rate.iloc[0]

        avg_win = trades[trades['PnL'] > 0]['PnL'].mean() if len(trades[trades['PnL'] > 0]) > 0 else 0
        avg_loss = trades[trades['PnL'] < 0]['PnL'].mean() if len(trades[trades['PnL'] < 0]) > 0 else 0

        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Win Rate: {win_rate*100:.1f}%")
        print(f"Average Win: ${avg_win:.4f} ({avg_win*100:.2f}%)")
        print(f"Average Loss: ${avg_loss:.4f} ({avg_loss*100:.2f}%)")
        print(f"Number of Trades: {n_trades}")

        # Verify bar lag
        print("\n" + "="*60)
        print("BAR LAG VERIFICATION")
        print("="*60)

        # Find first entry signal
        entry_signal_bars = np.where(entries)[0]
        if len(entry_signal_bars) > 0 and n_trades > 0:
            first_signal = entry_signal_bars[0]
            first_trade_entry = trades.iloc[0].get('Entry Index', 0)
            lag = first_trade_entry - first_signal

            print(f"First entry signal at bar: {first_signal}")
            print(f"First trade entry at bar: {first_trade_entry}")
            print(f"Measured lag: {lag} bars")
            print(f"Configured lag: {config['backtest']['signal_lag']} bars")

            if lag == config['backtest']['signal_lag']:
                print("✓ Bar lag correctly applied!")
            else:
                print("✗ Bar lag mismatch!")

    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())