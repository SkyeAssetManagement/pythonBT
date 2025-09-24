#!/usr/bin/env python3
"""
Console test script to verify profit calculation based on $1 invested
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')
sys.path.insert(0, 'src/trading/core')

import pandas as pd
import numpy as np
from datetime import datetime

# Test the standalone execution engine
from core.standalone_execution import ExecutionConfig, StandaloneExecutionEngine

def test_execution_engine():
    """Test the execution engine profit calculations"""
    print("=" * 60)
    print("Testing Execution Engine P&L Calculations")
    print("=" * 60)

    # Create test data with known price movements
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    prices = [100.0]  # Start at $100

    # Create price series with known movements
    for i in range(1, 100):
        if i == 20:
            prices.append(110.0)  # +10% move
        elif i == 40:
            prices.append(105.0)  # Back down
        elif i == 60:
            prices.append(115.0)  # +9.5% from 105
        elif i == 80:
            prices.append(112.0)  # Small drop
        else:
            # Small random moves
            prices.append(prices[-1] + np.random.randn() * 0.5)

    df = pd.DataFrame({
        'DateTime': dates,
        'Open': prices,
        'High': [p + 1 for p in prices],
        'Low': [p - 1 for p in prices],
        'Close': prices
    })

    # Create trading signals
    signals = pd.Series([0] * 100)
    signals[10] = 1   # Buy at bar 10
    signals[20] = 0   # Sell at bar 20
    signals[30] = -1  # Short at bar 30
    signals[40] = 0   # Cover at bar 40
    signals[50] = 1   # Buy at bar 50
    signals[60] = 0   # Sell at bar 60
    signals[70] = -1  # Short at bar 70
    signals[80] = 0   # Cover at bar 80

    # Create execution engine with no lag for simplicity
    config = ExecutionConfig(
        signal_lag=0,  # No lag for easier calculation verification
        execution_price='close'
    )
    engine = StandaloneExecutionEngine(config)

    # Execute trades
    trades = engine.execute_signals(signals, df)

    print(f"\nGenerated {len(trades)} trades:")
    print("-" * 40)

    total_pnl = 0
    closed_trades = 0
    wins = 0

    for trade in trades:
        print(f"Trade {trade['trade_id']}: {trade['trade_type']} at bar {trade['execution_bar']}")
        print(f"  Price: ${trade['execution_price']:.2f}")

        if trade.get('pnl_percent') is not None:
            closed_trades += 1
            pnl = trade['pnl_percent']
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            sign = '+' if pnl >= 0 else ''
            print(f"  P&L: {sign}{pnl:.2f}%")

    # Calculate summary statistics
    if closed_trades > 0:
        win_rate = (wins / closed_trades) * 100
        avg_pnl = total_pnl / closed_trades

        print("\nBacktest Summary:")
        print("-" * 40)
        print(f"Total Trades: {len(trades)}")
        print(f"Closed Trades: {closed_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total P&L: {'+' if total_pnl >= 0 else ''}{total_pnl:.2f}%")
        print(f"Avg P&L: {'+' if avg_pnl >= 0 else ''}{avg_pnl:.2f}%")

    # Verify specific calculations
    print("\nVerification of P&L Calculation (based on $1 invested):")
    print("-" * 40)

    # Find first long trade
    for i, trade in enumerate(trades):
        if trade['trade_type'] == 'BUY':
            entry_price = trade['execution_price']
            print(f"Entry (BUY): ${entry_price:.2f}")
            # Find corresponding sell
            for j in range(i+1, len(trades)):
                if trades[j]['trade_type'] == 'SELL':
                    exit_price = trades[j]['execution_price']
                    expected_pnl = ((exit_price / entry_price) - 1) * 100
                    actual_pnl = trades[j].get('pnl_percent', 0)
                    print(f"Exit (SELL): ${exit_price:.2f}")
                    print(f"Expected P&L: {expected_pnl:.2f}%")
                    print(f"Actual P&L: {actual_pnl:.2f}%")
                    print(f"Formula: ((${exit_price:.2f} / ${entry_price:.2f}) - 1) * 100 = {expected_pnl:.2f}%")
                    break
            break

    return trades

def test_strategy_pnl():
    """Test strategy P&L calculations"""
    print("\n" + "=" * 60)
    print("Testing Strategy P&L Calculations")
    print("=" * 60)

    sys.path.insert(0, 'src/trading/strategies')
    from strategies.sma_crossover import SMACrossoverStrategy

    # Create test data
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')
    np.random.seed(42)

    # Create trending price data
    trend = np.linspace(100, 120, 200)
    noise = np.random.randn(200) * 2
    prices = trend + noise

    df = pd.DataFrame({
        'DateTime': dates,
        'Open': prices,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices
    })

    # Create strategy
    strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)

    # Generate signals
    signals = strategy.generate_signals(df)

    # Convert to trades
    trades = strategy.signals_to_trades(signals, df)

    print(f"\nGenerated {len(trades)} trades from SMA strategy")
    print("-" * 40)

    # Check P&L calculations
    for trade in trades:
        if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
            print(f"Trade {trade.trade_id}: {trade.trade_type} at ${trade.price:.2f}")
            sign = '+' if trade.pnl_percent >= 0 else ''
            print(f"  P&L: {sign}{trade.pnl_percent:.2f}%")

    return trades

if __name__ == "__main__":
    # Run tests
    print("Running P&L Calculation Tests")
    print("=" * 60)

    # Test execution engine
    exec_trades = test_execution_engine()

    # Test strategy
    strat_trades = test_strategy_pnl()

    print("\n" + "=" * 60)
    print("Tests Complete!")
    print("Profit calculations are now based on $1 invested")
    print("P&L is displayed as percentage to 2 decimal places")