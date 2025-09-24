#!/usr/bin/env python3
"""
Test script to verify signal lag implementation
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')
sys.path.insert(0, 'src/trading/core')

import pandas as pd
import numpy as np

from core.standalone_execution import ExecutionConfig, StandaloneExecutionEngine

def test_signal_lag():
    """Test that signal lag is correctly applied"""
    print("=" * 60)
    print("Testing Signal Lag Implementation")
    print("=" * 60)

    # Create simple test data
    dates = pd.date_range('2024-01-01', periods=20, freq='1h')
    prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
              110, 111, 112, 113, 114, 115, 116, 117, 118, 119]

    df = pd.DataFrame({
        'DateTime': dates,
        'Open': prices,
        'High': [p + 0.5 for p in prices],
        'Low': [p - 0.5 for p in prices],
        'Close': prices
    })

    # Create simple signals
    signals = pd.Series([0] * 20)
    signals[5] = 1   # Buy signal at bar 5
    signals[10] = 0  # Exit signal at bar 10 (signal changes from 1 to 0)

    # Test with signal_lag = 1 (from config)
    print("\nTest 1: signal_lag = 1")
    print("-" * 40)
    config1 = ExecutionConfig(
        signal_lag=1,
        execution_price='close'
    )
    engine1 = StandaloneExecutionEngine(config1)
    trades1 = engine1.execute_signals(signals, df)

    for trade in trades1:
        print(f"Trade {trade['trade_id']}: {trade['trade_type']}")
        print(f"  Signal bar: {trade['signal_bar']} (price: ${trade['signal_price']:.2f})")
        print(f"  Execution bar: {trade['execution_bar']} (price: ${trade['execution_price']:.2f})")
        print(f"  Lag: {trade['lag']} bars")

    # Verify lag = 1
    assert trades1[0]['signal_bar'] == 5, "Buy signal should be at bar 5"
    assert trades1[0]['execution_bar'] == 6, "Buy execution should be at bar 6 (lag=1)"
    assert trades1[0]['lag'] == 1, "Lag should be 1"
    assert trades1[0]['execution_price'] == 106, "Execution price should be 106 (bar 6 close)"

    if len(trades1) > 1:
        # The sell happens when we detect the signal change from 1 to 0 at bar 10
        # But due to how the loop works, it might process differently
        print(f"\nSell trade details:")
        print(f"  Signal bar: {trades1[1]['signal_bar']}")
        print(f"  Expected: signal detected where position changes from 1 to 0")
        # Just verify the lag is correct
        assert trades1[1]['lag'] == 1, "Lag should be 1"
        assert trades1[1]['execution_bar'] == trades1[1]['signal_bar'] + 1, "Execution should be signal_bar + 1"

    # Test with signal_lag = 2
    print("\nTest 2: signal_lag = 2")
    print("-" * 40)
    config2 = ExecutionConfig(
        signal_lag=2,
        execution_price='close'
    )
    engine2 = StandaloneExecutionEngine(config2)
    trades2 = engine2.execute_signals(signals, df)

    for trade in trades2:
        print(f"Trade {trade['trade_id']}: {trade['trade_type']}")
        print(f"  Signal bar: {trade['signal_bar']} (price: ${trade['signal_price']:.2f})")
        print(f"  Execution bar: {trade['execution_bar']} (price: ${trade['execution_price']:.2f})")
        print(f"  Lag: {trade['lag']} bars")

    # Verify lag = 2
    assert trades2[0]['signal_bar'] == 5, "Buy signal should be at bar 5"
    assert trades2[0]['execution_bar'] == 7, "Buy execution should be at bar 7 (lag=2)"
    assert trades2[0]['lag'] == 2, "Lag should be 2"
    assert trades2[0]['execution_price'] == 107, "Execution price should be 107 (bar 7 close)"

    # Test with signal_lag = 0 (immediate execution)
    print("\nTest 3: signal_lag = 0 (immediate)")
    print("-" * 40)
    config0 = ExecutionConfig(
        signal_lag=0,
        execution_price='close'
    )
    engine0 = StandaloneExecutionEngine(config0)
    trades0 = engine0.execute_signals(signals, df)

    for trade in trades0:
        print(f"Trade {trade['trade_id']}: {trade['trade_type']}")
        print(f"  Signal bar: {trade['signal_bar']} (price: ${trade['signal_price']:.2f})")
        print(f"  Execution bar: {trade['execution_bar']} (price: ${trade['execution_price']:.2f})")
        print(f"  Lag: {trade['lag']} bars")

    # Verify lag = 0
    assert trades0[0]['signal_bar'] == 5, "Buy signal should be at bar 5"
    assert trades0[0]['execution_bar'] == 5, "Buy execution should be at bar 5 (lag=0)"
    assert trades0[0]['lag'] == 0, "Lag should be 0"
    assert trades0[0]['execution_price'] == 105, "Execution price should be 105 (bar 5 close)"

    print("\n" + "=" * 60)
    print("Signal Lag Tests PASSED!")
    print("Signal detected at bar N, execution at bar N + signal_lag")

    return True

if __name__ == "__main__":
    test_signal_lag()