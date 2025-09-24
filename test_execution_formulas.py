#!/usr/bin/env python3
"""
Test script to verify execution price formula implementation
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')
sys.path.insert(0, 'src/trading/core')

import pandas as pd
import numpy as np

from core.standalone_execution import ExecutionConfig, StandaloneExecutionEngine

def test_execution_formulas():
    """Test that execution price formulas work correctly"""
    print("=" * 60)
    print("Testing Execution Price Formulas")
    print("=" * 60)

    # Create test data with specific OHLC values
    dates = pd.date_range('2024-01-01', periods=10, freq='1h')

    # Create predictable OHLC data
    df = pd.DataFrame({
        'DateTime': dates,
        'Open': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
        'High': [101, 103, 105, 107, 109, 111, 113, 115, 117, 119],
        'Low': [99, 101, 103, 105, 107, 109, 111, 113, 115, 117],
        'Close': [100.5, 102.5, 104.5, 106.5, 108.5, 110.5, 112.5, 114.5, 116.5, 118.5]
    })

    # Create simple signals
    signals = pd.Series([0] * 10)
    signals[2] = 1   # Buy signal at bar 2
    signals[5] = 0   # Sell signal at bar 5

    # Test 1: Close price (default)
    print("\nTest 1: Close price execution")
    print("-" * 40)
    config1 = ExecutionConfig(
        signal_lag=1,
        execution_price='close',
        buy_execution_formula='C',
        sell_execution_formula='C'
    )
    engine1 = StandaloneExecutionEngine(config1)
    trades1 = engine1.execute_signals(signals, df)

    for trade in trades1:
        print(f"Trade {trade['trade_id']}: {trade['trade_type']}")
        print(f"  Execution bar: {trade['execution_bar']}")
        print(f"  Execution price: ${trade['execution_price']:.2f}")
        print(f"  Formula used: {trade['formula']}")

        # Verify close price
        expected_close = df['Close'].iloc[trade['execution_bar']]
        assert abs(trade['execution_price'] - expected_close) < 0.01, f"Expected close price {expected_close}"

    # Test 2: Formula - (H+L+C)/3 (Typical Price)
    print("\nTest 2: (H+L+C)/3 formula (Typical Price)")
    print("-" * 40)
    config2 = ExecutionConfig(
        signal_lag=1,
        execution_price='formula',
        buy_execution_formula='(H + L + C) / 3',
        sell_execution_formula='(H + L + C) / 3'
    )
    engine2 = StandaloneExecutionEngine(config2)
    trades2 = engine2.execute_signals(signals, df)

    for trade in trades2:
        exec_bar = trade['execution_bar']
        H = df['High'].iloc[exec_bar]
        L = df['Low'].iloc[exec_bar]
        C = df['Close'].iloc[exec_bar]
        expected_price = (H + L + C) / 3

        print(f"Trade {trade['trade_id']}: {trade['trade_type']}")
        print(f"  Execution bar: {exec_bar}")
        print(f"  H={H:.2f}, L={L:.2f}, C={C:.2f}")
        print(f"  Expected: ({H:.2f} + {L:.2f} + {C:.2f}) / 3 = {expected_price:.2f}")
        print(f"  Actual price: ${trade['execution_price']:.2f}")
        print(f"  Formula used: {trade['formula']}")

        assert abs(trade['execution_price'] - expected_price) < 0.01, f"Formula calculation mismatch"

    # Test 3: Different formulas for buy and sell
    print("\nTest 3: Different formulas - Buy at (H+L)/2, Sell at (O+C)/2")
    print("-" * 40)
    config3 = ExecutionConfig(
        signal_lag=1,
        execution_price='formula',
        buy_execution_formula='(H + L) / 2',  # Median price for buys
        sell_execution_formula='(O + C) / 2'   # Average of open/close for sells
    )
    engine3 = StandaloneExecutionEngine(config3)
    trades3 = engine3.execute_signals(signals, df)

    for trade in trades3:
        exec_bar = trade['execution_bar']
        O = df['Open'].iloc[exec_bar]
        H = df['High'].iloc[exec_bar]
        L = df['Low'].iloc[exec_bar]
        C = df['Close'].iloc[exec_bar]

        if trade['trade_type'] == 'BUY':
            expected_price = (H + L) / 2
            print(f"BUY Trade:")
            print(f"  Expected: ({H:.2f} + {L:.2f}) / 2 = {expected_price:.2f}")
        else:
            expected_price = (O + C) / 2
            print(f"SELL Trade:")
            print(f"  Expected: ({O:.2f} + {C:.2f}) / 2 = {expected_price:.2f}")

        print(f"  Actual price: ${trade['execution_price']:.2f}")
        print(f"  Formula used: {trade['formula']}")

        assert abs(trade['execution_price'] - expected_price) < 0.01, f"Formula calculation mismatch"

    # Test 4: Open price execution
    print("\nTest 4: Open price execution")
    print("-" * 40)
    config4 = ExecutionConfig(
        signal_lag=1,
        execution_price='open'
    )
    engine4 = StandaloneExecutionEngine(config4)
    trades4 = engine4.execute_signals(signals, df)

    for trade in trades4:
        exec_bar = trade['execution_bar']
        expected_open = df['Open'].iloc[exec_bar]

        print(f"Trade {trade['trade_id']}: {trade['trade_type']}")
        print(f"  Execution bar: {exec_bar}")
        print(f"  Expected open: ${expected_open:.2f}")
        print(f"  Actual price: ${trade['execution_price']:.2f}")

        assert abs(trade['execution_price'] - expected_open) < 0.01, f"Expected open price {expected_open}"

    print("\n" + "=" * 60)
    print("Execution Formula Tests PASSED!")
    print("Formulas correctly evaluate OHLC values")

    return True

if __name__ == "__main__":
    test_execution_formulas()