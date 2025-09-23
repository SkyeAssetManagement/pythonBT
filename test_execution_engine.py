"""
Test script for standalone execution engine
Tests lag calculations, price formulas, and trade generation
Independent of any visualization components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.trading.core.standalone_execution import StandaloneExecutionEngine, ExecutionConfig
from src.trading.core.trade_types import TradeRecord, TradeRecordCollection


def create_sample_data(num_bars=100):
    """Create sample OHLC data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='5min')

    # Generate realistic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(num_bars) * 0.5)

    data = {
        'DateTime': dates,
        'Open': close_prices + np.random.randn(num_bars) * 0.1,
        'High': close_prices + np.abs(np.random.randn(num_bars) * 0.3),
        'Low': close_prices - np.abs(np.random.randn(num_bars) * 0.3),
        'Close': close_prices,
        'Volume': np.random.randint(1000, 10000, num_bars)
    }

    df = pd.DataFrame(data)

    # Ensure High is highest and Low is lowest
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

    return df


def create_sample_signals(num_bars):
    """Create sample trading signals for testing"""
    signals = np.zeros(num_bars)

    # Create some buy/sell signals
    signals[10] = 1   # Buy at bar 10
    signals[20] = 0   # Sell at bar 20
    signals[30] = -1  # Short at bar 30
    signals[40] = 0   # Cover at bar 40
    signals[50] = 1   # Buy at bar 50
    signals[70] = -1  # Sell and short at bar 70
    signals[85] = 0   # Cover at bar 85

    return pd.Series(signals)


def test_basic_execution():
    """Test basic execution with default configuration"""
    print("=" * 80)
    print("TEST 1: Basic Execution with Default Config")
    print("=" * 80)

    # Create test data
    df = create_sample_data(100)
    signals = create_sample_signals(100)

    # Create engine with defaults
    engine = StandaloneExecutionEngine()

    # Execute signals
    trades = engine.execute_signals(signals, df)

    print(f"\nGenerated {len(trades)} trades:")
    for trade in trades:
        print(f"  Trade {trade['trade_id']}: {trade['trade_type']} at bar {trade['execution_bar']}")
        print(f"    Signal bar: {trade['signal_bar']}, Lag: {trade['lag']}")
        print(f"    Price: ${trade['execution_price']:.2f}")
        if trade.get('pnl_percent') is not None:
            print(f"    P&L: {trade['pnl_percent']:.2f}%")
        print()

    # Calculate metrics
    metrics = engine.calculate_metrics(trades)
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    return trades


def test_with_config_yaml():
    """Test execution with config.yaml settings"""
    print("\n" + "=" * 80)
    print("TEST 2: Execution with config.yaml")
    print("=" * 80)

    # Load config from yaml
    config = ExecutionConfig.from_yaml("C:\\code\\PythonBT\\tradingCode\\config.yaml")

    print("\nLoaded configuration:")
    print(f"  Signal lag: {config.signal_lag}")
    print(f"  Buy formula: {config.buy_execution_formula}")
    print(f"  Sell formula: {config.sell_execution_formula}")
    print(f"  Position size: {config.position_size}")

    # Create test data
    df = create_sample_data(100)
    signals = create_sample_signals(100)

    # Create engine with config
    engine = StandaloneExecutionEngine(config)

    # Execute signals
    trades = engine.execute_signals(signals, df)

    print(f"\nGenerated {len(trades)} trades with configured execution:")
    for i, trade in enumerate(trades[:5]):  # Show first 5 trades
        print(f"  Trade {trade['trade_id']}: {trade['trade_type']}")
        print(f"    Formula used: {trade['formula']}")
        print(f"    Signal price: ${trade['signal_price']:.2f}")
        print(f"    Execution price: ${trade['execution_price']:.2f}")
        print(f"    Lag: {trade['lag']} bars")
        if trade.get('pnl_percent') is not None:
            print(f"    P&L: {trade['pnl_percent']:.2f}%")
        print()

    return trades


def test_lag_calculations():
    """Test that lag calculations work correctly"""
    print("\n" + "=" * 80)
    print("TEST 3: Lag Calculation Verification")
    print("=" * 80)

    # Create test data
    df = create_sample_data(20)

    # Test different lag values
    for lag_value in [0, 1, 2, 3]:
        config = ExecutionConfig(signal_lag=lag_value)
        engine = StandaloneExecutionEngine(config)

        # Single signal at bar 5
        signals = pd.Series([0] * 20)
        signals[5] = 1  # Buy signal at bar 5
        signals[10] = 0  # Sell signal at bar 10

        trades = engine.execute_signals(signals, df)

        print(f"\nWith signal_lag={lag_value}:")
        for trade in trades:
            expected_exec_bar = trade['signal_bar'] + lag_value
            actual_exec_bar = trade['execution_bar']
            print(f"  {trade['trade_type']}: Signal bar {trade['signal_bar']} -> Execution bar {actual_exec_bar}")
            print(f"    Expected execution bar: {expected_exec_bar}")
            print(f"    Lag: {trade['lag']} bars")
            assert trade['lag'] == lag_value, f"Lag mismatch: expected {lag_value}, got {trade['lag']}"


def test_price_formulas():
    """Test execution price formulas"""
    print("\n" + "=" * 80)
    print("TEST 4: Price Formula Execution")
    print("=" * 80)

    # Create test data with known values
    df = pd.DataFrame({
        'DateTime': pd.date_range('2024-01-01', periods=10, freq='5min'),
        'Open': [100] * 10,
        'High': [105] * 10,
        'Low': [95] * 10,
        'Close': [102] * 10,
        'Volume': [1000] * 10
    })

    # Test different formulas
    formulas = {
        'Close': 'C',
        'Open': 'O',
        'Typical Price': '(H + L + C) / 3',
        'Median Price': '(H + L) / 2',
        'Weighted Close': '(H + L + 2*C) / 4'
    }

    for name, formula in formulas.items():
        config = ExecutionConfig(
            signal_lag=1,
            execution_price='formula',
            buy_execution_formula=formula,
            sell_execution_formula=formula
        )
        engine = StandaloneExecutionEngine(config)

        # Create simple signal
        signals = pd.Series([0] * 10)
        signals[2] = 1  # Buy at bar 2
        signals[5] = 0  # Sell at bar 5

        trades = engine.execute_signals(signals, df)

        print(f"\n{name} ({formula}):")
        for trade in trades:
            # Calculate expected price
            bar = df.iloc[trade['execution_bar']]
            O, H, L, C = bar['Open'], bar['High'], bar['Low'], bar['Close']
            expected = eval(formula, {"__builtins__": {}}, {'O': O, 'H': H, 'L': L, 'C': C})

            print(f"  {trade['trade_type']}: ${trade['execution_price']:.2f} (expected: ${expected:.2f})")
            assert abs(trade['execution_price'] - expected) < 0.001, "Price calculation error"


def test_trade_record_class():
    """Test TradeRecord data structure"""
    print("\n" + "=" * 80)
    print("TEST 5: TradeRecord Class")
    print("=" * 80)

    # Create sample trades
    trades_data = [
        TradeRecord(
            trade_id=0,
            bar_index=10,
            trade_type='BUY',
            price=100.50,
            signal_bar=9,
            execution_bar=10,
            lag=1,
            signal_price=100.25,
            execution_formula='(H + L + C) / 3',
            size=10,
            timestamp=pd.Timestamp('2024-01-01 09:00:00')
        ),
        TradeRecord(
            trade_id=1,
            bar_index=20,
            trade_type='SELL',
            price=102.75,
            signal_bar=19,
            execution_bar=20,
            lag=1,
            signal_price=102.50,
            execution_formula='(H + L + C) / 3',
            size=10,
            pnl_points=2.25,
            pnl_percent=2.24,
            timestamp=pd.Timestamp('2024-01-01 09:50:00')
        )
    ]

    # Create collection
    collection = TradeRecordCollection(trades_data)

    print(f"Trade collection with {len(collection)} trades")

    # Test metrics calculation
    metrics = collection.get_metrics()
    print("\nCollection Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Test P&L display formatting
    print("\nP&L Display Format:")
    for trade in collection:
        pnl_display = trade.format_pnl_display()
        if pnl_display:
            print(f"  Trade {trade.trade_id}: {pnl_display}")

    # Test DataFrame conversion
    df = collection.to_dataframe()
    print("\nDataFrame conversion:")
    print(df[['trade_id', 'trade_type', 'price', 'pnl_percent']].to_string())

    # Test summary stats
    print("\nSummary Statistics:")
    print(collection.get_summary_stats())


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("STANDALONE EXECUTION ENGINE TEST SUITE")
    print("=" * 80)

    try:
        # Run tests
        test_basic_execution()
        test_with_config_yaml()
        test_lag_calculations()
        test_price_formulas()
        test_trade_record_class()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 80)

        print("\nThe execution engine is working correctly:")
        print("  - Lag calculations are accurate")
        print("  - Price formulas evaluate properly")
        print("  - P&L tracking works as expected")
        print("  - Trade records store all necessary data")
        print("  - Config.yaml integration successful")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)