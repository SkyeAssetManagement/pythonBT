"""
Test strategy execution to identify and fix issues
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_strategy_runner_directly():
    """Test the strategy runner component directly"""
    print("="*60)
    print("TESTING STRATEGY RUNNER DIRECTLY")
    print("="*60)

    from PyQt5.QtWidgets import QApplication
    from src.trading.visualization.strategy_runner import StrategyRunner

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create strategy runner
    runner = StrategyRunner()

    # Create test data matching what the chart provides
    test_data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 100)
    }

    print(f"Created test data with keys: {list(test_data.keys())}")

    # Set the data
    runner.set_chart_data(test_data)

    # Check if data was set
    if runner.chart_data is None:
        print("ERROR: chart_data is None after setting!")
        return False

    print(f"Chart data set: {len(runner.chart_data)} rows")
    print(f"Chart data columns: {runner.chart_data.columns.tolist()}")

    # Try to run strategy
    print("\nAttempting to run SMA Crossover strategy...")
    runner.strategy_combo.setCurrentText("SMA Crossover")

    try:
        runner.run_strategy()
        print("Strategy executed without errors!")
        return True
    except Exception as e:
        print(f"ERROR running strategy: {e}")
        traceback.print_exc()
        return False

def test_strategy_classes():
    """Test strategy classes directly"""
    print("\n" + "="*60)
    print("TESTING STRATEGY CLASSES DIRECTLY")
    print("="*60)

    from src.trading.strategies.sma_crossover import SMACrossoverStrategy
    from src.trading.strategies.rsi_momentum import RSIMomentumStrategy

    # Create test DataFrame
    df = pd.DataFrame({
        'DateTime': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 101,
        'Low': np.random.randn(100).cumsum() + 99,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(100, 1000, 100)
    })

    print(f"Test DataFrame shape: {df.shape}")
    print(f"Test DataFrame columns: {df.columns.tolist()}")

    # Test SMA strategy
    print("\n1. Testing SMA Crossover Strategy...")
    try:
        sma_strategy = SMACrossoverStrategy(fast_period=10, slow_period=20)

        # Test signal generation
        signals = sma_strategy.generate_signals(df)
        print(f"   Generated signals shape: {signals.shape}")
        print(f"   Number of buy signals: {(signals == 1).sum()}")
        print(f"   Number of sell signals: {(signals == -1).sum()}")

        # Test trade conversion
        trades = sma_strategy.signals_to_trades(signals, df)
        print(f"   Generated {len(trades)} trades")

    except Exception as e:
        print(f"   ERROR: {e}")
        traceback.print_exc()

    # Test RSI strategy
    print("\n2. Testing RSI Momentum Strategy...")
    try:
        rsi_strategy = RSIMomentumStrategy(period=14, oversold=30, overbought=70)

        # Test signal generation
        signals = rsi_strategy.generate_signals(df)
        print(f"   Generated signals shape: {signals.shape}")
        print(f"   Number of buy signals: {(signals == 1).sum()}")
        print(f"   Number of sell signals: {(signals == -1).sum()}")

        # Test trade conversion
        trades = rsi_strategy.signals_to_trades(signals, df)
        print(f"   Generated {len(trades)} trades")

    except Exception as e:
        print(f"   ERROR: {e}")
        traceback.print_exc()

def test_in_full_app():
    """Test strategy execution in the full application"""
    print("\n" + "="*60)
    print("TESTING IN FULL APPLICATION")
    print("="*60)

    from PyQt5.QtWidgets import QApplication
    from launch_pyqtgraph_with_selector import ConfiguredChart
    import glob

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Find test data
    csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\range-ATR30x0.05\ES\diffAdjusted\*.csv")
    if not csv_files:
        csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\**\*.csv", recursive=True)

    if not csv_files:
        print("ERROR: No data files found")
        return False

    # Create chart with NO initial trades
    config = {
        'data_file': csv_files[0],
        'trade_source': 'none',  # Start with no trades
        'indicators': []
    }

    print(f"Loading chart with: {os.path.basename(config['data_file'])}")
    chart = ConfiguredChart(config)

    # Check if strategy runner exists
    if not hasattr(chart, 'trade_panel') or chart.trade_panel is None:
        print("ERROR: No trade panel found")
        return False

    # Access the strategy runner tab
    trade_panel = chart.trade_panel

    # Check if strategy runner tab exists
    if hasattr(trade_panel, 'strategy_runner'):
        print("Strategy runner tab found")
        runner = trade_panel.strategy_runner

        # Check if data is set
        if runner.chart_data is None:
            print("WARNING: chart_data is None in strategy runner")
        else:
            print(f"Strategy runner has data: {len(runner.chart_data)} rows")
            print(f"Data columns: {runner.chart_data.columns.tolist()}")

        # Try to run a strategy
        print("\nSimulating strategy execution...")
        runner.strategy_combo.setCurrentText("SMA Crossover")

        try:
            runner.run_strategy()
            print("SUCCESS: Strategy executed!")

            # Check status
            status_text = runner.status_label.text()
            print(f"Status: {status_text}")

            return True
        except Exception as e:
            print(f"ERROR executing strategy: {e}")
            traceback.print_exc()
            return False
    else:
        print("ERROR: No strategy runner found in trade panel")
        return False

def main():
    print("="*60)
    print("STRATEGY EXECUTION DEBUGGING")
    print("="*60)

    results = {}

    # Run tests
    tests = [
        ("Strategy Classes", test_strategy_classes),
        ("Strategy Runner Direct", test_strategy_runner_directly),
        ("Full Application", test_in_full_app)
    ]

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\nCRITICAL ERROR in {test_name}: {e}")
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")

    if all(results.values()):
        print("\nAll tests passed - strategies are working!")
    else:
        print("\nSome tests failed - check output for details")

    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)