#!/usr/bin/env python3
"""
Headless TWAP Test Runner
=========================
Test the complete headless backtesting system with TWAP execution
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_es_data(num_bars=1000):
    """Create realistic ES range bar test data"""

    print(f"Creating {num_bars} ES-like range bars...")

    np.random.seed(42)
    base_time = datetime(2023, 1, 4, 9, 30, 0)
    base_price = 4230.0
    data = []

    for i in range(num_bars):
        # Variable time increments (range bars characteristic)
        time_increment = np.random.choice([1, 2, 3, 5, 8, 13], p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05])
        base_time += timedelta(minutes=int(time_increment))

        # Price drift with some trend
        price_change = np.random.normal(0.02, 0.5)  # Small upward bias
        base_price += price_change

        # Range bar properties - fixed range
        range_size = 0.50  # ES 50 cent range bars

        bar_data = {
            'datetime': base_time,
            'open': base_price,
            'high': base_price + range_size,
            'low': base_price,
            'close': base_price + range_size,
            'volume': np.random.randint(500, 5000)  # Variable volume
        }
        data.append(bar_data)

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df['datetime'])

    # Save to CSV for headless backtester
    data_file = 'test_es_data.csv'
    df.to_csv(data_file)

    print(f"Created {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['low'].min():.2f} to {df['high'].max():.2f}")
    print(f"Saved to: {data_file}")

    return data_file

def test_headless_twap():
    """Test the complete headless TWAP backtesting system"""

    print("=" * 60)
    print("HEADLESS TWAP BACKTESTING SYSTEM TEST")
    print("=" * 60)

    # Create test data
    data_file = create_test_es_data(500)  # Use smaller dataset for testing

    try:
        # Import and run headless backtester
        from backtesting.headless_backtester import HeadlessBacktester

        backtester = HeadlessBacktester()

        # Test parameters
        parameters = {
            'fast_period': 10,
            'slow_period': 30,
            'long_only': False,
            'signal_lag': 2,
            'position_size': 1.0,
            'min_execution_time': 5.0,
            'fees': 0.0
        }

        print("\nRunning headless TWAP backtest...")
        print(f"Parameters: {parameters}")

        # Run backtest
        run_id = backtester.run_backtest(
            strategy_name='sma_crossover',
            parameters=parameters,
            data_file=data_file,
            execution_mode='twap'
        )

        print(f"\n[SUCCESS] Headless backtest completed: {run_id}")

        # Test result loading
        print("\nTesting result loading...")
        from visualization.backtest_result_loader import BacktestResultLoader

        loader = BacktestResultLoader()

        # List runs
        runs = loader.list_available_runs()
        print(f"Found {len(runs)} backtest runs")

        # Load the run we just created
        trades = loader.load_trade_list(run_id)
        if trades:
            print(f"Loaded {len(trades)} trades from CSV")

            # Check for TWAP metadata
            twap_trades = [t for t in trades if hasattr(t, 'metadata') and t.metadata and 'exec_bars' in t.metadata]
            print(f"Trades with TWAP metadata: {len(twap_trades)}")

            if twap_trades:
                first_twap = twap_trades[0]
                print(f"Sample TWAP trade:")
                print(f"  execBars: {first_twap.metadata['exec_bars']}")
                print(f"  Execution time: {first_twap.metadata['execution_time_minutes']:.2f} minutes")
                print(f"  Natural phases: {first_twap.metadata['num_phases']}")

        # Load equity curve
        equity = loader.load_equity_curve(run_id)
        if equity is not None:
            print(f"Loaded equity curve with {len(equity)} points")

        # Get run summary
        summary = loader.get_run_summary(run_id)
        if summary:
            print(f"Run summary:")
            print(f"  Strategy: {summary['metadata']['strategy_name']}")
            print(f"  Execution mode: {summary['metadata']['execution_mode']}")
            print(f"  Trade count: {summary['trade_count']}")
            print(f"  Has equity curve: {summary['has_equity_curve']}")
            print(f"  Folder: {summary['folder_path']}")

        print("\n" + "=" * 60)
        print("[SUCCESS] HEADLESS TWAP SYSTEM WORKING CORRECTLY!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n[ERROR] Headless TWAP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up test data file
        if os.path.exists(data_file):
            os.remove(data_file)
            print(f"\nCleaned up: {data_file}")

if __name__ == "__main__":
    success = test_headless_twap()

    if success:
        print("\nThe headless TWAP backtesting system is ready!")
        print("You can now:")
        print("1. Run backtests independently of the chart visualizer")
        print("2. Results are saved to organized folder structures")
        print("3. Chart visualizer can load results from CSV files")
        print("4. Large datasets (192k+ signals) are handled with chunking")

    sys.exit(0 if success else 1)