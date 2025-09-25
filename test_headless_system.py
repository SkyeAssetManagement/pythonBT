#!/usr/bin/env python3
"""
Test Headless Backtesting System - Command Line Interface
=========================================================
Tests the new headless backtesting architecture without GUI
"""

import sys
import os
sys.path.insert(0, 'src')

from src.trading.backtesting.headless_backtester import HeadlessBacktester
from src.trading.visualization.backtest_result_loader import BacktestResultLoader


def test_headless_backtest():
    """Test running a headless backtest"""
    print("\n" + "=" * 60)
    print("TESTING HEADLESS BACKTEST")
    print("=" * 60)

    try:
        # Create backtester
        backtester = HeadlessBacktester()

        # Test parameters
        params = {
            'fast_period': 10,
            'slow_period': 30,
            'long_only': False,
            'signal_lag': 2,
            'position_size': 1.0,
            'min_execution_time': 5.0
        }

        # Run backtest with sample data
        print("\n[TEST] Running headless backtest with sample data...")
        run_id = backtester.run_backtest(
            'sma_crossover',
            params,
            'data/sample_trading_data_small.csv',
            'standard'
        )

        print(f"\n[SUCCESS] Backtest completed: {run_id}")
        return run_id

    except Exception as e:
        print(f"\n[ERROR] Backtest failed: {e}")
        return None


def test_result_loading():
    """Test loading backtest results"""
    print("\n" + "=" * 60)
    print("TESTING RESULT LOADING")
    print("=" * 60)

    try:
        # Create loader
        loader = BacktestResultLoader()

        # List available runs
        print("\n[TEST] Listing available backtest runs...")
        runs = loader.list_available_runs()

        if not runs:
            print("[WARNING] No backtest runs found")
            return None

        print(f"\n[SUCCESS] Found {len(runs)} backtest runs:")
        for i, run in enumerate(runs[:5]):  # Show first 5
            print(f"  {i+1}. {run['run_id']} - {run['strategy_name']} ({run['execution_mode']})")

        # Load trades from latest run
        latest_run = runs[0]
        print(f"\n[TEST] Loading trades from latest run: {latest_run['run_id']}")

        trades = loader.load_trade_list(latest_run['run_id'])

        if trades and len(trades) > 0:
            print(f"\n[SUCCESS] Loaded {len(trades)} trades")
            print("\nFirst 3 trades:")
            for i, trade in enumerate(trades[:3]):
                print(f"  Trade {i+1}: {trade.trade_type} at {trade.price:.2f} (bar {trade.bar_index})")
                if hasattr(trade, 'metadata') and trade.metadata:
                    print(f"    Metadata: {trade.metadata}")

            return trades
        else:
            print("[WARNING] No trades loaded")
            return None

    except Exception as e:
        print(f"\n[ERROR] Result loading failed: {e}")
        return None


def test_p_and_l_verification():
    """Test P&L calculations in CSV files"""
    print("\n" + "=" * 60)
    print("TESTING P&L VERIFICATION")
    print("=" * 60)

    try:
        import pandas as pd
        from pathlib import Path

        # Find latest CSV file
        results_dir = Path("backtest_results")
        if not results_dir.exists():
            print("[WARNING] No backtest_results directory found")
            return

        csv_files = list(results_dir.glob("*/trades/trade_list.csv"))
        if not csv_files:
            print("[WARNING] No trade CSV files found")
            return

        # Use latest CSV file
        latest_csv = sorted(csv_files, key=lambda x: x.parent.parent.name)[-1]
        print(f"\n[TEST] Checking P&L in: {latest_csv}")

        # Read CSV
        df = pd.read_csv(latest_csv)
        print(f"\n[SUCCESS] CSV loaded with {len(df)} trades")
        print(f"Columns: {df.columns.tolist()}")

        # Check P&L columns
        required_cols = ['pnl', 'cumulative_profit']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"[ERROR] Missing P&L columns: {missing_cols}")
            return False

        # Show sample P&L data
        print("\nSample P&L data (first 5 rows):")
        pnl_cols = ['datetime', 'trade_type', 'price', 'pnl', 'cumulative_profit', 'is_entry', 'is_exit']
        available_cols = [col for col in pnl_cols if col in df.columns]
        print(df[available_cols].head().to_string(index=False))

        # Verify P&L format (no % signs)
        pnl_sample = df['pnl'].dropna().head(10)
        has_percent = any('%' in str(val) for val in pnl_sample)

        if has_percent:
            print("[ERROR] P&L values contain % signs - should be raw decimals")
            return False
        else:
            print("[SUCCESS] P&L values are raw decimals (no % signs)")

        # Check cumulative profit accumulation
        cumulative = df['cumulative_profit'].dropna()
        if len(cumulative) > 1:
            print(f"Cumulative profit range: {cumulative.min():.2f} to {cumulative.max():.2f}")

        return True

    except Exception as e:
        print(f"\n[ERROR] P&L verification failed: {e}")
        return False


def test_twap_error_handling():
    """Test TWAP error handling (no fallbacks)"""
    print("\n" + "=" * 60)
    print("TESTING TWAP ERROR HANDLING")
    print("=" * 60)

    try:
        # Create backtester
        backtester = HeadlessBacktester()

        # Test parameters
        params = {
            'fast_period': 10,
            'slow_period': 30,
            'signal_lag': 2,
            'min_execution_time': 5.0
        }

        # Try TWAP mode (should fail with error, not fallback)
        print("\n[TEST] Trying TWAP mode (should throw ImportError)...")

        try:
            run_id = backtester.run_backtest(
                'sma_crossover',
                params,
                'data/sample_trading_data_small.csv',
                'twap'
            )
            print(f"[ERROR] TWAP mode should have failed but got: {run_id}")
            return False

        except ImportError as e:
            print(f"[SUCCESS] Got expected ImportError: {e}")
            return True

        except Exception as e:
            print(f"[ERROR] Got unexpected error type: {type(e).__name__}: {e}")
            return False

    except Exception as e:
        print(f"\n[ERROR] TWAP error test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("HEADLESS BACKTESTING SYSTEM - COMMAND LINE TEST")
    print("Tests the new P&L system and CSV loading without GUI")

    results = {}

    # Test 1: Headless backtest
    results['backtest'] = test_headless_backtest() is not None

    # Test 2: Result loading
    results['loading'] = test_result_loading() is not None

    # Test 3: P&L verification
    results['pnl'] = test_p_and_l_verification() is True

    # Test 4: TWAP error handling
    results['twap_errors'] = test_twap_error_handling() is True

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name.upper()}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n[SUCCESS] All tests passed - headless system is working!")
        return True
    else:
        print(f"\n[WARNING] {total_tests - total_passed} tests failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)