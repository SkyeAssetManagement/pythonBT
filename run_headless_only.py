#!/usr/bin/env python3
"""
Headless Only Runner - NO GUI
=============================
Runs headless backtests and saves to CSV ONLY
NO chart display, NO GUI, NO hanging
"""

import sys
import os
sys.path.insert(0, 'src')

from src.trading.backtesting.headless_backtester import HeadlessBacktester
from src.trading.visualization.backtest_result_loader import BacktestResultLoader

def run_sma_backtest(data_file='data/sample_trading_data_small.csv'):
    """Run SMA crossover backtest"""
    print(f"\n[HEADLESS] Running SMA crossover backtest with {data_file}")

    if not os.path.exists(data_file):
        print(f"[ERROR] Data file not found: {data_file}")
        return None

    try:
        backtester = HeadlessBacktester()

        params = {
            'fast_period': 20,
            'slow_period': 50,
            'long_only': True,
            'signal_lag': 2,
            'position_size': 1.0,
            'min_execution_time': 5.0
        }

        print(f"[HEADLESS] Parameters: {params}")

        run_id = backtester.run_backtest(
            strategy_name='sma_crossover',
            parameters=params,
            data_file=data_file,
            execution_mode='standard'
        )

        print(f"[SUCCESS] Backtest completed: {run_id}")
        return run_id

    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        return None

def run_rsi_backtest(data_file='data/sample_trading_data_small.csv'):
    """Run RSI momentum backtest"""
    print(f"\n[HEADLESS] Running RSI momentum backtest with {data_file}")

    if not os.path.exists(data_file):
        print(f"[ERROR] Data file not found: {data_file}")
        return None

    try:
        backtester = HeadlessBacktester()

        params = {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70,
            'signal_lag': 2,
            'position_size': 1.0
        }

        print(f"[HEADLESS] Parameters: {params}")

        run_id = backtester.run_backtest(
            strategy_name='rsi_momentum',
            parameters=params,
            data_file=data_file,
            execution_mode='standard'
        )

        print(f"[SUCCESS] Backtest completed: {run_id}")
        return run_id

    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        return None

def list_results():
    """List all available backtest results"""
    print("\n[RESULTS] Available backtest runs:")

    try:
        loader = BacktestResultLoader()
        runs = loader.list_available_runs()

        if not runs:
            print("  No backtest results found")
            return

        for i, run in enumerate(runs):
            timestamp = run['timestamp']
            readable_time = f"{timestamp[6:8]}/{timestamp[4:6]} {timestamp[9:11]}:{timestamp[11:13]}"
            print(f"  {i+1}. {run['strategy_name']} - {readable_time} ({run['execution_mode']})")

        print(f"\nTotal: {len(runs)} backtest runs available")

    except Exception as e:
        print(f"[ERROR] Failed to list results: {e}")

def show_trade_summary(run_id):
    """Show summary of trades from a backtest run"""
    if not run_id:
        return

    print(f"\n[SUMMARY] Trade summary for {run_id}:")

    try:
        loader = BacktestResultLoader()
        trades = loader.load_trade_list(run_id)

        if not trades:
            print("  No trades found")
            return

        print(f"  Total trades: {len(trades)}")

        # Count trade types
        buy_trades = sum(1 for t in trades if t.trade_type == 'BUY')
        sell_trades = sum(1 for t in trades if t.trade_type == 'SELL')
        print(f"  BUY trades: {buy_trades}")
        print(f"  SELL trades: {sell_trades}")

        # Show first few trades
        print("\n  First 5 trades:")
        for i, trade in enumerate(trades[:5]):
            pnl_info = ""
            if hasattr(trade, 'metadata') and trade.metadata:
                exec_bars = trade.metadata.get('exec_bars', 'N/A')
                pnl_info = f", execBars: {exec_bars}"

            print(f"    {i+1}. {trade.trade_type} at bar {trade.bar_index}, price ${trade.price:.2f}{pnl_info}")

    except Exception as e:
        print(f"[ERROR] Failed to load trade summary: {e}")

def main():
    """Headless only runner"""
    print("=" * 70)
    print("HEADLESS ONLY BACKTESTING - NO GUI")
    print("Runs backtests and saves to CSV for chart display")
    print("=" * 70)

    # Check available data files
    data_files = []
    for file in ['data/sample_trading_data_small.csv', 'data/sample_trading_data.csv']:
        if os.path.exists(file):
            data_files.append(file)

    if not data_files:
        print("[ERROR] No sample data files found")
        return

    print(f"[INFO] Found data files: {data_files}")

    # Run backtests
    results = []

    # Run SMA backtest
    sma_result = run_sma_backtest(data_files[0])
    if sma_result:
        results.append(sma_result)
        show_trade_summary(sma_result)

    # Run RSI backtest (if we have strategy implemented)
    try:
        rsi_result = run_rsi_backtest(data_files[0])
        if rsi_result:
            results.append(rsi_result)
            show_trade_summary(rsi_result)
    except Exception as e:
        print(f"[SKIP] RSI backtest failed: {e}")

    # List all results
    list_results()

    # Summary
    print("\n" + "=" * 70)
    print("HEADLESS BACKTESTING COMPLETE")
    print("=" * 70)

    if results:
        print(f"✅ Generated {len(results)} backtest runs")
        print("📊 CSV files saved to backtest_results/")
        print("📈 Use launch_chart_only.py to display trades")
        print("\nNext steps:")
        print("  1. Run: python launch_chart_only.py")
        print("  2. Load your data file")
        print("  3. Click 'Load CSV Trades' to display results")
    else:
        print("❌ No backtests completed successfully")

if __name__ == '__main__':
    main()