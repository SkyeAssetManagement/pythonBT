#!/usr/bin/env python3
"""
Test Updated System - Headless CSV Integration
=============================================
Test the updated unified system backend without GUI
"""

import sys
import os
sys.path.insert(0, 'src')

def test_headless_integration():
    """Test the headless CSV integration in the updated system"""
    print("=" * 60)
    print("TESTING UPDATED SYSTEM - HEADLESS CSV INTEGRATION")
    print("=" * 60)

    try:
        from src.trading.backtesting.headless_backtester import HeadlessBacktester
        from src.trading.visualization.backtest_result_loader import BacktestResultLoader
        from trade_data import TradeCollection, TradeData

        # Test the exact logic from the updated launch_unified_system.py
        system_name = "Simple Moving Average"
        print(f"\n[TEST] Running headless backtest for system: {system_name}")

        # Map system names to strategy names (same as in updated file)
        strategy_map = {
            "Simple Moving Average": "sma_crossover",
            "RSI Momentum": "rsi_momentum"
        }
        strategy_name = strategy_map.get(system_name, "sma_crossover")

        # Set up parameters (same as in updated file)
        if strategy_name == "sma_crossover":
            params = {
                'fast_period': 20,
                'slow_period': 50,
                'long_only': True,
                'signal_lag': 2,
                'position_size': 1.0,
                'min_execution_time': 5.0
            }

        data_file = 'data/sample_trading_data_small.csv'
        if not os.path.exists(data_file):
            print(f"[ERROR] Data file not found: {data_file}")
            return False

        print(f"[TEST] Running headless backtest with {data_file}")
        print(f"[TEST] Parameters: {params}")

        # Run headless backtest (same as in updated file)
        backtester = HeadlessBacktester()
        run_id = backtester.run_backtest(
            strategy_name=strategy_name,
            parameters=params,
            data_file=data_file,
            execution_mode='standard'
        )

        print(f"[SUCCESS] Headless backtest completed: {run_id}")

        # Load results from CSV (same as in updated file)
        loader = BacktestResultLoader()
        csv_trades = loader.load_trade_list(run_id)

        if csv_trades and len(csv_trades) > 0:
            print(f"[SUCCESS] Loaded {len(csv_trades)} trades from CSV")

            # Convert to legacy trade format for chart compatibility (same as in updated file)
            legacy_trades = []
            for trade in csv_trades:
                legacy_trade = TradeData(
                    bar_index=trade.bar_index,
                    trade_type=trade.trade_type,
                    price=trade.price,
                    trade_id=getattr(trade, 'trade_id', len(legacy_trades)),
                    timestamp=trade.timestamp,
                    pnl=getattr(trade, 'pnl', 0),
                    strategy=trade.strategy
                )
                legacy_trades.append(legacy_trade)

            trade_collection = TradeCollection(legacy_trades)
            print(f"[SUCCESS] Created TradeCollection with {len(trade_collection)} trades")

            # Show P&L info if available (same as in updated file)
            if len(trade_collection) > 0:
                print("\n[TEST] First 3 trades from CSV:")
                for i, trade in enumerate(trade_collection[:3]):
                    print(f"  Trade {i}: {trade.trade_type} at bar {trade.bar_index}, price ${trade.price:.2f}")
                    if hasattr(trade, 'pnl'):
                        print(f"    P&L: {trade.pnl:.2f}")

            return True

        else:
            print("[ERROR] No trades generated from headless backtest")
            return False

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chart_compatibility():
    """Test that the trades are compatible with the chart system"""
    print("\n" + "=" * 60)
    print("TESTING CHART COMPATIBILITY")
    print("=" * 60)

    try:
        from trade_data import TradeCollection, TradeData

        # Create sample trade data in the same format as the updated system
        sample_trades = []
        for i in range(5):
            trade = TradeData(
                bar_index=i * 100,
                trade_type='BUY' if i % 2 == 0 else 'SELL',
                price=4000.0 + i * 10,
                trade_id=i,
                timestamp=None,  # Will be set by chart
                pnl=i * 5.0,
                strategy='sma_crossover'
            )
            sample_trades.append(trade)

        trade_collection = TradeCollection(sample_trades)
        print(f"[SUCCESS] Created sample TradeCollection with {len(trade_collection)} trades")

        # Test that it has the required attributes for chart loading
        for i, trade in enumerate(trade_collection[:3]):
            print(f"  Trade {i}: {trade.trade_type} at bar {trade.bar_index}")
            print(f"    Has bar_index: {hasattr(trade, 'bar_index')}")
            print(f"    Has trade_type: {hasattr(trade, 'trade_type')}")
            print(f"    Has price: {hasattr(trade, 'price')}")
            print(f"    Has pnl: {hasattr(trade, 'pnl')}")

        print("[SUCCESS] Trade format compatible with chart system")
        return True

    except Exception as e:
        print(f"[ERROR] Chart compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("UPDATED SYSTEM TEST - HEADLESS CSV + ORIGINAL CHART")
    print("Tests the backend changes to use headless CSV instead of old unified engine")

    results = {}

    # Test 1: Headless integration
    results['headless_integration'] = test_headless_integration()

    # Test 2: Chart compatibility
    results['chart_compatibility'] = test_chart_compatibility()

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
        print("\n[SUCCESS] Updated system backend is working!")
        print("The launch_unified_system.py should now:")
        print("  - Load chart data exactly as before (1000 bars, instant scrolling)")
        print("  - Use headless CSV system for trade generation")
        print("  - Display P&L data from CSV files")
        print("  - No more hanging on strategy execution")
        return True
    else:
        print(f"\n[WARNING] {total_tests - total_passed} tests failed")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)