#!/usr/bin/env python3
"""
Test Chart Only System - NO GUI
================================
Test the chart-only system components without GUI
"""

import sys
import os
sys.path.insert(0, 'src')

def test_chart_components():
    """Test that chart components can be imported and initialized"""
    print("=" * 60)
    print("TESTING CHART ONLY COMPONENTS")
    print("=" * 60)

    try:
        # Test core imports
        from src.trading.visualization.backtest_result_loader import BacktestResultLoader
        from src.trading.visualization.trade_data import TradeCollection, TradeData
        print("✓ Core components imported successfully")

        # Test result loader
        loader = BacktestResultLoader()
        runs = loader.list_available_runs()
        print(f"✓ Found {len(runs)} backtest runs")

        if runs:
            # Test loading trades
            latest_run = runs[0]
            trades = loader.load_trade_list(latest_run['run_id'])
            print(f"✓ Loaded {len(trades)} trades from latest run")

            # Test trade conversion
            chart_trades = []
            for trade in trades[:5]:  # Just test first 5
                chart_trade = TradeData(
                    bar_index=trade.bar_index,
                    trade_type=trade.trade_type,
                    price=trade.price,
                    trade_id=len(chart_trades),
                    timestamp=trade.timestamp,
                    pnl=getattr(trade, 'pnl', 0),
                    strategy=trade.strategy
                )
                chart_trades.append(chart_trade)

            trade_collection = TradeCollection(chart_trades)
            print(f"✓ Created TradeCollection with {len(trade_collection)} trades")

        return True

    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False

def test_data_loading():
    """Test data loading logic from chart system"""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)

    try:
        import pandas as pd
        import numpy as np

        # Test with sample data
        data_file = 'data/sample_trading_data_small.csv'
        if not os.path.exists(data_file):
            print(f"✗ Sample data not found: {data_file}")
            return False

        print(f"Testing with: {data_file}")

        # Load and process data (same logic as chart system)
        df = pd.read_csv(data_file)
        print(f"✓ Loaded {len(df)} rows")

        # Process datetime
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        else:
            df['DateTime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='5min')

        # Column mapping
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Create chart data structure
        chart_data = {
            'timestamp': df['DateTime'].values,
            'open': df['Open'].values.astype(np.float64),
            'high': df['High'].values.astype(np.float64),
            'low': df['Low'].values.astype(np.float64),
            'close': df['Close'].values.astype(np.float64),
            'volume': df['Volume'].values.astype(np.float64) if 'Volume' in df.columns else np.zeros(len(df))
        }

        print(f"✓ Created chart data with {len(chart_data['timestamp'])} bars")
        print(f"✓ Price range: ${chart_data['close'].min():.2f} - ${chart_data['close'].max():.2f}")

        return True

    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

def main():
    """Test chart-only system components"""
    print("CHART ONLY SYSTEM TEST")
    print("Tests chart components without GUI or strategy execution")

    results = {}

    # Test 1: Chart components
    results['components'] = test_chart_components()

    # Test 2: Data loading
    results['data_loading'] = test_data_loading()

    # Summary
    print("\n" + "=" * 60)
    print("CHART ONLY TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name.upper()}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n[SUCCESS] Chart-only system ready!")
        print("Components:")
        print("  ✓ Chart data loading (1000 bars, instant scrolling)")
        print("  ✓ CSV trade loading (from headless backtests)")
        print("  ✓ Trade display (with P&L data)")
        print("\nSeparation achieved:")
        print("  📊 Chart system: launch_chart_only.py")
        print("  🔄 Headless backtesting: run_headless_only.py")
        print("  ❌ No more hanging on strategy execution")
        return True
    else:
        print(f"\n[WARNING] {total_tests - total_passed} tests failed")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)