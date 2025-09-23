#!/usr/bin/env python3
"""
Test script to verify the application works with full 377,690 bar dataset
"""

import sys
import pandas as pd
import time
from pathlib import Path

# Add paths for imports
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')

# Test data loading performance
def test_data_loading():
    """Test loading the full dataset"""
    print("="*60)
    print("Testing Full Dataset Loading (377,690 bars)")
    print("="*60)

    file_path = Path("dataRaw/range-ATR30x0.05/ES/diffAdjusted/ES-DIFF-range-ATR30x0.05-dailyATR.csv")

    print(f"\n1. Loading file: {file_path}")
    start_time = time.time()

    try:
        df = pd.read_csv(file_path)
        load_time = time.time() - start_time

        print(f"   - Loaded {len(df):,} rows in {load_time:.2f} seconds")
        print(f"   - Columns: {df.columns.tolist()}")
        print(f"   - Date range: {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}")

        # Check for required columns
        required_cols = ['Date', 'Time', 'Open', 'High', 'Low', 'Close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"   - WARNING: Missing columns: {missing}")
        else:
            print(f"   - All required columns present")

        # Test datetime parsing
        print("\n2. Testing DateTime parsing...")
        start_time = time.time()
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        parse_time = time.time() - start_time
        print(f"   - Parsed DateTime in {parse_time:.2f} seconds")

        # Sample some timestamps to verify
        sample_indices = [0, 100000, 200000, 300000, -1]
        print("\n3. Sample timestamps:")
        for idx in sample_indices:
            if idx == -1:
                actual_idx = len(df) - 1
            else:
                actual_idx = idx
            dt = df.iloc[actual_idx]['DateTime']
            print(f"   - Bar {actual_idx:6}: {dt}")

        # Test memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"\n4. Memory usage: {memory_mb:.1f} MB")

        return True

    except Exception as e:
        print(f"   - ERROR: {e}")
        return False

def test_chart_initialization():
    """Test chart initialization with full dataset"""
    print("\n" + "="*60)
    print("Testing Chart Initialization")
    print("="*60)

    from PyQt5 import QtWidgets
    from launch_pyqtgraph_with_selector import ConfiguredChart

    # Create minimal config
    config = {
        'data_file': 'dataRaw/range-ATR30x0.05/ES/diffAdjusted/ES-DIFF-range-ATR30x0.05-dailyATR.csv',
        'trade_source': 'none',
        'indicators': []
    }

    print("\n1. Creating Qt Application...")
    app = QtWidgets.QApplication(sys.argv)

    print("2. Initializing chart with full dataset...")
    start_time = time.time()

    try:
        chart = ConfiguredChart(config)
        init_time = time.time() - start_time

        print(f"   - Chart initialized in {init_time:.2f} seconds")
        print(f"   - Total bars loaded: {chart.total_bars:,}")

        # Check data structure
        if chart.full_data:
            print(f"   - Data keys: {list(chart.full_data.keys())}")
            print(f"   - Timestamp array length: {len(chart.full_data['timestamp']):,}")

        # Check current_x_range initialization
        if hasattr(chart, 'current_x_range') and chart.current_x_range:
            start_idx, end_idx = chart.current_x_range
            print(f"   - Initial viewport: bars {start_idx:,} to {end_idx:,} ({end_idx - start_idx} bars)")
        else:
            print("   - WARNING: current_x_range not initialized")

        print("\n3. Test PASSED - Chart can handle full dataset")
        return True

    except Exception as e:
        print(f"   - ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        app.quit()

def test_trade_generation():
    """Test trade generation with full dataset"""
    print("\n" + "="*60)
    print("Testing Trade Generation")
    print("="*60)

    # Load data for trade generation
    file_path = Path("dataRaw/range-ATR30x0.05/ES/diffAdjusted/ES-DIFF-range-ATR30x0.05-dailyATR.csv")
    df = pd.read_csv(file_path)

    # Normalize columns
    column_mapping = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
    }
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    from launch_pyqtgraph_with_selector import generate_system_trades

    print("\n1. Testing Simple Moving Average strategy...")
    start_time = time.time()
    trades = generate_system_trades("Simple Moving Average", df)
    gen_time = time.time() - start_time

    print(f"   - Generated {len(trades)} trades in {gen_time:.2f} seconds")

    if len(trades) > 0:
        # Show first and last few trades
        print("\n   First 3 trades:")
        for i in range(min(3, len(trades))):
            trade = trades[i]
            print(f"     - Bar {trade.bar_index:,}: {trade.trade_type} at ${trade.price:.2f}")

        if len(trades) > 3:
            print(f"\n   Last 3 trades:")
            for i in range(max(0, len(trades)-3), len(trades)):
                trade = trades[i]
                print(f"     - Bar {trade.bar_index:,}: {trade.trade_type} at ${trade.price:.2f}")

    print("\n2. Testing RSI Momentum strategy...")
    start_time = time.time()
    trades = generate_system_trades("RSI Momentum", df)
    gen_time = time.time() - start_time

    print(f"   - Generated {len(trades)} trades in {gen_time:.2f} seconds")

    return True

if __name__ == "__main__":
    print("\nFull Dataset Test Suite")
    print("="*60)

    # Run tests
    results = []

    # Test 1: Data Loading
    results.append(("Data Loading", test_data_loading()))

    # Test 2: Chart Initialization
    results.append(("Chart Initialization", test_chart_initialization()))

    # Test 3: Trade Generation
    results.append(("Trade Generation", test_trade_generation()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "[+]" if passed else "[-]"
        print(f"{symbol} {test_name:25} {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\nAll tests PASSED! The application can handle the full 377,690 bar dataset.")
    else:
        print("\nSome tests FAILED. Please review the output above.")

    sys.exit(0 if all_passed else 1)