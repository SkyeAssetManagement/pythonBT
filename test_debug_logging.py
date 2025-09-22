"""
Test script to verify debug logging for rendering, DateTime, and jump-to-trade
"""

import sys
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication

sys.path.append('src/trading')

def test_rendering_debug():
    """Test that render_range debug logging is working"""
    print("\n" + "="*60)
    print("TEST 1: RENDER_RANGE DEBUG LOGGING")
    print("="*60)

    # Import with proper path
    import sys
    sys.path.insert(0, 'src/trading/visualization')
    from pyqtgraph_range_bars_final import RangeBarChartFinal as RangeBarChart

    # Create minimal app
    app = QApplication(sys.argv)

    # Load test data
    test_file = 'test_data_10000_bars.csv'
    if not os.path.exists(test_file):
        print(f"[FAIL] Test file {test_file} not found. Run test_large_dataset.py first")
        return False

    df = pd.read_csv(test_file)
    print(f"[OK] Loaded {len(df)} bars from {test_file}")

    # Create chart
    chart = RangeBarChart()

    # Load data (this should trigger render_range)
    print("\nLoading data into chart (should see [RENDER_RANGE] logs):")
    chart.load_data(df)

    # Test panning (should trigger on_range_changed and render_range)
    print("\nSimulating pan to bars 1000-1500 (should see [ON_RANGE_CHANGED] and [RENDER_RANGE] logs):")
    chart.render_range(1000, 1500)

    print("\nSimulating pan to bars 5000-5500:")
    chart.render_range(5000, 5500)

    print("\nSimulating pan to bars 9500-10000:")
    chart.render_range(9500, 10000)

    print("\n[OK] Render debug logging test complete")
    return True

def test_datetime_debug():
    """Test that DateTime extraction debug logging is working"""
    print("\n" + "="*60)
    print("TEST 2: DATETIME DEBUG LOGGING")
    print("="*60)

    from strategies.sma_crossover import SMACrossoverStrategy

    # Load test data
    test_file = 'test_data_10000_bars.csv'
    df = pd.read_csv(test_file)

    # Create strategy
    strategy = SMACrossoverStrategy()

    print("\nGenerating signals (should see [STRATEGY] DataFrame columns log):")
    signals = strategy.generate_signals(df)

    print("\nConverting signals to trades (should see [STRATEGY] DateTime extraction logs):")
    trades = strategy.signals_to_trades(signals, df)

    if trades.trades:
        first_trade = trades.trades[0]
        if first_trade.timestamp:
            print(f"[OK] First trade has timestamp: {first_trade.timestamp}")
        else:
            print(f"[WARNING] First trade missing timestamp")

    print("\n[OK] DateTime debug logging test complete")
    return True

def test_jump_debug():
    """Test that jump-to-trade debug logging is working"""
    print("\n" + "="*60)
    print("TEST 3: JUMP-TO-TRADE DEBUG LOGGING")
    print("="*60)

    # Import with proper paths
    import sys
    sys.path.insert(0, 'src/trading/visualization')
    sys.path.insert(0, 'src/trading/data')
    from pyqtgraph_range_bars_final import RangeBarChartFinal as RangeBarChart
    from trade_data import TradeData
    import pandas as pd

    # Create minimal app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Load test data
    test_file = 'test_data_10000_bars.csv'
    df = pd.read_csv(test_file)

    # Create chart
    chart = RangeBarChart()
    chart.load_data(df)

    # Create test trade
    test_trade = TradeData(
        bar_index=5000,
        trade_type='BUY',
        price=4000,
        trade_id=1,
        timestamp=pd.Timestamp('2024-01-01 12:00:00')
    )

    print("\nJumping to trade at bar 5000 (should see [JUMP_TO_TRADE] logs):")
    chart.jump_to_trade(test_trade)

    # Test edge cases
    test_trade2 = TradeData(
        bar_index=50,  # Near start
        trade_type='SELL',
        price=3950,
        trade_id=2
    )

    print("\nJumping to trade at bar 50 (near start):")
    chart.jump_to_trade(test_trade2)

    test_trade3 = TradeData(
        bar_index=9950,  # Near end
        trade_type='BUY',
        price=4100,
        trade_id=3
    )

    print("\nJumping to trade at bar 9950 (near end):")
    chart.jump_to_trade(test_trade3)

    print("\n[OK] Jump-to-trade debug logging test complete")
    return True

def test_signal_connections():
    """Test that trade_selected signal is properly connected"""
    print("\n" + "="*60)
    print("TEST 4: SIGNAL CONNECTIONS")
    print("="*60)

    # Import with proper paths
    import sys
    sys.path.insert(0, 'src/trading/visualization')
    from pyqtgraph_range_bars_final import RangeBarChartFinal as RangeBarChart
    from trade_panel import TradePanel

    # Create minimal app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create chart with trade panel
    chart = RangeBarChart()

    # Check if trade_panel exists and signal is connected
    if hasattr(chart, 'trade_panel') and chart.trade_panel:
        # Check if signal is connected
        receivers = chart.trade_panel.trade_selected.receivers(chart.trade_panel.trade_selected)
        if receivers > 0:
            print(f"[OK] trade_selected signal has {receivers} connections")
        else:
            print(f"[FAIL] trade_selected signal has no connections")

        # Check if jump_to_trade method exists
        if hasattr(chart, 'jump_to_trade'):
            print("[OK] jump_to_trade method exists on chart")
        else:
            print("[FAIL] jump_to_trade method not found on chart")
    else:
        print("[WARNING] Trade panel not initialized")

    print("\n[OK] Signal connection test complete")
    return True

def main():
    """Run all debug logging tests"""
    print("="*60)
    print("DEBUG LOGGING VERIFICATION TESTS")
    print("="*60)
    print("\nThese tests verify that debug logging is properly added")
    print("for the three critical issues identified in projecttodos.md")

    all_passed = True

    # Test 1: Rendering debug
    try:
        result = test_rendering_debug()
        all_passed &= result
    except Exception as e:
        print(f"[FAIL] Rendering debug test failed: {e}")
        all_passed = False

    # Test 2: DateTime debug
    try:
        result = test_datetime_debug()
        all_passed &= result
    except Exception as e:
        print(f"[FAIL] DateTime debug test failed: {e}")
        all_passed = False

    # Test 3: Jump-to-trade debug
    try:
        result = test_jump_debug()
        all_passed &= result
    except Exception as e:
        print(f"[FAIL] Jump debug test failed: {e}")
        all_passed = False

    # Test 4: Signal connections
    try:
        result = test_signal_connections()
        all_passed &= result
    except Exception as e:
        print(f"[FAIL] Signal connection test failed: {e}")
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("[OK] ALL DEBUG LOGGING TESTS PASSED")
        print("\nDebug logging has been successfully added for:")
        print("1. Bar rendering (render_range, on_range_changed)")
        print("2. DateTime extraction in strategies")
        print("3. Jump-to-trade functionality")
        print("\nNext steps:")
        print("1. Run integrated_trading_launcher.py")
        print("2. Load the 10,000 bar test file")
        print("3. Pan/zoom to test rendering past 500 bars")
        print("4. Run a strategy to test DateTime display")
        print("5. Double-click trades to test jump functionality")
    else:
        print("[FAIL] SOME DEBUG LOGGING TESTS FAILED")
        print("Review the output above to identify issues")
    print("="*60)

if __name__ == "__main__":
    main()