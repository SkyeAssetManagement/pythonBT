"""
Test all three critical fixes:
1. X-axis DateTime labels
2. Trade list DateTime display
3. Dynamic data loading when panning
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
import pandas as pd
import numpy as np

sys.path.append('src/trading')
sys.path.append('src/trading/visualization')

def test_fixes():
    """Test all three critical fixes"""
    print("="*60)
    print("TESTING ALL THREE CRITICAL FIXES")
    print("="*60)

    # Create app
    app = QApplication(sys.argv)

    # Import chart class
    from pyqtgraph_range_bars_final import RangeBarChartFinal

    # Create chart instance - this will auto-load the ES-DIFF data
    print("\n1. Creating chart (will auto-load ES-DIFF-range-ATR30x0.1-amibroker.parquet)...")
    chart = RangeBarChartFinal()

    print("\n" + "-"*60)
    print("TEST 1: X-AXIS DATETIME LABELS")
    print("-"*60)
    print("Check debug output above for [FORMAT_TIME_AXIS] logs")
    print("Should show:")
    print("- Timestamp type and first timestamp")
    print("- Label positions and formatted timestamps")
    print("- NOT just 0:00:00")

    print("\n" + "-"*60)
    print("TEST 2: TRADE LIST DATETIME")
    print("-"*60)

    # Run a simple strategy to generate trades
    from strategies.sma_crossover import SMACrossoverStrategy
    from visualization.strategy_runner import StrategyRunner

    # Create strategy runner and set data
    if hasattr(chart.trade_panel, 'strategy_runner'):
        runner = chart.trade_panel.strategy_runner

        # Get bar data from chart
        bar_data = {
            'timestamp': chart.full_data['timestamp'],
            'open': chart.full_data['open'],
            'high': chart.full_data['high'],
            'low': chart.full_data['low'],
            'close': chart.full_data['close'],
        }

        # Set data in runner
        runner.set_chart_data(bar_data)

        print("Running SMA Crossover strategy to generate trades...")
        runner.run_strategy()

        print("\nCheck if trades have timestamps:")
        print("- Look for [STRATEGY] logs showing DateTime column detection")
        print("- Trade list should show actual dates/times, not '-'")

    print("\n" + "-"*60)
    print("TEST 3: DYNAMIC DATA LOADING")
    print("-"*60)

    # Test panning to various ranges
    test_ranges = [
        (0, 500, "Initial range"),
        (450, 550, "Crossing 500 boundary"),
        (10000, 10500, "Mid-range"),
        (50000, 50500, "Far range"),
        (100000, 100500, "Very far range"),
        (122000, 122609, "End of data")
    ]

    for start, end, desc in test_ranges:
        print(f"\nPanning to {desc} (bars {start}-{end}):")
        chart.render_range(start, end)

        # Check for logs
        print("Check for:")
        print("- [ON_X_RANGE_CHANGED] logs")
        print("- [RENDER_RANGE] showing correct range")
        print("- Data slicing working properly")

    print("\n" + "="*60)
    print("SUMMARY OF FIXES APPLIED")
    print("="*60)

    print("\n1. X-AXIS DATETIME LABELS:")
    print("   - Fixed x-coordinate mapping in format_time_axis()")
    print("   - Added debug logging to trace timestamp formatting")
    print("   - Timestamps should now display correctly")

    print("\n2. TRADE LIST DATETIME:")
    print("   - StrategyRunner creates DataFrame with DateTime column")
    print("   - Strategy base class extracts DateTime and passes to trades")
    print("   - TradeTableModel should display timestamps")

    print("\n3. DYNAMIC DATA LOADING:")
    print("   - Added on_x_range_changed() handler")
    print("   - Connected sigXRangeChanged signal")
    print("   - ViewBox limits increased to 200,000 bars")
    print("   - Should now load data continuously when panning")

    print("\n" + "="*60)
    print("CHECK VISUAL RESULTS:")
    print("="*60)
    print("1. X-axis should show real dates/times")
    print("2. Trade list DateTime column should show timestamps")
    print("3. Chart should render continuously when panning right")
    print("4. No cutoff at any point in the 122,609 bars")

    return chart

if __name__ == "__main__":
    chart = test_fixes()
    print("\n\nChart window is open for manual testing.")
    print("Try:")
    print("- Pan left/right to see DateTime labels update")
    print("- Run a strategy and check trade timestamps")
    print("- Pan to the far right (bar 122,000+) to test dynamic loading")
    chart.show()
    sys.exit(QApplication.instance().exec_())