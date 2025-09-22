"""
Direct test of range bar rendering with ES-DIFF data
Tests the debug logging with actual 122,609 bars
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

# Add paths
sys.path.append('src/trading')
sys.path.append('src/trading/visualization')

def test_range_bar_rendering():
    """Test rendering with actual ES-DIFF range bar data"""
    print("\n" + "="*60)
    print("TESTING WITH ES-DIFF RANGE BAR DATA (122,609 bars)")
    print("="*60)

    # Create app
    app = QApplication(sys.argv)

    # Import chart class
    from pyqtgraph_range_bars_final import RangeBarChartFinal

    # Create chart instance - this will auto-load the ES-DIFF data
    print("\nCreating chart (will auto-load ES-DIFF-range-ATR30x0.1-amibroker.parquet)...")
    chart = RangeBarChartFinal()

    print("\nChart should have loaded 122,609 bars")
    print("Debug logs above should show:")
    print("- [RENDER_RANGE] with initial 500 bars")
    print("- [ON_RANGE_CHANGED] for initial viewport")

    # Test panning to different ranges
    print("\n" + "-"*60)
    print("TEST: Panning to bars around 500 boundary")
    print("-"*60)

    print("\nPanning to bars 400-600 (crossing the 500 boundary):")
    chart.render_range(400, 600)

    print("\nPanning to bars 490-510 (tight around 500):")
    chart.render_range(490, 510)

    print("\nPanning to bars 500-1000 (starting at 500):")
    chart.render_range(500, 1000)

    print("\n" + "-"*60)
    print("TEST: Panning to distant ranges")
    print("-"*60)

    print("\nPanning to bars 10000-10500:")
    chart.render_range(10000, 10500)

    print("\nPanning to bars 50000-50500:")
    chart.render_range(50000, 50500)

    print("\nPanning to bars 100000-100500:")
    chart.render_range(100000, 100500)

    print("\nPanning to bars 120000-122609 (end of data):")
    chart.render_range(120000, 122609)

    print("\n" + "-"*60)
    print("TEST: Testing jump-to-trade at various positions")
    print("-"*60)

    # Test jump-to-trade
    from data.trade_data import TradeData
    import pandas as pd

    # Create test trades at various positions
    test_trades = [
        (500, "At boundary"),
        (5000, "Mid-range"),
        (50000, "Far out"),
        (100000, "Very far"),
        (122000, "Near end")
    ]

    for bar_idx, desc in test_trades:
        trade = TradeData(
            bar_index=bar_idx,
            trade_type='BUY',
            price=4000,
            trade_id=bar_idx,
            timestamp=pd.Timestamp('2024-01-01 12:00:00')
        )
        print(f"\nJumping to trade at bar {bar_idx} ({desc}):")
        chart.jump_to_trade(trade)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nCheck debug output above to verify:")
    print("1. [RENDER_RANGE] logs show correct start/end for each pan")
    print("2. Data slicing works for all ranges (not just 0-500)")
    print("3. [JUMP_TO_TRADE] correctly calculates viewport for all positions")
    print("4. No errors when accessing bars beyond 500")

    return chart

if __name__ == "__main__":
    chart = test_range_bar_rendering()
    print("\nChart window is open for manual testing.")
    print("Try panning/zooming to see debug logs in real-time.")
    chart.show()
    sys.exit(QApplication.instance().exec_())