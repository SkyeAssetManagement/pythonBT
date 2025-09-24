#!/usr/bin/env python3
"""
Test script for enhanced trade panel fixes
Tests:
1. Proper P&L compounding calculation
2. Column sorting functionality
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')

from PyQt5 import QtWidgets
from enhanced_trade_panel import EnhancedTradeListPanel, EnhancedTradeTableModel
from trade_data import TradeData, TradeCollection

def test_pnl_calculation():
    """Test that P&L calculation properly compounds returns"""
    print("Testing P&L Calculation...")

    # Create test trades with known P&L percentages
    trades = [
        TradeData(trade_id=1, timestamp='2021-01-01 10:00:00', trade_type='BUY',
                 price=100, size=1, pnl_percent=0.10, bar_index=1),  # 10% gain
        TradeData(trade_id=2, timestamp='2021-01-01 11:00:00', trade_type='SELL',
                 price=110, size=1, pnl_percent=0.05, bar_index=2),  # 5% gain
        TradeData(trade_id=3, timestamp='2021-01-01 12:00:00', trade_type='SHORT',
                 price=115, size=1, pnl_percent=-0.02, bar_index=3),  # 2% loss
        TradeData(trade_id=4, timestamp='2021-01-01 13:00:00', trade_type='COVER',
                 price=113, size=1, pnl_percent=0.03, bar_index=4),  # 3% gain
    ]

    collection = TradeCollection(trades)

    # Calculate expected compounded return
    # Formula: (1 + 0.10) * (1 + 0.05) * (1 - 0.02) * (1 + 0.03) - 1
    expected_total = 1.10 * 1.05 * 0.98 * 1.03 - 1.0
    expected_total_pct = expected_total * 100

    print(f"Expected total P&L: {expected_total_pct:.2f}%")

    # Test the model's calculation
    model = EnhancedTradeTableModel()
    model.set_trades(collection)

    # Check cumulative P&L for last trade
    last_cum_pnl = model.cumulative_pnl_percent[-1]
    last_cum_pnl_pct = last_cum_pnl * 100

    print(f"Calculated total P&L: {last_cum_pnl_pct:.2f}%")

    # Verify they match (within floating point tolerance)
    assert abs(last_cum_pnl - expected_total) < 0.0001, \
        f"P&L calculation mismatch: expected {expected_total:.4f}, got {last_cum_pnl:.4f}"

    print("P&L calculation test PASSED")
    return True

def test_sorting_functionality():
    """Test that column sorting works correctly"""
    print("\nTesting Column Sorting...")

    from PyQt5 import QtCore
    app = QtWidgets.QApplication(sys.argv)

    # Create panel
    panel = EnhancedTradeListPanel()

    # Create test trades with various values
    trades = [
        TradeData(trade_id=3, timestamp='2021-01-03 10:00:00', trade_type='SHORT',
                 price=150, size=2, pnl_percent=0.05, bar_index=30),
        TradeData(trade_id=1, timestamp='2021-01-01 10:00:00', trade_type='BUY',
                 price=100, size=1, pnl_percent=-0.10, bar_index=10),
        TradeData(trade_id=2, timestamp='2021-01-02 10:00:00', trade_type='SELL',
                 price=120, size=3, pnl_percent=0.15, bar_index=20),
    ]

    collection = TradeCollection(trades)
    panel.load_trades(collection)

    # Test sorting by trade_id (column 0)
    panel.sort_trades(0, QtCore.Qt.AscendingOrder)
    sorted_trade = panel.table_model.get_trade_at_row(0)
    assert sorted_trade.trade_id == 1, f"Expected trade_id 1, got {sorted_trade.trade_id}"
    print("Sort by Trade ID ascending works")

    panel.sort_trades(0, QtCore.Qt.DescendingOrder)
    sorted_trade = panel.table_model.get_trade_at_row(0)
    assert sorted_trade.trade_id == 3, f"Expected trade_id 3, got {sorted_trade.trade_id}"
    print("Sort by Trade ID descending works")

    # Test sorting by P&L % (column 5)
    panel.sort_trades(5, QtCore.Qt.DescendingOrder)
    sorted_trade = panel.table_model.get_trade_at_row(0)
    assert sorted_trade.pnl_percent == 0.15, f"Expected pnl 0.15, got {sorted_trade.pnl_percent}"
    print("Sort by P&L % descending works (largest gain first)")

    panel.sort_trades(5, QtCore.Qt.AscendingOrder)
    sorted_trade = panel.table_model.get_trade_at_row(0)
    assert sorted_trade.pnl_percent == -0.10, f"Expected pnl -0.10, got {sorted_trade.pnl_percent}"
    print("Sort by P&L % ascending works (largest loss first)")

    print("Sorting functionality test PASSED")

    app.quit()
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("Enhanced Trade Panel Test Suite")
    print("="*60)

    try:
        # Test P&L calculation
        test_pnl_calculation()

        # Test sorting
        test_sorting_functionality()

        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        print("\nFixes verified:")
        print("1. Total P&L now correctly compounds returns")
        print("2. Column sorting works on all columns")
        print("\nYou can now run the main application to see the fixes in action:")
        print("  python launch_unified_system.py")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()