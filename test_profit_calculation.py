#!/usr/bin/env python3
"""
Test script to verify profit calculation based on $1 invested
and backtest summary display
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')
sys.path.insert(0, 'src/trading/visualization')

import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
from datetime import datetime, timedelta

# Import trade components
from data.trade_data import TradeData, TradeCollection
from visualization.enhanced_trade_panel import EnhancedTradeListPanel

def create_test_trades():
    """Create test trades with known P&L values"""
    trades = []

    # Test case 1: Long trade with 10% profit
    # Buy at $100, sell at $110 = 10% profit
    trades.append(TradeData(
        bar_index=10,
        trade_type='BUY',
        price=100.0,
        trade_id=0,
        timestamp=pd.Timestamp('2024-01-01 09:00:00'),
        strategy='Test'
    ))
    trades.append(TradeData(
        bar_index=20,
        trade_type='SELL',
        price=110.0,
        trade_id=1,
        timestamp=pd.Timestamp('2024-01-01 10:00:00'),
        pnl=10.0,  # This will be treated as percentage
        strategy='Test'
    ))
    # Set explicit percentage
    trades[-1].pnl_percent = 10.0

    # Test case 2: Long trade with 5% loss
    # Buy at $200, sell at $190 = -5% loss
    trades.append(TradeData(
        bar_index=30,
        trade_type='BUY',
        price=200.0,
        trade_id=2,
        timestamp=pd.Timestamp('2024-01-01 11:00:00'),
        strategy='Test'
    ))
    trades.append(TradeData(
        bar_index=40,
        trade_type='SELL',
        price=190.0,
        trade_id=3,
        timestamp=pd.Timestamp('2024-01-01 12:00:00'),
        pnl=-5.0,  # This will be treated as percentage
        strategy='Test'
    ))
    trades[-1].pnl_percent = -5.0

    # Test case 3: Short trade with 8% profit
    # Short at $150, cover at $138 = 8% profit
    trades.append(TradeData(
        bar_index=50,
        trade_type='SHORT',
        price=150.0,
        trade_id=4,
        timestamp=pd.Timestamp('2024-01-01 13:00:00'),
        strategy='Test'
    ))
    trades.append(TradeData(
        bar_index=60,
        trade_type='COVER',
        price=138.0,
        trade_id=5,
        timestamp=pd.Timestamp('2024-01-01 14:00:00'),
        pnl=8.0,  # This will be treated as percentage
        strategy='Test'
    ))
    trades[-1].pnl_percent = 8.0

    # Test case 4: Short trade with 3% loss
    # Short at $100, cover at $103 = -3% loss
    trades.append(TradeData(
        bar_index=70,
        trade_type='SHORT',
        price=100.0,
        trade_id=6,
        timestamp=pd.Timestamp('2024-01-01 15:00:00'),
        strategy='Test'
    ))
    trades.append(TradeData(
        bar_index=80,
        trade_type='COVER',
        price=103.0,
        trade_id=7,
        timestamp=pd.Timestamp('2024-01-01 16:00:00'),
        pnl=-3.0,  # This will be treated as percentage
        strategy='Test'
    ))
    trades[-1].pnl_percent = -3.0

    return TradeCollection(trades)

def test_profit_calculations():
    """Test profit calculations and display"""
    print("=" * 60)
    print("Testing Profit Calculations Based on $1 Invested")
    print("=" * 60)

    # Create test trades
    trades = create_test_trades()

    # Print trade details
    print("\nTrade Details:")
    print("-" * 40)
    for i, trade in enumerate(trades):
        print(f"Trade {trade.trade_id}: {trade.trade_type} at ${trade.price:.2f}")
        if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
            sign = '+' if trade.pnl_percent >= 0 else ''
            print(f"  P&L: {sign}{trade.pnl_percent:.2f}%")

    # Calculate expected summary
    pnl_values = []
    for trade in trades:
        if hasattr(trade, 'pnl_percent') and trade.pnl_percent is not None:
            pnl_values.append(trade.pnl_percent)

    if pnl_values:
        total_trades = len(trades)
        closed_trades = len(pnl_values)
        wins = [p for p in pnl_values if p > 0]
        losses = [p for p in pnl_values if p < 0]
        win_rate = (len(wins) / closed_trades) * 100 if closed_trades else 0
        total_pnl = sum(pnl_values)
        avg_pnl = total_pnl / closed_trades if closed_trades else 0

        print("\nExpected Summary Statistics:")
        print("-" * 40)
        print(f"Total Trades: {total_trades}")
        print(f"Closed Trades: {closed_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total P&L: {'+' if total_pnl >= 0 else ''}{total_pnl:.2f}%")
        print(f"Avg P&L: {'+' if avg_pnl >= 0 else ''}{avg_pnl:.2f}%")

        # Verify calculations
        print("\nVerification:")
        print("-" * 40)
        print(f"P&L values: {pnl_values}")
        print(f"Sum: 10.0 + (-5.0) + 8.0 + (-3.0) = {sum(pnl_values):.2f}%")
        print(f"Cumulative P&L should be: 10.00%")
        print(f"Average P&L should be: {10.0/4:.2f}%")
        print(f"Win rate should be: {2/4*100:.2f}% (2 wins out of 4 closed trades)")

    return trades

def test_gui_display():
    """Test the GUI display of trades and summary"""
    print("\n" + "=" * 60)
    print("Testing GUI Display")
    print("=" * 60)

    app = QtWidgets.QApplication(sys.argv)

    # Create enhanced trade panel
    panel = EnhancedTradeListPanel()

    # Load test trades
    trades = create_test_trades()
    panel.set_trades(trades)

    # Show panel
    panel.setWindowTitle("Trade Panel - P&L Test")
    panel.resize(600, 800)
    panel.show()

    print("\nGUI panel created with enhanced trade list and backtest summary.")
    print("Check that:")
    print("1. P&L column shows percentages to 2 decimal places")
    print("2. Cumulative P&L column shows running total")
    print("3. Backtest Summary at bottom shows:")
    print("   - Total Trades: 8")
    print("   - Win Rate: 50.00%")
    print("   - Total P&L: +10.00%")
    print("   - Avg P&L: +2.50%")

    sys.exit(app.exec_())

if __name__ == "__main__":
    # Run tests
    trades = test_profit_calculations()

    # Run GUI test
    test_gui_display()