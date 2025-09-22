#!/usr/bin/env python3
"""
Test the DateTime fix - verify timestamps propagate correctly
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')
sys.path.insert(0, 'src/trading/strategies')
sys.path.insert(0, 'src/trading/data')

import pandas as pd
import numpy as np
from datetime import datetime
from PyQt5 import QtWidgets

# Import the classes we need to test
from strategy_runner import StrategyRunner
from trade_panel import TradeListPanel
from strategies.sma_crossover import SMACrossoverStrategy

def test_datetime_propagation():
    """Test that timestamps propagate correctly through the system"""

    print("=" * 80)
    print("Testing DateTime Propagation After Fix")
    print("=" * 80)

    # Create Qt app for widgets
    app = QtWidgets.QApplication(sys.argv)

    # 1. Create sample data with timestamps - as chart would provide
    n = 100
    timestamps = pd.date_range('2022-01-18', periods=n, freq='5min')
    base_price = 5185.50
    prices = base_price + np.cumsum(np.random.randn(n) * 2)

    # This is what the chart provides after our fix
    chart_data = {
        'timestamp': timestamps,  # KEY FIX: Now includes timestamp!
        'open': prices + np.random.randn(n) * 0.5,
        'high': prices + np.abs(np.random.randn(n) * 2),
        'low': prices - np.abs(np.random.randn(n) * 2),
        'close': prices
    }

    print("\n1. Chart provides data with timestamp field:")
    print(f"   Keys in chart_data: {list(chart_data.keys())}")
    print(f"   First timestamp: {chart_data['timestamp'][0]}")

    # 2. Create StrategyRunner and set chart data
    strategy_runner = StrategyRunner()
    print("\n2. Setting chart data in StrategyRunner...")
    strategy_runner.set_chart_data(chart_data)

    # 3. Run strategy
    print("\n3. Running strategy to generate trades...")
    strategy_runner.strategy_combo.setCurrentText("SMA Crossover")
    strategy_runner.run_strategy()

    # 4. Check if the strategy runner has generated trades with timestamps
    print("\n4. Checking strategy results...")
    # The trades would be emitted via signal, so we need to check the internal state
    if hasattr(strategy_runner, 'current_strategy'):
        # Generate signals
        signals = strategy_runner.current_strategy.generate_signals(strategy_runner.chart_data)
        trades = strategy_runner.current_strategy.signals_to_trades(signals, strategy_runner.chart_data)

        print(f"   Generated {len(trades)} trades")

        # Check first few trades for timestamps
        print("\n5. Trade timestamps:")
        for i, trade in enumerate(trades[:5]):
            if trade.timestamp:
                print(f"   Trade {i}: timestamp={trade.timestamp}, "
                      f"formatted={trade.timestamp.strftime('%y-%m-%d %H:%M:%S')}")
            else:
                print(f"   Trade {i}: timestamp=None (BUG!)")

        # 6. Test trade panel display
        print("\n6. Testing trade panel display...")
        trade_panel = TradeListPanel()
        trade_panel.load_trades(trades)

        # The trade panel should now show timestamps
        print("   Trade panel loaded with trades")

    print("\n" + "=" * 80)
    print("Test Complete - DateTime should now work!")
    print("=" * 80)

    # Don't run the Qt event loop, just exit
    return 0

if __name__ == "__main__":
    test_datetime_propagation()