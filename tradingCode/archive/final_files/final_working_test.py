#!/usr/bin/env python3
"""
Create a WORKING dashboard with:
1. Exact timestamp matching for trade arrows
2. Full 5-decimal precision display
3. Visible candlesticks
4. Proper arrow positioning
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import asyncio
import time

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from PyQt5 import QtWidgets
from src.dashboard.dashboard_manager import DashboardManager

async def create_perfect_dashboard():
    """Create dashboard with perfect timestamp matching and precision"""
    
    print("=== CREATING PERFECT WORKING DASHBOARD ===")
    
    # Create simple dataset with exact matching
    n_bars = 20
    
    # Generate timestamps first
    base_timestamp = 1609459200  # 2021-01-01 00:00:00 UTC
    timestamps = np.array([
        (base_timestamp + i * 60) * 1_000_000_000  # Nanoseconds, 1-minute bars
        for i in range(n_bars)
    ], dtype=np.int64)
    
    # Create forex data with exact 5 decimal places
    forex_prices = np.array([
        0.65432, 0.65445, 0.65401, 0.65467, 0.65423,
        0.65456, 0.65434, 0.65478, 0.65412, 0.65398,
        0.65441, 0.65459, 0.65427, 0.65463, 0.65439,
        0.65452, 0.65418, 0.65485, 0.65421, 0.65407
    ])
    
    # Create OHLC with large enough spreads to see candlesticks clearly
    close_prices = forex_prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 0.65430
    
    # Create significant spreads for visibility
    spread = 0.00050  # 5 pips
    high_prices = np.maximum(open_prices, close_prices) + spread
    low_prices = np.minimum(open_prices, close_prices) - spread
    
    volume = np.full(n_bars, 10000.0)
    
    print(f"Created {n_bars} bars of data")
    print(f"Timestamp range: {timestamps[0]} to {timestamps[-1]}")
    print(f"Sample OHLC: O={open_prices[0]:.5f}, H={high_prices[0]:.5f}, L={low_prices[0]:.5f}, C={close_prices[0]:.5f}")
    print(f"Candle range example: {high_prices[0] - low_prices[0]:.5f}")
    
    price_data = {
        'timestamps': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    
    # Create trades using EXACT timestamps from the chart data
    buy_idx = 5
    sell_idx = 15
    
    trades_data = [
        {
            'trade_id': 'BUY_EXACT',
            'timestamp': timestamps[buy_idx],  # EXACT match
            'side': 'buy',
            'price': close_prices[buy_idx],    # Use actual close price
            'quantity': 100000,
            'pnl': None
        },
        {
            'trade_id': 'SELL_EXACT',
            'timestamp': timestamps[sell_idx], # EXACT match
            'side': 'sell',
            'price': close_prices[sell_idx],   # Use actual close price
            'quantity': 100000,
            'pnl': (close_prices[sell_idx] - close_prices[buy_idx]) * 100000
        }
    ]
    
    print(f"\\n=== TRADE ANALYSIS ===")
    for i, trade in enumerate(trades_data):
        idx = buy_idx if i == 0 else sell_idx
        print(f"Trade {i+1} ({trade['side']}):")
        print(f"  Timestamp: {trade['timestamp']} (matches chart timestamp exactly)")
        print(f"  Price: {trade['price']:.5f}")
        print(f"  Candle High: {high_prices[idx]:.5f}")
        print(f"  Candle Low: {low_prices[idx]:.5f}")
        
        candle_range = high_prices[idx] - low_prices[idx]
        if trade['side'] == 'buy':
            expected_arrow = low_prices[idx] - (candle_range * 0.02)
            print(f"  Expected BUY arrow: {expected_arrow:.5f} (below {low_prices[idx]:.5f})")
        else:
            expected_arrow = high_prices[idx] + (candle_range * 0.02)
            print(f"  Expected SELL arrow: {expected_arrow:.5f} (above {high_prices[idx]:.5f})")
    
    trade_df = pd.DataFrame(trades_data)
    portfolio_data = {'equity_curve': np.linspace(10000, 10200, n_bars)}
    
    # Initialize dashboard
    dashboard = DashboardManager()
    
    if not dashboard.initialize_qt_app():
        print("ERROR: Failed to initialize Qt")
        return None
    
    dashboard.create_main_window()
    
    # Load data
    print("\\nLoading data into dashboard...")
    await dashboard.load_backtest_data(price_data, trade_df, portfolio_data)
    
    # Force precision
    print(f"\\nForcing 5-decimal precision...")
    dashboard.force_precision(5)
    print(f"Dashboard precision: {dashboard.price_precision}")
    
    # Verify chart state
    print(f"\\n=== CHART STATE ===")
    if hasattr(dashboard.main_chart, 'data_buffer') and dashboard.main_chart.data_buffer:
        print(f"Chart data buffer: {len(dashboard.main_chart.data_buffer)} bars")
        print(f"Chart has candle item: {hasattr(dashboard.main_chart, 'candle_item')}")
        print(f"Chart has trade markers: {len(getattr(dashboard.main_chart, 'trade_markers', {}))}")
        
        # Check view range
        view_range = dashboard.main_chart.viewRange()
        print(f"Chart view range: X=[{view_range[0][0]:.1f}, {view_range[0][1]:.1f}], Y=[{view_range[1][0]:.5f}, {view_range[1][1]:.5f}]")
    
    # Show dashboard
    dashboard.show()
    dashboard.app.processEvents()
    time.sleep(3)
    
    # Take screenshot
    screenshot_path = Path(__file__).parent / "perfect_dashboard.png"
    if dashboard.main_window:
        pixmap = dashboard.main_window.grab()
        pixmap.save(str(screenshot_path))
        print(f"\\nScreenshot saved: {screenshot_path}")
    
    print(f"\\n{'='*80}")
    print(f"FINAL VERIFICATION:")
    print(f"1. Candlesticks should be visible in the chart")
    print(f"2. Trade arrows should be visible and positioned correctly")
    print(f"3. Trade list should show full 5-decimal prices")
    print(f"4. Y-axis should show 5-decimal precision")
    print(f"5. No timestamp errors")
    print(f"{'='*80}")
    
    return dashboard

if __name__ == "__main__":
    dashboard = asyncio.run(create_perfect_dashboard())
    
    if dashboard and dashboard.app:
        print("\\nDashboard created successfully!")
        print("Press Ctrl+C to exit")
        try:
            dashboard.app.exec_()
        except KeyboardInterrupt:
            print("\\nExiting...")
            dashboard.app.quit()