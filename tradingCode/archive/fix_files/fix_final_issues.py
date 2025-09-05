#!/usr/bin/env python3
"""
Fix the final two issues:
1. Trade list must show FULL 5 decimal places (e.g., 0.65432 not 0.6543)
2. Trade arrows must be positioned ABOVE/BELOW candle highs/lows, not at trade price
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

async def test_precision_and_arrows():
    """Create test with precise data to verify both fixes"""
    
    print("=== FIXING PRECISION AND ARROW POSITIONING ===")
    
    # Create data with EXACTLY 5 decimal places that are NOT truncated
    forex_prices = np.array([
        0.65432,  # 5 decimals
        0.65445,  # 5 decimals  
        0.65401,  # 5 decimals
        0.65467,  # 5 decimals
        0.65423,  # 5 decimals
        0.65456,  # 5 decimals
        0.65434,  # 5 decimals
        0.65478,  # 5 decimals
        0.65412,  # 5 decimals
        0.65398   # 5 decimals
    ])
    
    # Repeat pattern to create full dataset
    n_bars = 50
    close_prices = np.tile(forex_prices, n_bars // len(forex_prices) + 1)[:n_bars]
    
    # Generate OHLC with significant spreads for visible candlesticks
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 0.65430
    
    # Create realistic spreads (2-3 pips)
    spreads = np.random.uniform(0.00020, 0.00030, n_bars)
    high_prices = close_prices + spreads
    low_prices = close_prices - spreads
    
    # Round to 5 decimal places
    high_prices = np.round(high_prices, 5)
    low_prices = np.round(low_prices, 5)
    close_prices = np.round(close_prices, 5)
    open_prices = np.round(open_prices, 5)
    
    volume = np.full(n_bars, 10000.0)
    timestamps = np.arange(n_bars, dtype=np.int64) * 60 * 1000000000
    
    print(f"Created {n_bars} bars with precise 5-decimal prices")
    print(f"Sample close prices: {close_prices[:5]}")
    print(f"Sample high prices: {high_prices[:5]}")
    print(f"Sample low prices: {low_prices[:5]}")
    
    # Verify all prices have exactly 5 decimal places
    for i in range(5):
        print(f"  Close[{i}]: {close_prices[i]:.5f} (should show all 5 decimals)")
    
    price_data = {
        'timestamps': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    
    # Create trades with EXACT 5-decimal prices for testing
    trade_idx_1 = 15
    trade_idx_2 = 35
    
    trades_data = [
        {
            'trade_id': 'BUY_PRECISE',
            'timestamp': timestamps[trade_idx_1],
            'side': 'buy',
            'price': 0.65432,  # EXACTLY 5 decimals
            'quantity': 100000,
            'pnl': None
        },
        {
            'trade_id': 'SELL_PRECISE',
            'timestamp': timestamps[trade_idx_2],
            'side': 'sell',
            'price': 0.65467,  # EXACTLY 5 decimals
            'quantity': 100000,
            'pnl': 350.00
        }
    ]
    
    trade_df = pd.DataFrame(trades_data)
    
    print(f"\\n=== TRADE POSITIONING ANALYSIS ===")
    for i, trade in enumerate(trades_data):
        idx = trade_idx_1 if i == 0 else trade_idx_2
        candle_high = high_prices[idx]
        candle_low = low_prices[idx]
        candle_range = candle_high - candle_low
        
        print(f"Trade {i+1} ({trade['side']}):")
        print(f"  Trade price: {trade['price']:.5f}")
        print(f"  Candle high: {candle_high:.5f}")
        print(f"  Candle low:  {candle_low:.5f}")
        print(f"  Candle range: {candle_range:.5f}")
        
        if trade['side'] == 'buy':
            expected_y = candle_low - (candle_range * 0.02)
            print(f"  Expected BUY arrow: {expected_y:.5f} (BELOW low)")
            print(f"  Arrow should be BELOW {candle_low:.5f}")
        else:
            expected_y = candle_high + (candle_range * 0.02)
            print(f"  Expected SELL arrow: {expected_y:.5f} (ABOVE high)")
            print(f"  Arrow should be ABOVE {candle_high:.5f}")
    
    portfolio_data = {'equity_curve': np.linspace(10000, 10500, n_bars)}
    
    # Initialize dashboard
    dashboard = DashboardManager()
    
    if not dashboard.initialize_qt_app():
        print("ERROR: Failed to initialize Qt")
        return None
    
    dashboard.create_main_window()
    
    # Load data
    print("\\nLoading data into dashboard...")
    await dashboard.load_backtest_data(price_data, trade_df, portfolio_data)
    
    # Force precision to 5 explicitly
    print(f"\\n=== FORCING PRECISION TO 5 ===")
    print(f"Before force: Dashboard precision = {dashboard.price_precision}")
    dashboard.force_precision(5)
    print(f"After force: Dashboard precision = {dashboard.price_precision}")
    
    # Verify trade list precision
    if hasattr(dashboard.trade_list, 'price_precision'):
        print(f"Trade list precision: {dashboard.trade_list.price_precision}")
    
    # Show dashboard
    dashboard.show()
    dashboard.app.processEvents()
    time.sleep(2)
    
    # Take screenshot
    screenshot_path = Path(__file__).parent / "precision_arrow_fix.png"
    if dashboard.main_window:
        pixmap = dashboard.main_window.grab()
        pixmap.save(str(screenshot_path))
        print(f"\\nScreenshot saved: {screenshot_path}")
    
    print(f"\\n{'='*70}")
    print(f"VERIFICATION CHECKLIST:")
    print(f"1. Trade list MUST show: 0.65432 and 0.65467 (FULL 5 decimals)")
    print(f"2. Buy arrow MUST be positioned BELOW the candle low")
    print(f"3. Sell arrow MUST be positioned ABOVE the candle high")
    print(f"4. Y-axis should show 5 decimal places")
    print(f"5. Candlesticks should be clearly visible")
    print(f"{'='*70}")
    
    return dashboard

if __name__ == "__main__":
    dashboard = asyncio.run(test_precision_and_arrows())
    
    if dashboard and dashboard.app:
        print("\\nDashboard ready - check the screenshot!")
        print("Press Ctrl+C to exit")
        try:
            dashboard.app.exec_()
        except KeyboardInterrupt:
            print("\\nExiting...")
            dashboard.app.quit()