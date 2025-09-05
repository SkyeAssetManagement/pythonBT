#!/usr/bin/env python3
"""
Simple test to verify precision fix works
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

async def test_precision_fix():
    """Test the precision fix"""
    
    print("=== TESTING PRECISION FIX ===")
    
    # Create forex data that should trigger 5 decimal places
    forex_data = np.array([0.65432, 0.65445, 0.65401, 0.65467, 0.65423])
    
    # Test precision detection directly
    dm = DashboardManager()
    detected = dm._detect_precision(forex_data)
    print(f"Direct precision test: {detected} (expected: 5)")
    
    # Create full dataset for dashboard
    n_bars = 50
    close_prices = np.tile(forex_data, n_bars // 5 + 1)[:n_bars]
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 0.65430
    
    # Add small spreads
    spreads = np.full(n_bars, 0.00005)
    high_prices = close_prices + spreads
    low_prices = close_prices - spreads
    
    volume = np.full(n_bars, 10000.0)
    timestamps = np.arange(n_bars, dtype=np.int64) * 60 * 1000000000
    
    price_data = {
        'timestamps': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    
    # Create simple trades
    trades_data = [
        {
            'trade_id': 'TEST_BUY',
            'timestamp': timestamps[10],
            'side': 'buy',
            'price': 0.65432,
            'quantity': 100000,
            'pnl': None
        },
        {
            'trade_id': 'TEST_SELL',
            'timestamp': timestamps[30],
            'side': 'sell',
            'price': 0.65467,
            'quantity': 100000,
            'pnl': 350.00
        }
    ]
    
    trade_df = pd.DataFrame(trades_data)
    portfolio_data = {'equity_curve': np.linspace(10000, 10500, n_bars)}
    
    # Initialize dashboard
    if not dm.initialize_qt_app():
        print("ERROR: Failed to initialize Qt")
        return None
    
    dm.create_main_window()
    
    # Load data
    print("Loading data into dashboard...")
    await dm.load_backtest_data(price_data, trade_df, portfolio_data)
    
    print(f"\\n=== RESULTS ===")
    print(f"Dashboard price precision: {dm.price_precision}")
    print(f"Expected: 5")
    print(f"SUCCESS: {dm.price_precision == 5}")
    
    # Check trade list precision
    if hasattr(dm.trade_list, 'price_precision'):
        print(f"Trade list precision: {dm.trade_list.price_precision}")
    
    # Check chart precision  
    if hasattr(dm.main_chart, 'price_precision'):
        print(f"Chart precision: {dm.main_chart.price_precision}")
    
    # Show dashboard
    dm.show()
    dm.app.processEvents()
    time.sleep(2)
    
    # Take screenshot
    screenshot_path = Path(__file__).parent / "precision_fix_test.png"
    if dm.main_window:
        pixmap = dm.main_window.grab()
        pixmap.save(str(screenshot_path))
        print(f"\\nScreenshot saved: {screenshot_path}")
        
        # Check if trade list shows proper precision
        print("\\nCHECKLIST:")
        print("1. Trade list should show 0.65432 (not 0.6)")
        print("2. Chart should be visible with candlesticks")
        print("3. Y-axis should show 5 decimal places")
    
    return dm

if __name__ == "__main__":
    dashboard = asyncio.run(test_precision_fix())
    
    if dashboard and dashboard.app:
        print("\\nPress Ctrl+C to exit")
        try:
            dashboard.app.exec_()
        except KeyboardInterrupt:
            print("\\nExiting...")
            dashboard.app.quit()