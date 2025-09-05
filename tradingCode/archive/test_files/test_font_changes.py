#!/usr/bin/env python3
"""
Test font size changes for indicator panel and axes
Create a dashboard to verify that fonts are 2x larger
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

async def test_font_changes():
    """Test the font size changes in dashboard"""
    
    print("=== TESTING FONT SIZE CHANGES ===")
    print("This will create a dashboard to verify 2x larger fonts")
    
    # Create simple forex data
    n_bars = 20
    close_prices = np.array([
        0.65432, 0.65445, 0.65401, 0.65467, 0.65423,
        0.65456, 0.65434, 0.65478, 0.65412, 0.65398,
        0.65441, 0.65459, 0.65427, 0.65463, 0.65439,
        0.65452, 0.65418, 0.65485, 0.65421, 0.65407
    ])
    
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 0.65430
    
    spread = 0.00050
    high_prices = close_prices + spread
    low_prices = close_prices - spread
    
    volume = np.full(n_bars, 10000.0)
    
    # Simple timestamps
    base_time = 1609459200
    timestamps = np.array([
        (base_time + i * 60) * 1_000_000_000
        for i in range(n_bars)
    ], dtype=np.int64)
    
    price_data = {
        'timestamps': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    
    # Create trades
    trades_data = [
        {
            'trade_id': 'FONT_TEST_BUY',
            'timestamp': timestamps[5],
            'side': 'buy',
            'price': 0.65456,
            'quantity': 100000,
            'pnl': None
        },
        {
            'trade_id': 'FONT_TEST_SELL',
            'timestamp': timestamps[15],
            'side': 'sell',
            'price': 0.65421,
            'quantity': 100000,
            'pnl': -350.00
        }
    ]
    
    trade_df = pd.DataFrame(trades_data)
    portfolio_data = {'equity_curve': np.linspace(10000, 10100, n_bars)}
    
    # Initialize dashboard
    dashboard = DashboardManager()
    
    if not dashboard.initialize_qt_app():
        print("ERROR: Failed to initialize Qt")
        return None
    
    dashboard.create_main_window()
    
    print("Loading data...")
    await dashboard.load_backtest_data(price_data, trade_df, portfolio_data)
    
    print("Applying 5-decimal precision...")
    dashboard.force_precision(5)
    
    # Show dashboard
    dashboard.show()
    
    # Enhanced rendering
    for i in range(5):
        dashboard.app.processEvents()
        time.sleep(0.3)
    
    # Set chart range
    if hasattr(dashboard, 'main_chart'):
        chart = dashboard.main_chart
        y_min = float(np.min(low_prices))
        y_max = float(np.max(high_prices))
        y_padding = (y_max - y_min) * 0.05
        
        chart.setYRange(y_min - y_padding, y_max + y_padding)
        chart.setXRange(0, n_bars - 1)
        
        for i in range(3):
            chart.update()
            dashboard.app.processEvents()
            time.sleep(0.5)
    
    # Take screenshot
    screenshot_path = Path(__file__).parent / "font_test_result.png"
    if dashboard.main_window:
        pixmap = dashboard.main_window.grab()
        pixmap.save(str(screenshot_path))
        print(f"Screenshot saved: {screenshot_path}")
    
    print("\\n=== FONT SIZE VERIFICATION ===")
    print("Check screenshot for:")
    print("1. Y-axis labels should be larger and bold (16pt font)")
    print("2. X-axis (time) labels should be larger and bold (16pt font)")
    print("3. Indicator panel text should be larger and bold (16pt font)")
    print("4. 'VectorBT Pro Indicators' title should be more prominent")
    print("5. Dropdown and button text should be larger")
    
    return dashboard

if __name__ == "__main__":
    dashboard = asyncio.run(test_font_changes())
    
    if dashboard and dashboard.app:
        print("\\nFont test dashboard created!")
        print("Verify that fonts are 2x larger than before")
        print("Press Ctrl+C to exit")
        try:
            dashboard.app.exec_()
        except KeyboardInterrupt:
            print("\\nExiting...")
            dashboard.app.quit()