#!/usr/bin/env python3
"""
Test reduced font sizes to verify Y-axis text is visible again
Font sizes reduced from 16pt to 8pt (50% reduction)
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

async def test_reduced_fonts():
    """Test the reduced font sizes to ensure Y-axis is visible"""
    
    print("=== TESTING REDUCED FONT SIZES ===")
    print("Font sizes reduced from 16pt to 8pt (50% reduction)")
    print("This should restore Y-axis text visibility")
    
    # Create forex data for testing
    n_bars = 15
    close_prices = np.array([
        0.65432, 0.65445, 0.65401, 0.65467, 0.65423,
        0.65456, 0.65434, 0.65478, 0.65412, 0.65398,
        0.65441, 0.65459, 0.65427, 0.65463, 0.65439
    ])
    
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 0.65430
    
    spread = 0.00040
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
    
    # Create a simple trade for testing
    trades_data = [{
        'trade_id': 'FONT_REDUCE_TEST',
        'timestamp': timestamps[7],
        'side': 'buy',
        'price': 0.65456,
        'quantity': 100000,
        'pnl': None
    }]
    
    trade_df = pd.DataFrame(trades_data)
    portfolio_data = {'equity_curve': np.linspace(10000, 10075, n_bars)}
    
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
    
    # Enhanced rendering with more processing time
    for i in range(8):
        dashboard.app.processEvents()
        time.sleep(0.4)
    
    # Set chart range carefully
    if hasattr(dashboard, 'main_chart'):
        chart = dashboard.main_chart
        y_min = float(np.min(low_prices))
        y_max = float(np.max(high_prices))
        y_padding = (y_max - y_min) * 0.08
        
        print(f"Setting Y range: {y_min - y_padding:.5f} to {y_max + y_padding:.5f}")
        chart.setYRange(y_min - y_padding, y_max + y_padding)
        chart.setXRange(0, n_bars - 1)
        
        # Multiple updates for proper rendering
        for i in range(5):
            chart.update()
            dashboard.app.processEvents()
            time.sleep(0.3)
    
    # Take screenshot
    screenshot_path = Path(__file__).parent / "font_reduced_test.png"
    if dashboard.main_window:
        pixmap = dashboard.main_window.grab()
        pixmap.save(str(screenshot_path))
        print(f"Screenshot saved: {screenshot_path}")
    
    print("\\n=== FONT REDUCTION VERIFICATION ===")
    print("Check screenshot for:")
    print("1. Y-axis labels should be VISIBLE with 5-decimal precision")
    print("2. X-axis time labels should be VISIBLE")
    print("3. Indicator panel text should be readable but not too large")
    print("4. All fonts should be 8pt (moderate size, not tiny, not huge)")
    print("5. Candlesticks should be visible in the chart")
    print("6. Trade arrow should be visible")
    
    return dashboard

if __name__ == "__main__":
    dashboard = asyncio.run(test_reduced_fonts())
    
    if dashboard and dashboard.app:
        print("\\nFont reduction test dashboard created!")
        print("Y-axis text should now be visible with proper 5-decimal precision")
        print("Press Ctrl+C to exit")
        try:
            dashboard.app.exec_()
        except KeyboardInterrupt:
            print("\\nExiting...")
            dashboard.app.quit()