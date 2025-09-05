#!/usr/bin/env python3
"""
Simple font test using successful approach from comprehensive solution
Test reduced fonts with minimal complexity
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

async def simple_font_test():
    """Simple test with reduced font sizes"""
    
    print("=== SIMPLE FONT TEST ===")
    print("Testing 8pt fonts (reduced from 16pt)")
    
    # Use exact same data structure as our successful comprehensive solution
    n_bars = 10
    
    close_prices = np.array([
        0.65432, 0.65445, 0.65401, 0.65467, 0.65423,
        0.65456, 0.65434, 0.65478, 0.65412, 0.65398
    ])
    
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 0.65430
    
    spread = 0.00050
    high_prices = close_prices + spread
    low_prices = close_prices - spread
    
    volume = np.full(n_bars, 10000.0)
    
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
    
    # Empty trades to avoid data processing issues
    trade_df = pd.DataFrame()
    portfolio_data = {'equity_curve': np.linspace(10000, 10050, n_bars)}
    
    # Initialize dashboard
    dashboard = DashboardManager()
    
    if not dashboard.initialize_qt_app():
        print("ERROR: Failed to initialize Qt")
        return None
    
    dashboard.create_main_window()
    
    print("Loading data...")
    await dashboard.load_backtest_data(price_data, trade_df, portfolio_data)
    
    print("Forcing 5-decimal precision...")
    dashboard.force_precision(5)
    
    # Show dashboard
    dashboard.show()
    
    # Enhanced rendering
    for i in range(10):
        dashboard.app.processEvents()
        time.sleep(0.2)
    
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
    screenshot_path = Path(__file__).parent / "simple_font_test.png"
    if dashboard.main_window:
        pixmap = dashboard.main_window.grab()
        pixmap.save(str(screenshot_path))
        print(f"Screenshot saved: {screenshot_path}")
    
    print("\\n=== FONT VERIFICATION ===")
    print("Y-axis should show 5-decimal precision with 8pt font")
    print("Indicator panel should show readable 8pt font")
    
    return dashboard

if __name__ == "__main__":
    dashboard = asyncio.run(simple_font_test())
    
    if dashboard and dashboard.app:
        print("\\nSimple font test complete!")
        print("Press Ctrl+C to exit")
        try:
            dashboard.app.exec_()
        except KeyboardInterrupt:
            print("\\nExiting...")
            dashboard.app.quit()