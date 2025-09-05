#!/usr/bin/env python3
"""
Test with Real AD-style Data
Creates dashboard exactly as user would see with AD forex data
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import asyncio

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from PyQt5 import QtWidgets
from src.dashboard.dashboard_manager import DashboardManager

async def test_real_ad_data():
    """Test with realistic AD data that should show 5 decimal places"""
    
    print("=== REAL AD DATA TEST ===")
    
    # Real AD data with exactly 5 decimal places as user would have
    ad_prices = np.array([
        0.65432, 0.65445, 0.65401, 0.65467, 0.65423,
        0.65455, 0.65438, 0.65462, 0.65449, 0.65474,
        0.65441, 0.65456, 0.65430, 0.65463, 0.65447
    ])
    
    n_bars = len(ad_prices)
    
    # Create OHLC with small spreads
    close_prices = ad_prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 0.65430
    
    # Small forex spreads
    spread = 0.00008  # 0.8 pips
    high_prices = np.maximum(open_prices, close_prices) + spread/2
    low_prices = np.minimum(open_prices, close_prices) - spread/2
    
    volume = np.full(n_bars, 10000.0)  # Constant volume
    timestamps = np.arange(n_bars, dtype=np.int64) * 60 * 1000000000
    
    price_data = {
        'timestamps': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    
    print(f"AD Data Sample:")
    print(f"Close prices: {close_prices[:5]}")
    print(f"High prices: {high_prices[:5]}")
    print(f"Low prices: {low_prices[:5]}")
    
    # Test precision detection
    dm = DashboardManager()
    dm._update_precision_from_data(price_data)
    
    print(f"\nPrecision Detection:")
    print(f"Detected price precision: {dm.price_precision}")
    print(f"Expected: 5 decimal places")
    
    # Test formatting
    test_price = close_prices[0]
    formatted = f"{test_price:.{dm.price_precision}f}"
    print(f"Test price {test_price} formatted as: {formatted}")
    
    # Create trades
    trades_data = [
        {
            'trade_id': 'BUY_AD_001',
            'timestamp': timestamps[5],
            'side': 'buy',
            'price': 0.65455,
            'quantity': 100000,
            'pnl': None
        },
        {
            'trade_id': 'SELL_AD_001', 
            'timestamp': timestamps[10],
            'side': 'sell',
            'price': 0.65441,
            'quantity': 100000,
            'pnl': -140.00  # Loss
        }
    ]
    
    trade_df = pd.DataFrame(trades_data)
    
    print(f"\nTrade Data:")
    print(f"Buy price: {trades_data[0]['price']}")
    print(f"Sell price: {trades_data[1]['price']}")
    print(f"Expected PnL: ${trades_data[1]['pnl']}")
    
    # Create dashboard
    if not dm.initialize_qt_app():
        print("Failed to initialize Qt")
        return
        
    dm.create_main_window()
    
    # Load data
    await dm.load_backtest_data(price_data, trade_df, {})
    
    print(f"\n=== FINAL VERIFICATION ===")
    print(f"Dashboard precision: {dm.price_precision}")
    print(f"Chart precision: {dm.main_chart.price_precision}")
    print(f"Left axis precision: {dm.main_chart.left_price_axis.precision}")
    print(f"Right axis precision: {dm.main_chart.right_price_axis.precision}")
    print(f"Trade list precision: {dm.trade_list.price_precision}")
    
    # Test all formatting methods
    sample_price = 0.65432
    print(f"\nFormatting Tests:")
    print(f"Dashboard: ${sample_price:.{dm.price_precision}f}")
    print(f"Chart: ${sample_price:.{dm.main_chart.price_precision}f}")
    
    # Verify trade list formatting
    if hasattr(dm.trade_list, 'item') and dm.trade_list.rowCount() > 0:
        price_item = dm.trade_list.item(0, 3)  # Price column
        if price_item:
            print(f"Trade list price display: {price_item.text()}")
    
    dm.show()
    
    print(f"\n{'='*50}")
    print(f"AD DASHBOARD READY")
    print(f"Expected everywhere: 5 decimal places")
    print(f"Y-axis should show: 0.65432, 0.65445, etc.")
    print(f"Data window should show: O: $0.65432, etc.")
    print(f"Trade list should show: $0.65455, etc.")
    print(f"{'='*50}")
    
    return dm

if __name__ == "__main__":
    dm = asyncio.run(test_real_ad_data())
    if dm and dm.app:
        try:
            dm.app.exec_()
        except KeyboardInterrupt:
            dm.app.quit()