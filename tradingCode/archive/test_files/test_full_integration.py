#!/usr/bin/env python3
"""
Full Integration Test - Screenshot and Verify All Components
Tests the complete dashboard with AD-style forex data to verify precision and trade positioning
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

from PyQt5 import QtWidgets, QtCore, QtGui
from src.dashboard.dashboard_manager import DashboardManager
from src.dashboard.data_structures import TradeData

async def create_test_dashboard():
    """Create dashboard with AD-style forex data for testing"""
    
    print("=== FULL INTEGRATION TEST ===")
    print("Creating dashboard with AD forex data (5 decimal places expected)")
    
    # Create test data - AD-style forex with 5 decimal places
    n_bars = 1000
    base_price = 0.65400
    
    # Generate realistic forex price movement
    np.random.seed(42)  # Reproducible results
    price_changes = np.cumsum(np.random.normal(0, 0.00010, n_bars)) # Small movements
    close_prices = base_price + price_changes
    
    # Generate OHLC with proper relationships
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    # High/Low with small spreads (realistic for forex)
    spreads = np.random.uniform(0.00005, 0.00020, n_bars)  # 0.5-2 pip spreads
    high_prices = np.maximum(open_prices, close_prices) + spreads
    low_prices = np.minimum(open_prices, close_prices) - spreads
    
    # Volume  
    volume = np.random.randint(1000, 50000, n_bars).astype(float)
    
    # Timestamps
    timestamps = np.arange(n_bars, dtype=np.int64) * 60 * 1000000000  # 1 minute bars
    
    price_data = {
        'timestamps': timestamps,
        'open': open_prices,
        'high': high_prices, 
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    
    print(f"Created {n_bars} bars of AD forex data")
    print(f"Price range: {np.min(low_prices):.5f} - {np.max(high_prices):.5f}")
    print(f"Sample close prices: {close_prices[:5]}")
    
    # Create some test trades with precise positioning
    trades_data = []
    
    # Buy trade at bar 100
    buy_idx = 100
    buy_trade = {
        'trade_id': 'BUY_001',
        'timestamp': timestamps[buy_idx],
        'side': 'buy',
        'price': close_prices[buy_idx], 
        'quantity': 10000,
        'pnl': None
    }
    trades_data.append(buy_trade)
    
    # Sell trade at bar 200  
    sell_idx = 200
    sell_trade = {
        'trade_id': 'SELL_001',
        'timestamp': timestamps[sell_idx], 
        'side': 'sell',
        'price': close_prices[sell_idx],
        'quantity': 10000,
        'pnl': (close_prices[sell_idx] - close_prices[buy_idx]) * 10000
    }
    trades_data.append(sell_trade)
    
    trade_df = pd.DataFrame(trades_data)
    
    print(f"Created {len(trades_data)} test trades")
    print(f"Buy trade price: {buy_trade['price']:.5f}")
    print(f"Sell trade price: {sell_trade['price']:.5f}")
    print(f"Expected PnL: {sell_trade['pnl']:.2f}")
    
    # Create equity curve
    portfolio_data = {
        'equity_curve': np.cumsum(np.random.normal(100, 50, n_bars)) + 10000
    }
    
    # Initialize dashboard
    dashboard = DashboardManager()
    
    # Force Qt application
    if not dashboard.initialize_qt_app():
        print("Failed to initialize Qt")
        return None
        
    # Create main window
    dashboard.create_main_window()
    
    # Load data and verify precision
    print("\nLoading data into dashboard...")
    await dashboard.load_backtest_data(price_data, trade_df, portfolio_data)
    
    # Verify precision was detected correctly
    print(f"\nPrecision Detection Results:")
    print(f"Dashboard price precision: {dashboard.price_precision}")
    print(f"Dashboard volume precision: {dashboard.volume_precision}")
    print(f"Chart price precision: {dashboard.main_chart.price_precision}")
    print(f"Chart left axis precision: {dashboard.main_chart.left_price_axis.precision}")
    print(f"Chart right axis precision: {dashboard.main_chart.right_price_axis.precision}")
    
    # Test formatting
    test_price = close_prices[0]
    formatted_dashboard = f"{test_price:.{dashboard.price_precision}f}"
    formatted_chart = f"{test_price:.{dashboard.main_chart.price_precision}f}"
    
    print(f"\nFormatting Tests:")
    print(f"Test price: {test_price}")
    print(f"Dashboard formatted: {formatted_dashboard}")
    print(f"Chart formatted: {formatted_chart}")
    
    # Show dashboard
    dashboard.show()
    
    print(f"\nDashboard created and shown!")
    print(f"Window visible: {dashboard.main_window.isVisible()}")
    print(f"Chart has data: {dashboard.main_chart.data_buffer is not None}")
    print(f"Trade list has precision: {getattr(dashboard.trade_list, 'price_precision', 'NOT SET')}")
    
    return dashboard

async def main():
    """Main test function"""
    dashboard = await create_test_dashboard()
    
    if dashboard and dashboard.app:
        print("\n" + "="*50)
        print("Dashboard is now running with AD forex data")
        print("Expected: 5 decimal places throughout")
        print("Check Y-axis, data window, and trade list")
        print("Press Ctrl+C to exit")
        print("="*50)
        
        # Keep running for inspection
        try:
            dashboard.app.exec_()
        except KeyboardInterrupt:
            print("\nShutting down...")
            dashboard.app.quit()

if __name__ == "__main__":
    asyncio.run(main())