#!/usr/bin/env python3
"""
Detailed test of trade navigation with screenshot validation
"""

import sys
import time
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Import the main dashboard
from step6_complete_final import FinalTradingDashboard

def create_test_with_real_data():
    """Create test data that mirrors main.py behavior"""
    print("Creating test data similar to main.py...")
    
    # Create realistic OHLCV data with timestamps
    num_bars = 5000
    base_price = 1.2500
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, num_bars)
    prices = np.zeros(num_bars)
    prices[0] = base_price
    
    for i in range(1, num_bars):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    # Generate OHLC from prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, num_bars)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, num_bars)))
    opens = np.roll(prices, 1)
    closes = prices
    volumes = np.random.uniform(1000, 10000, num_bars)
    
    # Create datetime data (timestamps in nanoseconds like main.py would have)
    import pandas as pd
    start_date = pd.Timestamp('2024-10-01 09:00:00')
    datetime_index = pd.date_range(start=start_date, periods=num_bars, freq='1min')
    datetime_ns = datetime_index.astype(np.int64)
    
    ohlcv_data = {
        'datetime': np.arange(num_bars, dtype=np.int64),  # Bar indices for rendering
        'datetime_ns': datetime_ns,  # Actual timestamps for display
        'open': opens,
        'high': highs, 
        'low': lows,
        'close': closes,
        'volume': volumes
    }
    
    # Create realistic trades with actual timestamps
    trades_data = []
    num_trades = 20
    
    for i in range(num_trades):
        entry_idx = np.random.randint(100, num_bars - 200)
        exit_idx = entry_idx + np.random.randint(5, 50)
        
        entry_price = closes[entry_idx]
        exit_price = closes[exit_idx]
        
        direction = 'Long' if np.random.rand() > 0.5 else 'Short'
        size = np.random.uniform(0.5, 2.0)
        
        if direction == 'Long':
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size
        
        trades_data.append({
            'EntryTime': entry_idx,  # Use bar index like main.py
            'ExitTime': exit_idx,
            'Direction': direction,
            'Avg Entry Price': entry_price,
            'Avg Exit Price': exit_price,
            'Size': size,
            'PnL': pnl
        })
    
    # Save trades to CSV
    import pandas as pd
    trades_df = pd.DataFrame(trades_data)
    trades_csv = 'test_trades_detailed.csv'
    trades_df.to_csv(trades_csv, index=False)
    
    print(f"Created {num_bars} bars and {num_trades} trades")
    print(f"Trades saved to: {trades_csv}")
    
    return ohlcv_data, trades_csv

def test_trade_navigation_with_screenshots():
    """Test trade navigation and take screenshots"""
    print("="*60)
    print("TESTING TRADE NAVIGATION WITH SCREENSHOTS")
    print("="*60)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    # Create test data
    ohlcv_data, trades_csv = create_test_with_real_data()
    
    # Create dashboard
    dashboard = FinalTradingDashboard()
    dashboard.setWindowTitle("TRADE NAVIGATION VALIDATION TEST")
    dashboard.resize(1920, 1200)
    dashboard.show()
    
    # Load data
    success = dashboard.load_ultimate_dataset(ohlcv_data, trades_csv)
    print(f"Dataset loaded: {'SUCCESS' if success else 'FAILED'}")
    
    if success and dashboard.trades_data and len(dashboard.trades_data) >= 3:
        trades = dashboard.trades_data
        print(f"Loaded {len(trades)} trades for testing")
        
        def take_screenshot(name):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"trade_nav_{name}_{timestamp}.png"
            pixmap = dashboard.grab()
            success = pixmap.save(filename)
            print(f"Screenshot {'saved' if success else 'FAILED'}: {filename}")
            
            # Print current viewport for debugging
            if hasattr(dashboard, 'final_chart'):
                print(f"Current viewport: {dashboard.final_chart.viewport_start}-{dashboard.final_chart.viewport_end}")
            
            return filename
        
        def test_sequence():
            print("\n=== STARTING TRADE NAVIGATION TEST SEQUENCE ===")
            
            # Take initial screenshot
            print("\n1. Taking INITIAL screenshot...")
            take_screenshot("initial")
            
            # Click first trade
            print(f"\n2. Clicking FIRST trade: {trades[0].trade_id} (entry_time: {trades[0].entry_time})")
            dashboard.trade_list.trade_list_widget._on_trade_clicked(0, 0)
            
            # Wait and screenshot
            QTimer.singleShot(1000, lambda: (
                print("   Taking screenshot after FIRST trade click..."),
                take_screenshot("first_trade")
            ))
            
            # Click second trade
            def click_second_trade():
                print(f"\n3. Clicking SECOND trade: {trades[1].trade_id} (entry_time: {trades[1].entry_time})")
                dashboard.trade_list.trade_list_widget._on_trade_clicked(1, 0)
                
                # Wait and screenshot
                QTimer.singleShot(1000, lambda: (
                    print("   Taking screenshot after SECOND trade click..."),
                    take_screenshot("second_trade")
                ))
                
                # Click third trade
                def click_third_trade():
                    print(f"\n4. Clicking THIRD trade: {trades[2].trade_id} (entry_time: {trades[2].entry_time})")
                    dashboard.trade_list.trade_list_widget._on_trade_clicked(2, 0)
                    
                    # Wait and screenshot
                    QTimer.singleShot(1000, lambda: (
                        print("   Taking screenshot after THIRD trade click..."),
                        take_screenshot("third_trade"),
                        print("\n=== TEST SEQUENCE COMPLETE ==="),
                        print("Check the screenshots to see if chart actually moved between clicks!")
                    ))
                
                QTimer.singleShot(2000, click_third_trade)
            
            QTimer.singleShot(2000, click_second_trade)
        
        # Start test sequence after 3 seconds
        QTimer.singleShot(3000, test_sequence)
        
        # Run app
        app.exec_()
    
    print("Trade navigation test completed!")

if __name__ == "__main__":
    test_trade_navigation_with_screenshots()