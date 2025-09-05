#!/usr/bin/env python3
"""
Test multiple trade clicks with controlled data to isolate the issue
"""

import sys
import time
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Import dashboard
from step6_complete_final import FinalTradingDashboard

def create_simple_test_data():
    """Create simple test data to isolate trade clicking issues"""
    print("Creating simple test data...")
    
    num_bars = 2000
    
    # Create simple price data
    np.random.seed(42)
    base_price = 1.2500
    prices = np.zeros(num_bars)
    prices[0] = base_price
    
    for i in range(1, num_bars):
        change = np.random.normal(0, 0.001)
        prices[i] = prices[i-1] * (1 + change)
    
    # Create OHLC data
    opens = np.roll(prices, 1)
    closes = prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.003, num_bars)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.003, num_bars)))
    volumes = np.random.uniform(1000, 5000, num_bars)
    
    # Create datetime data
    import pandas as pd
    start_time = pd.Timestamp('2024-10-01 09:00:00')
    datetime_index = pd.date_range(start=start_time, periods=num_bars, freq='1min')
    datetime_ns = datetime_index.astype(np.int64)
    
    ohlcv_data = {
        'datetime': np.arange(num_bars, dtype=np.int64),  # Bar indices
        'datetime_ns': datetime_ns,  # Timestamps
        'open': opens,
        'high': highs,
        'low': lows, 
        'close': closes,
        'volume': volumes
    }
    
    # Create 3 trades at well-separated locations for easy testing
    trades_data = [
        {
            'EntryTime': 200,  # Bar index 200
            'ExitTime': 250,
            'Direction': 'Long',
            'Avg Entry Price': prices[200],
            'Avg Exit Price': prices[250],
            'Size': 1.0,
            'PnL': (prices[250] - prices[200]) * 1.0
        },
        {
            'EntryTime': 800,  # Bar index 800 - far from first trade
            'ExitTime': 850,
            'Direction': 'Short',
            'Avg Entry Price': prices[800],
            'Avg Exit Price': prices[850],
            'Size': 1.0,
            'PnL': (prices[800] - prices[850]) * 1.0
        },
        {
            'EntryTime': 1400,  # Bar index 1400 - far from second trade
            'ExitTime': 1450,
            'Direction': 'Long',
            'Avg Entry Price': prices[1400],
            'Avg Exit Price': prices[1450], 
            'Size': 1.0,
            'PnL': (prices[1450] - prices[1400]) * 1.0
        }
    ]
    
    # Save trades CSV
    import pandas as pd
    trades_df = pd.DataFrame(trades_data)
    trades_csv = 'test_multiple_clicks.csv'
    trades_df.to_csv(trades_csv, index=False)
    
    print(f"Created {num_bars} bars and {len(trades_data)} trades")
    print("Trades are at bar indices: 200, 800, 1400 (well separated)")
    print(f"Trades CSV: {trades_csv}")
    
    return ohlcv_data, trades_csv

def test_multiple_trade_clicks():
    """Test clicking multiple trades to isolate the issue"""
    print("="*80)
    print("TESTING MULTIPLE TRADE CLICKS")
    print("="*80)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    # Create test data
    ohlcv_data, trades_csv = create_simple_test_data()
    
    # Create dashboard
    dashboard = FinalTradingDashboard()
    dashboard.setWindowTitle("MULTIPLE TRADE CLICKS TEST")
    dashboard.resize(1400, 1000)
    dashboard.show()
    
    # Load data
    success = dashboard.load_ultimate_dataset(ohlcv_data, trades_csv)
    print(f"Dataset loaded: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        trades = dashboard.trades_data
        print(f"Loaded {len(trades)} trades:")
        for i, trade in enumerate(trades):
            print(f"  Trade {i+1}: {trade.trade_id} at bar {trade.entry_time}")
        print()
        
        # Function to simulate trade clicks
        def simulate_trade_clicks():
            print("STARTING AUTOMATED TRADE CLICK TEST")
            print("="*50)
            
            def click_trade_1():
                print(f"\n>>> CLICKING TRADE 1 (bar {trades[0].entry_time}) <<<")
                dashboard.trade_list.trade_list_widget._on_trade_clicked(0, 0)
                print("Waiting 2 seconds...")
                QTimer.singleShot(2000, click_trade_2)
                
            def click_trade_2(): 
                print(f"\n>>> CLICKING TRADE 2 (bar {trades[1].entry_time}) <<<")
                dashboard.trade_list.trade_list_widget._on_trade_clicked(1, 0)
                print("Waiting 2 seconds...")
                QTimer.singleShot(2000, click_trade_3)
                
            def click_trade_3():
                print(f"\n>>> CLICKING TRADE 3 (bar {trades[2].entry_time}) <<<")
                dashboard.trade_list.trade_list_widget._on_trade_clicked(2, 0)
                print("Waiting 2 seconds...")
                QTimer.singleShot(2000, test_complete)
                
            def test_complete():
                print("\n" + "="*50)
                print("AUTOMATED TEST COMPLETE!")
                print("="*50)
                print("Review the console output above:")
                print("1. Each trade click should show 'TRADE NAV: Setting viewport X-Y'")
                print("2. Viewport numbers should be DIFFERENT for each trade")
                print("3. Chart should regenerate with different vertex counts")
                print("4. If viewport changes but chart doesn't move visually, that's the bug!")
                print()
                print("Current viewport:", end=" ")
                if hasattr(dashboard, 'final_chart'):
                    print(f"{dashboard.final_chart.viewport_start}-{dashboard.final_chart.viewport_end}")
                else:
                    print("Unknown")
                print("Close window when done reviewing...")
            
            # Start the sequence
            click_trade_1()
        
        # Start automated test after 3 seconds
        print("Starting automated trade clicking test in 3 seconds...")
        QTimer.singleShot(3000, simulate_trade_clicks)
        
        # Run application
        app.exec_()
    
    print("Multiple trade clicks test completed!")

if __name__ == "__main__":
    test_multiple_trade_clicks()