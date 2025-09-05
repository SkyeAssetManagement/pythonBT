#!/usr/bin/env python
"""
Test script to verify:
1. Trade list displays up to 100,000 trades
2. Trade click jump-to functionality works
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication
from step6_complete_final import FinalTradingDashboard

def test_trade_list_fixes():
    """Test both trade list display limit and jump-to functionality"""
    
    print("=" * 70)
    print("TESTING TRADE LIST FIXES")
    print("=" * 70)
    
    # Create Qt app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create dashboard
    dashboard = FinalTradingDashboard()
    
    # Create test data with many trades
    print("\n1. Creating test data with many trades...")
    n_bars = 50000  # 50K bars to generate many trades
    
    # Create datetime array (1-minute bars)
    base_time = 1609459200_000_000_000  # 2021-01-01 in nanoseconds
    datetime_array = np.arange(n_bars, dtype=np.int64) * 60_000_000_000 + base_time
    
    # Create price data
    np.random.seed(42)
    price = 100.0
    prices = []
    for _ in range(n_bars):
        price *= (1 + np.random.randn() * 0.001)
        prices.append(price)
    prices = np.array(prices)
    
    # Create OHLCV data
    ohlcv_data = {
        'datetime': datetime_array,
        'open': prices * (1 + np.random.randn(n_bars) * 0.0001),
        'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.001),
        'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.001),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }
    
    # Create many trades (one every 20 bars)
    trades_data = []
    trade_indices = range(100, n_bars-100, 20)  # Trade every 20 bars
    
    print(f"2. Generating {len(trade_indices)} trades...")
    
    for i, entry_idx in enumerate(trade_indices):
        exit_idx = min(entry_idx + np.random.randint(5, 15), n_bars-1)
        
        # Create trade with exact timestamps from data
        trade = {
            'trade_id': f'T{i+1:05d}',
            'entry_time': datetime_array[entry_idx],  # Use exact timestamp
            'exit_time': datetime_array[exit_idx],
            'entry_price': prices[entry_idx],
            'exit_price': prices[exit_idx],
            'side': 'Long' if i % 2 == 0 else 'Short',
            'shares': 100,
            'pnl': (prices[exit_idx] - prices[entry_idx]) * 100
        }
        trades_data.append(trade)
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(trades_data)
    
    # Save as CSV
    trades_csv_path = Path('test_many_trades.csv')
    trades_df.to_csv(trades_csv_path, index=False)
    print(f"   Saved {len(trades_df)} trades to {trades_csv_path}")
    
    # Load into dashboard
    print("\n3. Loading data into dashboard...")
    success = dashboard.load_ultimate_dataset(ohlcv_data, str(trades_csv_path))
    
    if success:
        print("   Data loaded successfully!")
        
        # Check trade list
        if dashboard.trade_list:
            widget = dashboard.trade_list.trade_list_widget
            print(f"\n4. Trade list widget status:")
            print(f"   Max visible trades: {widget.max_visible_trades}")
            print(f"   Total trades loaded: {len(widget.trades_data)}")
            print(f"   All trades visible: {len(widget.trades_data) <= widget.max_visible_trades}")
            
            if len(widget.trades_data) > 10000:
                print(f"   [OK] Trade list can display more than 10,000 trades!")
            else:
                print(f"   [WARNING] Not enough trades to test limit")
        
        # Test trade navigation
        print(f"\n5. Testing trade click navigation:")
        print(f"   Click on any trade in the list")
        print(f"   The chart should jump to that trade location")
        print(f"   Watch the console for TRADE NAV debug messages")
        
        # Show dashboard
        dashboard.setWindowTitle("Trade List Fixes Test")
        dashboard.resize(1920, 1080)
        dashboard.show()
        
        print("\n" + "=" * 70)
        print("MANUAL TESTING REQUIRED:")
        print("=" * 70)
        print("1. Verify trade list shows all trades (scroll to check)")
        print("2. Click on different trades in the list")
        print("3. Verify chart jumps to the clicked trade")
        print("4. Check console for navigation debug messages")
        print("5. Close window when done testing")
        print("=" * 70)
        
        app.exec_()
        
    else:
        print("   ERROR: Failed to load data!")
    
    # Cleanup
    if trades_csv_path.exists():
        trades_csv_path.unlink()
        print(f"\nCleaned up temporary file: {trades_csv_path}")

if __name__ == "__main__":
    test_trade_list_fixes()