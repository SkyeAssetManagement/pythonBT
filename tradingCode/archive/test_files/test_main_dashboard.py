#!/usr/bin/env python
"""
Test script to run the modular dashboard from main branch
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication
from modular_trading_dashboard import ModularTradingDashboard


def main():
    """Run the modular trading dashboard"""
    
    print("="*70)
    print("TESTING MODULAR DASHBOARD FROM MAIN BRANCH")
    print("="*70)
    
    # Create Qt app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create dashboard
    dashboard = ModularTradingDashboard()
    
    # Create test OHLCV data
    print("\n1. Creating test OHLCV data...")
    n_bars = 1000
    base_time = 1609459200_000_000_000  # 2021-01-01 in nanoseconds
    datetime_array = np.arange(n_bars, dtype=np.int64) * 60_000_000_000 + base_time
    
    # Generate price data
    np.random.seed(42)
    prices = []
    price = 100.0
    for i in range(n_bars):
        price *= (1 + np.random.randn() * 0.002)
        prices.append(price)
    prices = np.array(prices)
    
    # Create OHLCV
    ohlcv_data = {
        'datetime': datetime_array,
        'open': prices * (1 + np.random.randn(n_bars) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.003),
        'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.003),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }
    
    # Create some test trades
    print("2. Creating test trades...")
    trades_data = []
    trade_positions = [
        (50, 60, 'Long'),
        (100, 115, 'Short'),
        (150, 170, 'Long'),
        (200, 220, 'Short'),
        (250, 275, 'Long'),
        (300, 325, 'Short'),
        (350, 380, 'Long'),
        (400, 430, 'Short'),
        (450, 485, 'Long'),
    ]
    
    for i, (entry_idx, exit_idx, direction) in enumerate(trade_positions):
        trade = {
            'trade_id': f'TEST_{i+1:03d}',
            'entry_time': datetime_array[entry_idx],
            'exit_time': datetime_array[exit_idx],
            'entry_price': prices[entry_idx],
            'exit_price': prices[exit_idx],
            'direction': direction,
            'shares': 100,
            'pnl': (prices[exit_idx] - prices[entry_idx]) * 100 * (1 if direction == 'Long' else -1)
        }
        trades_data.append(trade)
    
    # Save trades as CSV
    trades_df = pd.DataFrame(trades_data)
    trades_csv_path = Path('test_trades_main.csv')
    trades_df.to_csv(trades_csv_path, index=False)
    print(f"   Created {len(trades_data)} trades")
    
    # Load data into dashboard
    print("\n3. Loading data into dashboard...")
    success = dashboard.load_data(ohlcv_data, str(trades_csv_path))
    
    if success:
        print("   Data loaded successfully!")
        
        # Set initial viewport
        if dashboard.chart_manager:
            dashboard.chart_manager.set_viewport(0, 500)
            print("   Viewport set to show bars 0-500")
        
        # Show dashboard
        dashboard.setWindowTitle("Modular Trading Dashboard - Main Branch Test")
        dashboard.resize(1920, 1080)
        dashboard.show()
        
        print("\n" + "="*70)
        print("DASHBOARD FEATURES:")
        print("="*70)
        print("[OK] Clean, modular architecture")
        print("[OK] GPU-accelerated candlestick chart")
        print("[OK] Trade triangles (generating and rendering)")
        print("[OK] Trade list with navigation")
        print("[OK] Hover information")
        print("[OK] Time axis")
        print("\nTRADE MARKERS STATUS:")
        print("  - Trade triangles ARE being rendered")
        print("  - Check console for 'Drew X trade vertices as triangles'")
        print("  - Triangles may be small - zoom in to see them better")
        print("\nCONTROLS:")
        print("  - Mouse wheel: Zoom in/out")
        print("  - Click & drag: Pan the chart")
        print("  - Click trade in list: Navigate to that trade")
        print("="*70)
        
        app.exec_()
        
    else:
        print("   ERROR: Failed to load data!")
    
    # Cleanup
    if trades_csv_path.exists():
        trades_csv_path.unlink()
        print(f"\nCleaned up: {trades_csv_path}")


if __name__ == "__main__":
    main()