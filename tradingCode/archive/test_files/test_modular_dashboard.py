#!/usr/bin/env python
"""
Test script for the modular trading dashboard with visible trade triangles.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication
from modular_trading_dashboard import ModularTradingDashboard


def create_test_trades(n_bars, datetime_array, prices):
    """Create test trades with clear entry/exit points."""
    trades_data = []
    
    # Create trades at specific intervals
    trade_indices = [
        (50, 60, 'Long'),
        (100, 115, 'Short'),
        (150, 170, 'Long'),
        (200, 220, 'Short'),
        (250, 275, 'Long'),
        (300, 325, 'Short'),
        (350, 380, 'Long'),
        (400, 430, 'Short'),
        (450, 485, 'Long'),
        (500, 535, 'Short'),
        (550, 590, 'Long'),
        (600, 640, 'Short'),
        (650, 695, 'Long'),
        (700, 745, 'Short'),
        (750, 795, 'Long'),
    ]
    
    for i, (entry_idx, exit_idx, direction) in enumerate(trade_indices):
        if exit_idx >= n_bars:
            continue
            
        entry_price = prices[entry_idx]
        exit_price = prices[exit_idx]
        
        # Calculate PnL
        if direction == 'Long':
            pnl = (exit_price - entry_price) * 100
        else:
            pnl = (entry_price - exit_price) * 100
        
        trade = {
            'trade_id': f'T{i+1:03d}',
            'entry_time': datetime_array[entry_idx],
            'exit_time': datetime_array[exit_idx],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'shares': 100,
            'pnl': pnl
        }
        trades_data.append(trade)
    
    return trades_data


def test_modular_dashboard():
    """Test the modular dashboard with trade triangles."""
    
    print("="*70)
    print("TESTING MODULAR DASHBOARD WITH TRADE TRIANGLES")
    print("="*70)
    
    # Create Qt app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create dashboard
    dashboard = ModularTradingDashboard()
    
    # Create test data
    print("\n1. Creating test OHLCV data...")
    n_bars = 2000
    base_time = 1609459200_000_000_000  # 2021-01-01 in nanoseconds
    datetime_array = np.arange(n_bars, dtype=np.int64) * 60_000_000_000 + base_time
    
    # Create realistic price movement
    np.random.seed(42)
    prices = []
    price = 100.0
    trend = 0.0001  # Slight upward trend
    
    for i in range(n_bars):
        # Add trend and random walk
        price *= (1 + trend + np.random.randn() * 0.002)
        
        # Add some volatility clusters
        if i % 200 < 50:
            price *= (1 + np.random.randn() * 0.005)  # Higher volatility
        
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
    
    # Create trades
    print("2. Creating test trades...")
    trades_data = create_test_trades(n_bars, datetime_array, prices)
    print(f"   Generated {len(trades_data)} trades")
    
    # Save trades to CSV
    trades_df = pd.DataFrame(trades_data)
    trades_csv_path = Path('test_modular_trades.csv')
    trades_df.to_csv(trades_csv_path, index=False)
    
    # Print trade summary
    print("\n3. Trade Summary:")
    for i, trade in enumerate(trades_data[:5]):  # Show first 5 trades
        entry_idx = np.where(datetime_array == trade['entry_time'])[0][0]
        exit_idx = np.where(datetime_array == trade['exit_time'])[0][0]
        print(f"   {trade['trade_id']}: {trade['direction']:5} bars {entry_idx:3d}-{exit_idx:3d}, PnL: ${trade['pnl']:.2f}")
    if len(trades_data) > 5:
        print(f"   ... and {len(trades_data) - 5} more trades")
    
    # Load data into dashboard
    print("\n4. Loading data into modular dashboard...")
    success = dashboard.load_data(ohlcv_data, str(trades_csv_path))
    
    if success:
        print("   Data loaded successfully!")
        
        # Set initial viewport to show first trades
        if dashboard.chart_manager:
            dashboard.chart_manager.set_viewport(0, 500)
            print("   Viewport set to show bars 0-500")
        
        # Show dashboard
        dashboard.setWindowTitle("Modular Dashboard - Trade Triangle Test")
        dashboard.show()
        
        print("\n" + "="*70)
        print("EXPECTED RESULTS:")
        print("="*70)
        print("[OK] Clean, modular architecture")
        print("[OK] Candlestick chart with smooth rendering")
        print("[OK] VISIBLE trade triangles:")
        print("  - Long entries: Green triangles pointing UP below candles")
        print("  - Long exits: Red triangles pointing DOWN above candles")
        print("  - Short entries: Red triangles pointing DOWN above candles")
        print("  - Short exits: Green triangles pointing UP below candles")
        print("\nINTERACTIONS:")
        print("  - Mouse wheel: Zoom in/out")
        print("  - Click & drag: Pan the chart")
        print("  - Click trade in list: Navigate to that trade")
        print("  - Arrow keys: Navigate chart")
        print("  - Home/End: Jump to start/end")
        print("="*70)
        
        app.exec_()
        
    else:
        print("   ERROR: Failed to load data!")
    
    # Cleanup
    if trades_csv_path.exists():
        trades_csv_path.unlink()
        print(f"\nCleaned up: {trades_csv_path}")


if __name__ == "__main__":
    test_modular_dashboard()