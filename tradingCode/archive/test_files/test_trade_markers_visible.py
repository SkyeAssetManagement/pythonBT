#!/usr/bin/env python
"""
Test script to verify trade markers are now visible on the chart
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication
from step6_complete_final import FinalTradingDashboard

def test_trade_markers():
    """Test that trade markers are rendering correctly"""
    
    print("=" * 70)
    print("TESTING TRADE MARKER VISIBILITY")
    print("=" * 70)
    
    # Create Qt app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create dashboard
    dashboard = FinalTradingDashboard()
    
    # Create test data
    print("\n1. Creating test OHLCV data...")
    n_bars = 1000
    base_time = 1609459200_000_000_000  # 2021-01-01 in nanoseconds
    datetime_array = np.arange(n_bars, dtype=np.int64) * 60_000_000_000 + base_time
    
    # Create price data with clear trends
    prices = []
    price = 100.0
    for i in range(n_bars):
        # Create some volatility
        if i % 100 < 50:
            price *= 1.001  # Uptrend
        else:
            price *= 0.999  # Downtrend
        price *= (1 + np.random.randn() * 0.002)
        prices.append(price)
    prices = np.array(prices)
    
    # Create OHLCV
    ohlcv_data = {
        'datetime': datetime_array,
        'datetime_ns': datetime_array,  # Ensure we have datetime data
        'open': prices * (1 + np.random.randn(n_bars) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.003),
        'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.003),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }
    
    # Create trades with clear entry/exit points
    print("2. Creating test trades...")
    trades_data = []
    
    # Create trades at specific indices for easy verification
    trade_indices = [
        (50, 60),   # Trade 1: bars 50-60
        (100, 110), # Trade 2: bars 100-110
        (150, 165), # Trade 3: bars 150-165
        (200, 215), # Trade 4: bars 200-215
        (250, 270), # Trade 5: bars 250-270
        (300, 320), # Trade 6: bars 300-320
        (350, 375), # Trade 7: bars 350-375
        (400, 425), # Trade 8: bars 400-425
        (450, 480), # Trade 9: bars 450-480
    ]
    
    for i, (entry_idx, exit_idx) in enumerate(trade_indices):
        # Use exact timestamps from data
        trade = {
            'trade_id': f'TEST_{i+1:03d}',
            'entry_time': datetime_array[entry_idx],  # Use exact timestamp
            'exit_time': datetime_array[exit_idx],
            'entry_price': prices[entry_idx],
            'exit_price': prices[exit_idx],
            'direction': 'Long' if i % 2 == 0 else 'Short',
            'shares': 100,
            'pnl': (prices[exit_idx] - prices[entry_idx]) * 100 * (1 if i % 2 == 0 else -1)
        }
        trades_data.append(trade)
        print(f"   Trade {i+1}: bars {entry_idx}-{exit_idx} ({trade['direction']})")
    
    # Save trades as CSV
    trades_df = pd.DataFrame(trades_data)
    trades_csv_path = Path('test_trade_markers.csv')
    trades_df.to_csv(trades_csv_path, index=False)
    
    # Load into dashboard
    print("\n3. Loading data into dashboard...")
    success = dashboard.load_ultimate_dataset(ohlcv_data, str(trades_csv_path))
    
    if success:
        print("   Data loaded successfully!")
        
        # Set initial viewport to show first trades
        if dashboard.final_chart:
            dashboard.final_chart.viewport_start = 0
            dashboard.final_chart.viewport_end = 500
            dashboard.final_chart._generate_candlestick_geometry()
            dashboard.final_chart._generate_trade_markers()
            dashboard.final_chart._update_projection()
            dashboard.final_chart.canvas.update()
            print(f"   Viewport set to show bars 0-500")
        
        # Show dashboard
        dashboard.setWindowTitle("Trade Marker Visibility Test")
        dashboard.resize(1920, 1080)
        dashboard.show()
        
        print("\n" + "=" * 70)
        print("EXPECTED RESULTS:")
        print("=" * 70)
        print("You should see:")
        print("1. Candlestick chart with price data")
        print("2. Trade markers as colored triangles:")
        print("   - Long entries: Green triangles below candles")
        print("   - Long exits: Red triangles above candles")
        print("   - Short entries: Red triangles above candles")
        print("   - Short exits: Green triangles below candles")
        print("\nCheck console output for trade marker generation messages")
        print("\nIf you see 'ERROR drawing trade markers', the shader may have issues")
        print("If you see 'No trade vertices generated', timestamp conversion failed")
        print("=" * 70)
        
        app.exec_()
        
    else:
        print("   ERROR: Failed to load data!")
    
    # Cleanup
    if trades_csv_path.exists():
        trades_csv_path.unlink()
        print(f"\nCleaned up: {trades_csv_path}")

if __name__ == "__main__":
    test_trade_markers()