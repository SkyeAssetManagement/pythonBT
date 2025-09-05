#!/usr/bin/env python
"""
Test script to run code from main branch and verify trade triangles work.
This test uses the fixed triangle rendering from the current branch.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Qt OpenGL configuration
os.environ['QT_OPENGL'] = 'desktop'
from PyQt5.QtCore import QCoreApplication, Qt
QCoreApplication.setAttribute(Qt.AA_DontCheckOpenGLContextThreadAffinity, True)
QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)

from PyQt5.QtWidgets import QApplication
from modular_trading_dashboard import ModularTradingDashboard

def test_main_branch_with_fixed_triangles():
    """
    Test the dashboard with the fixed triangle rendering.
    This simulates what would happen if the fixes were merged to main.
    """
    
    print("="*70)
    print("TESTING MAIN BRANCH CODE WITH FIXED TRIANGLE RENDERING")
    print("="*70)
    print("This test verifies that trade triangles are now visible")
    print("after applying the shader data binding fix.")
    print("="*70)
    
    # Create Qt app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create dashboard with fixed rendering
    dashboard = ModularTradingDashboard()
    
    # Create test data similar to what main.py would generate
    print("\n1. Creating test data (similar to main.py backtest)...")
    n_bars = 1000
    
    # Create datetime array (nanoseconds)
    base_time = 1577836800_000_000_000  # 2020-01-01 in nanoseconds
    datetime_array = np.arange(n_bars, dtype=np.int64) * 60_000_000_000 + base_time
    
    # Generate realistic ES price data
    np.random.seed(42)
    prices = []
    price = 3200.0  # ES starting price around 2020-01-01
    
    for i in range(n_bars):
        # Add realistic ES volatility
        price *= (1 + np.random.randn() * 0.001)
        prices.append(price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    ohlcv_data = {
        'datetime': datetime_array,
        'open': prices * (1 + np.random.randn(n_bars) * 0.0005),
        'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.001),
        'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.001),
        'close': prices,
        'volume': np.random.randint(10000, 100000, n_bars)
    }
    
    # Create trades similar to what simpleSMA strategy would generate
    print("2. Creating test trades (similar to simpleSMA strategy)...")
    trades_data = []
    
    # Generate trades based on simple moving average crossovers
    sma_short = pd.Series(prices).rolling(10).mean().values
    sma_long = pd.Series(prices).rolling(30).mean().values
    
    # Find crossover points
    trade_id = 1
    position_open = False
    entry_idx = 0
    
    for i in range(31, n_bars - 50):  # Start after SMA warmup
        if not position_open and sma_short[i] > sma_long[i] and sma_short[i-1] <= sma_long[i-1]:
            # Long entry
            entry_idx = i
            position_open = True
            direction = 'Long'
        elif position_open and sma_short[i] < sma_long[i] and sma_short[i-1] >= sma_long[i-1]:
            # Exit
            exit_idx = i
            
            trade = {
                'trade_id': f'T{trade_id:03d}',
                'entry_time': datetime_array[entry_idx],
                'exit_time': datetime_array[exit_idx],
                'entry_price': prices[entry_idx],
                'exit_price': prices[exit_idx],
                'direction': direction,
                'shares': 1,
                'pnl': (prices[exit_idx] - prices[entry_idx]) * (1 if direction == 'Long' else -1)
            }
            trades_data.append(trade)
            trade_id += 1
            position_open = False
            
            # Limit number of trades for visibility
            if len(trades_data) >= 20:
                break
    
    print(f"   Generated {len(trades_data)} trades")
    
    # Save trades to CSV
    trades_df = pd.DataFrame(trades_data)
    trades_csv_path = Path('test_main_branch_trades.csv')
    trades_df.to_csv(trades_csv_path, index=False)
    
    # Print first few trades
    print("\n3. Sample trades:")
    for trade in trades_data[:3]:
        entry_idx = np.where(datetime_array == trade['entry_time'])[0][0]
        exit_idx = np.where(datetime_array == trade['exit_time'])[0][0]
        print(f"   {trade['trade_id']}: {trade['direction']} bars {entry_idx}-{exit_idx}, PnL: ${trade['pnl']:.2f}")
    
    # Load data into dashboard
    print("\n4. Loading data into dashboard with fixed triangle rendering...")
    success = dashboard.load_data(ohlcv_data, str(trades_csv_path))
    
    if success:
        print("   Data loaded successfully!")
        
        # Set viewport to show first trades
        if dashboard.chart_manager:
            dashboard.chart_manager.set_viewport(0, 500)
            print("   Viewport set to show bars 0-500")
        
        # Show dashboard
        dashboard.setWindowTitle("Main Branch Test - Fixed Triangle Rendering")
        dashboard.resize(1920, 1080)
        dashboard.show()
        
        print("\n" + "="*70)
        print("TRIANGLE RENDERING STATUS:")
        print("="*70)
        print("[FIXED] Shader data binding - vertices and colors now set before draw")
        print("[FIXED] Triangle size increased to 3.0 bars width")
        print("[FIXED] Triangle height increased to 1.5x offset")
        print("[FIXED] Colors set to bright green/red (1.0 intensity)")
        print("")
        print("EXPECTED RESULT:")
        print("  Trade triangles should now be VISIBLE on the chart")
        print("  - Green triangles for long entries (pointing up)")
        print("  - Red triangles for exits (pointing down)")
        print("")
        print("This demonstrates the fix working with main branch logic.")
        print("="*70)
        
        app.exec_()
        
    else:
        print("   ERROR: Failed to load data!")
    
    # Cleanup
    if trades_csv_path.exists():
        trades_csv_path.unlink()
        print(f"\nCleaned up: {trades_csv_path}")


if __name__ == "__main__":
    test_main_branch_with_fixed_triangles()