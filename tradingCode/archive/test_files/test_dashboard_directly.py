#!/usr/bin/env python3
"""Test the dashboard directly with main.py's logic"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

print("TESTING DASHBOARD WITH MAIN.PY LOGIC")
print("=" * 60)

# Set up environment
os.chdir("C:/Users/skyeAM/SkyeAM Dropbox/SAMresearch/ABtoPython/tradingCode")
print(f"Working directory: {os.getcwd()}")

# Check if results exist
results_dir = "results"
trades_csv = os.path.join(results_dir, "tradelist.csv")
equity_csv = os.path.join(results_dir, "equity_curve.csv")

if not os.path.exists(trades_csv):
    print("[ERROR] No trades found. Run main.py first!")
    sys.exit(1)

print(f"[OK] Found trades: {trades_csv}")

# Load trades to check indices
trades_df = pd.read_csv(trades_csv)
print(f"[OK] Loaded {len(trades_df)} trades")

# Check trade indices
if 'Entry Index' in trades_df.columns:
    min_idx = trades_df['Entry Index'].min()
    max_idx = trades_df['Exit Index'].max()
    print(f"[INFO] Trade indices range: {min_idx} to {max_idx}")

# Now test if dashboard would work
try:
    from PyQt5.QtWidgets import QApplication
    from step6_complete_final import FinalTradingDashboard
    
    # Create app
    app = QApplication([])
    
    # Create dashboard
    print("\nCreating dashboard...")
    dashboard = FinalTradingDashboard()
    
    # Create test OHLCV data that matches trade indices
    print("Creating OHLCV data that matches trade range...")
    
    # Create enough data to cover trades
    data_length = int(max_idx) + 1000 if 'Entry Index' in trades_df.columns else 20000
    print(f"[INFO] Creating {data_length} bars of OHLCV data")
    
    # Generate realistic OHLCV
    np.random.seed(42)
    base_price = 1580.0
    prices = base_price + np.cumsum(np.random.randn(data_length) * 0.5)
    
    ohlcv_data = {
        'open': prices + np.random.randn(data_length) * 0.2,
        'high': prices + np.abs(np.random.randn(data_length)) * 0.5,
        'low': prices - np.abs(np.random.randn(data_length)) * 0.5,
        'close': prices + np.random.randn(data_length) * 0.2,
        'volume': np.random.randint(1000, 10000, data_length)
    }
    
    # Fix high/low
    ohlcv_data['high'] = np.maximum(ohlcv_data['high'], np.maximum(ohlcv_data['open'], ohlcv_data['close']))
    ohlcv_data['low'] = np.minimum(ohlcv_data['low'], np.minimum(ohlcv_data['open'], ohlcv_data['close']))
    
    print(f"[OK] OHLCV data created: {len(ohlcv_data['close'])} bars")
    
    # Load into dashboard
    print("\nLoading data into dashboard...")
    success = dashboard.load_ultimate_dataset(ohlcv_data, trades_csv)
    
    if success:
        print("[SUCCESS] Data loaded into dashboard!")
        
        # Check what viewport was set
        if hasattr(dashboard.final_chart, 'viewport_start'):
            print(f"[INFO] Viewport: {dashboard.final_chart.viewport_start} to {dashboard.final_chart.viewport_end}")
        
        # Export screenshot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"C:/Users/skyeAM/SkyeAM Dropbox/SAMresearch/ABtoPython/2025-08-08_2/dashboard_test_{timestamp}.png"
        
        from PyQt5.QtCore import QTimer
        
        def capture():
            try:
                dashboard.export_dashboard_image(filename)
                print(f"\n[SUCCESS] Screenshot saved: dashboard_test_{timestamp}.png")
                print("\nCHECK THE SCREENSHOT FOR:")
                print("1. Candlesticks visible (not blank)")
                print("2. Trade arrows on chart")
                print("3. Trade list populated")
                print("4. No 'Invalid indices' error")
            except Exception as e:
                print(f"[ERROR] Screenshot failed: {e}")
            finally:
                app.quit()
        
        QTimer.singleShot(3000, capture)
        
        dashboard.show()
        app.exec_()
        
    else:
        print("[ERROR] Failed to load data into dashboard")
        
except Exception as e:
    print(f"\n[ERROR] Dashboard test failed: {e}")
    import traceback
    traceback.print_exc()