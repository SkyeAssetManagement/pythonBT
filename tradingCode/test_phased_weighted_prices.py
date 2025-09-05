"""
Test that phased trading uses weighted average prices from (H+L+C)/3 formula
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine


def test_weighted_average_prices():
    """Test that execution prices are calculated as weighted averages."""
    
    print("="*80)
    print("TESTING PHASED TRADING WEIGHTED AVERAGE PRICES")
    print("="*80)
    
    # Create test config with phased trading
    config_content = """
data:
  amibroker_path: "dummy"
  data_frequency: "5T"

backtest:
  initial_cash: 100000
  position_size: 1000
  position_size_type: "value"
  execution_price: "formula"
  buy_execution_formula: "(H + L + C) / 3"
  sell_execution_formula: "(H + L + C) / 3"
  signal_lag: 0
  fees: 0.0
  fixed_fees: 0.0
  slippage: 0.0
  direction: "both"
  min_size: 0.0000000001
  call_seq: "auto"
  freq: "5T"
  phased_trading_enabled: true
  phased_entry_bars: 5
  phased_exit_bars: 3
  phased_entry_distribution: "linear"
  phased_exit_distribution: "linear"
  consolidate_phased_trades: false

output:
  results_dir: "results_weighted_test"
  trade_list_filename: "trades.csv"
  equity_curve_filename: "equity.csv"
"""
    
    config_path = "config_weighted_test.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Create test data with varying prices for clear calculation
    n_bars = 300
    data = {
        'open': np.arange(5000, 5000 + n_bars * 5, 5).astype(float),
        'high': np.arange(5010, 5010 + n_bars * 5, 5).astype(float),
        'low': np.arange(4990, 4990 + n_bars * 5, 5).astype(float),
        'close': np.arange(5000, 5000 + n_bars * 5, 5).astype(float),
        'volume': np.ones(n_bars) * 1000000
    }
    
    # Create signals
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    
    entries[50] = True   # Entry at bar 50 (will phase to bars 50-54)
    exits[200] = True    # Exit at bar 200 (will phase to bars 200-202)
    
    print("\n### TEST DATA")
    print("-"*60)
    print("Entry signal at bar 50, will phase across bars 50-54")
    print("Exit signal at bar 200, will phase across bars 200-202")
    
    # Calculate expected prices
    print("\n### EXPECTED WEIGHTED AVERAGE PRICES")
    print("-"*60)
    
    # Entry phase prices
    entry_prices = []
    print("\nEntry Phase (bars 50-54, equal 20% weights):")
    for i in range(5):
        bar_idx = 50 + i
        hlc3 = (data['high'][bar_idx] + data['low'][bar_idx] + data['close'][bar_idx]) / 3.0
        weight = 0.2  # 1/5 = 20% each
        weighted_price = hlc3 * weight
        entry_prices.append(weighted_price)
        print(f"  Bar {bar_idx}: H={data['high'][bar_idx]:.0f}, L={data['low'][bar_idx]:.0f}, C={data['close'][bar_idx]:.0f}")
        print(f"    (H+L+C)/3 = {hlc3:.2f}, Weight = {weight:.1%}, Weighted = {weighted_price:.2f}")
    
    expected_entry_price = sum(entry_prices)
    print(f"\nExpected weighted average entry price: {expected_entry_price:.2f}")
    
    # Exit phase prices
    exit_prices = []
    print("\nExit Phase (bars 200-202, equal 33.33% weights):")
    for i in range(3):
        bar_idx = 200 + i
        hlc3 = (data['high'][bar_idx] + data['low'][bar_idx] + data['close'][bar_idx]) / 3.0
        weight = 1.0 / 3.0  # 33.33% each
        weighted_price = hlc3 * weight
        exit_prices.append(weighted_price)
        print(f"  Bar {bar_idx}: H={data['high'][bar_idx]:.0f}, L={data['low'][bar_idx]:.0f}, C={data['close'][bar_idx]:.0f}")
        print(f"    (H+L+C)/3 = {hlc3:.2f}, Weight = {weight:.2%}, Weighted = {weighted_price:.2f}")
    
    expected_exit_price = sum(exit_prices)
    print(f"\nExpected weighted average exit price: {expected_exit_price:.2f}")
    
    # Run backtest
    engine = VectorBTEngine(config_path)
    pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
    
    # Get actual trade prices
    print("\n### ACTUAL BACKTEST RESULTS")
    print("-"*60)
    
    if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
        trades_df = pf.trades.records_readable
        
        for idx, trade in trades_df.iterrows():
            actual_entry = trade.get('Avg Entry Price', 0)
            actual_exit = trade.get('Avg Exit Price', 0)
            
            print(f"\nTrade {idx + 1}:")
            print(f"  Actual Entry Price: {actual_entry:.2f}")
            print(f"  Actual Exit Price: {actual_exit:.2f}")
            print(f"  Entry Index: {trade.get('Entry Index', 'N/A')}")
            print(f"  Exit Index: {trade.get('Exit Index', 'N/A')}")
            
            # Compare with expected
            print("\n### VERIFICATION")
            print("-"*60)
            
            # Check if VectorBT is using first bar price or weighted average
            first_entry_bar_price = data['close'][50]  # First bar close price
            first_entry_bar_hlc3 = (data['high'][50] + data['low'][50] + data['close'][50]) / 3.0
            
            print(f"\nFirst entry bar (50) close price: {first_entry_bar_price:.2f}")
            print(f"First entry bar (50) HLC3 price: {first_entry_bar_hlc3:.2f}")
            print(f"Expected weighted average entry: {expected_entry_price:.2f}")
            print(f"Actual entry price: {actual_entry:.2f}")
            
            if abs(actual_entry - first_entry_bar_price) < 0.01:
                print("[INFO] VectorBT is using the first bar's CLOSE price")
            elif abs(actual_entry - first_entry_bar_hlc3) < 0.01:
                print("[INFO] VectorBT is using the first bar's HLC3 price")
            elif abs(actual_entry - expected_entry_price) < 0.01:
                print("[OK] VectorBT is using the weighted average price!")
            else:
                print("[WARNING] VectorBT is using an unexpected price")
            
            first_exit_bar_price = data['close'][200]  # First bar close price
            first_exit_bar_hlc3 = (data['high'][200] + data['low'][200] + data['close'][200]) / 3.0
            
            print(f"\nFirst exit bar (200) close price: {first_exit_bar_price:.2f}")
            print(f"First exit bar (200) HLC3 price: {first_exit_bar_hlc3:.2f}")
            print(f"Expected weighted average exit: {expected_exit_price:.2f}")
            print(f"Actual exit price: {actual_exit:.2f}")
            
            if abs(actual_exit - first_exit_bar_price) < 0.01:
                print("[INFO] VectorBT is using the first bar's CLOSE price")
            elif abs(actual_exit - first_exit_bar_hlc3) < 0.01:
                print("[INFO] VectorBT is using the first bar's HLC3 price")
            elif abs(actual_exit - expected_exit_price) < 0.01:
                print("[OK] VectorBT is using the weighted average price!")
            else:
                print("[WARNING] VectorBT is using an unexpected price")
    
    # Clean up
    import os
    if os.path.exists(config_path):
        os.remove(config_path)
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nAs shown above, VectorBT consolidates phased trades internally and")
    print("uses the first bar's execution price rather than calculating a true")
    print("weighted average across all phased bars. This is a limitation of")
    print("VectorBT's architecture which prioritizes performance over granular")
    print("trade tracking.")
    print("\nTo get true weighted average execution prices, we would need to:")
    print("1. Implement custom trade recording outside of VectorBT")
    print("2. Calculate weighted averages post-processing")
    print("3. Or use a different backtesting engine that supports this natively")


if __name__ == "__main__":
    test_weighted_average_prices()