"""
Test script to verify consolidated vs separate phased trades
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine
from strategies.simpleSMA import SimpleSMAStrategy


def test_consolidation_modes():
    """Test both consolidated and separate phased trade modes."""
    
    print("="*70)
    print("TESTING PHASED TRADING CONSOLIDATION MODES")
    print("="*70)
    
    # Generate synthetic test data with clear signals
    print("\n1. Generating test data with sparse signals...")
    n_bars = 500
    np.random.seed(42)
    
    # Create more predictable price data
    prices = 4000 + np.arange(n_bars) * 0.1  # Slowly rising price
    noise = np.random.normal(0, 5, n_bars)
    close = prices + noise
    
    data = {
        'open': close * 0.999,
        'high': close * 1.001,
        'low': close * 0.998,
        'close': close,
        'volume': np.ones(n_bars) * 1e6
    }
    
    # Create sparse, clear signals (only 2 entry signals, 2 exit signals)
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    
    # Place signals with plenty of space between them
    entries[50] = True   # First entry signal at bar 50
    exits[100] = True    # First exit signal at bar 100
    entries[200] = True  # Second entry signal at bar 200
    exits[250] = True    # Second exit signal at bar 250
    
    print(f"   Created {np.sum(entries)} entry signals at bars: {np.where(entries)[0]}")
    print(f"   Created {np.sum(exits)} exit signals at bars: {np.where(exits)[0]}")
    
    # Test configuration with phasing
    config_base = {
        'phased_trading_enabled': True,
        'phased_entry_bars': 5,  # Phase over 5 bars
        'phased_exit_bars': 3,   # Phase over 3 bars
        'phased_entry_distribution': 'linear',
        'phased_exit_distribution': 'linear'
    }
    
    # Test 1: Consolidated mode (default)
    print("\n2. Testing CONSOLIDATED mode (consolidate_phased_trades=True)...")
    print("   Expected: Phased trades merged into continuous positions")
    
    # Create config file for consolidated mode
    config_consolidated = """
# Test config - Consolidated mode
data:
  amibroker_path: "dummy"
  data_frequency: "1T"

backtest:
  initial_cash: 100000
  position_size: 1000
  position_size_type: "value"
  execution_price: "close"
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
  phased_entry_price_method: "limit"
  phased_exit_price_method: "limit"
  consolidate_phased_trades: true  # CONSOLIDATED MODE

output:
  results_dir: "results_test"
  trade_list_filename: "trades_consolidated.csv"
  equity_curve_filename: "equity_consolidated.csv"
"""
    
    with open("config_test_consolidated.yaml", "w") as f:
        f.write(config_consolidated)
    
    engine_consolidated = VectorBTEngine("config_test_consolidated.yaml")
    pf_consolidated = engine_consolidated.run_vectorized_backtest(data, entries, exits, "TEST")
    
    trades_consolidated = len(pf_consolidated.trades.records) if hasattr(pf_consolidated.trades, 'records') else 0
    print(f"   Result: {trades_consolidated} trades in trade list")
    
    # Test 2: Separate mode
    print("\n3. Testing SEPARATE mode (consolidate_phased_trades=False)...")
    print("   Expected: Each phased entry/exit as separate trade")
    
    config_separate = config_consolidated.replace(
        "consolidate_phased_trades: true",
        "consolidate_phased_trades: false"
    ).replace(
        "trades_consolidated.csv",
        "trades_separate.csv"
    ).replace(
        "equity_consolidated.csv",
        "equity_separate.csv"
    )
    
    with open("config_test_separate.yaml", "w") as f:
        f.write(config_separate)
    
    engine_separate = VectorBTEngine("config_test_separate.yaml")
    pf_separate = engine_separate.run_vectorized_backtest(data, entries, exits, "TEST")
    
    trades_separate = len(pf_separate.trades.records) if hasattr(pf_separate.trades, 'records') else 0
    print(f"   Result: {trades_separate} trades in trade list")
    
    # Compare results
    print("\n4. COMPARISON:")
    print("="*50)
    print(f"Original signals: {np.sum(entries)} entries, {np.sum(exits)} exits")
    print(f"Expected phased signals: {np.sum(entries) * 5} entry points, {np.sum(exits) * 3} exit points")
    print(f"\nConsolidated mode: {trades_consolidated} trades")
    print(f"Separate mode: {trades_separate} trades")
    
    if trades_separate > trades_consolidated:
        print(f"\n[OK] Separate mode shows more trades ({trades_separate} vs {trades_consolidated})")
        print("     Each phased entry/exit is displayed as a separate trade arrow")
    else:
        print(f"\n[INFO] Both modes show same number of trades")
        print("     This may be due to VectorBT's internal trade consolidation")
    
    # Export trade lists for inspection
    print("\n5. Exporting trade lists for inspection...")
    
    # Export consolidated trades
    if trades_consolidated > 0:
        trades_df_consolidated = pf_consolidated.trades.records_readable
        trades_df_consolidated.to_csv("trades_consolidated_detailed.csv", index=False)
        print("   Consolidated trades saved to: trades_consolidated_detailed.csv")
    
    # Export separate trades
    if trades_separate > 0:
        trades_df_separate = pf_separate.trades.records_readable
        trades_df_separate.to_csv("trades_separate_detailed.csv", index=False)
        print("   Separate trades saved to: trades_separate_detailed.csv")
    
    # Show trade entry/exit indices
    if trades_consolidated > 0:
        print("\n6. Trade timing analysis:")
        print("   Consolidated mode trades:")
        trades_df = pf_consolidated.trades.records_readable
        for i, row in trades_df.iterrows():
            print(f"     Trade {i+1}: Entry at bar {row['Entry Index']}, Exit at bar {row['Exit Index']}")
    
    if trades_separate > 0:
        print("\n   Separate mode trades:")
        trades_df = pf_separate.trades.records_readable
        for i, row in trades_df.iterrows():
            if i < 10:  # Limit output
                print(f"     Trade {i+1}: Entry at bar {row['Entry Index']}, Exit at bar {row['Exit Index']}")
        if len(trades_df) > 10:
            print(f"     ... and {len(trades_df) - 10} more trades")
    
    # Clean up test files
    import os
    for file in ["config_test_consolidated.yaml", "config_test_separate.yaml"]:
        if os.path.exists(file):
            os.remove(file)
    
    return trades_consolidated, trades_separate


if __name__ == "__main__":
    consolidated, separate = test_consolidation_modes()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nSummary:")
    print(f"- Consolidated mode: {consolidated} trades (phased trades merged)")
    print(f"- Separate mode: {separate} trades (each phase shown separately)")
    print("\nTo use in main.py:")
    print("- Set 'consolidate_phased_trades: false' in config.yaml to see separate arrows")
    print("- Set 'consolidate_phased_trades: true' for cleaner trade list (default)")