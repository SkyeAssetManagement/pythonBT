"""
Test both phased trading modes:
1. consolidate_phased_trades=true: Use VBT's natural consolidation
2. consolidate_phased_trades=false: Force separate trades at 1/n size
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine


def test_consolidated_mode():
    """Test with consolidate_phased_trades=true (VBT's natural consolidation)"""
    
    print("="*70)
    print("TEST 1: CONSOLIDATED MODE (VBT Natural Consolidation)")
    print("="*70)
    
    # Create simple test data
    n_bars = 300
    np.random.seed(42)
    close = 100 + np.arange(n_bars) * 0.01
    
    data = {
        'open': close,
        'high': close * 1.001,
        'low': close * 0.999,
        'close': close,
        'volume': np.ones(n_bars) * 1e6
    }
    
    # Create signals
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    
    entries[20] = True   # Entry signal 1
    exits[100] = True    # Exit signal 1
    entries[150] = True  # Entry signal 2
    exits[250] = True    # Exit signal 2
    
    # Create config with consolidate=true
    config_consolidated = """
data:
  amibroker_path: "dummy"
  data_frequency: "5T"

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
  consolidate_phased_trades: true  # Use VBT consolidation

output:
  results_dir: "results_consolidated"
  trade_list_filename: "trades_consolidated.csv"
  equity_curve_filename: "equity_consolidated.csv"
"""
    
    with open("config_consolidated.yaml", "w") as f:
        f.write(config_consolidated)
    
    print("\nRunning backtest with consolidate_phased_trades=true...")
    print("Expected: VBT will consolidate phased entries/exits naturally")
    
    engine = VectorBTEngine("config_consolidated.yaml")
    pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
    
    # Get trade details
    if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
        trades_df = pf.trades.records_readable
        num_trades = len(trades_df)
        
        print(f"\nResults:")
        print(f"  Number of trades: {num_trades}")
        print(f"  Expected: ~2 trades (VBT consolidates phases)")
        
        if num_trades > 0:
            print(f"\nTrade details:")
            print("-"*60)
            for i, row in trades_df.iterrows():
                print(f"Trade {i+1}:")
                print(f"  Entry: bar {row['Entry Index']}")
                print(f"  Exit: bar {row['Exit Index']}")
                print(f"  Size: {row['Size']:.2f}")
                print()
        
        # Export for inspection
        trades_df.to_csv("trades_consolidated.csv", index=False)
        print(f"Trade list exported to: trades_consolidated.csv")
    
    # Clean up
    import os
    if os.path.exists("config_consolidated.yaml"):
        os.remove("config_consolidated.yaml")
    
    return num_trades if 'num_trades' in locals() else 0


def test_separate_mode():
    """Test with consolidate_phased_trades=false (force separate trades)"""
    
    print("\n" + "="*70)
    print("TEST 2: SEPARATE MODE (Force Individual Trades)")
    print("="*70)
    
    # Same test data
    n_bars = 300
    np.random.seed(42)
    close = 100 + np.arange(n_bars) * 0.01
    
    data = {
        'open': close,
        'high': close * 1.001,
        'low': close * 0.999,
        'close': close,
        'volume': np.ones(n_bars) * 1e6
    }
    
    # Same signals
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    
    entries[20] = True
    exits[100] = True
    entries[150] = True
    exits[250] = True
    
    # Create config with consolidate=false
    config_separate = """
data:
  amibroker_path: "dummy"
  data_frequency: "5T"

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
  consolidate_phased_trades: false  # Force separate trades
  phased_min_separation_bars: 1
  phased_force_separate_trades: true

output:
  results_dir: "results_separate"
  trade_list_filename: "trades_separate.csv"
  equity_curve_filename: "equity_separate.csv"
"""
    
    with open("config_separate.yaml", "w") as f:
        f.write(config_separate)
    
    print("\nRunning backtest with consolidate_phased_trades=false...")
    print("Expected: Multiple separate trades at 1/n size")
    
    engine = VectorBTEngine("config_separate.yaml")
    pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
    
    # Get trade details
    if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
        trades_df = pf.trades.records_readable
        num_trades = len(trades_df)
        
        print(f"\nResults:")
        print(f"  Number of trades: {num_trades}")
        print(f"  Expected: 10 trades (2 signals × 5 phases each)")
        
        if num_trades > 0:
            print(f"\nTrade details:")
            print("-"*60)
            for i, row in trades_df.iterrows():
                print(f"Trade {i+1}:")
                print(f"  Entry: bar {row['Entry Index']}")
                print(f"  Exit: bar {row['Exit Index']}")
                print(f"  Size: {row['Size']:.2f}")
                expected_size = 1000 / 5  # 1/5 of total position
                if abs(row['Size'] - expected_size) < 1:
                    print(f"  [OK] Size is ~{expected_size:.0f} (1/5 of 1000)")
                print()
            
            # Check if we got the expected number
            if num_trades == 10:
                print("\n[SUCCESS] Got exactly 10 separate trades as expected!")
            elif num_trades > 2:
                print(f"\n[PARTIAL SUCCESS] Got {num_trades} trades (more than consolidated)")
            else:
                print(f"\n[INFO] Still got {num_trades} trades (VBT may be consolidating)")
        
        # Export for inspection
        trades_df.to_csv("trades_separate.csv", index=False)
        print(f"Trade list exported to: trades_separate.csv")
    
    # Clean up
    import os
    if os.path.exists("config_separate.yaml"):
        os.remove("config_separate.yaml")
    
    return num_trades if 'num_trades' in locals() else 0


def compare_results():
    """Compare the two modes side by side"""
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    # Check if CSV files exist
    consolidated_exists = Path("trades_consolidated.csv").exists()
    separate_exists = Path("trades_separate.csv").exists()
    
    if consolidated_exists and separate_exists:
        consolidated_df = pd.read_csv("trades_consolidated.csv")
        separate_df = pd.read_csv("trades_separate.csv")
        
        print(f"\nConsolidated Mode:")
        print(f"  Total trades: {len(consolidated_df)}")
        if len(consolidated_df) > 0:
            print(f"  Average size: {consolidated_df['Size'].mean():.2f}")
            print(f"  Total P&L: {consolidated_df['PnL'].sum():.2f}")
        
        print(f"\nSeparate Mode:")
        print(f"  Total trades: {len(separate_df)}")
        if len(separate_df) > 0:
            print(f"  Average size: {separate_df['Size'].mean():.2f}")
            print(f"  Total P&L: {separate_df['PnL'].sum():.2f}")
        
        if len(separate_df) > len(consolidated_df):
            print(f"\n[SUCCESS] Separate mode created {len(separate_df) - len(consolidated_df)} more trades!")
        else:
            print(f"\n[INFO] Both modes created similar number of trades")


if __name__ == "__main__":
    print("TESTING PHASED TRADING - BOTH MODES")
    print("="*70)
    
    # Test 1: Consolidated mode
    consolidated_trades = test_consolidated_mode()
    
    # Test 2: Separate mode
    separate_trades = test_separate_mode()
    
    # Compare results
    compare_results()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    if separate_trades > consolidated_trades:
        print(f"\n[SUCCESS] Separate mode ({separate_trades} trades) > Consolidated ({consolidated_trades} trades)")
        print("The consolidate_phased_trades flag is working correctly!")
    else:
        print(f"\n[INFO] Consolidated: {consolidated_trades} trades, Separate: {separate_trades} trades")
        print("Check the CSV files for detailed trade information")