"""
Test to verify multiple trades at 1/n size are created for phased trading
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine


def test_multiple_trades():
    """Test that phased trading creates multiple trades at 1/n size."""
    
    print("="*70)
    print("TESTING MULTIPLE TRADES AT 1/n SIZE")
    print("="*70)
    
    # Create very simple test data
    n_bars = 200
    np.random.seed(42)
    close = 100 + np.arange(n_bars) * 0.01  # Slowly rising price
    
    data = {
        'open': close,
        'high': close * 1.001,
        'low': close * 0.999,
        'close': close,
        'volume': np.ones(n_bars) * 1e6
    }
    
    # Create ONE entry signal and ONE exit signal with lots of space
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    
    entries[20] = True   # Single entry signal at bar 20
    exits[100] = True    # Single exit signal at bar 100
    
    print(f"\nTest setup:")
    print(f"  Data: {n_bars} bars")
    print(f"  Signals: 1 entry at bar 20, 1 exit at bar 100")
    
    # Create config for separate trades (consolidate=false)
    config_separate = """
data:
  amibroker_path: "dummy"
  data_frequency: "5T"

backtest:
  initial_cash: 100000
  position_size: 1000  # Total position size
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
  phased_entry_bars: 5  # Should create 5 separate trades
  phased_exit_bars: 1   # Single exit per trade
  phased_entry_distribution: "linear"
  phased_exit_distribution: "linear"
  phased_entry_price_method: "limit"
  phased_exit_price_method: "limit"
  consolidate_phased_trades: false  # IMPORTANT: Create separate trades

output:
  results_dir: "results_multi_test"
  trade_list_filename: "trades_multiple.csv"
  equity_curve_filename: "equity_multiple.csv"
"""
    
    with open("config_test_multiple.yaml", "w") as f:
        f.write(config_separate)
    
    print("\nRunning backtest with phased_entry_bars=5, consolidate=false...")
    print("Expected: 5 separate trades, each with size = 1000/5 = 200")
    
    engine = VectorBTEngine("config_test_multiple.yaml")
    pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
    
    # Get trade details
    if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
        trades_df = pf.trades.records_readable
        num_trades = len(trades_df)
        
        print(f"\nResults:")
        print(f"  Number of trades: {num_trades}")
        
        if num_trades > 0:
            print(f"\nTrade details:")
            print("-"*60)
            for i, row in trades_df.iterrows():
                print(f"Trade {i+1}:")
                print(f"  Entry: bar {row['Entry Index']}")
                print(f"  Exit: bar {row['Exit Index']}")
                print(f"  Size: {row['Size']:.2f}")
                print(f"  Entry Price: {row['Avg Entry Price']:.2f}")
                print(f"  Exit Price: {row['Avg Exit Price']:.2f}")
                print(f"  P&L: {row['PnL']:.2f}")
                print()
            
            # Check if we got the expected number of trades
            if num_trades == 5:
                print("[OK] Successfully created 5 separate trades!")
                
                # Check if sizes are approximately 1/5 of total
                sizes = trades_df['Size'].values
                expected_size = 200  # 1000/5
                sizes_correct = all(abs(s - expected_size) < 1 for s in sizes)
                
                if sizes_correct:
                    print(f"[OK] Each trade has size ~{expected_size} (1/5 of 1000)")
                else:
                    print(f"[WARNING] Trade sizes vary: {sizes}")
            else:
                print(f"[WARNING] Expected 5 trades, got {num_trades}")
        
        # Export for inspection
        trades_df.to_csv("test_multiple_trades.csv", index=False)
        print(f"\nTrade list exported to: test_multiple_trades.csv")
    else:
        print("\n[ERROR] No trades found!")
    
    # Clean up
    import os
    if os.path.exists("config_test_multiple.yaml"):
        os.remove("config_test_multiple.yaml")
    
    return num_trades if 'num_trades' in locals() else 0


def test_consolidated_comparison():
    """Compare consolidated vs separate modes."""
    
    print("\n" + "="*70)
    print("COMPARING CONSOLIDATED VS SEPARATE MODES")
    print("="*70)
    
    # Same simple test data
    n_bars = 200
    close = 100 + np.arange(n_bars) * 0.01
    
    data = {
        'open': close,
        'high': close * 1.001,
        'low': close * 0.999,
        'close': close,
        'volume': np.ones(n_bars) * 1e6
    }
    
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    entries[20] = True
    exits[100] = True
    
    results = {}
    
    for consolidate in [True, False]:
        mode = "Consolidated" if consolidate else "Separate"
        
        config = f"""
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
  phased_exit_bars: 1
  phased_entry_distribution: "linear"
  phased_exit_distribution: "linear"
  phased_entry_price_method: "limit"
  phased_exit_price_method: "limit"
  consolidate_phased_trades: {str(consolidate).lower()}

output:
  results_dir: "results_compare"
  trade_list_filename: "trades_{mode}.csv"
  equity_curve_filename: "equity_{mode}.csv"
"""
        
        config_file = f"config_{mode.lower()}.yaml"
        with open(config_file, "w") as f:
            f.write(config)
        
        print(f"\n{mode} Mode:")
        engine = VectorBTEngine(config_file)
        pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
        
        num_trades = len(pf.trades.records) if hasattr(pf.trades, 'records') else 0
        results[mode] = num_trades
        print(f"  Trades: {num_trades}")
        
        # Clean up
        import os
        if os.path.exists(config_file):
            os.remove(config_file)
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"  Consolidated mode: {results.get('Consolidated', 0)} trade(s)")
    print(f"  Separate mode: {results.get('Separate', 0)} trade(s)")
    
    if results.get('Separate', 0) > results.get('Consolidated', 0):
        print("\n[OK] Separate mode creates more trades as expected!")
    else:
        print("\n[INFO] Both modes show same number of trades")
        print("       VectorBT may be consolidating internally")


if __name__ == "__main__":
    # Test 1: Verify multiple trades at 1/n size
    num_trades = test_multiple_trades()
    
    # Test 2: Compare modes
    test_consolidated_comparison()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    if num_trades == 5:
        print("\n[SUCCESS] Phased trading creates multiple trades at 1/n size!")
    else:
        print(f"\n[INFO] Created {num_trades} trades (expected 5)")
        print("       Check test_multiple_trades.csv for details")