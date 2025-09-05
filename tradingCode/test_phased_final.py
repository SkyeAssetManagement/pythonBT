"""
Test the final phased trading solution with the user's config
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine


def test_phased_separate_trades():
    """Test that we get separate trades at 1/n size"""
    
    print("="*70)
    print("TESTING FINAL PHASED TRADING SOLUTION")
    print("="*70)
    
    # Create simple test data
    n_bars = 500
    np.random.seed(42)
    close = 100 + np.arange(n_bars) * 0.01
    
    data = {
        'open': close * 1.001,  # Open slightly higher
        'high': close * 1.002,
        'low': close * 0.998,
        'close': close,
        'volume': np.ones(n_bars) * 1e6
    }
    
    # Create a few well-spaced signals
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    
    # Signal 1
    entries[50] = True
    exits[150] = True
    
    # Signal 2
    entries[200] = True  
    exits[350] = True
    
    print(f"\nTest setup:")
    print(f"  Data: {n_bars} bars")
    print(f"  Signals: 2 entry signals, 2 exit signals")
    print(f"  Using config_phased_test.yaml settings")
    
    # Load and run with the user's config
    config_path = "config_phased_test.yaml"
    
    print(f"\nConfiguration from {config_path}:")
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    backtest_config = config['backtest']
    print(f"  phased_trading_enabled: {backtest_config.get('phased_trading_enabled')}")
    print(f"  phased_entry_bars: {backtest_config.get('phased_entry_bars')}")
    print(f"  phased_exit_bars: {backtest_config.get('phased_exit_bars')}")
    print(f"  consolidate_phased_trades: {backtest_config.get('consolidate_phased_trades')}")
    print(f"  position_size: {backtest_config.get('position_size')}")
    
    entry_bars = backtest_config.get('phased_entry_bars', 1)
    position_size = backtest_config.get('position_size', 1000)
    expected_size_per_phase = position_size / entry_bars
    expected_total_trades = 2 * entry_bars  # 2 signals × n phases each
    
    print(f"\nExpected results:")
    print(f"  Total trades: {expected_total_trades}")
    print(f"  Size per trade: {expected_size_per_phase:.2f} (1/{entry_bars} of {position_size})")
    
    print("\nRunning backtest...")
    engine = VectorBTEngine(config_path)
    pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
    
    # Get trade details
    if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
        trades_df = pf.trades.records_readable
        num_trades = len(trades_df)
        
        print(f"\n" + "="*70)
        print(f"RESULTS:")
        print(f"  Actual number of trades: {num_trades}")
        print(f"  Expected number of trades: {expected_total_trades}")
        
        if num_trades == expected_total_trades:
            print(f"\n[SUCCESS] Got exactly {expected_total_trades} separate trades!")
        elif num_trades > 2:
            print(f"\n[PARTIAL SUCCESS] Got {num_trades} trades (more than base signals)")
        else:
            print(f"\n[INFO] Got {num_trades} trades (VBT may still be consolidating)")
        
        print(f"\nDetailed trade list:")
        print("-"*70)
        for i, row in trades_df.iterrows():
            print(f"\nTrade {i+1}:")
            print(f"  Entry Index: {row['Entry Index']}")
            print(f"  Exit Index: {row['Exit Index']}")
            print(f"  Size: {row['Size']:.2f}")
            print(f"  Entry Price: ${row['Avg Entry Price']:.2f}")
            print(f"  Exit Price: ${row['Avg Exit Price']:.2f}")
            print(f"  P&L: ${row['PnL']:.2f}")
            
            # Check if size matches expected
            if abs(row['Size'] - expected_size_per_phase) < 1:
                print(f"  [OK] Size is correct (~{expected_size_per_phase:.0f})")
            else:
                print(f"  [WARN] Size mismatch (expected {expected_size_per_phase:.0f})")
        
        # Export for detailed inspection
        output_file = "results_phased/tradelist_final_test.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\n" + "="*70)
        print(f"Trade list exported to: {output_file}")
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Total trades: {num_trades}")
        print(f"  Average trade size: {trades_df['Size'].mean():.2f}")
        print(f"  Min trade size: {trades_df['Size'].min():.2f}")
        print(f"  Max trade size: {trades_df['Size'].max():.2f}")
        print(f"  Total P&L: ${trades_df['PnL'].sum():.2f}")
        
        # Check if we achieved the goal
        if num_trades >= expected_total_trades:
            sizes = trades_df['Size'].values
            sizes_correct = all(abs(s - expected_size_per_phase) < 1 for s in sizes)
            if sizes_correct:
                print(f"\n" + "="*70)
                print(f"[SUCCESS] GOAL ACHIEVED!")
                print(f"  ✓ {num_trades} separate trades in tradelist")
                print(f"  ✓ Each trade has size ~{expected_size_per_phase:.0f} (1/{entry_bars} of {position_size})")
                print(f"  ✓ Trades are visible in: results_phased/tradelist.csv")
                return True
        
    else:
        print("\n[ERROR] No trades found!")
    
    return False


if __name__ == "__main__":
    success = test_phased_separate_trades()
    
    if not success:
        print("\n" + "="*70)
        print("TROUBLESHOOTING:")
        print("1. Check that consolidate_phased_trades is set to false")
        print("2. Verify phased_trading_enabled is true")
        print("3. Check the exported CSV file for trade details")
        print("4. VectorBT may still be consolidating - this is a core limitation")