"""
Test that phased trading is now properly integrated and working
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine
import vectorbtpro as vbt


def test_phased_signal_splitting():
    """Test that signals are actually being split across multiple bars."""
    
    print("="*80)
    print("TESTING PHASED TRADING INTEGRATION")
    print("="*80)
    
    # Create test config with phased trading enabled
    config_content = """
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
  consolidate_phased_trades: false

output:
  results_dir: "results_phased_test"
  trade_list_filename: "trades.csv"
  equity_curve_filename: "equity.csv"
"""
    
    config_path = "config_phased_test.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Create simple test data
    n_bars = 500
    data = {
        'open': np.full(n_bars, 100.0),
        'high': np.full(n_bars, 101.0),
        'low': np.full(n_bars, 99.0),
        'close': np.full(n_bars, 100.0),
        'volume': np.ones(n_bars) * 1000000
    }
    
    # Create ONE entry signal and ONE exit signal
    original_entries = np.zeros(n_bars, dtype=bool)
    original_exits = np.zeros(n_bars, dtype=bool)
    
    original_entries[50] = True   # Single entry at bar 50
    original_exits[200] = True    # Single exit at bar 200
    
    print("\n### ORIGINAL SIGNALS")
    print("-"*60)
    print(f"Entry signals at bars: {np.where(original_entries)[0].tolist()}")
    print(f"Exit signals at bars: {np.where(original_exits)[0].tolist()}")
    print(f"Total entry signals: {np.sum(original_entries)}")
    print(f"Total exit signals: {np.sum(original_exits)}")
    
    # Capture what VBT actually receives
    original_from_signals = vbt.Portfolio.from_signals
    captured_data = {}
    
    def capture_vbt_inputs(close, entries, exits, **kwargs):
        """Capture what VBT actually receives."""
        captured_data['entries'] = entries.copy()
        captured_data['exits'] = exits.copy()
        captured_data['kwargs'] = kwargs.copy()
        
        print("\n### SIGNALS RECEIVED BY VECTORBT")
        print("-"*60)
        print(f"Entries shape: {entries.shape}")
        print(f"Entry signals at bars: {np.where(entries)[0].tolist() if entries.ndim == 1 else 'Multi-dimensional'}")
        print(f"Number of entry signals: {np.sum(entries)}")
        print(f"Exits shape: {exits.shape}")
        print(f"Exit signals at bars: {np.where(exits)[0].tolist() if exits.ndim == 1 else 'Multi-dimensional'}")
        print(f"Number of exit signals: {np.sum(exits)}")
        
        # Check size parameter
        if 'size' in kwargs:
            size = kwargs['size']
            if isinstance(size, np.ndarray):
                print(f"\nSize parameter is an array with shape: {size.shape}")
                nonzero_sizes = size[size != 0]
                if len(nonzero_sizes) > 0:
                    print(f"Non-zero sizes found at {len(nonzero_sizes)} positions")
                    print(f"First 10 non-zero sizes: {nonzero_sizes[:10]}")
                    print(f"Sum of all sizes: {np.sum(nonzero_sizes):.2f}")
            else:
                print(f"\nSize parameter: {size}")
        
        return original_from_signals(close=close, entries=entries, exits=exits, **kwargs)
    
    vbt.Portfolio.from_signals = capture_vbt_inputs
    
    try:
        # Run the engine
        engine = VectorBTEngine(config_path)
        
        # Check if phased engine is initialized
        if hasattr(engine, 'phased_engine'):
            print(f"\n[OK] Engine has phased_engine attribute")
            print(f"  Phased trading enabled: {engine.phased_engine.config.enabled}")
            print(f"  Entry bars: {engine.phased_engine.config.entry_bars}")
            print(f"  Exit bars: {engine.phased_engine.config.exit_bars}")
        else:
            print(f"\n[ERROR] Engine does NOT have phased_engine attribute")
        
        # Run backtest
        pf = engine.run_vectorized_backtest(data, original_entries, original_exits, "TEST")
        
        # Check trades
        if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
            trades_df = pf.trades.records_readable
            print(f"\n### RESULTING TRADES")
            print("-"*60)
            print(f"Number of trades created: {len(trades_df)}")
            
            # Show trade details
            for idx, trade in trades_df.iterrows():
                print(f"\nTrade {idx + 1}:")
                print(f"  Entry Index: {trade.get('Entry Index', 'N/A')}")
                print(f"  Exit Index: {trade.get('Exit Index', 'N/A')}")
                print(f"  Size: {trade.get('Size', 'N/A')}")
                print(f"  Entry Price: {trade.get('Avg Entry Price', 'N/A')}")
                print(f"  Exit Price: {trade.get('Avg Exit Price', 'N/A')}")
                print(f"  PnL: {trade.get('PnL', 'N/A')}")
        
    finally:
        # Restore
        vbt.Portfolio.from_signals = original_from_signals
    
    # Clean up
    import os
    if os.path.exists(config_path):
        os.remove(config_path)
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if 'entries' in captured_data:
        entries_array = captured_data['entries']
        exits_array = captured_data['exits']
        
        # Flatten if needed
        if entries_array.ndim > 1:
            entries_array = entries_array.flatten()
        if exits_array.ndim > 1:
            exits_array = exits_array.flatten()
        
        n_entries = np.sum(entries_array)
        n_exits = np.sum(exits_array)
        
        print(f"\nOriginal signals: 1 entry, 1 exit")
        print(f"After phasing: {n_entries} entries, {n_exits} exits")
        
        if n_entries == 5 and n_exits == 3:
            print("\n[OK] SUCCESS: Signals ARE being split correctly!")
            print(f"  1 original entry -> 5 phased entries (as configured)")
            print(f"  1 original exit -> 3 phased exits (as configured)")
            
            # Show where the phased signals are
            entry_indices = np.where(entries_array)[0]
            exit_indices = np.where(exits_array)[0]
            print(f"\n  Phased entry signals at bars: {entry_indices.tolist()}")
            print(f"  Phased exit signals at bars: {exit_indices.tolist()}")
            
            # Check if they're consecutive
            if len(entry_indices) > 1:
                entry_diffs = np.diff(entry_indices)
                if np.all(entry_diffs == 1):
                    print(f"  [OK] Entry signals are consecutive")
                else:
                    print(f"  [WARNING] Entry signals are not consecutive: gaps = {entry_diffs.tolist()}")
            
            if len(exit_indices) > 1:
                exit_diffs = np.diff(exit_indices)
                if np.all(exit_diffs == 1):
                    print(f"  [OK] Exit signals are consecutive")
                else:
                    print(f"  [WARNING] Exit signals are not consecutive: gaps = {exit_diffs.tolist()}")
                    
            return True
        else:
            print("\n[ERROR] FAILURE: Signals are NOT being split correctly")
            print(f"  Expected: 5 entries and 3 exits")
            print(f"  Got: {n_entries} entries and {n_exits} exits")
            return False
    else:
        print("Could not capture VBT inputs")
        return False


if __name__ == "__main__":
    success = test_phased_signal_splitting()
    
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if success:
        print("[OK] PHASED TRADING IS NOW WORKING!")
        print("\nThe implementation has been successfully restored to vbt_engine.py")
        print("Signals are being split across multiple bars as configured")
        print("\nHowever, note that VectorBT may still consolidate these in the final")
        print("trade list due to its internal trade management. To see individual")
        print("phase trades, you would need to use the phased_force_separate_trades option")
        print("or implement custom trade recording.")
    else:
        print("[ERROR] PHASED TRADING IS STILL NOT WORKING")
        print("\nPlease check the implementation and configuration")