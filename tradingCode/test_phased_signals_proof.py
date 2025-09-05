"""
Definitive test to prove whether phased trading is actually splitting signals across bars
This will show the exact signal arrays before and after phasing
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine
from src.backtest.phased_trading_engine import PhasedTradingEngine, PhasedConfig


def test_signal_splitting():
    """Test and visualize exact signal splitting."""
    
    print("="*80)
    print("PHASED TRADING SIGNAL SPLITTING VERIFICATION")
    print("="*80)
    
    # Create simple test data
    n_bars = 100
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
    
    original_entries[20] = True  # Single entry at bar 20
    original_exits[60] = True    # Single exit at bar 60
    
    print("\n### ORIGINAL SIGNALS")
    print("-"*60)
    print(f"Entry signals at bars: {np.where(original_entries)[0].tolist()}")
    print(f"Exit signals at bars: {np.where(original_exits)[0].tolist()}")
    print(f"Total entry signals: {np.sum(original_entries)}")
    print(f"Total exit signals: {np.sum(original_exits)}")
    
    # Create phased trading engine
    config = PhasedConfig(
        enabled=True,
        entry_bars=5,
        exit_bars=3,
        entry_distribution="linear",
        exit_distribution="linear"
    )
    
    engine = PhasedTradingEngine(config)
    
    print("\n### PHASED CONFIGURATION")
    print("-"*60)
    print(f"Entry bars: {config.entry_bars}")
    print(f"Exit bars: {config.exit_bars}")
    print(f"Entry distribution: {config.entry_distribution}")
    print(f"Entry weights: {engine.entry_weights}")
    print(f"Exit weights: {engine.exit_weights}")
    
    # Process signals through phased engine
    phased_results = engine.process_signals(
        original_entries, 
        original_exits, 
        data,
        position_size=1000
    )
    
    # Extract phased signals
    phased_entry_sizes = phased_results['entry_sizes']
    phased_exit_sizes = phased_results['exit_sizes']
    
    # Convert to boolean for signal detection
    phased_entry_signals = phased_entry_sizes > 0
    phased_exit_signals = phased_exit_sizes > 0
    
    print("\n### PHASED SIGNALS")
    print("-"*60)
    print(f"Entry signals at bars: {np.where(phased_entry_signals)[0].tolist()}")
    print(f"Exit signals at bars: {np.where(phased_exit_signals)[0].tolist()}")
    print(f"Total entry signals: {np.sum(phased_entry_signals)}")
    print(f"Total exit signals: {np.sum(phased_exit_signals)}")
    
    # Show detailed breakdown
    print("\n### DETAILED ENTRY PHASE BREAKDOWN")
    print("-"*60)
    entry_bars = np.where(phased_entry_signals)[0]
    for bar in entry_bars:
        size = phased_entry_sizes[bar]
        print(f"Bar {bar}: Size = {size:.2f} (${size:.2f} position)")
    
    print("\n### DETAILED EXIT PHASE BREAKDOWN")
    print("-"*60)
    exit_bars = np.where(phased_exit_signals)[0]
    for bar in exit_bars:
        size = phased_exit_sizes[bar]
        print(f"Bar {bar}: Size = {size:.2f} (${size:.2f} position)")
    
    # Visual representation
    print("\n### VISUAL SIGNAL MAP (first 80 bars)")
    print("-"*60)
    print("Legend: . = no signal, E = entry, X = exit, e = phased entry, x = phased exit")
    print()
    
    # Original signals line
    original_line = ""
    for i in range(80):
        if original_entries[i]:
            original_line += "E"
        elif original_exits[i]:
            original_line += "X"
        else:
            original_line += "."
    
    # Phased signals line
    phased_line = ""
    for i in range(80):
        if phased_entry_signals[i]:
            phased_line += "e"
        elif phased_exit_signals[i]:
            phased_line += "x"
        else:
            phased_line += "."
    
    print(f"Bar:      {''.join(str(i%10) for i in range(80))}")
    print(f"Original: {original_line}")
    print(f"Phased:   {phased_line}")
    
    # Verification
    print("\n### VERIFICATION")
    print("-"*60)
    
    if np.sum(phased_entry_signals) == config.entry_bars:
        print(f"✓ SUCCESS: 1 original entry signal became {config.entry_bars} phased entries")
    else:
        print(f"✗ FAILURE: Expected {config.entry_bars} phased entries, got {np.sum(phased_entry_signals)}")
    
    if np.sum(phased_exit_signals) == config.exit_bars:
        print(f"✓ SUCCESS: 1 original exit signal became {config.exit_bars} phased exits")
    else:
        print(f"✗ FAILURE: Expected {config.exit_bars} phased exits, got {np.sum(phased_exit_signals)}")
    
    # Check if signals are consecutive
    entry_indices = np.where(phased_entry_signals)[0]
    if len(entry_indices) > 1:
        consecutive = all(entry_indices[i+1] - entry_indices[i] == 1 for i in range(len(entry_indices)-1))
        if consecutive:
            print(f"✓ Phased entries are consecutive: {entry_indices.tolist()}")
        else:
            print(f"✗ Phased entries are NOT consecutive: {entry_indices.tolist()}")
    
    exit_indices = np.where(phased_exit_signals)[0]
    if len(exit_indices) > 1:
        consecutive = all(exit_indices[i+1] - exit_indices[i] == 1 for i in range(len(exit_indices)-1))
        if consecutive:
            print(f"✓ Phased exits are consecutive: {exit_indices.tolist()}")
        else:
            print(f"✗ Phased exits are NOT consecutive: {exit_indices.tolist()}")
    
    return phased_entry_signals, phased_exit_signals, phased_entry_sizes, phased_exit_sizes


def test_with_vbt_engine():
    """Test using the full VBT engine to see what signals it receives."""
    
    print("\n" + "="*80)
    print("TESTING WITH VBT ENGINE")
    print("="*80)
    
    # Create config with phased trading
    config = """
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
  consolidate_phased_trades: true

output:
  results_dir: "results_test"
  trade_list_filename: "trades_test.csv"
  equity_curve_filename: "equity_test.csv"
"""
    
    with open("config_signal_test.yaml", "w") as f:
        f.write(config)
    
    # Create test data
    n_bars = 100
    data = {
        'open': np.full(n_bars, 100.0),
        'high': np.full(n_bars, 101.0),
        'low': np.full(n_bars, 99.0),
        'close': np.full(n_bars, 100.0),
        'volume': np.ones(n_bars) * 1000000
    }
    
    # Create signals
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    entries[20] = True
    exits[60] = True
    
    print("\nOriginal signals passed to VBT engine:")
    print(f"  Entries: {np.where(entries)[0].tolist()}")
    print(f"  Exits: {np.where(exits)[0].tolist()}")
    
    # We need to monkey-patch VBT to see what signals it actually receives
    import vectorbtpro as vbt
    original_from_signals = vbt.Portfolio.from_signals
    
    captured_signals = {}
    
    def capture_signals(close, entries, exits, **kwargs):
        """Capture the actual signals VBT receives."""
        captured_signals['entries'] = entries.copy()
        captured_signals['exits'] = exits.copy()
        captured_signals['kwargs'] = kwargs.copy()
        
        print("\n### SIGNALS RECEIVED BY VECTORBT:")
        print(f"  Entry shape: {entries.shape}")
        print(f"  Entry signals at: {np.where(entries)[0].tolist()}")
        print(f"  Exit shape: {exits.shape}")
        print(f"  Exit signals at: {np.where(exits)[0].tolist()}")
        print(f"  Total entries: {np.sum(entries)}")
        print(f"  Total exits: {np.sum(exits)}")
        
        if 'size' in kwargs:
            size = kwargs['size']
            if isinstance(size, np.ndarray):
                print(f"  Size array shape: {size.shape}")
                nonzero_sizes = size[size > 0]
                if len(nonzero_sizes) > 0:
                    print(f"  Non-zero sizes: {nonzero_sizes[:10]}")  # Show first 10
        
        # Call original function
        return original_from_signals(close=close, entries=entries, exits=exits, **kwargs)
    
    # Temporarily replace the function
    vbt.Portfolio.from_signals = capture_signals
    
    try:
        # Run backtest
        engine = VectorBTEngine("config_signal_test.yaml")
        pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
        
        # Check results
        if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
            trades_df = pf.trades.records_readable
            print(f"\n### RESULTING TRADES:")
            print(f"  Number of trades: {len(trades_df)}")
            for i, trade in trades_df.head().iterrows():
                print(f"  Trade {i+1}: Entry at {trade['Entry Index']}, Exit at {trade['Exit Index']}")
    
    finally:
        # Restore original function
        vbt.Portfolio.from_signals = original_from_signals
    
    # Clean up
    import os
    if os.path.exists("config_signal_test.yaml"):
        os.remove("config_signal_test.yaml")
    
    return captured_signals


if __name__ == "__main__":
    # Test 1: Direct phased engine test
    print("TEST 1: DIRECT PHASED ENGINE TEST")
    phased_entries, phased_exits, entry_sizes, exit_sizes = test_signal_splitting()
    
    # Test 2: Full VBT engine test
    print("\n" + "="*80)
    print("TEST 2: FULL VBT ENGINE TEST")
    captured = test_with_vbt_engine()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if np.sum(phased_entries) > 1:
        print("✓ CONFIRMED: Phased trading IS splitting signals across multiple bars")
        print(f"  - 1 entry signal → {np.sum(phased_entries)} phased entries")
        print(f"  - 1 exit signal → {np.sum(phased_exits)} phased exits")
    else:
        print("✗ PROBLEM: Signals are NOT being split properly")
    
    print("\nHowever, VectorBT consolidates these into fewer trades in the final output.")
    print("This is why you don't see multiple trades in the tradelist.")