"""
Test to definitively prove whether phased trading is working
by examining the actual implementation in vbt_engine.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_current_implementation():
    """Test what the current VBT engine actually does with phased trading config."""
    
    print("="*80)
    print("TESTING CURRENT PHASED TRADING IMPLEMENTATION")
    print("="*80)
    
    # Check what's in the current vbt_engine.py
    vbt_engine_path = Path("src/backtest/vbt_engine.py")
    
    print("\nChecking vbt_engine.py for phased trading implementation...")
    print("-"*60)
    
    with open(vbt_engine_path, 'r') as f:
        content = f.read()
    
    # Look for phased trading references
    has_phased_config = 'phased_trading_enabled' in content
    has_phased_engine = 'PhasedTradingEngine' in content or 'phased_trading_engine' in content
    has_phased_import = 'from .phased_trading' in content
    has_process_signals = 'process_signals' in content
    
    print(f"Has phased_trading_enabled config check: {has_phased_config}")
    print(f"Has PhasedTradingEngine reference: {has_phased_engine}")
    print(f"Has phased trading import: {has_phased_import}")
    print(f"Has process_signals method: {has_process_signals}")
    
    if not (has_phased_config or has_phased_engine):
        print("\n⚠️ WARNING: No phased trading implementation found in vbt_engine.py!")
        print("The phased trading code appears to have been removed or disabled.")
        return False
    
    # Now test with actual config
    print("\n" + "="*80)
    print("RUNNING ACTUAL TEST WITH CONFIG")
    print("="*80)
    
    from src.backtest.vbt_engine import VectorBTEngine
    import vectorbtpro as vbt
    
    # Create test config
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
  consolidate_phased_trades: false

output:
  results_dir: "results_test"
  trade_list_filename: "trades.csv"
  equity_curve_filename: "equity.csv"
"""
    
    config_path = "test_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Create simple test data
    n_bars = 100
    data = {
        'open': np.full(n_bars, 100.0),
        'high': np.full(n_bars, 100.0),
        'low': np.full(n_bars, 100.0),
        'close': np.full(n_bars, 100.0),
        'volume': np.ones(n_bars) * 1000000
    }
    
    # Create ONE signal
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    entries[20] = True
    exits[60] = True
    
    print(f"\nOriginal signals:")
    print(f"  Entry at bar: {np.where(entries)[0].tolist()}")
    print(f"  Exit at bar: {np.where(exits)[0].tolist()}")
    
    # Capture what VBT receives
    original_from_signals = vbt.Portfolio.from_signals
    captured_data = {}
    
    def capture_vbt_inputs(close, entries, exits, **kwargs):
        """Capture what VBT actually receives."""
        captured_data['entries'] = entries.copy()
        captured_data['exits'] = exits.copy()
        
        print(f"\nSignals received by VectorBT:")
        print(f"  Entries shape: {entries.shape}")
        print(f"  Entry signals at bars: {np.where(entries)[0].flatten().tolist()}")
        print(f"  Number of entry signals: {np.sum(entries)}")
        print(f"  Exits shape: {exits.shape}")
        print(f"  Exit signals at bars: {np.where(exits)[0].flatten().tolist()}")
        print(f"  Number of exit signals: {np.sum(exits)}")
        
        return original_from_signals(close=close, entries=entries, exits=exits, **kwargs)
    
    vbt.Portfolio.from_signals = capture_vbt_inputs
    
    try:
        # Run the engine
        engine = VectorBTEngine(config_path)
        
        # Check if engine has phased trading attributes
        if hasattr(engine, 'phased_engine'):
            print(f"\n✓ Engine has phased_engine attribute")
        else:
            print(f"\n✗ Engine does NOT have phased_engine attribute")
        
        # Run backtest
        pf = engine.run_vectorized_backtest(data, entries, exits, "TEST")
        
        # Check trades
        if hasattr(pf.trades, 'records'):
            print(f"\nNumber of trades created: {len(pf.trades.records)}")
        
    finally:
        # Restore
        vbt.Portfolio.from_signals = original_from_signals
    
    # Clean up
    import os
    if os.path.exists(config_path):
        os.remove(config_path)
    
    # Analyze results
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if 'entries' in captured_data:
        n_entries = np.sum(captured_data['entries'])
        n_exits = np.sum(captured_data['exits'])
        
        if n_entries > 1:
            print(f"✓ SUCCESS: Signals ARE being split!")
            print(f"  1 original entry → {n_entries} signals to VBT")
            print(f"  1 original exit → {n_exits} signals to VBT")
            return True
        else:
            print(f"✗ FAILURE: Signals are NOT being split")
            print(f"  VBT received only {n_entries} entry signal(s)")
            return False
    else:
        print("Could not capture VBT inputs")
        return False


if __name__ == "__main__":
    success = test_current_implementation()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if not success:
        print("❌ PHASED TRADING IS NOT WORKING")
        print("\nThe implementation appears to be missing or disabled.")
        print("The phased trading configuration in config.yaml is not being processed.")
        print("\nTo fix this, the vbt_engine.py needs to:")
        print("1. Import and initialize a phased trading engine")
        print("2. Process signals through the phased engine before passing to VBT")
        print("3. Split single signals into multiple phased signals")
    else:
        print("✅ PHASED TRADING IS WORKING")
        print("\nSignals are being split across multiple bars as configured.")