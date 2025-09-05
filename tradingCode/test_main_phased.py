"""
Direct test of phased trading using main.py infrastructure
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.vbt_engine import VectorBTEngine
from strategies.simpleSMA import SimpleSMAStrategy


def test_phased_config():
    """Test that phased configuration is being loaded and applied."""
    
    print("="*70)
    print("TESTING PHASED TRADING WITH MAIN.PY INFRASTRUCTURE")
    print("="*70)
    
    # Generate synthetic test data
    print("\n1. Generating synthetic test data...")
    n_bars = 1000
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, n_bars)
    close = 4000 * np.exp(np.cumsum(returns))
    
    data = {
        'open': close * (1 + np.random.normal(0, 0.001, n_bars)),
        'high': close * (1 + np.abs(np.random.normal(0, 0.005, n_bars))),
        'low': close * (1 - np.abs(np.random.normal(0, 0.005, n_bars))),
        'close': close,
        'volume': np.random.uniform(1e5, 1e6, n_bars)
    }
    
    # Initialize strategy
    print("\n2. Initializing strategy...")
    strategy = SimpleSMAStrategy()
    entries, exits = strategy.generate_signals(data)
    print(f"   Generated {np.sum(entries)} entry signals, {np.sum(exits)} exit signals")
    
    # Test with normal config
    print("\n3. Testing with NORMAL configuration (no phasing)...")
    engine_normal = VectorBTEngine("config.yaml")
    print(f"   Phased trading enabled: {engine_normal.config['backtest'].get('phased_trading_enabled', False)}")
    
    pf_normal = engine_normal.run_vectorized_backtest(data, entries, exits, "TEST")
    trades_normal = len(pf_normal.trades.records) if hasattr(pf_normal.trades, 'records') else 0
    print(f"   Result: {trades_normal} trades")
    
    # Test with phased config
    print("\n4. Testing with PHASED configuration...")
    engine_phased = VectorBTEngine("config_phased_test.yaml")
    config_bt = engine_phased.config['backtest']
    print(f"   Phased trading enabled: {config_bt.get('phased_trading_enabled', False)}")
    print(f"   Entry bars: {config_bt.get('phased_entry_bars', 1)}")
    print(f"   Exit bars: {config_bt.get('phased_exit_bars', 1)}")
    print(f"   Entry distribution: {config_bt.get('phased_entry_distribution', 'linear')}")
    print(f"   Exit distribution: {config_bt.get('phased_exit_distribution', 'linear')}")
    
    pf_phased = engine_phased.run_vectorized_backtest(data, entries, exits, "TEST")
    trades_phased = len(pf_phased.trades.records) if hasattr(pf_phased.trades, 'records') else 0
    print(f"   Result: {trades_phased} trades")
    
    # Compare metrics
    print("\n5. COMPARISON:")
    print("="*50)
    
    metrics_normal = {
        'return': float(pf_normal.total_return.iloc[0] if hasattr(pf_normal.total_return, 'iloc') else pf_normal.total_return) * 100,
        'sharpe': float(pf_normal.sharpe_ratio.iloc[0] if hasattr(pf_normal.sharpe_ratio, 'iloc') else pf_normal.sharpe_ratio),
        'trades': trades_normal
    }
    
    metrics_phased = {
        'return': float(pf_phased.total_return.iloc[0] if hasattr(pf_phased.total_return, 'iloc') else pf_phased.total_return) * 100,
        'sharpe': float(pf_phased.sharpe_ratio.iloc[0] if hasattr(pf_phased.sharpe_ratio, 'iloc') else pf_phased.sharpe_ratio),
        'trades': trades_phased
    }
    
    print(f"Normal Trading:")
    print(f"  Return: {metrics_normal['return']:.4f}%")
    print(f"  Sharpe: {metrics_normal['sharpe']:.2f}")
    print(f"  Trades: {metrics_normal['trades']}")
    
    print(f"\nPhased Trading:")
    print(f"  Return: {metrics_phased['return']:.4f}%")
    print(f"  Sharpe: {metrics_phased['sharpe']:.2f}")
    print(f"  Trades: {metrics_phased['trades']}")
    
    print(f"\nDifferences:")
    print(f"  Return diff: {metrics_phased['return'] - metrics_normal['return']:.4f}%")
    print(f"  Sharpe diff: {metrics_phased['sharpe'] - metrics_normal['sharpe']:.2f}")
    print(f"  Trade diff: {metrics_phased['trades'] - metrics_normal['trades']}")
    
    # Check if phased engine was actually created
    if engine_phased.phased_engine is not None:
        print("\n[OK] PHASED TRADING ENGINE SUCCESSFULLY INITIALIZED")
        print(f"   Entry weights: {engine_phased.phased_engine.entry_weights}")
        print(f"   Exit weights: {engine_phased.phased_engine.exit_weights}")
    else:
        print("\n[ERROR] PHASED TRADING ENGINE NOT INITIALIZED")
    
    return metrics_normal, metrics_phased


if __name__ == "__main__":
    test_phased_config()
    print("\n[OK] Test complete! You can now use phased trading with main.py")