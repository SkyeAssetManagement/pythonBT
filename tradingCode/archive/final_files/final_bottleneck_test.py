#!/usr/bin/env python3
"""
Final bottleneck identification for single run since 2000
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data.parquet_converter import ParquetConverter
from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy

def test_large_dataset_bottleneck():
    """Test what happens with the full 2000 dataset"""
    
    print("FINAL BOTTLENECK TEST: Single Run Since 2000")
    print("=" * 50)
    
    # Load data
    print("1. Loading data since 2000...")
    start_time = time.time()
    
    parquet_converter = ParquetConverter()
    data = parquet_converter.load_or_convert("ES", "1m", "diffAdjusted")
    data = parquet_converter.filter_data_by_date(data, "2000-01-01", None)
    
    load_time = time.time() - start_time
    n_bars = len(data['close'])
    
    print(f"   SUCCESS: {n_bars:,} bars loaded in {load_time:.2f}s")
    print(f"   Data size: {n_bars/1000000:.1f}M bars")
    
    # Test individual components
    strategy = TimeWindowVectorizedStrategy()
    params = strategy.get_parameter_combinations(use_defaults_only=True)[0]
    
    print(f"\n2. Testing signal generation only...")
    start_time = time.time()
    
    try:
        entry_signals_2d, exit_signals_2d, prices_2d = strategy._generate_gradual_signals(data, params)
        signal_time = time.time() - start_time
        
        print(f"   SUCCESS: Signal generation in {signal_time:.2f}s")
        print(f"   Throughput: {n_bars/signal_time:,.0f} bars/sec")
        print(f"   Entry signals shape: {entry_signals_2d.shape}")
        print(f"   Total entry signals: {entry_signals_2d.sum()}")
        
        # Test VectorBT portfolio creation
        print(f"\n3. Testing VectorBT portfolio creation...")
        
        config = {'initial_capital': 100000, 'commission': 0.001, 'slippage': 0.0001}
        
        start_time = time.time()
        
        # This is likely where the bottleneck is
        close_prices = data['close']
        n_bars, n_entry_bars = entry_signals_2d.shape
        
        # Create fractional position sizing
        fractional_size = 1.0 / n_entry_bars
        
        # Set up signals for VectorBT
        if params['direction'] == 'long':
            long_entries = entry_signals_2d
            short_entries = np.zeros_like(entry_signals_2d)
            long_exits = exit_signals_2d
            short_exits = np.zeros_like(exit_signals_2d)
        else:
            long_entries = np.zeros_like(entry_signals_2d)
            short_entries = entry_signals_2d
            long_exits = np.zeros_like(exit_signals_2d)
            short_exits = exit_signals_2d
        
        # VectorBT portfolio creation (suspected bottleneck)
        import vectorbtpro as vbt
        
        portfolio_config = {
            'init_cash': config.get('initial_capital', 100000),
            'fees': config.get('commission', 0.001),
            'slippage': config.get('slippage', 0.0001),
            'freq': '1min',
            'size': fractional_size,
            'size_type': 'amount'
        }
        
        pf = vbt.Portfolio.from_signals(
            close=close_prices,
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            price=prices_2d[:, :n_entry_bars] if prices_2d.shape[1] >= n_entry_bars else prices_2d,
            **portfolio_config
        )
        
        portfolio_time = time.time() - start_time
        trades = len(pf.trades.records_readable)
        
        print(f"   SUCCESS: VectorBT portfolio in {portfolio_time:.2f}s")
        print(f"   Total trades: {trades}")
        
        total_time = signal_time + portfolio_time
        print(f"\n4. PERFORMANCE BREAKDOWN:")
        print(f"   Signal generation: {signal_time:.2f}s ({signal_time/total_time*100:.1f}%)")
        print(f"   VectorBT portfolio: {portfolio_time:.2f}s ({portfolio_time/total_time*100:.1f}%)")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Overall throughput: {n_bars/total_time:,.0f} bars/sec")
        
        if portfolio_time > signal_time * 2:
            print(f"\n   BOTTLENECK: VectorBT portfolio creation is the main bottleneck")
        else:
            print(f"\n   BOTTLENECK: Signal generation is the main bottleneck")
            
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import numpy as np
    test_large_dataset_bottleneck()