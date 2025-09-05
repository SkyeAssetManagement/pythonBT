#!/usr/bin/env python3
"""
Test Timestamp Enhancement for Trade List
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add strategies to path
strategies_path = Path(__file__).parent / "strategies"
if str(strategies_path) not in sys.path:
    sys.path.insert(0, str(strategies_path))

from backtest.vbt_engine import VectorBTEngine
from time_window_strategy_vectorized import TimeWindowVectorizedSingle

def test_timestamp_formatting():
    """Test the timestamp formatting function"""
    
    print("TESTING TIMESTAMP FORMATTING")
    print("=" * 50)
    
    # Create VBT engine instance
    try:
        engine = VectorBTEngine()
    except FileNotFoundError:
        print("Config file not found, using minimal config for testing")
        # Create minimal config for testing
        import yaml
        test_config = {
            'execution_price': 'close',
            'signal_lag': 0,
            'position_size': 1,
            'initial_cash': 100000,
            'fees': 0.001,
            'slippage': 0.0005
        }
        engine.config = test_config
    
    # Test timestamp conversion
    test_timestamps = [
        1751362260000000000,  # From actual trade list
        1751364060000000000,  # From actual trade list
        1640995200000000000,  # Jan 1, 2022 00:00:00 UTC
        1609459200000000000,  # Jan 1, 2021 00:00:00 UTC
    ]
    
    expected_formats = [
        "2025-06-30 09:31:00",  # Expected for first timestamp
        "2025-06-30 10:01:00",  # Expected for second timestamp  
        "2022-01-01 00:00:00",  # Expected for third timestamp
        "2021-01-01 00:00:00",  # Expected for fourth timestamp
    ]
    
    print("Testing timestamp conversion:")
    for i, timestamp_ns in enumerate(test_timestamps):
        result = engine._format_timestamp_readable(timestamp_ns)
        print(f"  {timestamp_ns} -> {result}")
        
        # Verify format is correct (yyyy-mm-dd hh:mm:ss)
        if len(result) == 19 and result[4] == '-' and result[7] == '-' and result[10] == ' ' and result[13] == ':' and result[16] == ':':
            print(f"    Format OK")
        else:
            print(f"    Format ERROR: Expected 'yyyy-mm-dd hh:mm:ss', got '{result}'")
            return False
    
    # Test edge cases
    print("\nTesting edge cases:")
    edge_cases = [
        0,  # Epoch start
        -1, # Negative timestamp
        999999999999999999999,  # Very large timestamp
    ]
    
    for timestamp in edge_cases:
        result = engine._format_timestamp_readable(timestamp)
        print(f"  {timestamp} -> {result}")
        # Should not crash and should return a string
        if isinstance(result, str):
            print(f"    Edge case handled OK")
        else:
            print(f"    Edge case ERROR: Expected string, got {type(result)}")
            return False
    
    return True

def test_trade_list_integration():
    """Test integration with actual trade list generation"""
    
    print("\nTESTING TRADE LIST INTEGRATION")
    print("=" * 50)
    
    # Create test data
    n_bars = 100
    base_price = 6300.0
    
    timestamps = np.arange(n_bars, dtype=np.int64) * 60 * 1_000_000_000  # Every minute
    close_prices = base_price + np.cumsum(np.random.normal(0, 0.01, n_bars))
    high_prices = close_prices + np.random.uniform(0.5, 2.0, n_bars)
    low_prices = close_prices - np.random.uniform(0.5, 2.0, n_bars)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    volume = np.random.randint(500, 1500, n_bars).astype(float)
    
    data = {
        'datetime': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    
    print(f"Created test data: {n_bars} bars")
    
    # Create strategy and run backtest
    strategy = TimeWindowVectorizedSingle()
    
    config = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0001
    }
    
    print("Running backtest...")
    pf = strategy.run_vectorized_backtest(data, config)
    
    # Create VBT engine and generate trade list
    try:
        engine = VectorBTEngine()
    except FileNotFoundError:
        engine = VectorBTEngine.__new__(VectorBTEngine)
        engine.config = config
    
    print("Generating trade list...")
    trades_df = engine.generate_trade_list(pf, timestamps=timestamps)
    
    # Check if new columns were added
    expected_columns = ['entryTime_text', 'exitTime_text']
    
    print("Checking for new timestamp columns:")
    for col in expected_columns:
        if col in trades_df.columns:
            print(f"  {col}: FOUND")
            
            # Check format of first few entries
            sample_values = trades_df[col].dropna().head(3)
            for idx, value in sample_values.items():
                if pd.notna(value):
                    print(f"    Sample: {value}")
                    # Verify format
                    if isinstance(value, str) and len(value) == 19:
                        print(f"    Format: OK")
                    else:
                        print(f"    Format: ERROR - Expected 19-char string, got {type(value)} '{value}'")
                        return False
        else:
            print(f"  {col}: MISSING")
            return False
    
    print(f"\nTotal trades generated: {len(trades_df)}")
    print(f"Trade list columns: {list(trades_df.columns)}")
    
    return True

if __name__ == "__main__":
    print("TIMESTAMP ENHANCEMENT TEST")
    print("=" * 60)
    
    success = True
    
    # Test 1: Timestamp formatting
    if not test_timestamp_formatting():
        success = False
    
    # Test 2: Trade list integration
    if not test_trade_list_integration():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: All timestamp enhancement tests passed!")
        print("Trade list will now include human-readable timestamp fields:")
        print("- entryTime_text (yyyy-mm-dd hh:mm:ss)")  
        print("- exitTime_text (yyyy-mm-dd hh:mm:ss)")
    else:
        print("FAILURE: Some timestamp enhancement tests failed")
        print("Review errors above and fix before using")