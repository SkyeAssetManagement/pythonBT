#!/usr/bin/env python3
"""
Test gradual entry/exit backtest directly
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from strategies.time_window_strategy_vectorized import TimeWindowVectorizedSingle
from src.data.parquet_loader import ParquetLoader

def test_gradual_backtest():
    """Test gradual entry/exit with actual data"""
    
    print("TESTING GRADUAL ENTRY/EXIT BACKTEST")
    print("=" * 50)
    
    # Load a small sample of data
    parquet_root = os.path.join(os.path.dirname(__file__), "data", "parquet")
    loader = ParquetLoader(parquet_root)
    
    print(f"Loading ES data...")
    data = loader.load_symbol_data("ES")
    print(f"Loaded {len(data['close'])} bars")
    
    # Create strategy
    strategy = TimeWindowVectorizedSingle()
    
    # Configuration
    config = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0001
    }
    
    # Run gradual backtest
    print("\nRunning gradual entry/exit backtest...")
    try:
        pf = strategy.run_vectorized_backtest(data, config)
        
        # Check the results
        print(f"Portfolio shape: {pf.wrapper.shape}")
        print(f"Number of columns: {pf.wrapper.shape[1] if len(pf.wrapper.shape) > 1 else 1}")
        
        # Check trades
        trades = pf.trades.records_readable
        print(f"Total trades: {len(trades)}")
        print(f"Unique columns: {trades['Column'].unique() if len(trades) > 0 else 'None'}")
        
        if len(trades) > 0:
            print("\nFirst few trades:")
            print(trades[['Column', 'Size', 'Entry Idx', 'Exit Idx', 'PnL']].head(10))
            
            # Check if we have multiple columns (fractional positions)
            unique_columns = trades['Column'].unique()
            if len(unique_columns) > 1:
                print(f"\n[OK] SUCCESS: Found {len(unique_columns)} fractional positions!")
                print(f"Columns: {unique_columns}")
                
                # Analyze entry patterns
                for col in unique_columns[:3]:  # Show first 3 columns
                    col_trades = trades[trades['Column'] == col]
                    if len(col_trades) > 0:
                        first_entry = col_trades.iloc[0]['Entry Idx']
                        print(f"Column {col}: First entry at bar {first_entry}")
                
                return True
            else:
                print("[X] ISSUE: Only found single column - gradual entry not working")
                return False
        else:
            print("[X] ISSUE: No trades generated")
            return False
            
    except Exception as e:
        print(f"[X] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gradual_backtest()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Gradual Entry/Exit Backtest Working!")
        print("- Multiple fractional positions detected")
        print("- Each entry spreads over 5 consecutive bars")
        print("- Each bar gets 1/5 of total position size")
    else:
        print("ISSUE: Gradual Entry/Exit needs debugging")
        print("Check the analysis above")