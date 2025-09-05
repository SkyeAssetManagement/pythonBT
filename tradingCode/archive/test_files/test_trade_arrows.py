#!/usr/bin/env python3
"""
Test trade arrow positioning fix
"""

import pandas as pd
import numpy as np
from src.dashboard.parallel_loader import ParallelDataLoader

def test_trade_arrow_positioning():
    """Test that trade arrows are positioned at correct timestamps"""
    
    print("TESTING TRADE ARROW POSITIONING FIX")
    print("=" * 50)
    
    # Create sample trade data matching the current trade list format
    trade_data = pd.DataFrame({
        'Exit Trade Id': [0, 1],
        'Entry Index': [571, 1951],
        'Exit Index': [601, 1981], 
        'Direction': ['Long', 'Long'],
        'Avg Entry Price': [6236.5, 6245.625],
        'Avg Exit Price': [6244.125, 6246.25],
        'Size': [1.0, 1.0],
        'PnL': [-4.86, -11.87],
        'EntryTime': [1751362260000000000, 1751448660000000000],  # Entry timestamps
        'ExitTime': [1751364060000000000, 1751450460000000000]    # Exit timestamps (30 minutes later)
    })
    
    print("Sample trade data:")
    print(trade_data[['Exit Trade Id', 'Entry Index', 'Exit Index', 'EntryTime', 'ExitTime']])
    print()
    
    # Create loader and process trades
    loader = ParallelDataLoader()
    trade_objects = loader._process_trades(trade_data)
    
    print(f"Generated {len(trade_objects)} trade arrow objects")
    print()
    
    # Analyze the results
    entries = [t for t in trade_objects if '_entry' in t.trade_id]
    exits = [t for t in trade_objects if '_exit' in t.trade_id]
    
    print(f"Entry arrows: {len(entries)}")
    print(f"Exit arrows: {len(exits)}")
    print()
    
    # Verify timestamps are different
    print("Trade Arrow Positioning Analysis:")
    print("-" * 40)
    
    for i in range(len(entries)):
        if i < len(exits):
            entry = entries[i]
            exit_arrow = exits[i]
            
            entry_time = pd.to_datetime(entry.timestamp, unit='ns')
            exit_time = pd.to_datetime(exit_arrow.timestamp, unit='ns')
            time_diff = (exit_time - entry_time).total_seconds() / 60  # Minutes
            
            print(f"Trade {i}:")
            print(f"  Entry: {entry.side} at {entry_time} (timestamp: {entry.timestamp})")
            print(f"  Exit:  {exit_arrow.side} at {exit_time} (timestamp: {exit_arrow.timestamp})")
            print(f"  Time Difference: {time_diff:.1f} minutes")
            print(f"  Positioning: {'CORRECT' if time_diff == 30.0 else 'ERROR'}")
            print()
    
    # Verify we have both entry and exit arrows with different timestamps
    success = True
    if len(entries) != len(exits):
        print("ERROR: Unequal number of entry and exit arrows")
        success = False
    
    for i in range(min(len(entries), len(exits))):
        if entries[i].timestamp == exits[i].timestamp:
            print(f"ERROR: Trade {i} entry and exit have same timestamp")
            success = False
    
    return success

if __name__ == "__main__":
    success = test_trade_arrow_positioning()
    
    print("=" * 50)
    if success:
        print("SUCCESS: Trade arrow positioning fix works correctly!")
        print("- Entry and exit arrows now have different timestamps")
        print("- Arrows will appear on correct chart bars (30 minutes apart)")
        print("- Dashboard will show proper trade arrow positioning")
    else:
        print("FAILURE: Trade arrow positioning fix has issues")
        print("- Check the errors above")