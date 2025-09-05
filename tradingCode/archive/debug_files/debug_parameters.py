#!/usr/bin/env python3
"""
Debug which parameter combination produces 17:31 entries with 30min hold
"""

import numpy as np
import pandas as pd
from datetime import datetime
from strategies.time_window_strategy_vectorized import TimeWindowVectorizedStrategy

def debug_parameter_selection():
    """Investigate which parameters match the tradelist.csv results"""
    
    print("DEBUGGING PARAMETER SELECTION")
    print("=" * 50)
    
    # Load strategy and get all parameter combinations
    strategy = TimeWindowVectorizedStrategy()
    combinations = strategy.get_parameter_combinations()
    
    print(f"Total parameter combinations: {len(combinations)}")
    print()
    
    # Look for combinations that could produce 30min hold + specific entry times
    print("Analyzing parameter combinations:")
    print("-" * 40)
    
    for i, params in enumerate(combinations):
        entry_time = params['entry_time']
        hold_time = params['hold_time']
        direction = params['direction']
        
        print(f"Combination {i+1:2d}: Entry={entry_time}, Hold={hold_time}min, Direction={direction}")
        
        # Check if this matches our mystery parameters
        if hold_time == 30:
            print(f"  *** MATCHES 30min hold time from tradelist.csv")
    
    print()
    print("MYSTERY ANALYSIS:")
    print("- tradelist.csv shows 17:31 entry times")
    print("- tradelist.csv shows 30min hold times")
    print("- But 17:31 is NOT in the entry_time values!")
    print()
    
    # Check entry_time values
    entry_times = strategy.parameters['entry_time']['values']
    print(f"Available entry_time values: {entry_times}")
    print("17:30 or 17:31 is NOT in this list!")
    print()
    
    # Check if there's a timezone or data processing issue
    print("POSSIBLE EXPLANATIONS:")
    print("1. Timezone conversion issue (UTC vs local time)")
    print("2. Data timestamp issue")
    print("3. The actual entry time was different but got processed as 17:31")
    print("4. The best combination selected had 14:30 entry, but timezone shifted to 17:30")
    
    # Check timezone hypothesis
    print()
    print("TIMEZONE HYPOTHESIS CHECK:")
    for entry_time in entry_times:
        hour, minute = map(int, entry_time.split(':'))
        print(f"  {entry_time} + 3 hours (EST->UTC) = {hour+3:02d}:{minute:02d}")
        if hour + 3 == 17 and minute == 30:
            print(f"    *** This could explain 17:30 entries!")
    
    return combinations

def check_tradelist_timing():
    """Check the actual timestamps in tradelist.csv"""
    
    print("\nTRADELIST TIMING ANALYSIS:")
    print("=" * 30)
    
    try:
        # Read the tradelist
        import pandas as pd
        tradelist = pd.read_csv("results/tradelist.csv")
        
        if 'entryTime_text' in tradelist.columns:
            print("Entry times from tradelist:")
            for i, entry_time in enumerate(tradelist['entryTime_text'].head()):
                print(f"  Trade {i}: {entry_time}")
            
            # Parse first entry time
            first_entry = tradelist['entryTime_text'].iloc[0]
            print(f"\nFirst entry: {first_entry}")
            
            # Extract time component
            time_part = first_entry.split(' ')[1]  # Get "17:31:00"
            hour, minute, second = time_part.split(':')
            print(f"Time: {hour}:{minute}:{second}")
            
            # Check if this could be a timezone converted value
            original_hour = int(hour) - 3  # Subtract 3 hours (EST to UTC)
            if original_hour < 0:
                original_hour += 24
            print(f"If this was UTC and original was EST: {original_hour:02d}:{minute}:{second}")
            
    except Exception as e:
        print(f"Error reading tradelist: {e}")

if __name__ == "__main__":
    combinations = debug_parameter_selection()
    check_tradelist_timing()
    
    print("\n" + "=" * 50)
    print("CONCLUSION:")
    print("Need to investigate timezone handling and best combination selection")
    print("The 17:31 entries are likely timezone-converted from 14:30 EST entries")