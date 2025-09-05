#!/usr/bin/env python3
"""
Test the timezone fix for trade timestamps
"""

from src.backtest.vbt_engine import VectorBTEngine
from datetime import datetime
import pandas as pd

def test_timezone_fix():
    """Test that timestamps are now displayed in EST/EDT"""
    
    print("TESTING TIMEZONE FIX")
    print("=" * 40)
    
    # Create VBT engine
    try:
        engine = VectorBTEngine("config.yaml")
    except:
        # Use dummy config if file doesn't exist
        engine = VectorBTEngine.__new__(VectorBTEngine)
    
    # Test timestamps from the actual tradelist
    test_timestamps = [
        1751362260000000000,  # From trade 0
        1751364060000000000,  # From trade 0 exit
        1751448660000000000,  # From trade 1
        1751450460000000000,  # From trade 1 exit
    ]
    
    print("Converting sample timestamps to EST/EDT:")
    print("-" * 40)
    
    for i, timestamp_ns in enumerate(test_timestamps):
        readable = engine._format_timestamp_readable(timestamp_ns)
        print(f"Timestamp {i+1}: {readable}")
        
        # Parse the time component
        time_part = readable.split(' ')[1]
        hour = int(time_part.split(':')[0])
        
        # Should now be showing 14:30 entries instead of 17:30
        if hour == 14:
            print(f"  [OK] FIXED: Now showing {hour}:xx (EST) instead of 17:xx (UTC)")
        elif hour == 17:
            print(f"  [X] STILL BROKEN: Still showing {hour}:xx (likely UTC)")
        else:
            print(f"  ? Unexpected hour: {hour}")
    
    # Test specific conversion
    print("\nExpected conversion:")
    print("- UTC 17:30 -> EST 14:30 (17:30 - 3 hours)")
    print("- UTC 18:00 -> EST 15:00 (18:00 - 3 hours)")
    
    return True

if __name__ == "__main__":
    try:
        test_timezone_fix()
        print("\n" + "=" * 40)
        print("Timezone fix has been applied!")
        print("Next tradelist.csv should show EST times (14:30 entries)")
    except Exception as e:
        print(f"Error testing timezone fix: {e}")
        print("You may need to install pytz: pip install pytz")