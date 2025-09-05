#!/usr/bin/env python3
"""
Debug timestamp conversion to understand the timezone issue
"""

from datetime import datetime, timezone
import pytz

def debug_timestamp_conversion():
    """Debug what the timestamps actually represent"""
    
    print("TIMESTAMP CONVERSION DEBUG")
    print("=" * 50)
    
    # Sample timestamp from tradelist
    timestamp_ns = 1751362260000000000
    timestamp_sec = timestamp_ns / 1_000_000_000
    
    print(f"Original timestamp (ns): {timestamp_ns}")
    print(f"Converted to seconds: {timestamp_sec}")
    print()
    
    # Different interpretations
    print("Different interpretations:")
    print("-" * 30)
    
    # 1. As UTC
    utc_dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
    print(f"1. As UTC: {utc_dt}")
    
    # 2. As local system time
    local_dt = datetime.fromtimestamp(timestamp_sec)
    print(f"2. As local: {local_dt}")
    
    # 3. Convert UTC to EST
    eastern_tz = pytz.timezone('US/Eastern')
    est_dt = utc_dt.astimezone(eastern_tz)
    print(f"3. UTC -> EST: {est_dt}")
    
    # 4. Treat as EST originally
    est_native = eastern_tz.localize(datetime.fromtimestamp(timestamp_sec))
    print(f"4. As EST native: {est_native}")
    
    print()
    print("ANALYSIS:")
    print("We want 14:30 entries (EST), but we're seeing different times.")
    print("The issue might be that timestamps are already in EST, not UTC.")
    
    # Test: if timestamp is already EST, just format it directly
    print()
    print("HYPOTHESIS TEST:")
    print("If timestamps are already in EST/local time:")
    direct_dt = datetime.fromtimestamp(timestamp_sec)
    print(f"Direct formatting: {direct_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return timestamp_sec

if __name__ == "__main__":
    debug_timestamp_conversion()