#!/usr/bin/env python3
"""
Properly debug and fix the timezone conversion
"""

from datetime import datetime, timezone

def analyze_conversion():
    """Analyze the proper conversion needed"""
    
    print("PROPER TIMEZONE CONVERSION ANALYSIS")
    print("=" * 50)
    
    timestamp_ns = 1751362260000000000
    timestamp_sec = timestamp_ns / 1_000_000_000
    
    print(f"Timestamp (ns): {timestamp_ns}")
    print(f"Timestamp (sec): {timestamp_sec}")
    print()
    
    # What we currently get
    current = datetime.fromtimestamp(timestamp_sec)
    print(f"Current result: {current} (system local time)")
    
    # What we want: 14:30 EST entries
    print()
    print("DESIRED OUTCOME:")
    print("- Strategy entry_time: 14:30")
    print("- Actual entries: 14:31 (1 minute after entry_time)")
    print("- Hold time: 30 minutes")
    print("- Exit time: 15:01")
    print()
    
    # The timestamp represents 09:31 UTC
    utc_time = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
    print(f"Timestamp as UTC: {utc_time}")
    
    # We want it to display as 14:31 EST
    # That means we need: UTC 09:31 -> EST 14:31
    # So we need to ADD 5 hours to the UTC time, then display without timezone
    
    # Method 1: Convert UTC to EST properly
    utc_hour = utc_time.hour
    est_hour = utc_hour + 5  # EST is UTC-5, but we want to show local EST time
    
    print(f"UTC hour: {utc_hour}")
    print(f"EST hour (UTC+5): {est_hour}")
    
    # Method 2: Simple offset
    est_timestamp = timestamp_sec + (5 * 3600)  # Add 5 hours in seconds
    est_datetime = datetime.utcfromtimestamp(est_timestamp)  # Use UTC to avoid system timezone
    
    print(f"Method 2 result: {est_datetime}")
    
    return est_datetime

if __name__ == "__main__":
    analyze_conversion()