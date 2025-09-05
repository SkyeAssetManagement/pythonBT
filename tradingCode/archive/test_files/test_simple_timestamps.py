#!/usr/bin/env python3
"""
Simple test for timestamp functionality
"""

from datetime import datetime

def test_timestamp_conversion():
    """Test timestamp conversion logic directly"""
    
    print("TESTING TIMESTAMP CONVERSION")
    print("=" * 40)
    
    # Test the exact logic from the VBT engine
    def format_timestamp_readable(timestamp_ns):
        try:
            # Convert nanoseconds to seconds
            timestamp_sec = timestamp_ns / 1_000_000_000
            
            # Create datetime object and format
            dt_obj = datetime.fromtimestamp(timestamp_sec)
            return dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            
        except (ValueError, OSError, OverflowError) as e:
            # Handle edge cases like invalid timestamps
            return f"Invalid timestamp ({timestamp_ns})"
    
    # Test with actual timestamps from the trade list
    test_timestamps = [
        1751362260000000000,  # From actual trade list
        1751364060000000000,  # From actual trade list
        1751448660000000000,  # From actual trade list
        1751450460000000000,  # From actual trade list
    ]
    
    print("Converting sample timestamps:")
    for timestamp_ns in test_timestamps:
        result = format_timestamp_readable(timestamp_ns)
        print(f"  {timestamp_ns} -> {result}")
        
        # Verify format
        if len(result) == 19 and result[4] == '-' and result[7] == '-' and result[10] == ' ':
            print(f"    Format: CORRECT")
        else:
            print(f"    Format: ERROR")
            return False
    
    print("\nTesting edge cases:")
    edge_cases = [0, -1, 9999999999999999999]
    for timestamp in edge_cases:
        result = format_timestamp_readable(timestamp)
        print(f"  {timestamp} -> {result}")
    
    return True

if __name__ == "__main__":
    success = test_timestamp_conversion()
    
    if success:
        print("\nSUCCESS: Timestamp conversion logic works correctly")
        print("Format: yyyy-mm-dd hh:mm:ss")
    else:
        print("\nFAILURE: Timestamp conversion has issues")