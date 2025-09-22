#!/usr/bin/env python3
"""
Test complete DateTime functionality including CSV loading
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Test CSV date/time parsing
def test_csv_datetime_parsing():
    """Test that Date and Time columns are properly combined"""
    print("=" * 80)
    print("Testing CSV DateTime Parsing")
    print("=" * 80)

    # Simulate CSV data as it would be loaded
    csv_data = {
        'Date': ['2021/01/03', '2021/01/03', '2021/01/03', '2021/01/04', '2021/01/04'],
        'Time': ['17:01:00', '17:02:00', '17:03:00', '09:30:00', '09:31:00'],
        'Open': [4300.75, 4302.50, 4301.00, 4305.00, 4306.00],
        'High': [4307.50, 4304.25, 4303.00, 4307.00, 4308.00],
        'Low': [4300.75, 4300.75, 4300.00, 4304.00, 4305.00],
        'Close': [4302.75, 4301.00, 4302.00, 4306.00, 4307.00]
    }

    df = pd.DataFrame(csv_data)
    print(f"\n1. Original CSV data columns: {df.columns.tolist()}")
    print(f"   Date sample: {df['Date'].iloc[0]}")
    print(f"   Time sample: {df['Time'].iloc[0]}")

    # Combine Date and Time as our fix does
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    print(f"\n2. Combined DateTime:")
    print(f"   First DateTime: {df['DateTime'].iloc[0]}")
    print(f"   DateTime type: {df['DateTime'].dtype}")

    # Check formatting
    print(f"\n3. DateTime formatting tests:")
    for i in range(min(3, len(df))):
        dt = df['DateTime'].iloc[i]
        formatted = dt.strftime('%y-%m-%d %H:%M:%S')
        print(f"   Row {i}: {dt} -> '{formatted}'")

    # Test x-axis label formatting (hour:minute:second)
    print(f"\n4. X-axis label format (h:mm:ss):")
    for i in range(min(3, len(df))):
        dt = df['DateTime'].iloc[i]
        hour = dt.hour
        time_str = f"{hour}:{dt.strftime('%M:%S')}"
        print(f"   Row {i}: {time_str}")

    print("\n" + "=" * 80)
    print("DateTime parsing test complete!")
    print("Expected: Times like '17:01:00' and '9:30:00' are preserved")
    print("=" * 80)

if __name__ == "__main__":
    test_csv_datetime_parsing()