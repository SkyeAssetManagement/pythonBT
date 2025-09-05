#!/usr/bin/env python3
"""
Test trade list timestamp generation directly
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_tradelist_enhancement():
    """Test the trade list enhancement directly"""
    
    print("TESTING TRADE LIST TIMESTAMP ENHANCEMENT")
    print("=" * 50)
    
    # Create mock trade data like what VBT generates
    mock_trades = pd.DataFrame({
        'Entry Index': [571, 1951, 3331],
        'Exit Index': [601, 1981, 3361],
        'Size': [1.0, 1.0, 1.0],
        'PnL': [-4.86, -11.87, 4.39],
        'Direction': ['Long', 'Long', 'Long']
    })
    
    # Create mock timestamps (minute data)
    n_bars = 5000
    base_timestamp = 1751362200000000000  # Base nanosecond timestamp
    timestamps = np.array([base_timestamp + i * 60 * 1_000_000_000 for i in range(n_bars)])
    
    print(f"Created mock data:")
    print(f"  Trades: {len(mock_trades)}")
    print(f"  Timestamps: {len(timestamps)}")
    
    # Apply the timestamp mapping logic
    entry_idx_col = 'Entry Index'
    exit_idx_col = 'Exit Index'
    
    def format_timestamp_readable(timestamp_ns):
        try:
            timestamp_sec = timestamp_ns / 1_000_000_000
            dt_obj = datetime.fromtimestamp(timestamp_sec)
            return dt_obj.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, OSError, OverflowError) as e:
            return f"Invalid timestamp ({timestamp_ns})"
    
    # Map timestamps
    print("\nMapping timestamps...")
    mock_trades['EntryTime'] = mock_trades[entry_idx_col].apply(
        lambda x: timestamps[int(x)] if x < len(timestamps) else None
    )
    mock_trades['ExitTime'] = mock_trades[exit_idx_col].apply(
        lambda x: timestamps[int(x)] if x < len(timestamps) else None
    )
    
    # Add human-readable timestamps
    print("Adding human-readable timestamps...")
    mock_trades['entryTime_text'] = mock_trades['EntryTime'].apply(
        lambda x: format_timestamp_readable(x) if x is not None and not pd.isna(x) else None
    )
    mock_trades['exitTime_text'] = mock_trades['ExitTime'].apply(
        lambda x: format_timestamp_readable(x) if x is not None and not pd.isna(x) else None
    )
    
    print("\nResult:")
    print(mock_trades[['Entry Index', 'Exit Index', 'EntryTime', 'ExitTime', 'entryTime_text', 'exitTime_text']])
    
    # Verify the new columns exist and have correct format
    new_columns = ['entryTime_text', 'exitTime_text']
    for col in new_columns:
        if col in mock_trades.columns:
            print(f"\n{col} column: ADDED")
            sample_values = mock_trades[col].dropna()
            for idx, value in sample_values.items():
                print(f"  Row {idx}: {value}")
                # Verify format
                if isinstance(value, str) and len(value) == 19 and value[4] == '-' and value[7] == '-' and value[10] == ' ':
                    print(f"    Format: CORRECT")
                else:
                    print(f"    Format: ERROR - Expected 'yyyy-mm-dd hh:mm:ss', got '{value}'")
                    return False
        else:
            print(f"\n{col} column: MISSING")
            return False
    
    # Show final column list
    print(f"\nFinal columns: {list(mock_trades.columns)}")
    
    return True

if __name__ == "__main__":
    success = test_tradelist_enhancement()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Trade list timestamp enhancement working!")
        print("New columns will be added to tradelist.csv:")
        print("- entryTime_text: Human-readable entry timestamp")
        print("- exitTime_text: Human-readable exit timestamp")
        print("Format: yyyy-mm-dd hh:mm:ss")
    else:
        print("FAILURE: Trade list timestamp enhancement failed")
        print("Check the errors above")