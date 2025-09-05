#!/usr/bin/env python
"""
Test that the debug CSV export fix works with different formats
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.OMtree_validation import DirectionalValidator

# Test with old format (walkforward_results format)
print("Testing debug CSV export with old walkforward format...")

# Create sample data like old format
old_format_data = {
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
    'time': ['09:55', '09:55', '09:55', '09:55'],
    'prediction': [0.6, -0.3, 0.8, 0.2],  # Probabilities
    'actual': [1.5, -0.8, 2.1, -1.2],     # Actual returns 
    'signal': [1, 0, 1, 1],               # Trade signals
    'pnl': [1.5, 0, 2.1, -1.2]            # P&L values
}

old_df = pd.DataFrame(old_format_data)

# Test with new format (OMtree_results format)
print("\nTesting with new format (OMtree_results)...")

new_format_data = {
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
    'prediction': [1, 0, 1, 1],           # Binary predictions
    'target_value': [1.5, -0.8, 2.1, -1.2],  # Target values
    'actual_profitable': [1, 0, 1, 0]     # Profitability flags
}

new_df = pd.DataFrame(new_format_data)

# Create validator instance
validator = DirectionalValidator(
    data_file='processed_data.csv',
    config_file='OMtree_config.ini', 
    model_type='longonly'
)

# Test both formats
print("\n" + "="*60)
print("Testing OLD FORMAT (walkforward_results style)")
print("="*60)

try:
    validator._export_returns_debug_csv(old_df, verbose=True)
    print("SUCCESS: Old format handled correctly")
except Exception as e:
    print(f"ERROR with old format: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing NEW FORMAT (OMtree_results style)")  
print("="*60)

try:
    validator._export_returns_debug_csv(new_df, verbose=True)
    print("SUCCESS: New format handled correctly")
except Exception as e:
    print(f"ERROR with new format: {e}")
    import traceback
    traceback.print_exc()
    
# Check if files were created
if os.path.exists('results/returns_debug_longonly.csv'):
    debug_df = pd.read_csv('results/returns_debug_longonly.csv')
    print(f"\n✓ Debug CSV created with {len(debug_df)} rows")
    print(f"Columns: {debug_df.columns.tolist()}")
else:
    print("\n✗ Debug CSV was not created")

print("\n" + "="*60)
print("Test complete - debug CSV export should now work at end of walk-forward")
print("="*60)