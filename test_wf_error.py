#!/usr/bin/env python
"""
Test to identify the error at end of walk-forward test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try running a minimal walk-forward test
from src.OMtree_validation import DirectionalWalkForwardValidator

try:
    print("Testing walk-forward completion...")
    
    # Create validator instance
    validator = DirectionalWalkForwardValidator(
        data_file='processed_data.csv',
        config_file='OMtree_config.ini',
        model_type='longonly'
    )
    
    # Check if the debug export method exists and would work
    import pandas as pd
    import numpy as np
    
    # Create a sample dataframe like what would be passed
    sample_data = {
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'prediction': [1, 0, 1],
        'target_value': [0.5, 0, -0.3],
        'actual_profitable': [1, 0, 0],
        'probability': [0.6, 0.4, 0.55]
    }
    
    pred_df = pd.DataFrame(sample_data)
    
    # Try calling the debug export
    print("Calling _export_returns_debug_csv...")
    validator._export_returns_debug_csv(pred_df, verbose=True)
    print("Success! No error in debug CSV export")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
    
print("\nNow let's check the actual walk-forward completion code...")

# Check if there's an issue with the validation run method
import inspect

# Get the run_validation source
validation_module = sys.modules['src.OMtree_validation']
source = inspect.getsource(validation_module.DirectionalWalkForwardValidator.run_validation)

# Find where errors might occur
if '_export_returns_debug_csv' in source:
    print("✓ Debug CSV export is called in run_validation")
    
    # Check for potential issues
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if '_export_returns_debug_csv' in line:
            print(f"Line {i}: {line.strip()}")
            # Check context around this line
            if i > 0:
                print(f"  Previous: {lines[i-1].strip()}")
            if i < len(lines) - 1:
                print(f"  Next: {lines[i+1].strip()}")