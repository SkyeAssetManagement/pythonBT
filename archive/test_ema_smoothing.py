"""
Test that EMA smoothing alpha is correctly calculated from lookback window
"""

import numpy as np
import pandas as pd
from OMtree_preprocessing import DataPreprocessor
import configparser

print("Testing EMA Smoothing Alpha Calculation")
print("="*60)

# Test different lookback windows
lookback_windows = [10, 20, 30, 50, 100]

for lookback in lookback_windows:
    # Create config
    config = configparser.ConfigParser()
    config['data'] = {
        'csv_file': 'test.csv',
        'feature_columns': 'test_feature',
        'target_column': 'test_target',
        'selected_features': 'test_feature'
    }
    config['preprocessing'] = {
        'normalize_features': 'true',
        'normalize_target': 'false',
        'normalization_method': 'IQR',
        'vol_window': '50',
        'smoothing_type': 'exponential',
        'recent_iqr_lookback': str(lookback),
        'percentile_upper': '75',
        'percentile_lower': '25',
        'winsorize_enabled': 'false',
        'iqr_weighting_enabled': 'false',
        'avs_slow_window': '60',
        'avs_fast_window': '20',
        'vol_signal_window': '0'
    }
    
    # Save temp config
    with open('temp_config.ini', 'w') as f:
        config.write(f)
    
    # Create preprocessor and check alpha
    preprocessor = DataPreprocessor('temp_config.ini')
    calculated_alpha = preprocessor.smoothing_alpha
    expected_alpha = 2.0 / (lookback + 1)
    
    print(f"\nLookback window: {lookback}")
    print(f"  Expected alpha (2/(n+1)): {expected_alpha:.4f}")
    print(f"  Calculated alpha:         {calculated_alpha:.4f}")
    print(f"  Match: {'YES' if abs(calculated_alpha - expected_alpha) < 0.0001 else 'NO'}")
    
    # Verify it matches
    assert abs(calculated_alpha - expected_alpha) < 0.0001, f"Alpha mismatch for lookback {lookback}"

# Clean up
import os
os.remove('temp_config.ini')

print("\n" + "="*60)
print("All tests passed! Alpha is correctly calculated as 2/(n+1)")
print("="*60)