"""
Test that preprocessing handles empty/missing config values gracefully
"""

import configparser
from OMtree_preprocessing import DataPreprocessor

print("Testing Config Robustness")
print("="*60)

# Test 1: Config with empty values
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
    'recent_iqr_lookback': '20',
    'percentile_upper': '75',
    'percentile_lower': '25',
    'winsorize_enabled': 'false',
    'winsorize_percentile': '',  # Empty value
    'iqr_weighting_enabled': 'false',
    'iqr_decay_factor': '',  # Empty value
    'avs_slow_window': '',  # Empty value
    'avs_fast_window': '',  # Empty value
    'vol_signal_window': '0'
}

# Save temp config
with open('temp_config_test.ini', 'w') as f:
    config.write(f)

try:
    # Create preprocessor - should handle empty values
    preprocessor = DataPreprocessor('temp_config_test.ini')
    
    print("Empty value handling:")
    print(f"  iqr_decay_factor (empty -> 0.98): {preprocessor.iqr_decay_factor}")
    print(f"  winsorize_percentile (empty -> 5.0): {preprocessor.winsorize_percentile}")
    print(f"  avs_slow_window (empty -> 60): {preprocessor.avs_slow_window}")
    print(f"  avs_fast_window (empty -> 20): {preprocessor.avs_fast_window}")
    print("[PASS] Successfully handled empty values")
    
except Exception as e:
    print(f"[FAIL] Failed with error: {e}")

print("\n" + "-"*60)

# Test 2: Config with missing keys
config2 = configparser.ConfigParser()
config2['data'] = {
    'csv_file': 'test.csv',
    'feature_columns': 'test_feature',
    'target_column': 'test_target',
    'selected_features': 'test_feature'
}
config2['preprocessing'] = {
    'normalize_features': 'true',
    'normalize_target': 'false',
    'normalization_method': 'IQR',
    'vol_window': '50',
    'smoothing_type': 'exponential',
    'recent_iqr_lookback': '20',
    'percentile_upper': '75',
    'percentile_lower': '25',
    'vol_signal_window': '0'
    # Missing: winsorize_enabled, iqr_weighting_enabled, etc.
}

with open('temp_config_test2.ini', 'w') as f:
    config2.write(f)

try:
    preprocessor2 = DataPreprocessor('temp_config_test2.ini')
    
    print("Missing key handling:")
    print(f"  winsorize_enabled (missing -> False): {preprocessor2.winsorize_enabled}")
    print(f"  iqr_weighting_enabled (missing -> False): {preprocessor2.iqr_weighting_enabled}")
    print(f"  iqr_decay_factor (missing -> 0.98): {preprocessor2.iqr_decay_factor}")
    print("[PASS] Successfully handled missing keys")
    
except Exception as e:
    print(f"[FAIL] Failed with error: {e}")

# Clean up
import os
os.remove('temp_config_test.ini')
os.remove('temp_config_test2.ini')

print("\n" + "="*60)
print("Config robustness test completed successfully!")
print("="*60)