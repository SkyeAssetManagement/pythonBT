"""
Test detrending combined with normalization
"""

import pandas as pd
import numpy as np
import configparser
from OMtree_preprocessing import DataPreprocessor

print("=" * 80)
print("TESTING DETRENDING + NORMALIZATION")
print("=" * 80)

# Load sample data
df = pd.read_csv('DTSmlDATA7x7.csv')

# Test configuration combinations
test_configs = [
    {'detrend': 'false', 'normalize': 'false', 'method': 'IQR'},
    {'detrend': 'true', 'normalize': 'false', 'method': 'IQR'},
    {'detrend': 'false', 'normalize': 'true', 'method': 'IQR'},
    {'detrend': 'true', 'normalize': 'true', 'method': 'IQR'},
    {'detrend': 'true', 'normalize': 'true', 'method': 'AVS'},
]

config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

test_config = 'test_combinations.ini'

for i, test in enumerate(test_configs, 1):
    print(f"\nTest {i}: Detrend={test['detrend']}, Normalize={test['normalize']}, Method={test['method']}")
    print("-" * 60)
    
    # Set config
    config['preprocessing']['detrend_features'] = test['detrend']
    config['preprocessing']['normalize_features'] = test['normalize']
    config['preprocessing']['normalization_method'] = test['method']
    
    with open(test_config, 'w') as f:
        config.write(f)
    
    # Process data
    preprocessor = DataPreprocessor(test_config)
    processed = preprocessor.process_data(df)
    
    # Check what columns were created
    detrend_cols = [c for c in processed.columns if '_detrend' in c]
    vol_adj_cols = [c for c in processed.columns if '_vol_adj' in c and 'detrend' in c]
    
    print(f"  Detrended columns: {len(detrend_cols)}")
    print(f"  Vol-adjusted detrended columns: {len(vol_adj_cols)}")
    
    # Sample a feature to show the processing chain
    if test['detrend'] == 'true':
        feature = 'Ret_0-1hr'
        print(f"\n  Processing chain for {feature}:")
        
        # Original
        orig_val = df[feature].iloc[200]
        print(f"    Original value: {orig_val:.4f}")
        
        # After detrending
        if f'{feature}_detrend' in processed.columns:
            detrend_val = processed[f'{feature}_detrend'].iloc[200]
            print(f"    After detrending: {detrend_val:.4f}")
            
            # After normalization (if enabled)
            if test['normalize'] == 'true':
                if f'{feature}_detrend_vol_adj' in processed.columns:
                    norm_val = processed[f'{feature}_detrend_vol_adj'].iloc[200]
                    print(f"    After normalization: {norm_val:.4f}")

# Clean up
import os
if os.path.exists(test_config):
    os.remove(test_config)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nProcessing Pipeline:")
print("1. If detrend_features=true: Subtract rolling median from features")
print("2. If normalize_features=true: Apply volatility normalization (IQR or AVS)")
print("3. Features get suffixes: _detrend and/or _vol_adj based on processing")
print("4. Target is never detrended, only normalized if normalize_target=true")
print("\nKey Points:")
print("- Detrending removes bias/trend while preserving volatility patterns")
print("- Normalization scales volatility to make data more stationary")
print("- Both can be used together for maximum stationarity")
print("- Order: Detrend first, then normalize the detrended values")