"""
Test walkforward with DTSnnData.csv to diagnose error
"""

import pandas as pd
import numpy as np
import configparser
from OMtree_preprocessing import DataPreprocessor
from OMtree_validation import DirectionalValidator
import traceback

print("=" * 80)
print("TESTING WALKFORWARD WITH DTSnnData.csv")
print("=" * 80)

# Load DTSnnData to check its structure
df = pd.read_csv('DTSnnData.csv')
print(f"\nDTSnnData.csv shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"First few rows:")
print(df.head(3))

# Create test config for DTSnnData
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

# Update config for DTSnnData
config['data']['csv_file'] = 'DTSnnData.csv'
# Note: config has target_column = RetForward and selected_features = Overnight
# which should exist in DTSnnData

test_config = 'test_dtsnn.ini'
with open(test_config, 'w') as f:
    config.write(f)

print("\n" + "=" * 40)
print("TEST 1: PREPROCESSING")
print("=" * 40)

try:
    preprocessor = DataPreprocessor(test_config)
    processed = preprocessor.process_data(df)
    print(f"Processed shape: {processed.shape}")
    print(f"Processed columns: {list(processed.columns)}")
    
    # Check for expected columns
    print("\nChecking for expected columns:")
    print(f"  RetForward in data: {'RetForward' in df.columns}")
    print(f"  Overnight in data: {'Overnight' in df.columns}")
    print(f"  RetForward_vol_adj in processed: {'RetForward_vol_adj' in processed.columns}")
    print(f"  Overnight_vol_adj in processed: {'Overnight_vol_adj' in processed.columns}")
    
except Exception as e:
    print(f"[ERROR] Preprocessing failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 40)
print("TEST 2: VALIDATION SETUP")
print("=" * 40)

try:
    validator = DirectionalValidator(test_config)
    print(f"Validator settings:")
    print(f"  Selected features: {validator.selected_features}")
    print(f"  Target column: {validator.target_column}")
    print(f"  Train size: {validator.train_size}")
    print(f"  Test size: {validator.test_size}")
    
    # Try to load and prepare data
    data = validator.load_and_prepare_data()
    print(f"\nPrepared data shape: {data.shape}")
    print(f"Prepared columns: {list(data.columns)[:10]}...")
    
except Exception as e:
    print(f"[ERROR] Validation setup failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 40)
print("TEST 3: SINGLE VALIDATION STEP")
print("=" * 40)

try:
    if 'data' in locals():
        # Try a single validation step
        test_end_idx = min(1200, len(data) - 50)
        print(f"Attempting split at index {test_end_idx}")
        
        result = validator.get_train_test_split(data, test_end_idx)
        
        if result[0] is not None:
            train_X, train_y, test_X, test_y, test_indices = result
            print(f"[SUCCESS] Split successful:")
            print(f"  Train X shape: {train_X.shape if hasattr(train_X, 'shape') else len(train_X)}")
            print(f"  Train y shape: {train_y.shape if hasattr(train_y, 'shape') else len(train_y)}")
            print(f"  Test X shape: {test_X.shape if hasattr(test_X, 'shape') else len(test_X)}")
            
            # Try to train a model
            from OMtree_model import DirectionalTreeEnsemble
            model = DirectionalTreeEnsemble(test_config, verbose=False)
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            print(f"  Predictions: {predictions[:5]}")
            
        else:
            print("[ERROR] Split returned None")
            
except Exception as e:
    print(f"[ERROR] Validation step failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 40)
print("TEST 4: FULL WALKFORWARD")
print("=" * 40)

try:
    # Run a short walkforward
    config['validation']['train_size'] = '100'
    config['validation']['test_size'] = '20'
    config['validation']['step_size'] = '20'
    
    with open(test_config, 'w') as f:
        config.write(f)
    
    # Create new validator with updated config
    validator = DirectionalValidator(test_config)
    
    # Run validation with verbose output
    results = validator.run_validation(verbose=True)
    
    if results is not None and not results.empty:
        print(f"\n[SUCCESS] Walkforward completed with {len(results)} steps")
        print(f"Results shape: {results.shape}")
        print(f"Results columns: {list(results.columns)}")
        print(f"First few results:")
        print(results.head())
    else:
        print("\n[ERROR] Walkforward returned no results")
        
except Exception as e:
    print(f"[ERROR] Full walkforward failed: {e}")
    traceback.print_exc()

# Clean up
import os
if os.path.exists(test_config):
    os.remove(test_config)
    
print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)