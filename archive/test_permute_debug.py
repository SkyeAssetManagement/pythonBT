"""
Debug test for PermuteAlpha functionality
"""

import configparser
import tempfile
import os
import pandas as pd
import numpy as np
from OMtree_validation import DirectionalValidator

print("="*60)
print("PERMUTE ALPHA DEBUG TEST")
print("="*60)

# Create a test config
config = configparser.ConfigParser()
config.read('OMtree_config.ini')

# Create temp config for testing
temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
temp_path = temp_config.name

# Set specific test parameters
config['data']['target_column'] = 'Ret_fwd3hr'
config['model']['model_type'] = 'longonly'

# Set a specific ticker and hour if they exist
if 'ticker_filter' in config['data']:
    config['data']['ticker_filter'] = 'ES'
if 'hour_filter' in config['data']:
    config['data']['hour_filter'] = '10'

# Write config
config.write(temp_config)
temp_config.close()

print(f"\nTest config created: {temp_path}")
print(f"  Target: {config['data']['target_column']}")
print(f"  Model type: {config['model']['model_type']}")
print(f"  Ticker filter: {config['data'].get('ticker_filter', 'None')}")
print(f"  Hour filter: {config['data'].get('hour_filter', 'None')}")

try:
    # Create validator
    print("\n1. Creating validator...")
    validator = DirectionalValidator(temp_path)
    
    # Load data
    print("2. Loading and preparing data...")
    data = validator.load_and_prepare_data()
    
    if data is None:
        print("ERROR: No data returned from load_and_prepare_data()")
    else:
        print(f"   Data shape: {data.shape}")
        print(f"   Columns: {list(data.columns)[:10]}...")
        print(f"   Feature columns: {validator.processed_feature_columns}")
        print(f"   Target column: {validator.processed_target_column}")
        
        # Check for required columns
        if 'Date' not in data.columns:
            print("   WARNING: No 'Date' column found")
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            if date_cols:
                print(f"   Found date-like columns: {date_cols}")
        
        # Check data validity
        print(f"\n3. Checking data validity...")
        feature_cols = validator.processed_feature_columns
        target_col = validator.processed_target_column
        
        if feature_cols and target_col:
            # Check for NaN values
            feature_nans = data[feature_cols].isna().sum().sum()
            target_nans = data[target_col].isna().sum()
            print(f"   Feature NaNs: {feature_nans}")
            print(f"   Target NaNs: {target_nans}")
            
            # Try a single window
            print(f"\n4. Testing single window...")
            test_end_idx = validator.train_size + validator.test_size
            
            if test_end_idx <= len(data):
                test_start_idx = test_end_idx - validator.test_size
                train_end_idx = test_start_idx
                train_start_idx = max(0, train_end_idx - validator.train_size)
                
                print(f"   Train indices: {train_start_idx} to {train_end_idx} ({train_end_idx - train_start_idx} samples)")
                print(f"   Test indices: {test_start_idx} to {test_end_idx} ({test_end_idx - test_start_idx} samples)")
                
                train_data = data.iloc[train_start_idx:train_end_idx]
                test_data = data.iloc[test_start_idx:test_end_idx]
                
                train_X = train_data[feature_cols].values
                train_y = train_data[target_col].values
                test_X = test_data[feature_cols].values
                test_y = test_data[target_col].values
                
                print(f"   Train X shape: {train_X.shape}")
                print(f"   Train y shape: {train_y.shape}")
                print(f"   Test X shape: {test_X.shape}")
                print(f"   Test y shape: {test_y.shape}")
                
                # Check for valid data
                valid_train = ~(np.isnan(train_X).any(axis=1) | np.isnan(train_y))
                print(f"   Valid training samples: {valid_train.sum()}/{len(valid_train)}")
                
                if valid_train.sum() >= validator.min_training_samples:
                    # Try to create and fit model
                    print(f"\n5. Testing model creation and training...")
                    from OMtree_model import DirectionalTreeEnsemble
                    
                    # Use config path like the GUI does
                    model = DirectionalTreeEnsemble(
                        config_path=temp_path,
                        verbose=False
                    )
                    
                    train_X_clean = train_X[valid_train]
                    train_y_clean = train_y[valid_train]
                    
                    print(f"   Fitting model with {len(train_X_clean)} samples...")
                    model.fit(train_X_clean, train_y_clean)
                    
                    print(f"   Making predictions...")
                    predictions = model.predict(test_X)
                    print(f"   Predictions shape: {predictions.shape}")
                    print(f"   Unique predictions: {np.unique(predictions)[:10]}")
                    
                    # Check if we can get raw target values
                    raw_target_col = validator.config['data']['target_column']
                    if raw_target_col in test_data.columns:
                        print(f"   Raw target column '{raw_target_col}' found")
                        test_y_raw = test_data[raw_target_col].values
                        print(f"   Raw target sample: {test_y_raw[:5]}")
                    else:
                        print(f"   WARNING: Raw target column '{raw_target_col}' not found")
                        print(f"   Available columns: {list(test_data.columns)}")
                    
                    print("\n[SUCCESS] Model training and prediction worked!")
                else:
                    print(f"\n[ERROR] Not enough valid training samples")
            else:
                print(f"\n[ERROR] Not enough data for walk-forward (need {test_end_idx} rows, have {len(data)})")
        else:
            print("\n[ERROR] No feature or target columns found")
            
except Exception as e:
    print(f"\n[ERROR] Test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # Clean up temp file
    if os.path.exists(temp_path):
        os.unlink(temp_path)
        print(f"\nCleaned up temp config")

print("="*60)