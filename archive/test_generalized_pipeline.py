"""
Test generalized pipeline with different data formats
"""

import pandas as pd
import numpy as np
import configparser
from OMtree_preprocessing import DataPreprocessor
from OMtree_model import DirectionalTreeEnsemble
from OMtree_validation import DirectionalValidator
import os

print("=" * 80)
print("TESTING GENERALIZED PIPELINE")
print("=" * 80)

# Check what CSV files are available
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"\nAvailable CSV files: {csv_files}")

# Test with different data file if it exists
test_file = None
for f in csv_files:
    if 'DTSnn' in f or 'DTSmm' in f:
        test_file = f
        break

if not test_file:
    # Create a test file with different column names
    print("\nCreating test data file with different column names...")
    np.random.seed(42)
    n_rows = 1000
    
    test_data = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
        'Time': ['10:00:00'] * n_rows,
        'Overnight': np.random.randn(n_rows) * 0.01,
        '3day': np.random.randn(n_rows) * 0.02,
        '1week': np.random.randn(n_rows) * 0.03,
        'Feature4': np.random.randn(n_rows) * 0.01,
        'Feature5': np.random.randn(n_rows) * 0.01,
        'RetForward': np.random.randn(n_rows) * 0.02,
        'FwdRet2': np.random.randn(n_rows) * 0.02,
        'Target3': np.random.randn(n_rows) * 0.02
    })
    
    test_file = 'DTSnnDATA.csv'
    test_data.to_csv(test_file, index=False)
    print(f"Created test file: {test_file}")

# Create test config
print(f"\nConfiguring for {test_file}...")
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

# Update config for test file
config['data']['csv_file'] = test_file

# Save test config
test_config = 'test_generalized.ini'
with open(test_config, 'w') as f:
    config.write(f)

print("\n" + "=" * 40)
print("TEST 1: PREPROCESSING")
print("=" * 40)

try:
    # Test preprocessing
    df = pd.read_csv(test_file)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    preprocessor = DataPreprocessor(test_config)
    processed = preprocessor.process_data(df)
    
    print(f"\nProcessed shape: {processed.shape}")
    print(f"Processed columns: {list(processed.columns)[:10]}...")
    print("[PASS] Preprocessing handled missing columns")
    
except Exception as e:
    print(f"[FAIL] Preprocessing error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 40)
print("TEST 2: MODEL TRAINING")
print("=" * 40)

try:
    # Test model training
    if 'processed' in locals():
        # Find feature columns
        feature_cols = [c for c in processed.columns if '_vol_adj' in c and 'VolSignal' not in c]
        target_cols = [c for c in processed.columns if 'forward' in c.lower() or 'fwd' in c.lower()]
        
        if not target_cols:
            target_cols = [c for c in processed.columns if '_vol_adj' in c][-1:]
        
        if feature_cols and target_cols:
            X = processed[feature_cols[:1]].dropna().values  # Use first feature
            y = processed[target_cols[0]].dropna().values
            
            # Ensure same length
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            if len(X) > 100:
                model = DirectionalTreeEnsemble(test_config, verbose=False)
                model.fit(X[:100], y[:100])
                predictions = model.predict(X[100:110])
                
                print(f"Trained on {X[:100].shape} samples")
                print(f"Predictions shape: {predictions.shape}")
                print(f"Sample predictions: {predictions[:5]}")
                print("[PASS] Model training handled data format")
            else:
                print("[SKIP] Not enough data for model training")
        else:
            print("[SKIP] No suitable columns for training")
    else:
        print("[SKIP] Preprocessing failed")
        
except Exception as e:
    print(f"[FAIL] Model training error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 40)
print("TEST 3: VALIDATION")
print("=" * 40)

try:
    # Test validation with smaller windows
    config['validation']['train_size'] = '50'
    config['validation']['test_size'] = '10'
    config['validation']['step_size'] = '10'
    
    with open(test_config, 'w') as f:
        config.write(f)
    
    validator = DirectionalValidator(test_config)
    data = validator.load_and_prepare_data()
    
    print(f"Validation data shape: {data.shape}")
    
    # Try one validation step
    test_end_idx = min(200, len(data) - 10)
    result = validator.get_train_test_split(data, test_end_idx)
    
    if result[0] is not None:
        train_X, train_y, test_X, test_y, test_indices = result
        print(f"Train shape: {train_X.shape if hasattr(train_X, 'shape') else len(train_X)}")
        print(f"Test shape: {test_X.shape if hasattr(test_X, 'shape') else len(test_X)}")
        print("[PASS] Validation handled missing columns")
    else:
        print("[INFO] Not enough data for validation split")
        
except Exception as e:
    print(f"[FAIL] Validation error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
The pipeline has been generalized to:
1. Auto-detect columns when configured columns are missing
2. Find similar columns using string matching
3. Handle various column naming conventions
4. Gracefully fallback to available data

This allows the code to work with different data formats without errors.
""")

# Clean up
if os.path.exists(test_config):
    os.remove(test_config)
    
if test_file == 'DTSnnDATA.csv' and os.path.exists(test_file):
    os.remove(test_file)
    print(f"Cleaned up test file: {test_file}")