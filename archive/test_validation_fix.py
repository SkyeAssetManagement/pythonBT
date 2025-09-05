"""
Test if the validation fix works
"""

import sys
import time
from OMtree_validation import DirectionalValidator

print("Testing validation after fix...")
print("="*60)

start_time = time.time()

try:
    print("1. Creating validator...")
    sys.stdout.flush()
    
    validator = DirectionalValidator('OMtree_config.ini')
    
    print("2. Loading data...")
    sys.stdout.flush()
    
    data = validator.load_and_prepare_data()
    
    print("3. Data loaded successfully")
    print(f"   Data shape: {data.shape}")
    print(f"   Processed features: {len(validator.processed_feature_columns)} columns")
    print(f"   Processed target: {validator.processed_target_column}")
    sys.stdout.flush()
    
    print("\n4. Testing train/test split...")
    test_end_idx = validator.train_size + validator.test_size + 250
    train_X, train_y, test_X, test_y, test_y_raw = validator.get_train_test_split(data, test_end_idx)
    
    if train_X is not None:
        print(f"   Train X shape: {train_X.shape}")
        print(f"   Train y shape: {train_y.shape}")
        print(f"   Test X shape: {test_X.shape}")
        print(f"   Test y shape: {test_y.shape}")
    else:
        print("   Split returned None (not enough data)")
    
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Validation test completed in {elapsed:.2f} seconds")
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n[ERROR] Failed after {elapsed:.2f} seconds")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()

print("="*60)