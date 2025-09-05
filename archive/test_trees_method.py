"""
Test the n_trees_method feature (absolute vs per_feature)
"""
import numpy as np
import pandas as pd
import configparser
from OMtree_model import DirectionalTreeEnsemble

print("="*80)
print("TESTING N_TREES_METHOD: ABSOLUTE VS PER_FEATURE")
print("="*80)

# Create test data
np.random.seed(42)
n_samples = 500

# Test configurations
test_cases = [
    {'n_features': 1, 'n_trees': 50, 'method': 'absolute'},
    {'n_features': 1, 'n_trees': 50, 'method': 'per_feature'},
    {'n_features': 3, 'n_trees': 50, 'method': 'absolute'},
    {'n_features': 3, 'n_trees': 50, 'method': 'per_feature'},
    {'n_features': 5, 'n_trees': 30, 'method': 'absolute'},
    {'n_features': 5, 'n_trees': 30, 'method': 'per_feature'},
]

for test in test_cases:
    n_features = test['n_features']
    n_trees_base = test['n_trees']
    method = test['method']
    
    print(f"\n{'='*60}")
    print(f"TEST: {n_features} features, n_trees={n_trees_base}, method={method}")
    print(f"{'='*60}")
    
    # Create data with specified number of features
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.1
    
    # Setup config
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config['data'] = {
        'csv_file': 'test.csv',
        'feature_columns': ','.join([f'f{i+1}' for i in range(n_features)]),
        'selected_features': ','.join([f'f{i+1}' for i in range(n_features)]),
        'target_column': 'target',
    }
    
    config['preprocessing'] = {
        'normalize_features': 'false',
        'normalize_target': 'false',
        'detrend_features': 'false',
    }
    
    config['model'] = {
        'model_type': 'longonly',
        'algorithm': 'decision_trees',
        'probability_aggregation': 'mean',
        'balanced_bootstrap': 'false',
        'n_trees_method': method,
        'n_trees': str(n_trees_base),
        'max_depth': '2',
        'bootstrap_fraction': '0.8',
        'min_leaf_fraction': '0.05',
        'target_threshold': '0.0',
        'vote_threshold': '0.5',
        'random_seed': '42'
    }
    
    test_config = 'test_trees_method.ini'
    with open(test_config, 'w') as f:
        config.write(f)
    
    # Train model
    model = DirectionalTreeEnsemble(test_config, verbose=True)
    model.fit(X, y)
    
    # Verify tree count
    expected_trees = n_trees_base * n_features if method == 'per_feature' else n_trees_base
    actual_trees = len(model.trees)
    
    print(f"\nExpected trees: {expected_trees}")
    print(f"Actual trees: {actual_trees}")
    
    if actual_trees == expected_trees:
        print("[SUCCESS] Tree count matches expected!")
    else:
        print(f"[ERROR] Mismatch! Expected {expected_trees}, got {actual_trees}")
    
    # Test predictions work
    X_test = np.random.randn(10, n_features)
    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Mean prediction: {np.mean(predictions):.3f}")

# Clean up
import os
if os.path.exists('test_trees_method.ini'):
    os.remove('test_trees_method.ini')

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
The n_trees_method feature allows two modes:

1. ABSOLUTE (default):
   - Fixed number of trees regardless of features
   - Good for: consistent model complexity
   - Example: 100 trees always

2. PER_FEATURE:
   - Trees = n_trees × n_features
   - Good for: scaling complexity with feature count
   - Example: 50 × 3 features = 150 trees
   
Use cases:
- Use ABSOLUTE when you want consistent training time
- Use PER_FEATURE when more features need more trees to explore interactions
""")