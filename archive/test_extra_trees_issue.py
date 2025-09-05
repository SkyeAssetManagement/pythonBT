"""
Investigate Extra Trees prediction issue
"""

import numpy as np
import configparser
from OMtree_model import DirectionalTreeEnsemble

print("=" * 80)
print("INVESTIGATING EXTRA TREES ISSUE")
print("=" * 80)

# Create simple test data
X_positive = np.ones((100, 2)) * 0.2  # Clearly positive
y_positive = np.ones(100) * 0.15  # Above threshold

X_test = np.ones((20, 2)) * 0.2

# Setup config
config = configparser.ConfigParser(inline_comment_prefixes='#')
config['data'] = {
    'csv_file': 'test.csv',
    'feature_columns': 'f1,f2',
    'selected_features': 'f1,f2',
    'target_column': 'target',
}

config['preprocessing'] = {
    'normalize_features': 'false',
    'normalize_target': 'false',
    'detrend_features': 'false',
}

config['model'] = {
    'model_type': 'longonly',
    'algorithm': 'extra_trees',
    'probability_aggregation': 'mean',
    'n_trees': '50',
    'max_depth': '1',
    'bootstrap_fraction': '0.8',
    'min_leaf_fraction': '0.1',
    'target_threshold': '0.1',
    'vote_threshold': '0.6',
    'random_seed': '42'
}

test_config = 'test_extra_trees.ini'
with open(test_config, 'w') as f:
    config.write(f)

# Test Extra Trees
print("\n1. Testing Extra Trees Model:")
print("-" * 40)

try:
    model = DirectionalTreeEnsemble(test_config, verbose=False)
    
    print(f"X shape: {X_positive.shape}")
    print(f"y shape: {y_positive.shape}")
    print(f"y values: min={y_positive.min():.3f}, max={y_positive.max():.3f}")
    print(f"Target threshold: {model.target_threshold}")
    
    # Create directional labels
    y_labels = model.create_directional_labels(y_positive)
    print(f"Directional labels: unique={np.unique(y_labels)}, mean={np.mean(y_labels):.3f}")
    
    # Train model
    model.fit(X_positive, y_positive)
    
    # Check if model was fitted
    if model.algorithm == 'extra_trees':
        print(f"Model fitted: {hasattr(model, 'model')}")
        if hasattr(model, 'model'):
            print(f"Number of estimators: {len(model.model.estimators_)}")
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"\nPredictions: {predictions[:10]}")
    print(f"Probabilities: {np.round(probabilities[:10], 3)}")
    print(f"Mean prediction: {np.mean(predictions):.3f}")
    print(f"Mean probability: {np.mean(probabilities):.3f}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Now test with Decision Trees for comparison
print("\n2. Testing Decision Trees Model (for comparison):")
print("-" * 40)

config['model']['algorithm'] = 'decision_trees'
with open(test_config, 'w') as f:
    config.write(f)

try:
    model_dt = DirectionalTreeEnsemble(test_config, verbose=False)
    model_dt.fit(X_positive, y_positive)
    
    predictions_dt = model_dt.predict(X_test)
    probabilities_dt = model_dt.predict_proba(X_test)
    
    print(f"Predictions: {predictions_dt[:10]}")
    print(f"Probabilities: {np.round(probabilities_dt[:10], 3)}")
    print(f"Mean prediction: {np.mean(predictions_dt):.3f}")
    print(f"Mean probability: {np.mean(probabilities_dt):.3f}")
    
except Exception as e:
    print(f"Error: {e}")

# Clean up
import os
if os.path.exists(test_config):
    os.remove(test_config)

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("""
The issue appears to be that Extra Trees is predicting all zeros.
This could be due to:
1. Different behavior with constant/near-constant features
2. Min samples leaf being too high for the data
3. Bootstrap sampling creating insufficient variation

Let's test with more varied data...
""")

# Test with varied data
print("\n3. Testing with Varied Data:")
print("-" * 40)

# Create data with variation
np.random.seed(42)
X_varied = np.random.randn(200, 2) * 0.1 + 0.15  # Centered around positive
y_varied = np.where(X_varied[:, 0] + X_varied[:, 1] > 0.3, 0.2, 0.05)  # Clear pattern

X_test_varied = np.random.randn(20, 2) * 0.1 + 0.15

config['model']['algorithm'] = 'extra_trees'
with open(test_config, 'w') as f:
    config.write(f)

try:
    model_varied = DirectionalTreeEnsemble(test_config, verbose=False)
    model_varied.fit(X_varied, y_varied)
    
    predictions_varied = model_varied.predict(X_test_varied)
    probabilities_varied = model_varied.predict_proba(X_test_varied)
    
    print(f"Predictions: {predictions_varied[:10]}")
    print(f"Mean prediction: {np.mean(predictions_varied):.3f}")
    print(f"Mean probability: {np.mean(probabilities_varied):.3f}")
    
    if np.mean(predictions_varied) > 0:
        print("\n[SUCCESS] Extra Trees works with varied data")
    else:
        print("\n[ISSUE] Extra Trees still predicting all zeros")
        
except Exception as e:
    print(f"Error: {e}")

# Clean up
if os.path.exists(test_config):
    os.remove(test_config)