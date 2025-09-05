"""
Test actual model behavior with median aggregation
==================================================
"""

import numpy as np
import pandas as pd
import configparser
from OMtree_model import DirectionalTreeEnsemble

print("="*80)
print("TESTING ACTUAL MODEL MEDIAN BEHAVIOR")
print("="*80)

# Create diverse test data to get varied tree predictions
np.random.seed(42)
n_samples = 200

# Create features with varying patterns
X_train = np.random.randn(n_samples, 3) * 0.5
# Target with some noise to create diversity in tree predictions
y_train = 0.1 * X_train[:, 0] + 0.05 * X_train[:, 1] + np.random.randn(n_samples) * 0.05

# Test data - specifically designed to get split votes
X_test = np.array([
    [0.0, 0.0, 0.0],    # Neutral - should get mixed votes
    [0.01, 0.01, 0.01], # Very slightly positive
    [-0.01, -0.01, -0.01], # Very slightly negative
    [0.5, -0.5, 0.0],   # Mixed signals
    [0.1, 0.0, -0.1],   # Mixed signals
])

# Test with different numbers of trees
tree_counts = [10, 20, 50, 100, 200]

for n_trees in tree_counts:
    print(f"\n{'='*40}")
    print(f"Testing with {n_trees} trees (EVEN number)")
    print(f"{'='*40}")
    
    # Setup config
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config['data'] = {
        'csv_file': 'test.csv',
        'feature_columns': 'f1,f2,f3',
        'selected_features': 'f1,f2,f3',
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
        'probability_aggregation': 'median',
        'n_trees': str(n_trees),
        'max_depth': '2',  # Shallow trees for more variation
        'bootstrap_fraction': '0.7',  # More variation between trees
        'min_leaf_fraction': '0.05',  # Allow smaller leaves
        'target_threshold': '0.0',
        'vote_threshold': '0.5',
        'random_seed': '42'
    }
    
    test_config = f'test_median_{n_trees}.ini'
    with open(test_config, 'w') as f:
        config.write(f)
    
    # Train model
    model = DirectionalTreeEnsemble(test_config, verbose=False)
    model.fit(X_train, y_train)
    
    # Get predictions
    probabilities = model.predict_proba(X_test)
    
    # Also get individual tree votes for analysis
    predictions_per_tree = []
    for tree in model.trees:
        pred = tree.predict(X_test)
        predictions_per_tree.append(pred)
    
    predictions_per_tree = np.array(predictions_per_tree)
    
    print(f"Probabilities from model: {probabilities}")
    print(f"\nPossible values with {n_trees} trees:")
    if n_trees % 2 == 0:
        # Even number of trees
        possible = [i/n_trees for i in range(0, n_trees//2)] + [0.5] + [i/n_trees for i in range(n_trees//2 + 1, n_trees + 1)]
        print(f"  0.0, ..., {(n_trees//2-1)/n_trees:.3f}, 0.5, {(n_trees//2+1)/n_trees:.3f}, ..., 1.0")
    else:
        # Odd number of trees
        print(f"  Only 0.0 or 1.0 (binary)")
    
    # Check for 0.5 values
    has_half = np.any(probabilities == 0.5)
    print(f"\nContains 0.5 probability: {has_half}")
    
    # Detailed analysis for first test sample
    print(f"\nDetailed analysis for sample 1 (neutral features):")
    votes = predictions_per_tree[:, 0]
    print(f"  Individual tree votes: {votes}")
    print(f"  Votes for 1: {np.sum(votes)}/{n_trees}")
    print(f"  Manual median: {np.median(votes)}")
    print(f"  Model probability: {probabilities[0]}")
    
    # Clean up
    import os
    os.remove(test_config)

# Now test with actual config settings
print(f"\n{'='*80}")
print("TESTING WITH ACTUAL CONFIG (200 trees)")
print(f"{'='*80}")

# Create more challenging test data
X_challenging = np.random.randn(1000, 5) * 0.1  # Small variations
y_challenging = np.random.randn(1000) * 0.05  # Small target variations

# Load actual config
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

# Temporarily set to test with known features
config['data']['feature_columns'] = 'f1,f2,f3,f4,f5'
config['data']['selected_features'] = 'f1,f2,f3,f4,f5'
config['model']['n_trees'] = '200'
config['model']['probability_aggregation'] = 'median'

with open('test_actual_median.ini', 'w') as f:
    config.write(f)

model = DirectionalTreeEnsemble('test_actual_median.ini', verbose=False)
model.fit(X_challenging[:500], y_challenging[:500])

# Test on remaining data
test_probs = model.predict_proba(X_challenging[500:600])

print(f"Probability distribution with 200 trees:")
unique_probs = np.unique(test_probs)
print(f"Unique probability values: {unique_probs}")

# Count occurrences
prob_counts = {}
for p in test_probs:
    p_rounded = round(p, 3)
    prob_counts[p_rounded] = prob_counts.get(p_rounded, 0) + 1

print(f"\nProbability value counts:")
for prob, count in sorted(prob_counts.items()):
    print(f"  {prob:.3f}: {count} occurrences")

# Check for 0.5
n_half = np.sum(test_probs == 0.5)
print(f"\nSamples with exactly 0.5 probability: {n_half}/100")

if n_half == 0:
    print("\nWhy no 0.5 values?")
    print("With 200 trees, getting exactly 100 votes for each class is rare.")
    print("Most samples will lean one way or another, giving 0.0 or 1.0.")
    print("This makes median aggregation MOSTLY binary in practice,")
    print("even though 0.5 is theoretically possible.")

# Clean up
import os
if os.path.exists('test_actual_median.ini'):
    os.remove('test_actual_median.ini')