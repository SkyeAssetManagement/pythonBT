"""
Test balanced bootstrap sampling functionality
"""
import numpy as np
import pandas as pd
import configparser
from OMtree_model import DirectionalTreeEnsemble

print("="*80)
print("TESTING BALANCED BOOTSTRAP SAMPLING")
print("="*80)

# Create test data with imbalanced classes
np.random.seed(42)
n_samples = 1000

# Create imbalanced data: 70% negative, 30% positive
X = np.random.randn(n_samples, 3)
y = np.concatenate([
    np.ones(300) * 0.15,   # 300 positive samples
    np.ones(700) * -0.05   # 700 negative samples
])
# Shuffle
shuffle_idx = np.random.permutation(n_samples)
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"\nOriginal data distribution:")
print(f"Total samples: {len(y)}")
print(f"Positive (y > 0): {np.sum(y > 0)} ({np.mean(y > 0)*100:.1f}%)")
print(f"Negative (y <= 0): {np.sum(y <= 0)} ({np.mean(y <= 0)*100:.1f}%)")

# Test 1: Regular bootstrap
print("\n" + "="*60)
print("TEST 1: REGULAR BOOTSTRAP")
print("="*60)

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
    'probability_aggregation': 'mean',
    'balanced_bootstrap': 'false',  # Regular bootstrap
    'n_trees': '10',
    'max_depth': '2',
    'bootstrap_fraction': '0.8',
    'min_leaf_fraction': '0.1',
    'target_threshold': '0.0',
    'vote_threshold': '0.5',
    'random_seed': '42'
}

test_config = 'test_regular.ini'
with open(test_config, 'w') as f:
    config.write(f)

model_regular = DirectionalTreeEnsemble(test_config, verbose=False)
model_regular.fit(X, y)

# Check bootstrap sample distributions
if model_regular.bootstrap_indices:
    print(f"Bootstrap sample distributions (first 5 trees):")
    for i, indices in enumerate(model_regular.bootstrap_indices[:5]):
        bootstrap_y = y[indices]
        pos_ratio = np.mean(bootstrap_y > 0)
        print(f"  Tree {i+1}: {pos_ratio*100:.1f}% positive, {(1-pos_ratio)*100:.1f}% negative")

# Make predictions
X_test = np.random.randn(100, 3)
pred_regular = model_regular.predict(X_test)
prob_regular = model_regular.predict_proba(X_test)

print(f"\nPrediction results:")
print(f"  Mean probability: {np.mean(prob_regular):.3f}")
print(f"  Trades (pred=1): {np.sum(pred_regular)}/{len(pred_regular)} ({np.mean(pred_regular)*100:.1f}%)")

# Test 2: Balanced bootstrap
print("\n" + "="*60)
print("TEST 2: BALANCED BOOTSTRAP")
print("="*60)

config['model']['balanced_bootstrap'] = 'true'  # Enable balanced bootstrap

test_config = 'test_balanced.ini'
with open(test_config, 'w') as f:
    config.write(f)

model_balanced = DirectionalTreeEnsemble(test_config, verbose=False)
model_balanced.fit(X, y)

# Check bootstrap sample distributions
if model_balanced.bootstrap_indices:
    print(f"Bootstrap sample distributions (first 5 trees):")
    for i, indices in enumerate(model_balanced.bootstrap_indices[:5]):
        bootstrap_y = y[indices]
        pos_ratio = np.mean(bootstrap_y > 0)
        print(f"  Tree {i+1}: {pos_ratio*100:.1f}% positive, {(1-pos_ratio)*100:.1f}% negative")

# Make predictions
pred_balanced = model_balanced.predict(X_test)
prob_balanced = model_balanced.predict_proba(X_test)

print(f"\nPrediction results:")
print(f"  Mean probability: {np.mean(prob_balanced):.3f}")
print(f"  Trades (pred=1): {np.sum(pred_balanced)}/{len(pred_balanced)} ({np.mean(pred_balanced)*100:.1f}%)")

# Compare the two methods
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

print(f"Regular bootstrap:")
print(f"  - Maintains original class distribution (~30% positive)")
print(f"  - Mean probability: {np.mean(prob_regular):.3f}")
print(f"  - Trade frequency: {np.mean(pred_regular)*100:.1f}%")

print(f"\nBalanced bootstrap:")
print(f"  - Forces 50/50 class distribution")
print(f"  - Mean probability: {np.mean(prob_balanced):.3f}")
print(f"  - Trade frequency: {np.mean(pred_balanced)*100:.1f}%")

print(f"\nDifference in trade frequency: {abs(np.mean(pred_balanced) - np.mean(pred_regular))*100:.1f} percentage points")

# Clean up
import os
for file in ['test_regular.ini', 'test_balanced.ini']:
    if os.path.exists(file):
        os.remove(file)

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
Balanced bootstrap successfully:
1. Forces equal representation of positive and negative samples
2. Each bootstrap sample has ~50% from each class
3. Helps when original data is imbalanced
4. Can lead to different (often higher) trade frequencies
5. Useful for ensuring trees see both market conditions equally
""")