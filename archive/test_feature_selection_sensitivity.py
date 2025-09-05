"""
Test script to verify that feature selection changes with different min_leaf settings
"""
import numpy as np
import pandas as pd
from feature_selector import FeatureSelector
import configparser

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 8

# Create features with varying importance
X = np.random.randn(n_samples, n_features)
# Make some features more important
y = 3 * X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.1

feature_names = [f'Feature_{i+1}' for i in range(n_features)]

print("Testing feature selection with different min_samples_split values:\n")

# Test with different min_samples_split values
for min_split in [2, 10, 50, 100, 200]:
    print(f"\nMin samples split: {min_split}")
    print("-" * 50)
    
    selector = FeatureSelector(
        n_features=4,
        min_features=1,
        max_features=8,
        rf_n_estimators=100,
        rf_max_depth=1,
        rf_min_samples_split=min_split,
        rf_bootstrap_fraction=0.8,
        importance_threshold=0.15
    )
    
    selected_indices, selected_names = selector.select_features(
        X, y, feature_names=feature_names, verbose=False
    )
    
    print(f"Selected features: {selected_names}")
    print(f"Importances: {selector.selection_scores}")
    print(f"All importances: {selector.all_importances}")

# Now test with actual data
print("\n" + "="*60)
print("Testing with actual trading data:")
print("="*60)

# Load config
config = configparser.ConfigParser()
config.read('OMtree_config.ini')

# Load data
csv_file = config['data']['csv_file']
df = pd.read_csv(csv_file)

# Get selected features
selected_features = [f.strip() for f in config['data']['selected_features'].split(',')]
target_column = config['data']['target_column']

# Prepare data (simplified - no preprocessing)
X = df[selected_features].iloc[1000:2000].values
y = df[target_column].iloc[1000:2000].values

# Remove NaN
mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
X = X[mask]
y = y[mask]

print(f"\nData shape: {X.shape}")
print(f"Features: {selected_features}")

# Test with different settings
for min_leaf_pct in [0.01, 0.05, 0.1, 0.2, 0.4]:
    min_split = max(2, int(1000 * min_leaf_pct))
    print(f"\nMin leaf %: {min_leaf_pct:.1%} -> Min samples split: {min_split}")
    print("-" * 50)
    
    selector = FeatureSelector(
        n_features=4,
        min_features=1, 
        max_features=8,
        rf_n_estimators=400,  # 50 * 8 features
        rf_max_depth=1,
        rf_min_samples_split=min_split,
        rf_bootstrap_fraction=0.8,
        importance_threshold=0.15
    )
    
    selected_indices, selected_names = selector.select_features(
        X, y, feature_names=selected_features, verbose=False
    )
    
    print(f"Selected: {selected_names}")
    # Show top 3 importance scores
    sorted_scores = sorted(selector.selection_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"Top scores: {sorted_scores}")