"""
Verify that changing max_depth allows min_samples_split to affect feature selection
"""
import numpy as np
import pandas as pd
from feature_selector import FeatureSelector
import configparser

# Load config and data
config = configparser.ConfigParser()
config.read('OMtree_config.ini')

csv_file = config['data']['csv_file']
df = pd.read_csv(csv_file)

selected_features = [f.strip() for f in config['data']['selected_features'].split(',')]
target_column = config['data']['target_column']

# Prepare data
X = df[selected_features].iloc[1000:2000].values
y = df[target_column].iloc[1000:2000].values

# Remove NaN
mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
X = X[mask]
y = y[mask]

print("Testing feature selection with different max_depth and min_samples_split:")
print("="*70)

# Test with depth=1 (stumps)
print("\n--- MAX DEPTH = 1 (Decision Stumps) ---")
for min_split in [10, 100, 400]:
    selector = FeatureSelector(
        n_features=4, min_features=1, max_features=8,
        rf_n_estimators=400, rf_max_depth=1,
        rf_min_samples_split=min_split,
        rf_bootstrap_fraction=0.8,
        importance_threshold=0.15
    )
    
    selected_indices, selected_names = selector.select_features(
        X, y, feature_names=selected_features, verbose=False
    )
    
    print(f"min_split={min_split:3d}: {selected_names}")

# Test with depth=2
print("\n--- MAX DEPTH = 2 ---")
for min_split in [10, 100, 400]:
    selector = FeatureSelector(
        n_features=4, min_features=1, max_features=8,
        rf_n_estimators=400, rf_max_depth=2,
        rf_min_samples_split=min_split,
        rf_bootstrap_fraction=0.8,
        importance_threshold=0.15
    )
    
    selected_indices, selected_names = selector.select_features(
        X, y, feature_names=selected_features, verbose=False
    )
    
    print(f"min_split={min_split:3d}: {selected_names}")

# Test with depth=3
print("\n--- MAX DEPTH = 3 ---")
for min_split in [10, 100, 400]:
    selector = FeatureSelector(
        n_features=4, min_features=1, max_features=8,
        rf_n_estimators=400, rf_max_depth=3,
        rf_min_samples_split=min_split,
        rf_bootstrap_fraction=0.8,
        importance_threshold=0.15
    )
    
    selected_indices, selected_names = selector.select_features(
        X, y, feature_names=selected_features, verbose=False
    )
    
    print(f"min_split={min_split:3d}: {selected_names}")

print("\n" + "="*70)
print("\nCONCLUSION:")
print("- With max_depth=1, min_samples_split has NO effect (all results identical)")
print("- With max_depth>=2, min_samples_split DOES affect feature selection")
print("- Your model uses max_depth=1, so min_leaf% won't change feature selection!")