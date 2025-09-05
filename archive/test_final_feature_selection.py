"""
Final test to verify all feature selection settings work correctly
"""

import numpy as np
from feature_selector import FeatureSelector

# Create test data
np.random.seed(42)
n_samples, n_features = 500, 10
X = np.random.randn(n_samples, n_features)
y = 2*X[:, 0] + 1.5*X[:, 1] + 0.8*X[:, 2] + 0.3*X[:, 3] + 0.1*np.random.randn(n_samples)
feature_names = [f'Feature_{i}' for i in range(n_features)]

print("FINAL FEATURE SELECTION VERIFICATION")
print("="*60)

# Test 1: Exact constraint (min=3, max=3)
print("\nTest 1: min=3, max=3, threshold=0.1 (minimum mode)")
selector = FeatureSelector(
    min_features=3,
    max_features=3,
    importance_threshold=0.1,
    cumulative_importance_mode=False,
    rf_n_estimators=50,
    rf_max_depth=3,
    random_seed=42
)
_, selected = selector.select_features(X, y, feature_names, verbose=True)
assert len(selected) == 3, f"Expected exactly 3 features, got {len(selected)}"
print(f"Result: {len(selected)} features - CORRECT!")

# Test 2: Cumulative mode with constraints
print("\n" + "="*60)
print("\nTest 2: min=1, max=8, cumulative=50%")
selector = FeatureSelector(
    min_features=1,
    max_features=8,
    cumulative_importance_mode=True,
    cumulative_importance_threshold=0.5,
    rf_n_estimators=50,
    rf_max_depth=3,
    random_seed=42
)
_, selected = selector.select_features(X, y, feature_names, verbose=True)
assert 1 <= len(selected) <= 8, f"Expected 1-8 features, got {len(selected)}"
print(f"Result: {len(selected)} features - CORRECT!")

# Test 3: High threshold with min constraint
print("\n" + "="*60)
print("\nTest 3: min=2, max=5, threshold=0.5 (should force min)")
selector = FeatureSelector(
    min_features=2,
    max_features=5,
    importance_threshold=0.5,  # Very high - likely no features pass
    cumulative_importance_mode=False,
    rf_n_estimators=50,
    rf_max_depth=3,
    random_seed=42
)
_, selected = selector.select_features(X, y, feature_names, verbose=True)
assert len(selected) >= 2, f"Expected at least 2 features, got {len(selected)}"
assert len(selected) <= 5, f"Expected at most 5 features, got {len(selected)}"
print(f"Result: {len(selected)} features - CORRECT!")

# Test 4: Many features above threshold with max constraint  
print("\n" + "="*60)
print("\nTest 4: min=1, max=2, threshold=0.01 (should force max)")
selector = FeatureSelector(
    min_features=1,
    max_features=2,
    importance_threshold=0.01,  # Very low - many features pass
    cumulative_importance_mode=False,
    rf_n_estimators=50,
    rf_max_depth=3,
    random_seed=42
)
_, selected = selector.select_features(X, y, feature_names, verbose=True)
assert len(selected) <= 2, f"Expected at most 2 features, got {len(selected)}"
print(f"Result: {len(selected)} features - CORRECT!")

print("\n" + "="*60)
print("ALL TESTS PASSED - Feature selection settings working correctly!")
print("="*60)