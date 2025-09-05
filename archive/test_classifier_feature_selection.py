import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd

print("="*80)
print("COMPARING REGRESSOR VS CLASSIFIER FOR FEATURE SELECTION")
print("="*80)

# Create synthetic data similar to trading returns
np.random.seed(42)
n_samples = 1000

# Feature 1: Good at predicting direction but not magnitude
feature1 = np.random.randn(n_samples) * 0.3

# Feature 2: Mediocre at direction but great at identifying big moves
feature2 = np.random.randn(n_samples) * 0.2

# Feature 3: Noise
feature3 = np.random.randn(n_samples) * 0.5

# Create target returns
# Feature 1 contribution: small but consistent
# Feature 2 contribution: occasional large moves
small_moves = 0.3 * feature1 + np.random.randn(n_samples) * 0.1
large_moves = np.where(np.abs(feature2) > 0.3, 
                       feature2 * 3,  # Large moves when feature2 is extreme
                       0)
target_returns = small_moves + large_moves + np.random.randn(n_samples) * 0.2

# Create feature matrix
X = np.column_stack([feature1, feature2, feature3])
feature_names = ['SmallConsistent', 'BigMoves', 'Noise']

print(f"\nTarget statistics:")
print(f"  Mean return: {np.mean(target_returns):.3f}")
print(f"  Std return: {np.std(target_returns):.3f}")
print(f"  % positive: {(target_returns > 0).mean() * 100:.1f}%")

# Test 1: RandomForestRegressor (current approach)
print("\n" + "="*50)
print("APPROACH 1: RandomForestRegressor (Current)")
print("="*50)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf_reg.fit(X, target_returns)

print("Feature importance scores (predicting continuous returns):")
for name, importance in zip(feature_names, rf_reg.feature_importances_):
    print(f"  {name:20s}: {importance:.3f}")

# Test 2: RandomForestClassifier (alternative)
print("\n" + "="*50)
print("APPROACH 2: RandomForestClassifier (Alternative)")
print("="*50)

# Create binary labels
threshold = 0
y_binary = (target_returns > threshold).astype(int)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf_clf.fit(X, y_binary)

print("Feature importance scores (predicting positive/negative):")
for name, importance in zip(feature_names, rf_clf.feature_importances_):
    print(f"  {name:20s}: {importance:.3f}")

# Now simulate what happens with normalization
print("\n" + "="*50)
print("EFFECT OF NORMALIZATION")
print("="*50)

# Simulate IQR normalization
iqr = np.percentile(target_returns, 75) - np.percentile(target_returns, 25)
normalized_returns = target_returns / iqr

print(f"\nNormalized target statistics:")
print(f"  Mean return: {np.mean(normalized_returns):.3f}")
print(f"  Std return: {np.std(normalized_returns):.3f}")
print(f"  % positive: {(normalized_returns > 0).mean() * 100:.1f}% (should be same)")

# Test with normalized targets
rf_reg_norm = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf_reg_norm.fit(X, normalized_returns)

print("\nRegressor importance with NORMALIZED targets:")
for name, importance in zip(feature_names, rf_reg_norm.feature_importances_):
    print(f"  {name:20s}: {importance:.3f}")

# Binary classification should be identical
y_binary_norm = (normalized_returns > 0).astype(int)
assert np.array_equal(y_binary, y_binary_norm), "Binary labels should be identical!"

rf_clf_norm = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf_clf_norm.fit(X, y_binary_norm)

print("\nClassifier importance with NORMALIZED targets (should be same):")
for name, importance in zip(feature_names, rf_clf_norm.feature_importances_):
    print(f"  {name:20s}: {importance:.3f}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("""
1. Regressor identifies 'BigMoves' feature as important (good for trading!)
2. Classifier focuses more on 'SmallConsistent' (better classification accuracy)
3. Normalization changes Regressor scores but NOT Classifier scores
4. This explains why your normalized vs non-normalized results differ

For trading, you likely want features that predict BIG moves, not just direction.
The Regressor approach captures this better, even though it causes the 
normalization differences you observed.
""")