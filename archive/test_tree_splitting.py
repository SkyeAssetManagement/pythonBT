import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

print("="*80)
print("TESTING WHY DECISION TREES SPLIT DIFFERENTLY WITH NORMALIZATION")
print("="*80)

# Simulate some data similar to our returns
np.random.seed(42)
n_samples = 1000

# Create features (like our return features)
X = np.random.randn(n_samples, 3) * 0.5  # 3 features

# Create target (returns)
raw_target = np.random.randn(n_samples) * 0.3

# Create binary classification at threshold=0
y_raw = (raw_target > 0).astype(int)

# Simulate IQR normalization (divide by volatility)
# This changes the scale but not the sign
iqr = 0.5 + 0.2 * np.random.random(n_samples)  # Varying IQR like in real data
normalized_target = raw_target / iqr

# Check that signs are preserved
y_normalized = (normalized_target > 0).astype(int)
assert np.array_equal(y_raw, y_normalized), "Signs should be identical!"
print("[OK] Confirmed: Binary labels are identical (signs preserved)")

# But here's the key issue: Trees use FEATURES to predict the target
# When we normalize the target during training, the tree learns different relationships

print("\n" + "="*50)
print("TRAINING DECISION STUMPS (max_depth=1)")
print("="*50)

# Train tree on raw data
tree_raw = DecisionTreeClassifier(max_depth=1, random_state=42)
tree_raw.fit(X, y_raw)

# Train tree on "normalized" scenario
# In reality, the tree sees the same binary labels, but during training
# it may use different sample weights or handle edge cases differently
tree_norm = DecisionTreeClassifier(max_depth=1, random_state=42)
tree_norm.fit(X, y_normalized)

# The trees should be identical in this simple case
pred_raw = tree_raw.predict(X)
pred_norm = tree_norm.predict(X)
match_rate = (pred_raw == pred_norm).mean() * 100

print(f"Prediction match rate: {match_rate:.1f}%")

# Now let's see what happens with BOOTSTRAP SAMPLING
print("\n" + "="*50)
print("THE REAL ISSUE: BOOTSTRAP SAMPLING + FEATURE SELECTION")
print("="*50)

print("""
The key insight: Even though threshold=0 creates the same binary labels,
the walk-forward validation has these differences:

1. FEATURE SELECTION: The MDI importance scores are calculated differently
   - With normalized targets, the variance is different
   - This changes which features get selected
   - Different features = different models

2. BOOTSTRAP SAMPLING: When using balanced bootstrap
   - The positive/negative class balance might vary slightly
   - Edge cases (values very close to 0) might be handled differently
   
3. TEMPORAL EFFECTS: The 250-sample offset means:
   - Training windows contain slightly different market regimes
   - Even 1 sample difference can change feature selection
""")

# Demonstrate feature importance difference
from sklearn.ensemble import RandomForestRegressor

print("\nDemonstrating feature importance changes:")

# Use regression for feature importance (MDI)
rf_raw = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
rf_raw.fit(X, raw_target)  # Train on continuous target

rf_norm = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
rf_norm.fit(X, normalized_target)  # Train on normalized continuous target

print(f"Feature importance with raw target:       {rf_raw.feature_importances_}")
print(f"Feature importance with normalized target: {rf_norm.feature_importances_}")
print(f"Importance difference: {np.abs(rf_raw.feature_importances_ - rf_norm.feature_importances_)}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
Even with threshold=0 and perfect sign preservation, models differ because:

1. Feature selection uses MDI on the continuous target values (before thresholding)
   - Normalized targets have different variance patterns
   - This leads to different features being selected

2. The actual tree training sees binary labels (post-threshold) but:
   - Different features are available (from step 1)
   - Bootstrap sampling may select slightly different samples
   - Ensemble voting amplifies these small differences

This is WHY you see 20% different predictions even though the fundamental
classification task (positive vs negative) is identical!
""")