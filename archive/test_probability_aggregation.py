"""
Test probability aggregation methods (mean vs median)
"""

import numpy as np
import pandas as pd
import configparser
from OMtree_model import DirectionalTreeEnsemble
from OMtree_preprocessing import DataPreprocessor

print("=" * 80)
print("TESTING PROBABILITY AGGREGATION METHODS")
print("=" * 80)

# Load sample data
df = pd.read_csv('DTSmlDATA7x7.csv')
print(f"\nLoaded data: {df.shape}")

# Prepare data
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

test_config = 'test_aggregation.ini'

# Preprocess data once
preprocessor = DataPreprocessor('OMtree_config.ini')
processed = preprocessor.process_data(df)

# Get features and target
feature_cols = [c for c in processed.columns if '_vol_adj' in c and 'fwd' not in c and 'VolSignal' not in c]
target_col = 'Ret_fwd6hr_vol_adj' if 'Ret_fwd6hr_vol_adj' in processed.columns else 'Ret_fwd6hr'

# Take a sample for testing
sample_size = 500
X = processed[feature_cols[:1]].iloc[:sample_size].values  # Use first feature
y = processed[target_col].iloc[:sample_size].values

# Remove NaN values
mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X_clean = X[mask]
y_clean = y[mask]

print(f"Training data shape: {X_clean.shape}")

# Test 1: Mean aggregation
print("\n" + "=" * 60)
print("TEST 1: MEAN AGGREGATION")
print("=" * 60)

config['model']['probability_aggregation'] = 'mean'
config['model']['n_trees'] = '100'  # Use fewer trees for testing
config['model']['vote_threshold'] = '0.7'

with open(test_config, 'w') as f:
    config.write(f)

model_mean = DirectionalTreeEnsemble(test_config, verbose=False)
model_mean.fit(X_clean[:300], y_clean[:300])

# Get predictions and probabilities
pred_mean = model_mean.predict(X_clean[300:350])
prob_mean = model_mean.predict_proba(X_clean[300:350])

print(f"Predictions: {pred_mean[:10]}")
print(f"Probabilities: {np.round(prob_mean[:10], 3)}")
print(f"Trade signals: {np.sum(pred_mean)} out of {len(pred_mean)}")
print(f"Average probability: {np.mean(prob_mean):.3f}")
print(f"Std of probabilities: {np.std(prob_mean):.3f}")

# Test 2: Median aggregation
print("\n" + "=" * 60)
print("TEST 2: MEDIAN AGGREGATION")
print("=" * 60)

config['model']['probability_aggregation'] = 'median'

with open(test_config, 'w') as f:
    config.write(f)

model_median = DirectionalTreeEnsemble(test_config, verbose=False)
model_median.fit(X_clean[:300], y_clean[:300])

# Get predictions and probabilities
pred_median = model_median.predict(X_clean[300:350])
prob_median = model_median.predict_proba(X_clean[300:350])

print(f"Predictions: {pred_median[:10]}")
print(f"Probabilities: {np.round(prob_median[:10], 3)}")
print(f"Trade signals: {np.sum(pred_median)} out of {len(pred_median)}")
print(f"Average probability: {np.mean(prob_median):.3f}")
print(f"Std of probabilities: {np.std(prob_median):.3f}")

# Test 3: Compare the two methods
print("\n" + "=" * 60)
print("TEST 3: COMPARISON")
print("=" * 60)

# Compare probabilities
diff = prob_mean - prob_median
print(f"Mean - Median probability difference:")
print(f"  Average difference: {np.mean(diff):.3f}")
print(f"  Max difference: {np.max(np.abs(diff)):.3f}")
print(f"  Samples where mean > median: {np.sum(diff > 0)}")
print(f"  Samples where median > mean: {np.sum(diff < 0)}")

# Compare predictions
agree = pred_mean == pred_median
print(f"\nPrediction agreement: {np.sum(agree)}/{len(agree)} ({np.mean(agree)*100:.1f}%)")

# Show cases where they disagree
disagree_idx = np.where(~agree)[0]
if len(disagree_idx) > 0:
    print(f"\nDisagreement examples (first 5):")
    print("Index | Mean Prob | Median Prob | Mean Pred | Median Pred")
    print("-" * 60)
    for idx in disagree_idx[:5]:
        print(f"{idx:5d} | {prob_mean[idx]:9.3f} | {prob_median[idx]:11.3f} | "
              f"{pred_mean[idx]:9d} | {pred_median[idx]:11d}")

# Test 4: Simulate edge cases
print("\n" + "=" * 60)
print("TEST 4: EDGE CASE SIMULATION")
print("=" * 60)

# Simulate votes from 100 trees
n_trees = 100

# Case 1: Balanced votes (50-50)
votes_balanced = np.array([1]*50 + [0]*50)
mean_balanced = np.mean(votes_balanced)
median_balanced = np.median(votes_balanced)
print(f"Balanced (50-50) votes:")
print(f"  Mean: {mean_balanced:.3f}, Median: {median_balanced:.3f}")

# Case 2: Slight majority (51-49)
votes_slight = np.array([1]*51 + [0]*49)
mean_slight = np.mean(votes_slight)
median_slight = np.median(votes_slight)
print(f"Slight majority (51-49):")
print(f"  Mean: {mean_slight:.3f}, Median: {median_slight:.3f}")

# Case 3: Strong majority (70-30)
votes_strong = np.array([1]*70 + [0]*30)
mean_strong = np.mean(votes_strong)
median_strong = np.median(votes_strong)
print(f"Strong majority (70-30):")
print(f"  Mean: {mean_strong:.3f}, Median: {median_strong:.3f}")

# Case 4: Near unanimous (90-10)
votes_unanimous = np.array([1]*90 + [0]*10)
mean_unanimous = np.mean(votes_unanimous)
median_unanimous = np.median(votes_unanimous)
print(f"Near unanimous (90-10):")
print(f"  Mean: {mean_unanimous:.3f}, Median: {median_unanimous:.3f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nKey Differences:")
print("1. MEAN AGGREGATION:")
print("   - Gives continuous probabilities (0.0 to 1.0)")
print("   - More sensitive to the exact vote distribution")
print("   - Can detect subtle differences in confidence")
print("")
print("2. MEDIAN AGGREGATION:")
print("   - More binary (tends toward 0 or 1)")
print("   - Robust to outliers")
print("   - Requires >50% agreement to give probability > 0.5")
print("   - Better for binary decision making")
print("")
print("Recommendation:")
print("- Use MEAN for nuanced probability estimates")
print("- Use MEDIAN for more decisive, robust signals")

# Clean up
import os
if os.path.exists(test_config):
    os.remove(test_config)