"""
Test where the tree actually splits with different settings
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('C:/Users/jd/OM3/DTSmlDATA7x7.csv')
X = df[['Ret_8-16hr']].values
y = df['Ret_fwd6hr'].values

# Remove NaN
valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X = X[valid_mask][:1000]  # First 1000 samples
y = y[valid_mask][:1000]

# Create binary labels (threshold at 0)
y_binary = (y > 0).astype(int)

print("="*60)
print("WHERE DOES THE TREE SPLIT?")
print("="*60)

# Sort X to understand percentiles
X_sorted = np.sort(X.flatten())
print(f"\nFeature (Ret_8-16hr) percentiles:")
print(f"  25th: {np.percentile(X, 25):.4f}")
print(f"  50th (median): {np.percentile(X, 50):.4f}")
print(f"  75th: {np.percentile(X, 75):.4f}")

print(f"\nTarget distribution:")
print(f"  Positive (y > 0): {np.sum(y_binary)}/{len(y_binary)} ({np.mean(y_binary)*100:.1f}%)")

# Test different min_samples_leaf values
min_leaf_values = [50, 100, 200, 250, 300, 400, 450]

print("\n" + "="*40)
print("SPLIT POINTS WITH DIFFERENT min_samples_leaf:")
print("="*40)

for min_samples in min_leaf_values:
    tree = DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=min_samples,
        random_state=42
    )
    
    try:
        tree.fit(X, y_binary)
        
        if tree.tree_.node_count > 1:
            # Tree made a split
            threshold = tree.tree_.threshold[0]
            
            # Where is this threshold in the distribution?
            percentile = np.mean(X <= threshold) * 100
            
            # How many samples on each side?
            n_left = np.sum(X <= threshold)
            n_right = np.sum(X > threshold)
            
            print(f"\nmin_samples_leaf = {min_samples} ({min_samples/len(X)*100:.0f}%):")
            print(f"  Split at: {threshold:.4f}")
            print(f"  This is the {percentile:.1f}th percentile")
            print(f"  Left: {n_left} samples ({n_left/len(X)*100:.1f}%)")
            print(f"  Right: {n_right} samples ({n_right/len(X)*100:.1f}%)")
        else:
            print(f"\nmin_samples_leaf = {min_samples} ({min_samples/len(X)*100:.0f}%):")
            print(f"  NO SPLIT (single leaf)")
    except Exception as e:
        print(f"\nmin_samples_leaf = {min_samples}: Error - {e}")

# Now test what determines the split location
print("\n" + "="*60)
print("WHAT DETERMINES THE SPLIT LOCATION?")
print("="*60)

tree = DecisionTreeClassifier(
    max_depth=1,
    min_samples_leaf=100,  # 10% minimum
    random_state=42
)
tree.fit(X, y_binary)

if tree.tree_.node_count > 1:
    threshold = tree.tree_.threshold[0]
    
    # Calculate impurity reduction for different split points
    print("\nThe tree chooses the split that maximizes information gain")
    print("(reduces impurity the most), subject to min_samples_leaf constraint.")
    
    # Test some potential split points
    test_thresholds = np.percentile(X, [20, 30, 40, 50, 60, 70, 80])
    
    print("\nInformation gain at different split points:")
    for test_thresh in test_thresholds:
        left_mask = X.flatten() <= test_thresh
        right_mask = ~left_mask
        
        if np.sum(left_mask) >= 100 and np.sum(right_mask) >= 100:
            # Calculate Gini impurity
            left_labels = y_binary[left_mask]
            right_labels = y_binary[right_mask]
            
            p_left = np.mean(left_labels)
            p_right = np.mean(right_labels)
            
            gini_left = 2 * p_left * (1 - p_left)
            gini_right = 2 * p_right * (1 - p_right)
            
            n_left = len(left_labels)
            n_right = len(right_labels)
            n_total = n_left + n_right
            
            weighted_impurity = (n_left/n_total) * gini_left + (n_right/n_total) * gini_right
            
            percentile = np.mean(X <= test_thresh) * 100
            chosen = " <-- CHOSEN" if abs(test_thresh - threshold) < 0.0001 else ""
            print(f"  {percentile:4.1f}th percentile ({test_thresh:7.4f}): impurity = {weighted_impurity:.4f}{chosen}")

print("\n" + "="*60)
print("KEY INSIGHT:")
print("="*60)
print("""
min_leaf_fraction (or min_samples_leaf) only sets CONSTRAINTS.
Within those constraints, the tree finds the split that best
separates the positive and negative labels (maximizes information gain).

The split will NOT necessarily be at the median!
It will be wherever the feature value best predicts the target,
as long as both sides have enough samples.
""")