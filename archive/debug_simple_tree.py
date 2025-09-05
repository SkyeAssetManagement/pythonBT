"""
Debug why the simple tree is giving all 1s
"""
import numpy as np
import pandas as pd
from OMtree_model import DirectionalTreeEnsemble

# Load data
df = pd.read_csv('C:/Users/jd/OM3/DTSmlDATA7x7.csv')

# Get feature and target
X = df[['Ret_8-16hr']].values
y = df['Ret_fwd6hr'].values

# Remove NaN
valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X = X[valid_mask]
y = y[valid_mask]

print("="*60)
print("DEBUGGING SIMPLE TREE BEHAVIOR")
print("="*60)

# Train model
model = DirectionalTreeEnsemble('OMtree_config.ini', verbose=False)

# Check first 1000 samples
X_train = X[:1000]
y_train = y[:1000]

print(f"\nTraining data stats:")
print(f"X_train min: {X_train.min():.4f}, max: {X_train.max():.4f}")
print(f"X_train median: {np.median(X_train):.4f}")
print(f"y_train min: {y_train.min():.4f}, max: {y_train.max():.4f}")
print(f"y_train median: {np.median(y_train):.4f}")

# With target_threshold = 0, what are the labels?
print(f"\nTarget threshold: {model.target_threshold}")
labels = model.create_directional_labels(y_train)
print(f"Label distribution: {np.unique(labels, return_counts=True)}")
print(f"Proportion of 1s: {np.mean(labels):.2%}")

# Fit the model
model.fit(X_train, y_train)

# Check tree structure
if len(model.trees) > 0:
    tree = model.trees[0]
    print(f"\nTree information:")
    print(f"Number of nodes: {tree.tree_.node_count}")
    print(f"Max depth reached: {tree.tree_.max_depth}")
    
    # Get split information
    if tree.tree_.node_count > 1:
        # Root node (index 0) contains the split
        feature = tree.tree_.feature[0]
        threshold = tree.tree_.threshold[0]
        print(f"Split feature index: {feature}")
        print(f"Split threshold: {threshold:.6f}")
        
        # Left child (values <= threshold)
        left_child = tree.tree_.children_left[0]
        left_value = tree.tree_.value[left_child][0][0]
        
        # Right child (values > threshold)
        right_child = tree.tree_.children_right[0]
        right_value = tree.tree_.value[right_child][0][0]
        
        print(f"\nLeft child (X <= {threshold:.6f}): predicts {left_value}")
        print(f"Right child (X > {threshold:.6f}): predicts {right_value}")
        
        # How many samples go each way?
        n_left = np.sum(X_train <= threshold)
        n_right = np.sum(X_train > threshold)
        print(f"\nTraining samples distribution:")
        print(f"Left (<=): {n_left} samples ({n_left/len(X_train)*100:.1f}%)")
        print(f"Right (>): {n_right} samples ({n_right/len(X_train)*100:.1f}%)")
    else:
        # No split made - tree is just a leaf
        print("Tree has no splits - it's just a single leaf!")
        print(f"Leaf value: {tree.tree_.value[0][0][0]}")

# Test predictions
X_test = X[1000:1100]
predictions = model.predict(X_test)
probs = model.predict_proba(X_test)

print(f"\nTest predictions:")
print(f"Unique predictions: {np.unique(predictions)}")
print(f"Unique probabilities: {np.unique(probs)}")
print(f"Prediction distribution: {np.unique(predictions, return_counts=True)}")

# Why all 1s?
print(f"\n" + "="*40)
print("EXPLANATION:")
print("="*40)

if model.target_threshold == 0:
    print("With target_threshold = 0:")
    print("- Any positive return (y > 0) becomes label 1")
    print("- Any non-positive return (y <= 0) becomes label 0")
    print(f"- In your data, {np.mean(y_train > 0)*100:.1f}% of targets are positive")
    
    if np.mean(labels) > 0.5:
        print("\nSince majority of labels are 1, and min_leaf_fraction = 0.5,")
        print("the tree cannot make a split that would put all 1s on one side!")
        print("So it becomes a single leaf predicting the majority class (1).")

# Test with different target threshold
print(f"\n" + "="*40)
print("TESTING WITH MEDIAN TARGET THRESHOLD")
print("="*40)

model2 = DirectionalTreeEnsemble('OMtree_config.ini', verbose=False)
model2.target_threshold = np.median(y_train)
print(f"Setting target_threshold to median: {model2.target_threshold:.4f}")

labels2 = model2.create_directional_labels(y_train)
print(f"New label distribution: {np.unique(labels2, return_counts=True)}")
print(f"Proportion of 1s: {np.mean(labels2):.2%}")

model2.fit(X_train, y_train)
predictions2 = model2.predict(X_test)
print(f"Predictions with median threshold: {np.unique(predictions2, return_counts=True)}")