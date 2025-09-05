"""
Test ultra-simple model behavior
"""
import numpy as np
import pandas as pd
from OMtree_model import DirectionalTreeEnsemble
from OMtree_preprocessing import DataPreprocessor

# Load actual data
df = pd.read_csv('C:/Users/jd/OM3/DTSmlDATA7x7.csv')

# Get just the feature and target we need
feature_col = 'Ret_8-16hr'
target_col = 'Ret_fwd6hr'

# Ensure we have these columns
if feature_col not in df.columns or target_col not in df.columns:
    print(f"Missing required columns. Available: {df.columns.tolist()}")
else:
    # Extract data
    X = df[[feature_col]].values
    y = df[target_col].values
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    print("="*60)
    print("ULTRA-SIMPLE MODEL TEST")
    print("="*60)
    print(f"Using {len(X)} samples")
    print(f"Feature: {feature_col}")
    print(f"Target: {target_col}")
    
    # Stats on the data
    print(f"\nData Statistics:")
    print(f"Feature mean: {X.mean():.4f}")
    print(f"Feature median: {np.median(X):.4f}")
    print(f"Target mean: {y.mean():.4f}")
    print(f"Target median: {np.median(y):.4f}")
    
    # Train the model with current config
    model = DirectionalTreeEnsemble('OMtree_config.ini', verbose=False)
    
    print(f"\nModel Configuration:")
    print(f"n_trees: {model.n_trees}")
    print(f"max_depth: {model.max_depth}")
    print(f"min_leaf_fraction: {model.min_leaf_fraction}")
    print(f"vote_threshold: {model.vote_threshold}")
    print(f"target_threshold: {model.target_threshold}")
    
    # Use first 1000 samples for training
    train_size = min(1000, len(X) - 100)
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    X_test = X[train_size:train_size+100]
    y_test = y[train_size:train_size+100]
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"\n" + "="*40)
    print("RESULTS WITH CURRENT SETTINGS")
    print("="*40)
    
    # With single tree, probability = tree output (0 or 1)
    unique_probs = np.unique(probabilities)
    print(f"Unique probability values: {unique_probs}")
    
    # Prediction statistics
    n_trades = np.sum(predictions == 1)
    print(f"\nPredictions:")
    print(f"  Trade signals (1): {n_trades}/{len(predictions)} ({n_trades/len(predictions)*100:.1f}%)")
    print(f"  No trade (0): {len(predictions)-n_trades}/{len(predictions)} ({(len(predictions)-n_trades)/len(predictions)*100:.1f}%)")
    
    # Find the split point
    if len(model.trees) > 0:
        tree = model.trees[0]
        if hasattr(tree, 'tree_'):
            # Get the threshold used for splitting
            threshold_idx = 0  # Root node
            threshold = tree.tree_.threshold[threshold_idx]
            print(f"\nTree split threshold: {threshold:.6f}")
            print(f"Feature median in training: {np.median(X_train):.6f}")
            
    # Test with different vote thresholds
    print(f"\n" + "="*40)
    print("EFFECT OF VOTE THRESHOLD")
    print("="*40)
    
    for vt in [0.0, 0.5, 0.99, 1.0]:
        model.vote_threshold = vt
        preds = model.predict(X_test)
        n_trades = np.sum(preds == 1)
        print(f"vote_threshold = {vt:.2f}: {n_trades} trades ({n_trades/len(preds)*100:.1f}%)")
    
    # What happens with vote_threshold = 1.0?
    print(f"\n" + "="*40)
    print("WITH vote_threshold = 1.0:")
    print("="*40)
    print("Since you have 1 tree that outputs 0 or 1:")
    print("- If tree outputs 1, probability = 1.0 → TRADE")
    print("- If tree outputs 0, probability = 0.0 → NO TRADE")
    print("Result: Same as any threshold between 0.01 and 1.0")
    print("\nOnly vote_threshold = 0 would trade on EVERY sample")
    print("(but that would be meaningless)")