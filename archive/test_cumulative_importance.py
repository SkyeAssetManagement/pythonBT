"""
Test script for cumulative importance threshold feature selection
"""

import numpy as np
import pandas as pd
from feature_selector import FeatureSelector

def test_cumulative_importance():
    """Test the cumulative importance threshold feature selection"""
    
    print("="*60)
    print("Testing Cumulative Importance Feature Selection")
    print("="*60)
    
    # Create synthetic data with known feature importance patterns
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create features with decreasing importance
    X = np.random.randn(n_samples, n_features)
    
    # Create target with strong dependency on first few features
    # Feature 0: weight = 5.0 (most important)
    # Feature 1: weight = 3.0
    # Feature 2: weight = 2.0
    # Feature 3: weight = 1.0
    # Feature 4: weight = 0.5
    # Others: noise
    y = (5.0 * X[:, 0] + 
         3.0 * X[:, 1] + 
         2.0 * X[:, 2] + 
         1.0 * X[:, 3] + 
         0.5 * X[:, 4] + 
         0.1 * np.random.randn(n_samples))
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    print("\n" + "="*60)
    print("TEST 1: Minimum Importance Threshold Mode (Traditional)")
    print("="*60)
    
    # Test with minimum importance threshold
    selector_min = FeatureSelector(
        n_features=10,
        min_features=2,
        max_features=8,
        rf_n_estimators=100,
        rf_max_depth=5,
        rf_min_samples_leaf=10,
        rf_bootstrap_fraction=0.8,
        importance_threshold=0.05,  # 5% minimum importance
        cumulative_importance_mode=False,  # Use traditional mode
        random_seed=42
    )
    
    selected_indices, selected_names = selector_min.select_features(
        X, y, feature_names, verbose=True
    )
    
    print(f"\nSelected {len(selected_names)} features using minimum threshold:")
    for name in selected_names:
        print(f"  - {name}")
    
    print("\n" + "="*60)
    print("TEST 2: Cumulative Importance Mode (95% threshold)")
    print("="*60)
    
    # Test with cumulative importance threshold
    selector_cum = FeatureSelector(
        n_features=10,
        min_features=2,
        max_features=8,
        rf_n_estimators=100,
        rf_max_depth=5,
        rf_min_samples_leaf=10,
        rf_bootstrap_fraction=0.8,
        importance_threshold=0.05,  # Not used in cumulative mode
        cumulative_importance_mode=True,  # Use cumulative mode
        cumulative_importance_threshold=0.95,  # Capture 95% of importance
        random_seed=42
    )
    
    selected_indices, selected_names = selector_cum.select_features(
        X, y, feature_names, verbose=True
    )
    
    print(f"\nSelected {len(selected_names)} features using cumulative threshold:")
    for name in selected_names:
        print(f"  - {name}")
    
    print("\n" + "="*60)
    print("TEST 3: Cumulative Importance Mode (80% threshold)")
    print("="*60)
    
    # Test with lower cumulative threshold
    selector_cum_80 = FeatureSelector(
        n_features=10,
        min_features=2,
        max_features=8,
        rf_n_estimators=100,
        rf_max_depth=5,
        rf_min_samples_leaf=10,
        rf_bootstrap_fraction=0.8,
        cumulative_importance_mode=True,
        cumulative_importance_threshold=0.80,  # Capture 80% of importance
        random_seed=42
    )
    
    selected_indices, selected_names = selector_cum_80.select_features(
        X, y, feature_names, verbose=True
    )
    
    print(f"\nSelected {len(selected_names)} features using 80% cumulative threshold:")
    for name in selected_names:
        print(f"  - {name}")
    
    # Compare all importances
    print("\n" + "="*60)
    print("COMPARISON OF FEATURE IMPORTANCES")
    print("="*60)
    
    if selector_cum.all_importances is not None:
        importances = selector_cum.all_importances
        sorted_idx = np.argsort(importances)[::-1]
        
        cumsum = np.cumsum(importances[sorted_idx])
        total = np.sum(importances)
        
        print("\nFeature Importance Rankings:")
        print("-" * 50)
        print(f"{'Rank':<6} {'Feature':<12} {'Importance':<12} {'Cumulative %':<12}")
        print("-" * 50)
        
        for rank, idx in enumerate(sorted_idx, 1):
            imp = importances[idx]
            cum_pct = (cumsum[rank-1] / total) * 100 if total > 0 else 0
            print(f"{rank:<6} {feature_names[idx]:<12} {imp:.4f}{'':3} {cum_pct:>8.1f}%")
        
        print("-" * 50)
        print(f"Total importance: {total:.4f}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    test_cumulative_importance()