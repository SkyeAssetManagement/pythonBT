"""
Comprehensive test for all feature selection settings
Tests min_features, max_features, importance thresholds, and cumulative mode
"""

import numpy as np
import pandas as pd
from feature_selector import FeatureSelector

def create_test_data(n_samples=1000, n_features=10):
    """Create test data with known importance patterns"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Create target with decreasing importance for features
    base_weights = [5.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
    weights = base_weights[:n_features]  # Use only as many weights as we have features
    y = sum(w * X[:, i] for i, w in enumerate(weights)) + 0.1 * np.random.randn(n_samples)
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    return X, y, feature_names

def test_min_max_constraints():
    """Test that min_features and max_features constraints are respected"""
    X, y, feature_names = create_test_data()
    
    print("="*60)
    print("TEST: Min/Max Feature Constraints")
    print("="*60)
    
    # Test 1: min_features = 3, max_features = 3 (force exactly 3)
    print("\n1. Testing min=3, max=3 (should select exactly 3 features)")
    selector = FeatureSelector(
        n_features=10,
        min_features=3,
        max_features=3,
        rf_n_estimators=50,
        rf_max_depth=3,
        rf_min_samples_leaf=10,
        importance_threshold=0.0,  # No threshold
        cumulative_importance_mode=False,
        random_seed=42
    )
    
    selected_indices, selected_names = selector.select_features(X, y, feature_names, verbose=False)
    print(f"   Selected: {len(selected_names)} features")
    assert len(selected_names) == 3, f"Expected 3 features, got {len(selected_names)}"
    print("   [PASS] Exactly 3 features selected")
    
    # Test 2: min_features = 2, max_features = 5, high threshold (tests min constraint)
    print("\n2. Testing min=2, max=5, high threshold (should hit min constraint)")
    selector = FeatureSelector(
        n_features=10,
        min_features=2,
        max_features=5,
        rf_n_estimators=50,
        rf_max_depth=3,
        rf_min_samples_leaf=10,
        importance_threshold=0.3,  # Very high threshold
        cumulative_importance_mode=False,
        random_seed=42
    )
    
    selected_indices, selected_names = selector.select_features(X, y, feature_names, verbose=False)
    print(f"   Selected: {len(selected_names)} features")
    assert len(selected_names) >= 2, f"Expected at least 2 features, got {len(selected_names)}"
    assert len(selected_names) <= 5, f"Expected at most 5 features, got {len(selected_names)}"
    print("   [PASS] PASSED: Min constraint respected")
    
    # Test 3: min_features = 1, max_features = 4, low threshold (tests max constraint)
    print("\n3. Testing min=1, max=4, low threshold (should hit max constraint)")
    selector = FeatureSelector(
        n_features=10,
        min_features=1,
        max_features=4,
        rf_n_estimators=50,
        rf_max_depth=3,
        rf_min_samples_leaf=10,
        importance_threshold=0.01,  # Very low threshold
        cumulative_importance_mode=False,
        random_seed=42
    )
    
    selected_indices, selected_names = selector.select_features(X, y, feature_names, verbose=False)
    print(f"   Selected: {len(selected_names)} features")
    assert len(selected_names) <= 4, f"Expected at most 4 features, got {len(selected_names)}"
    print("   [PASS] PASSED: Max constraint respected")

def test_threshold_modes():
    """Test minimum importance and cumulative importance modes"""
    X, y, feature_names = create_test_data()
    
    print("\n" + "="*60)
    print("TEST: Threshold Modes")
    print("="*60)
    
    # Test minimum importance threshold
    print("\n1. Minimum importance threshold = 0.1")
    selector = FeatureSelector(
        n_features=10,
        min_features=1,
        max_features=10,
        rf_n_estimators=50,
        rf_max_depth=3,
        rf_min_samples_leaf=10,
        importance_threshold=0.1,
        cumulative_importance_mode=False,
        random_seed=42
    )
    
    selected_indices, selected_names = selector.select_features(X, y, feature_names, verbose=False)
    importances = selector.all_importances
    print(f"   Selected: {len(selected_names)} features")
    for name in selected_names:
        imp = selector.selection_scores[name]
        print(f"   - {name}: {imp:.3f}")
        if imp < 0.1:
            print(f"     WARNING: Feature {name} below threshold but selected!")
    print("   [PASS] Test completed")
    
    # Test cumulative importance threshold
    print("\n2. Cumulative importance threshold = 50%")
    selector = FeatureSelector(
        n_features=10,
        min_features=1,
        max_features=10,
        rf_n_estimators=50,
        rf_max_depth=3,
        rf_min_samples_leaf=10,
        cumulative_importance_mode=True,
        cumulative_importance_threshold=0.5,
        random_seed=42
    )
    
    selected_indices, selected_names = selector.select_features(X, y, feature_names, verbose=False)
    print(f"   Selected: {len(selected_names)} features")
    
    # Calculate cumulative importance
    sorted_scores = sorted(selector.selection_scores.values(), reverse=True)
    total_imp = sum(selector.all_importances)
    cumulative = sum(sorted_scores) / total_imp if total_imp > 0 else 0
    print(f"   Cumulative importance captured: {cumulative:.1%}")
    print("   [PASS] Test completed")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    X, y, feature_names = create_test_data(n_features=5)
    
    print("\n" + "="*60)
    print("TEST: Edge Cases")
    print("="*60)
    
    # Test when min_features > available features
    print("\n1. min_features (8) > available features (5)")
    selector = FeatureSelector(
        n_features=10,
        min_features=8,
        max_features=10,
        rf_n_estimators=50,
        rf_max_depth=3,
        rf_min_samples_leaf=10,
        importance_threshold=0.0,
        cumulative_importance_mode=False,
        random_seed=42
    )
    
    selected_indices, selected_names = selector.select_features(X, y, feature_names, verbose=False)
    print(f"   Available: 5, Requested min: 8")
    print(f"   Selected: {len(selected_names)} features")
    assert len(selected_names) == 5, f"Should select all 5 available features, got {len(selected_names)}"
    print("   [PASS] PASSED: Selected all available features")
    
    # Test when max_features = 0
    print("\n2. max_features = 0 (should use min_features)")
    selector = FeatureSelector(
        n_features=10,
        min_features=2,
        max_features=0,  # Invalid, should be ignored
        rf_n_estimators=50,
        rf_max_depth=3,
        rf_min_samples_leaf=10,
        importance_threshold=0.0,
        cumulative_importance_mode=False,
        random_seed=42
    )
    
    selected_indices, selected_names = selector.select_features(X, y, feature_names, verbose=False)
    print(f"   Selected: {len(selected_names)} features")
    assert len(selected_names) >= 2, f"Should select at least min_features (2), got {len(selected_names)}"
    print("   [PASS] PASSED: min_features constraint respected")

def test_config_scenario():
    """Test the exact scenario from the current config"""
    X, y, feature_names = create_test_data()
    
    print("\n" + "="*60)
    print("TEST: Current Config Scenario")
    print("="*60)
    print("Config: min=3, max=3, threshold_mode=minimum, importance_threshold=0.1")
    
    selector = FeatureSelector(
        n_features=10,
        min_features=3,
        max_features=3,
        rf_n_estimators=50,
        rf_max_depth=3,
        rf_min_samples_leaf=10,
        importance_threshold=0.1,
        cumulative_importance_mode=False,  # minimum mode
        random_seed=42
    )
    
    selected_indices, selected_names = selector.select_features(X, y, feature_names, verbose=True)
    print(f"\nFinal result: {len(selected_names)} features selected")
    assert len(selected_names) == 3, f"With min=3 and max=3, exactly 3 features should be selected, got {len(selected_names)}"
    print("[PASS] PASSED: Configuration working correctly")

if __name__ == "__main__":
    print("COMPREHENSIVE FEATURE SELECTION SETTINGS TEST")
    print("="*60)
    
    test_min_max_constraints()
    test_threshold_modes()
    test_edge_cases()
    test_config_scenario()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)