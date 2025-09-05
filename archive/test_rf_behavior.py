"""
Test Random Forest behavior with different parameters
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Generate simple test data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 8)
# Feature 0 and 1 are important, others are noise
y = 3 * X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.1

print("Testing Random Forest with different parameters:\n")
print("Expected: Feature 0 should be most important, then 1, then 2, others near 0")
print("="*60)

# Test different combinations
test_cases = [
    {"max_depth": 1, "min_samples_split": 2, "desc": "Depth=1, min_split=2"},
    {"max_depth": 1, "min_samples_split": 100, "desc": "Depth=1, min_split=100"}, 
    {"max_depth": 1, "min_samples_split": 400, "desc": "Depth=1, min_split=400"},
    {"max_depth": 3, "min_samples_split": 2, "desc": "Depth=3, min_split=2"},
    {"max_depth": 3, "min_samples_split": 100, "desc": "Depth=3, min_split=100"},
    {"max_depth": None, "min_samples_split": 2, "desc": "Depth=None, min_split=2"},
    {"max_depth": None, "min_samples_split": 100, "desc": "Depth=None, min_split=100"},
]

for params in test_cases:
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        max_samples=int(0.8 * n_samples),
        random_state=42
    )
    
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    print(f"\n{params['desc']}:")
    print(f"  Importances: {importances.round(3)}")
    print(f"  Top 3: {np.argsort(importances)[-3:][::-1]} (indices)")
    
    # Check if trees actually split
    tree = rf.estimators_[0]
    n_nodes = tree.tree_.node_count
    print(f"  First tree nodes: {n_nodes} (1 = no splits)")

print("\n" + "="*60)
print("\nKey insights:")
print("- With max_depth=1 (stumps), min_samples_split has NO effect")
print("- Trees with depth=1 can only make 1 split -> max 3 nodes")
print("- min_samples_split only matters when depth allows multiple splits")