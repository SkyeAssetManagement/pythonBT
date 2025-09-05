"""
Explain how min_leaf works with different tree depths
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

print("="*80)
print("UNDERSTANDING MIN_LEAF WITH TREE DEPTH")
print("="*80)

# Create sample data
np.random.seed(42)
n_samples = 750  # After bootstrap
X = np.random.randn(n_samples, 1)
y = (X.flatten() + np.random.randn(n_samples) * 0.5 > 0).astype(int)

print(f"\nTraining samples: {n_samples}")
print(f"Class distribution: {np.sum(y==0)} zeros, {np.sum(y==1)} ones")

# Test different min_samples_leaf values
min_leaf_values = [50, 100, 188, 250, 375]  # 6.7%, 13.3%, 25%, 33%, 50%

for min_leaf in min_leaf_values:
    print(f"\n{'='*60}")
    print(f"MIN_SAMPLES_LEAF = {min_leaf} ({min_leaf/n_samples*100:.1f}% of data)")
    print(f"{'='*60}")
    
    for max_depth in [1, 2, 3]:
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=42
        )
        
        try:
            tree.fit(X, y)
            
            # Analyze tree structure
            n_nodes = tree.tree_.node_count
            n_leaves = np.sum(tree.tree_.feature == -2)  # -2 indicates leaf
            actual_depth = tree.tree_.max_depth
            
            print(f"\nDepth {max_depth} tree:")
            print(f"  Actual depth: {actual_depth}")
            print(f"  Total nodes: {n_nodes}")
            print(f"  Leaf nodes: {n_leaves}")
            
            # Show the sample counts at each node
            if n_nodes <= 15:  # Only show for small trees
                print(f"  Node sample counts:")
                for i in range(n_nodes):
                    n_samples_node = tree.tree_.n_node_samples[i]
                    is_leaf = tree.tree_.feature[i] == -2
                    node_type = "LEAF" if is_leaf else "split"
                    print(f"    Node {i} ({node_type}): {n_samples_node} samples", end="")
                    if is_leaf and n_samples_node < min_leaf:
                        print(" <- VIOLATION!", end="")
                    print()
                    
        except Exception as e:
            print(f"\nDepth {max_depth}: Error - {e}")

# Detailed example with depth=2
print("\n" + "="*80)
print("DETAILED EXAMPLE: DEPTH=2, MIN_LEAF=25%")
print("="*80)

min_leaf = int(750 * 0.25)  # 188 samples
tree = DecisionTreeClassifier(
    max_depth=2,
    min_samples_leaf=min_leaf,
    random_state=42
)
tree.fit(X, y)

print(f"\nTree structure with {min_leaf} min samples per leaf:")
print("""
                    Root (750 samples)
                    /                \\
            Split at X <= a
                /          \\
        Node1 (≥188)      Node2 (≥188)
           /    \\            /    \\
    Leaf1   Leaf2      Leaf3   Leaf4
    (≥188)  (≥188)     (≥188)  (≥188)
""")

# What happens at each level?
print("\nConstraints at each level:")
print("1. ROOT SPLIT (depth 0→1):")
print(f"   - Each child must have ≥ {min_leaf} samples")
print(f"   - So split must be between {min_leaf} and {n_samples-min_leaf}")
print(f"   - Possible split range: {min_leaf/n_samples*100:.1f}% to {(n_samples-min_leaf)/n_samples*100:.1f}%")

print("\n2. SECOND SPLIT (depth 1→2):")
print("   - Each grandchild must have ≥ 188 samples")
print("   - If Node1 has 400 samples:")
print("     - Can split between 188 and 212 (400-188)")
print("   - If Node1 has 300 samples:")
print("     - Can split between 188 and 112... IMPOSSIBLE!")
print("     - Node becomes a leaf instead")

# Demonstrate the impossibility
print("\n" + "="*80)
print("WHY TREES OFTEN CAN'T REACH MAX DEPTH")
print("="*80)

# Calculate minimum samples needed at each depth
print("\nMinimum samples needed to reach depth:")
for depth in range(1, 5):
    min_needed = min_leaf * (2 ** depth)
    print(f"  Depth {depth}: {min_needed} samples (need {2**depth} leaves × {min_leaf} each)")
    if min_needed > n_samples:
        print(f"    → IMPOSSIBLE with {n_samples} samples!")

# Practical example
print("\n" + "="*80)
print("PRACTICAL IMPLICATIONS")
print("="*80)

print(f"""
With your settings:
- 750 samples (after bootstrap)
- min_leaf = 188 (25%)
- max_depth = 2

DEPTH 1 (2 leaves):
✓ Needs 376 samples minimum (2 × 188)
✓ CAN achieve this depth

DEPTH 2 (4 leaves):
✗ Needs 752 samples minimum (4 × 188)
✗ CANNOT achieve this depth!

RESULT: Tree will stop at depth 1, creating only 2 leaves.

To get depth 2 with 750 samples:
- Max min_leaf_fraction = 750/4 = 187.5 samples
- So min_leaf_fraction must be < 0.25 (actually < 187.5/750 = 0.249)

RECOMMENDATION:
- For depth 2: Use min_leaf_fraction ≤ 0.20
- For depth 3: Use min_leaf_fraction ≤ 0.10
- For depth 4: Use min_leaf_fraction ≤ 0.05
""")