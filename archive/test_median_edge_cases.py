"""
Test median aggregation edge cases
===================================
Specifically testing what happens with even number of trees
and equal split of votes.
"""

import numpy as np

print("="*80)
print("TESTING MEDIAN AGGREGATION EDGE CASES")
print("="*80)

# Test case 1: 10 trees with 50/50 split
votes_10_split = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
median_10 = np.median(votes_10_split)
print(f"\n10 trees with 50/50 split:")
print(f"Votes: {votes_10_split}")
print(f"Median: {median_10}")
print(f"Expected: 0.5")
print(f"Correct: {median_10 == 0.5}")

# Test case 2: 10 trees with 4/6 split
votes_10_minority = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
median_10_min = np.median(votes_10_minority)
print(f"\n10 trees with 4/6 split:")
print(f"Votes: {votes_10_minority}")
print(f"Median: {median_10_min}")
print(f"Expected: 1.0")
print(f"Correct: {median_10_min == 1.0}")

# Test case 3: 11 trees (odd number)
votes_11 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
median_11 = np.median(votes_11)
print(f"\n11 trees (odd number):")
print(f"Votes: {votes_11}")
print(f"Median: {median_11}")
print(f"Expected: 1.0")
print(f"Correct: {median_11 == 1.0}")

# Test case 4: 100 trees with various splits
print("\n100 trees with various splits:")
for n_ones in [0, 25, 49, 50, 51, 75, 100]:
    votes = np.array([0]*(100-n_ones) + [1]*n_ones)
    median_val = np.median(votes)
    print(f"  {n_ones}/100 votes for 1: median = {median_val}")

# Test case 5: Small even numbers
print("\nSmall even numbers of trees:")
for n_trees in [2, 4, 6, 8]:
    for n_ones in range(n_trees + 1):
        votes = np.array([0]*(n_trees-n_ones) + [1]*n_ones)
        median_val = np.median(votes)
        print(f"  {n_trees} trees, {n_ones} ones: median = {median_val}")

print("\n" + "="*80)
print("IMPLICATIONS FOR THE MODEL")
print("="*80)

print("""
With EVEN number of trees:
- When votes are tied (e.g., 50/50), median = 0.5
- This means median aggregation CAN produce non-binary probabilities
- But only specific values are possible based on tree count

With ODD number of trees:
- Median will always be either 0 or 1 (binary)
- No intermediate values possible

Current config has n_trees = 200 (EVEN), so possible median values are:
- 0.0 (if < 100 trees vote 1)
- 0.5 (if exactly 100 trees vote 1)  
- 1.0 (if > 100 trees vote 1)

This is different from what the test showed earlier!
Let me verify with the actual model...
""")