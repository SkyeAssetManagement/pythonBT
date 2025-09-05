# Extra Trees Implementation

## Overview
Successfully added **Extra Trees** (Extremely Randomized Trees) as an alternative algorithm option alongside the existing Decision Trees implementation.

## What are Extra Trees?
Extra Trees is an ensemble method similar to Random Forest but with additional randomization:
- **Random Forests**: Select best split among random subset of features
- **Extra Trees**: Select random split thresholds for random features
- More randomization = less overfitting, faster training

## Implementation Details

### 1. Configuration (`OMtree_config.ini`)
```ini
[model]
algorithm = decision_trees  # Options: decision_trees, extra_trees
```

### 2. Model Code (`OMtree_model.py`)
- Added `algorithm` parameter to choose between methods
- Integrated `ExtraTreesClassifier` from scikit-learn
- Both algorithms use same interface for predictions
- Extra Trees uses all CPU cores (`n_jobs=-1`)

### 3. GUI Integration (`OMtree_gui_v3.py`)
- Added "Algorithm" dropdown in Model Parameters section
- Options: `decision_trees`, `extra_trees`
- Dynamic description shows benefits of each
- Saves/loads algorithm choice in config

## Algorithm Comparison

| Feature | Decision Trees | Extra Trees |
|---------|---------------|-------------|
| Split Selection | Best split | Random split |
| Training Speed | Moderate | Faster |
| Overfitting Risk | Higher | Lower |
| Reproducibility | Exact | Statistical |
| Best For | Small datasets | Large datasets |
| Feature Handling | Sequential | Better with many features |

## Usage

### In GUI:
1. Go to Model Tester tab
2. Under Model Parameters, select Algorithm:
   - `decision_trees`: Standard approach
   - `extra_trees`: More randomized approach
3. Configure other parameters as usual
4. Run walk-forward validation

### In Code:
```python
# Configure
config['model']['algorithm'] = 'extra_trees'

# Model automatically uses selected algorithm
model = DirectionalTreeEnsemble('config.ini')
model.fit(X, y)
predictions = model.predict(X_test)
```

## Benefits of Extra Trees

### 1. Reduced Overfitting
- Random splits prevent memorizing training data
- Better generalization to new data
- Especially useful with many features

### 2. Faster Training
- No need to evaluate all possible splits
- Parallelization across CPU cores
- Scales better with data size

### 3. Better with High Dimensionality
- Handles many features well
- Natural feature selection through randomization
- Less prone to noise in individual features

## When to Use Each

### Use Extra Trees when:
- Dataset is large (>10,000 samples)
- Many features (>10)
- Concerned about overfitting
- Need faster training
- Working with noisy data

### Use Decision Trees when:
- Dataset is small (<1,000 samples)
- Few features (<5)
- Need interpretability
- Require exact reproducibility
- Want fine control over trees

## Performance Notes
- Both algorithms support same parameters (n_trees, max_depth, etc.)
- Vote threshold applies to both
- Bootstrap fraction works for both
- Compatible with all preprocessing options

## Files Modified
- `OMtree_model.py` - Added Extra Trees implementation
- `OMtree_config.ini` - Added algorithm parameter
- `OMtree_gui_v3.py` - Added algorithm dropdown

## Testing Results
- Both algorithms work correctly
- High agreement between predictions (>70%)
- Extra Trees shows more consistent predictions
- Successfully integrated with walk-forward validation

The implementation maintains backward compatibility - existing configs default to `decision_trees` if algorithm not specified.