# Why RandomForestRegressor for Feature Selection?

## Current Design

### Feature Selection Stage:
- Uses **RandomForestRegressor** on **continuous target values** (returns)
- Measures MDI (Mean Decrease in Impurity) for predicting the actual return magnitude
- Selects features that best predict the continuous return values

### Model Training Stage:
- Uses **ExtraTreesClassifier** on **binary labels** (after applying threshold)
- Trains on profitable (1) vs unprofitable (0) based on threshold
- Makes binary predictions for trading decisions

## Why This Design Makes Sense

### 1. Information Preservation
- **Continuous values contain more information** than binary labels
- A feature that predicts +5% vs +0.1% returns (both positive) is more valuable
- Binary classification would treat both as simply "1" (profitable)

### 2. Magnitude Matters
- In trading, you want features that predict not just direction but also magnitude
- Features that identify large moves are more valuable than those identifying small moves
- Regressor captures this; classifier doesn't

### 3. Example Scenario
Consider two features:
- Feature A: Predicts whether return > 0 with 60% accuracy
- Feature B: Predicts whether return > 0 with 55% accuracy BUT identifies the big moves

With **Classifier** for feature selection:
- Feature A scores higher (better classification accuracy)

With **Regressor** for feature selection:
- Feature B scores higher (better at predicting magnitude)
- This is likely better for trading!

## Alternative Approach: Using Classifier

If you wanted feature selection to match the final task exactly:

```python
# Change in feature_selector.py:
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# In select_features method:
# Apply threshold first
y_binary = (y > threshold).astype(int)

if self.algorithm == 'extra_trees':
    self.rf_model = ExtraTreesClassifier(...)
else:
    self.rf_model = RandomForestClassifier(...)

self.rf_model.fit(X, y_binary)  # Fit on binary labels
```

### Pros of Classifier Approach:
- Feature selection matches the final classification task
- Normalization wouldn't affect feature selection (since threshold=0 preserves signs)
- More consistent results between normalized/non-normalized

### Cons of Classifier Approach:
- Loses information about return magnitude
- May select features that are good at direction but bad at identifying big moves
- Could lead to more frequent but smaller profitable trades

## Recommendation

The current design (Regressor for selection, Classifier for prediction) is actually **sophisticated and sensible** for trading:

1. **Feature Selection**: Find features that predict return magnitude (more informative)
2. **Model Training**: Use those features for binary classification (actionable signals)

This two-stage approach extracts maximum information from the data while providing clear trading signals.

## If You Want Identical Results with Normalization

Options:
1. **Keep current design**: Accept that normalization changes feature selection (this is actually informative!)
2. **Switch to Classifier**: Would make results more similar but potentially less profitable
3. **Disable feature selection**: Use fixed features when comparing normalization effects
4. **Hybrid approach**: Use both regression and classification scores for feature selection