# Why Target Normalization Changes Results Even With Threshold=0

## Executive Summary
You are absolutely correct that logically, with `target_threshold=0` and normalization that only divides by volatility (not subtracting mean/median), the classification should be identical. However, the models produce different results due to **how feature selection works**.

## The Root Cause: Feature Selection Uses Continuous Values

### What Happens:
1. **Feature Selection Stage** (BEFORE thresholding):
   - Uses `RandomForestRegressor` on **continuous target values**
   - Calculates MDI (Mean Decrease in Impurity) scores
   - These scores measure how well features predict the continuous target

2. **With Normalization**:
   - Target values are divided by IQR: `normalized = raw / IQR`
   - This changes the variance and distribution of targets
   - MDI scores change because the regression task is different
   - **Different features get selected**

3. **Model Training** (AFTER thresholding):
   - Uses binary classification (threshold=0)
   - But trains with **different features** selected in step 1
   - Different features → Different models → Different predictions

## Proof From Our Tests

### Feature Selection Differences (from actual run):
- **Without normalization**: Ret_16-32hr selected 93.8% of time
- **With normalization**: Ret_0-1hr selected 100% of time

### Why MDI Scores Change:
```python
# Example from test_tree_splitting.py:
Feature importance with raw target:       [0.227, 0.568, 0.205]
Feature importance with normalized target: [0.113, 0.599, 0.287]
Importance difference:                     [0.114, 0.031, 0.083]
```

The normalization changes how much each feature contributes to reducing variance in the regression tree, even though the signs remain identical.

## The Complete Chain of Events

1. **Data Processing**:
   - Raw target: `[0.393, -0.637, 0.392, ...]`
   - Normalized: `[0.568, -0.933, 0.577, ...]` (divided by IQR ~0.68)
   - Signs preserved: ✓

2. **Feature Selection** (RandomForestRegressor):
   - Fits on continuous values (not binary)
   - Normalized targets have different variance patterns
   - MDI scores change significantly
   - Different features selected

3. **Model Training** (ExtraTreesClassifier):
   - Threshold applied: both become `[1, 0, 1, ...]`
   - BUT different features available from step 2
   - Trees learn different patterns

4. **Prediction**:
   - 79.8% match rate (from our test)
   - 20.2% differ due to different features and tree structures

## Why This Design?

The system uses continuous values for feature selection because:
- MDI on continuous values captures more nuanced relationships
- Binary classification would lose information about magnitude
- This allows selecting features that predict both direction AND magnitude

## Solutions

If you want identical results with and without normalization:

1. **Option 1**: Modify feature selection to use binary classification
   - Change `RandomForestRegressor` to `RandomForestClassifier`
   - Apply threshold before feature selection
   - This would make feature selection less informative

2. **Option 2**: Disable feature selection when comparing
   - Set `feature_selection.enabled = false`
   - This ensures same features used in both cases
   - Results should be much more similar

3. **Option 3**: Accept the differences
   - The current design is actually sensible
   - Normalization genuinely changes the regression task
   - Different features may be optimal for normalized vs raw data

## Conclusion

Your intuition is correct: the classification task (positive vs negative) is identical. However, the feature selection stage operates on continuous values where normalization does make a difference. This cascades into different models even though the fundamental classification boundary (zero) remains the same.

The 20% difference in predictions is not a bug but a consequence of:
1. Different features being selected (due to MDI on continuous values)
2. Different tree structures learned with those features
3. Ensemble voting amplifying these differences