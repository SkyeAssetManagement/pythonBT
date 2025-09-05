# Feature Selector Fixed to Match Model Logic

## The Problem
You correctly identified that the feature selector should use the same logic as the model. Previously:
- **Feature Selection**: Used `RandomForestRegressor` on continuous target values
- **Model Training**: Used `ExtraTreesClassifier` on binary labels (after threshold)
- This mismatch caused different features to be selected with/without normalization

## The Solution
Modified `feature_selector.py` to:
1. Apply the target threshold FIRST (creating binary labels)
2. Use `RandomForestClassifier` or `ExtraTreesClassifier` (matching model)
3. Select features based on classification importance, not regression

## Code Changes

### feature_selector.py
```python
# OLD (Regressor on continuous values):
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
self.rf_model = RandomForestRegressor(...)
self.rf_model.fit(X, y)  # y is continuous

# NEW (Classifier on binary labels):
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
y_binary = (y > self.target_threshold).astype(int)  # Apply threshold first!
self.rf_model = RandomForestClassifier(...)
self.rf_model.fit(X, y_binary)  # y_binary matches model training
```

## Test Results

### With Classifier-Based Feature Selection
| Metric | Without Normalization | With Normalization | Difference |
|--------|----------------------|--------------------|------------|
| Trades | 906 | 906 | **0** |
| Hit Rate | 55.7% | 55.7% | **0.0%** |
| Total P&L | 16.17 | 16.17 | **0.00** |
| **Prediction Match** | - | - | **100.0%** |

## Key Benefits

1. **Consistency**: Feature selection now matches model training exactly
2. **Predictability**: Normalization doesn't affect results when threshold=0
3. **Logical Coherence**: The entire pipeline uses the same classification task
4. **Same Performance**: No loss in performance, just more consistent behavior

## Feature Selection Comparison

### Before (Different features selected):
- **Without norm**: Ret_2-4hr, Ret_4-8hr dominated
- **With norm**: Ret_0-1hr dominated
- Led to 20% different predictions

### After (Same features selected):
- **Both**: Ret_0-1hr (81.2%), Ret_2-4hr (43.8%), Ret_4-8hr (43.8%)
- 100% identical predictions

## Conclusion

Your instinct was absolutely correct - the feature selector should use the same logic as the model. By switching from regression to classification for feature selection, we've achieved perfect consistency between normalized and non-normalized runs when threshold=0.

The system now behaves exactly as you logically expected: since normalization only divides by volatility (doesn't change signs), and the threshold is at zero, the classification task is identical, leading to identical features being selected and identical models being trained.