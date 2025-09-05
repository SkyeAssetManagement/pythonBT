# Regression Mode Implementation

## Overview
Added an option to use RandomForestRegressor/ExtraTreesRegressor for both feature selection and model training/prediction, providing an alternative to the classification approach.

## Configuration
In `OMtree_config.ini`:
```ini
[model]
regression_mode = false  # Set to true to use regression mode
```

## How It Works

### Classification Mode (regression_mode = false)
1. **Feature Selection**: Uses RandomForestClassifier on binary labels (threshold applied)
2. **Model Training**: Uses ExtraTreesClassifier on binary labels
3. **Prediction**: Direct binary classification with vote threshold

### Regression Mode (regression_mode = true)
1. **Feature Selection**: Uses RandomForestRegressor on continuous target values
2. **Model Training**: Uses ExtraTreesRegressor on continuous target values
3. **Prediction**: 
   - Trees predict continuous values
   - Apply threshold to convert to binary (value > threshold → 1, else → 0)
   - Apply vote threshold as usual

## Code Changes

### 1. feature_selector.py
- Added `regression_mode` parameter
- Conditionally uses Regressor or Classifier based on mode
- In regression mode: fits on continuous values
- In classification mode: fits on binary labels (threshold applied)

### 2. OMtree_model.py
- Added `regression_mode` parameter
- Supports both DecisionTreeRegressor and DecisionTreeClassifier
- ExtraTreesRegressor and ExtraTreesClassifier support
- Prediction applies threshold to regression outputs

### 3. OMtree_validation.py
- Reads `regression_mode` from config
- Passes to both FeatureSelector and DirectionalTreeEnsemble
- Displays mode in validation output

## Key Differences Between Modes

| Aspect | Classification Mode | Regression Mode |
|--------|-------------------|-----------------|
| Feature Selection | Binary labels (threshold applied) | Continuous values |
| Model Training | Binary classification | Continuous regression |
| Tree Output | 0 or 1 | Continuous value |
| Final Prediction | Direct voting | Threshold then voting |
| Normalization Effect | No effect (signs preserved) | Affects feature selection |

## When to Use Each Mode

### Use Classification Mode When:
- You want consistent results regardless of normalization
- The binary classification boundary is most important
- You prefer simpler, more interpretable models

### Use Regression Mode When:
- Magnitude of returns matters for feature selection
- You want to identify features that predict large moves
- You're willing to accept normalization affecting results

## Test Results

### Classification Mode (2 features, no feature selection):
- Hit Rate: 55.5%
- Total P&L: 29.45
- 881 trades

### Regression Mode (same 2 features):
- Hit Rate: 56.6%
- Total P&L: 17.20
- 355 trades (higher vote threshold = fewer but better trades)

## Usage Example

To switch between modes:

```python
# In config file
[model]
regression_mode = true   # Enable regression mode
# or
regression_mode = false  # Use classification mode (default)
```

The system automatically handles all the complexity - just set the flag and run!

## Benefits of This Implementation

1. **Flexibility**: Choose the approach that best fits your strategy
2. **Consistency**: Same configuration and workflow for both modes
3. **Compatibility**: Works with all existing features (normalization, feature selection, etc.)
4. **Performance**: Can optimize for different objectives (frequency vs quality)