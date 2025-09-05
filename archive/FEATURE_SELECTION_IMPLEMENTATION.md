# Automatic Feature Selection Implementation

## Overview
Added adaptive feature selection capability to the OMtree model that allows automatic selection of the best features at each walk-forward validation step. This enables the model to adapt to changing market conditions by dynamically choosing which features to use.

## Components Added

### 1. Feature Selection Module (`feature_selector.py`)
- **Methods Available:**
  - `correlation`: Select features with highest Pearson correlation to target
  - `spearman`: Use Spearman rank correlation
  - `mutual_info`: Mutual information between features and target
  - `rf_importance`: Random Forest feature importance
  - `forward_selection`: Iteratively add features that improve performance
  - `stability`: Select uncorrelated features with good target correlation
  - `top_n_variance`: Select features with highest variance

### 2. Configuration Settings (`OMtree_config.ini`)
```ini
[feature_selection]
enabled = false                # Enable/disable feature selection
method = correlation           # Selection method to use
n_features = 5                # Target number of features to select
min_features = 2              # Minimum features to always keep
max_features = 8              # Maximum features to allow
selection_lookback = 500      # Number of recent samples for selection
```

### 3. Validation Pipeline Integration (`OMtree_validation.py`)
- Automatically selects features at each walk-forward step
- Uses recent data (lookback window) to determine best features
- Tracks selection history throughout validation
- Generates feature selection report showing:
  - Feature usage frequency across all steps
  - Average selection scores
  - Selection patterns over time

### 4. GUI Controls (`OMtree_gui_v3.py`)
- Added "Automatic Feature Selection" section in Model Tester tab
- Controls for:
  - Enable/disable toggle
  - Method selection dropdown
  - Feature count parameters (target, min, max)
  - Lookback window setting
- Widgets automatically enable/disable based on selection toggle

## How It Works

1. **During Walk-Forward Validation:**
   - At each step, before training the model
   - Takes the most recent `selection_lookback` samples
   - Applies the selected method to rank features
   - Selects top N features (respecting min/max constraints)
   - Trains model using only selected features
   - Stores selection history for reporting

2. **Feature Selection Process:**
   - Each method scores features based on different criteria
   - Features are ranked by score
   - Top N features are selected for that training window
   - Selection adapts as market conditions change

3. **Reporting:**
   - Feature importance chart aggregates across all selected features
   - Selection report shows which features were most frequently selected
   - Provides insights into feature stability and relevance over time

## Benefits

1. **Adaptive Models**: Automatically adjusts to changing market conditions
2. **Reduced Overfitting**: Uses only the most relevant features at each point
3. **Feature Discovery**: Identifies which features are consistently useful
4. **Computational Efficiency**: Can reduce model complexity when fewer features are selected

## Usage Example

1. Enable feature selection in GUI or config file
2. Choose selection method (e.g., "correlation")
3. Set target number of features (e.g., 4 out of 8 available)
4. Run walk-forward validation
5. Review feature selection report to see patterns

## Test Results
The test script (`test_feature_selection.py`) confirms:
- All selection methods work correctly
- Features are selected adaptively during walk-forward
- Selection history is properly tracked
- Reports are generated successfully
- GUI controls integrate seamlessly

## Files Modified/Created
- Created: `feature_selector.py`
- Created: `test_feature_selection.py`
- Modified: `OMtree_config.ini` (added [feature_selection] section)
- Modified: `OMtree_validation.py` (integrated selection, fixed importance charts)
- Modified: `OMtree_gui_v3.py` (added GUI controls)

## Next Steps
The feature selection system is fully operational and ready for use. Users can now:
- Experiment with different selection methods
- Analyze which features are most stable/useful
- Build more adaptive trading models
- Reduce feature space for faster computation