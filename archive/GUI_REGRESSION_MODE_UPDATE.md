# GUI Update: Regression Mode Option Added

## What Was Added

The `regression_mode` option has been successfully integrated into the GUI (OMtree_gui_v3.py).

## Location in GUI

In the **Model Parameters** section, you'll now see:
- **Regression Mode** dropdown (3rd option, after Algorithm)
- Options: `false` (default) or `true`
- Dynamic description that updates based on selection:
  - `false`: "Use classifiers (binary labels)"
  - `true`: "Use regressors (continuous targets)"

## GUI Layout

```
Model Parameters
├── Model Type: [longonly/shortonly]
├── Algorithm: [decision_trees/extra_trees]
├── Regression Mode: [false/true]  ← NEW!
├── Probability Aggregation: [mean/median]
├── Balanced Bootstrap: [false/true]
├── Trees Method: [absolute/per_feature]
├── Number of Trees: [10-1000]
├── Max Depth: [1-10]
├── Bootstrap Fraction: [0.1-1.0]
├── Min Leaf Fraction: [0.01-0.5]
├── Target Threshold: [0.0-0.5]
├── Vote Threshold: [0.5-1.0]
└── CPU Cores (n_jobs): [-1 to 16]
```

## How to Use

1. **Open the GUI**: Run `python OMtree_gui_v3.py`
2. **Navigate to Model Parameters** section
3. **Select Regression Mode**:
   - `false` (default): Uses classifiers throughout (consistent results with/without normalization)
   - `true`: Uses regressors throughout (better for identifying large moves)
4. **Save Configuration**: Click "Save Config" to persist your choice
5. **Run Validation**: The system will automatically use the appropriate models

## Benefits

- **Easy Toggle**: Switch between classification and regression modes with a simple dropdown
- **Visual Feedback**: Description updates to show what each mode does
- **Persistent**: Settings are saved to and loaded from the config file
- **Integrated**: Works seamlessly with all other GUI features

## Technical Implementation

### Code Changes in OMtree_gui_v3.py:
1. Added to `model_params` list (line 377)
2. Added description update function (lines 424-436)
3. Automatically handled by existing `load_config` and `save_config` methods

### Integration Points:
- Config file: `model.regression_mode = true/false`
- Feature selector: Uses appropriate model type
- Model training: Uses appropriate tree type
- Predictions: Handles threshold application for regression mode

## Verification

The GUI update has been verified:
- ✅ Parameter appears in Model Parameters section
- ✅ Description updates dynamically
- ✅ Saves to config file
- ✅ Loads from config file
- ✅ Integrates with validation workflow