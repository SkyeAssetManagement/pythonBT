# VolSignal Feature - Complete GUI Integration

## Overview
The VolSignal feature is now fully integrated into the GUI workflow with proper user control and clear separation between normalized features and engineered features.

## What Changed

### 1. Configuration File (`OMtree_config.ini`)
```ini
[preprocessing]
# Engineered features
add_volatility_signal = true  # Enable/disable VolSignal feature
vol_signal_window = 250       # Lookback window for percentile rank
vol_signal_decay = 0.995      # Exponential decay for recency bias
```

### 2. GUI Updates (`OMtree_gui_v3.py`)

#### New "Engineered Features" Section
- Located in Model Tester tab
- Clear separation from regular features
- Shows as "Immune to Normalization"
- Checkbox to enable/disable VolSignal
- Adjustable parameters (window, decay)

#### Feature Display
- Regular features show: "(features will be normalized based on settings)"
- Engineered features show: "(auto-included if enabled)"
- Visual indicator when VolSignal is enabled: "[ENABLED] VolSignal_Max250d"

### 3. Preprocessing Updates (`OMtree_preprocessing.py`)
- Respects `add_volatility_signal` config setting
- Only creates VolSignal when enabled
- Remains immune to normalization settings

### 4. Validation Updates (`OMtree_validation.py`)
- Checks config to include VolSignal only if enabled
- Automatically adds to feature list when present

## User Workflow

### Step 1: Launch GUI
```bash
python launch_gui.py
```

### Step 2: Navigate to Model Tester Tab
Look for the new sections:
- **Preprocessing Settings** - for normalization of regular features
- **Engineered Features** - for features immune to normalization

### Step 3: Configure VolSignal
In the "Engineered Features" section:
- ✓ Check "Add VolSignal_Max250d" to enable
- Adjust Window (default: 250 days)
- Adjust Decay (default: 0.995)

### Step 4: Configure Regular Features
- Select which features to use for model
- These will be normalized based on preprocessing settings
- VolSignal will be added automatically if enabled

### Step 5: Run Walk-Forward Validation
- VolSignal is calculated from RAW data before normalization
- Added to model features automatically
- Appears in results as `VolSignal_Max250d_value`

## Key Benefits

### 1. Clear Separation
- Users can see which features are normalized vs engineered
- No confusion about what preprocessing affects

### 2. Full Control
- Enable/disable engineered features
- Adjust parameters
- Save/load configurations

### 3. Proper Pipeline
```
Raw Data → Calculate VolSignal → Normalize Features → Combine → Model
                ↓                        ↓
          (Immune to norm)        (Based on settings)
```

### 4. Verification
Run `python test_volsignal_workflow.py` to verify:
- Config settings work
- Enable/disable works
- Normalization immunity works
- Validation integration works

## Technical Details

### VolSignal Calculation
1. For each feature, calculate |value|
2. Compare to 250-day history with exponential decay
3. Get percentile rank (0-100)
4. Take MAX across all features
5. Result: Super-responsive volatility indicator

### Why It's Immune to Normalization
- Calculated from raw data BEFORE normalization
- Already normalized (percentile rank 0-100)
- Provides orthogonal information to normalized features

## Files Modified
- `OMtree_config.ini` - Added engineered feature settings
- `OMtree_gui_v3.py` - Added engineered features section
- `OMtree_preprocessing.py` - Respects enable/disable setting
- `OMtree_validation.py` - Conditionally includes VolSignal

## Testing
```bash
# Test workflow integration
python test_volsignal_workflow.py

# Test in GUI
python launch_gui.py
# Then check/uncheck VolSignal and run validation
```

## Summary
The VolSignal feature is now properly integrated with:
- Clear GUI controls
- Separation from normalized features  
- Full user control (enable/disable/parameters)
- Proper workflow pipeline
- Configuration persistence

Users can now easily understand and control engineered features that are immune to normalization while regular features follow standard preprocessing.