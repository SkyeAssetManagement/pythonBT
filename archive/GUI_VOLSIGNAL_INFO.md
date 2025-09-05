# VolSignal Feature in GUI - Important Information

## Current Behavior

The **VolSignal_Max250d** feature is an engineered feature that is **automatically calculated during preprocessing**, not when loading raw data.

## How to See/Use VolSignal in GUI

### Option 1: Load Preprocessed Data
1. Run: `python test_gui_volsignal.py` to create preprocessed file
2. Launch GUI: `python launch_gui.py`
3. Load: `DTSmlDATA7x7_with_volsignal.csv`
4. VolSignal_Max250d will appear in features list

### Option 2: Automatic During Walk-Forward
1. Launch GUI with regular data file
2. Configure your model settings
3. Run Walk-Forward Validation
4. **VolSignal is automatically added during preprocessing**
5. Check results CSV - it will contain VolSignal_Max250d_value

## Why It's Not Visible Initially

- VolSignal is calculated from 250-day rolling volatility percentiles
- Requires preprocessing to compute
- Only appears after data goes through preprocessing pipeline
- This is by design - it's an engineered feature, not raw data

## Verification

To confirm VolSignal is working:
1. Run any walk-forward validation
2. Check the generated `OMtree_results.csv`
3. Look for column: `VolSignal_Max250d_value`
4. Values should be 0-100 (percentile ranks)

## Technical Details

- **Calculation**: MAX(percentile_rank of |feature| vs 250-day history) across all features
- **Range**: 0-100 (percentile rank)
- **Immunity**: Not affected by normalization settings
- **Automatic**: Always included when preprocessing runs

## GUI Updates Made

1. Added detection for VolSignal columns in feature list
2. Added note in Model Tester tab about automatic inclusion
3. Feature will show when loading preprocessed data

The feature is working correctly - it's just created during preprocessing, not visible in raw data.