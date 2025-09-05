# OMtree Workspace State - Saved 2025-08-19

## Current Project Status

### Recent Accomplishments
1. **Fixed Sharpe Ratio Calculation**: Updated from monthly to annualized (multiply by √12)
   - Old: `sharpe = monthly_mean / monthly_std` 
   - New: `sharpe = (monthly_mean * 12) / (monthly_std * √12)`
   - Now shows ~0.872 instead of ~0.25

2. **Created Comprehensive GUI Application**
   - 4 tabs: Configuration, Run Validation, Performance Stats, Charts
   - Constrained input widgets (dropdowns, spinboxes) for valid values only
   - Real-time progress tracking during validation
   - Auto-loads results and charts after completion

3. **Completed All Optimization Experiments**
   - Tested 5 parameters with full walkforward validation
   - Best configuration: min_leaf_fraction=0.20, Sharpe=0.894
   - All results saved in experiment_reproduce/ folder

## Key Files

### Core Model Files
- `OMtree_model.py` - Main model implementation (decision tree ensemble)
- `OMtree_validation.py` - Validation logic with FIXED annualized Sharpe
- `OMtree_preprocessing.py` - Volatility normalization using IQR
- `OMtree_walkforward.py` - Complete walkforward validation script
- `OMtree_config.ini` - Configuration file (currently vol_window=10)

### GUI Files
- `OMtree_gui.py` - Main GUI application with constrained widgets
- `launch_gui.py` - GUI launcher with dependency checking
- `GUI_README.md` - GUI documentation
- `GUI_UPDATES.md` - Latest GUI improvements documentation

### Data & Results
- `DTSnnData.csv` - Input trading data
- `OMtree_results.csv` - Latest validation results
- `OMtree_performance.csv` - Performance metrics log (150+ entries)
- `OMtree_comprehensive_longonly.png` - 6-panel analysis chart
- `OMtree_progression_longonly.png` - 4-panel yearly progression

### Experiments (in experiment_reproduce/)
- `min_leaf_fraction_results.csv` - Best: 0.20, Sharpe 0.894
- `vol_window_results.csv` - Best: 120, Sharpe 0.347
- `bootstrap_fraction_results.csv` - Best: 0.575, Sharpe 0.345
- `target_threshold_results.csv` - Best: 0.05, Sharpe 0.331
- `vote_threshold_results.csv` - Best: 0.6, Sharpe 0.337

## Current Configuration (OMtree_config.ini)

```ini
[model]
model_type = longonly
n_trees = 200
max_depth = 1
bootstrap_fraction = 0.8
min_leaf_fraction = 0.2
target_threshold = 0.05
vote_threshold = 0.6

[preprocessing]
normalize_features = true
normalize_target = true
vol_window = 10  # Note: Currently set to 10, was 50 earlier
smoothing_type = exponential
smoothing_alpha = 0.1

[validation]
train_size = 1000
test_size = 100
step_size = 50
validation_start_date = 2010-01-01
```

## Latest Performance Metrics
- **Annualized Sharpe Ratio**: 0.872 (properly annualized)
- **Hit Rate**: 54.8% (vs 42% base rate)
- **Edge**: +12.8%
- **Trading Frequency**: 57.3% of days
- **Monthly Win Rate**: 67.1% positive months

## To Resume Work

### Launch GUI:
```bash
python launch_gui.py
```

### Run Validation:
```bash
python OMtree_walkforward.py
```

### View Latest Results:
- Check `OMtree_performance.csv` for historical runs
- View charts: `OMtree_comprehensive_longonly.png`

## Important Notes

1. **Sharpe Calculation Fixed**: Now properly annualized in OMtree_validation.py
2. **GUI Has Constrained Inputs**: Dropdowns and spinboxes prevent invalid values
3. **Config Recently Modified**: vol_window changed from 50 to 10
4. **Model Type**: Only supports longonly/shortonly (no bidirectional)
5. **Best Parameters Found**: min_leaf_fraction=0.20 gives Sharpe ~0.894

## Next Steps (Optional)
- Consider reverting vol_window back to 50 (better Sharpe in experiments)
- Run more experiments with the optimal min_leaf_fraction=0.20
- Test shortonly mode performance
- Add more visualizations to GUI

## Archive Folders
- `backup/` - Old version with original Sharpe calculation
- `archive_old/` - Previous experiments and versions
- `experiment_reproduce/` - All optimization experiment results

---
Workspace saved successfully. Ready to resume after reboot.