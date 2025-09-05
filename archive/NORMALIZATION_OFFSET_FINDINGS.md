# Target Normalization 90-Sample Offset Issue

## Summary
When using target normalization with `target_threshold=0`, different results are obtained compared to running without normalization, even though theoretically they should be identical. This is caused by a permanent 90-sample offset in the walk-forward validation windows.

## Root Cause
1. **IQR Normalization Requirement**: The IQR normalization uses a 90-sample `vol_window` for calculating rolling statistics
2. **Initial NaN Values**: This creates 90 NaN values at the beginning of the normalized data
3. **Permanent Index Offset**: The walk-forward validation uses index-based windows that become permanently offset by 90 samples
4. **Different Time Periods**: Throughout the entire validation (not just at the start), normalized and non-normalized runs test on different time periods

## Key Findings

### 1. IQR Normalization Process
- The normalization ONLY divides by IQR, does NOT subtract median
- Formula: `normalized_value = data[col] / smoothed_iqr`
- First 90 samples become NaN due to `vol_window` requirement

### 2. Walk-Forward Window Misalignment
Without normalization:
- Train indices: [0:2000], Test indices: [2000:2100]
- Train indices: [100:2100], Test indices: [2100:2200]
- etc.

With normalization (90 NaN rows removed):
- Train indices: [0:2000], Test indices: [2000:2100] (but maps to original rows 90:2090 and 2090:2190)
- Train indices: [100:2100], Test indices: [2100:2200] (but maps to original rows 190:2190 and 2190:2290)
- etc.

### 3. Proof of Concept Test Results
When manually aligning windows to use EXACT SAME indices:
```
Train indices: [1000:3000], Test indices: [3001:3100]
Train classification match rate: 100.0%
Test classification match rate: 100.0%
Model predictions match rate: 100.0%
```

This proves definitively that the different results are solely due to the offset causing different time windows.

## Bug Discovery
The `validation_start_date` parameter in the config is completely ignored in the code. The walk-forward validation always starts from the beginning of the data, regardless of the `validation_start_date` setting.

## Implications
1. **Backtesting Validity**: Results from normalized vs non-normalized runs are not directly comparable as they test on different time periods
2. **Performance Metrics**: Any performance differences may be due to different market conditions in the offset time periods rather than the normalization itself
3. **Feature Selection**: Different features may be selected due to training on different historical periods

## Recommendations
1. **Fix the Offset**: Modify the walk-forward validation to account for the 90-sample burn-in period
2. **Use Date-Based Windows**: Implement true date-based windowing instead of index-based to ensure alignment
3. **Fix validation_start_date**: Implement proper filtering based on the validation_start_date parameter
4. **Add Validation Checks**: Add checks to ensure train/test windows align properly between normalized and non-normalized runs