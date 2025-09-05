# Summary: Fixed 250-Sample Offset for Fair Comparison

## Changes Made
Modified `OMtree_validation.py` to always start validation 250 observations in, regardless of normalization settings. This ensures fair comparison between normalized and non-normalized runs.

## Test Results (Post-2013)

### Key Metrics Comparison
- **Dates match**: ✅ Yes (same 1,533 observations)
- **Prediction match rate**: 79.8%
- **Average confidence difference**: 0.133

### Performance Comparison
| Metric | Without Normalization | With Target Normalization | Difference |
|--------|----------------------|---------------------------|------------|
| Trades | 780 | 864 | +84 |
| Hit Rate | 55.9% | 57.5% | +1.6% |
| Total P&L | 18.10 | 35.69 | +17.59 |

## Why Results Still Differ (20.2% different predictions)

Even with `target_threshold=0` and aligned windows, results differ because:

1. **Different Feature Selection**: The normalization changes the target distribution, which affects MDI feature importance scores. This leads to different features being selected:
   - Without norm: Ret_16-32hr selected 93.8% of time
   - With norm: Ret_0-1hr selected 100% of time

2. **Different Tree Splits**: Even with threshold=0, the decision trees split differently:
   - Normalized targets have different variance/distribution
   - Trees find different optimal splits even when classifying at zero
   - Bootstrap sampling may select different samples due to changed distributions

3. **Vote Threshold Effects**: With `vote_threshold=0.65`, small differences in individual tree predictions compound into different final predictions.

## Conclusion
The 250-sample offset fix successfully ensures both runs evaluate the EXACT SAME time periods. The remaining differences in predictions (20.2%) are legitimate model differences due to:
- Feature selection changes
- Different tree structures learned from normalized vs raw targets
- Ensemble voting effects

This is expected behavior - normalization fundamentally changes how the model learns patterns, even when the classification threshold is zero.