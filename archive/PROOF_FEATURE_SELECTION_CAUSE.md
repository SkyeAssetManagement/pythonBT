# PROOF: Feature Selection Causes the Differences

## Test Setup
- **Fixed 2 features**: Ret_0-1hr, Ret_4-8hr
- **Feature selection**: DISABLED
- **Target threshold**: 0
- **250-sample offset**: Applied to both runs
- **Only difference**: Target normalization ON vs OFF

## Results: 100% PERFECT MATCH!

### Performance Metrics (Post-2013)
| Metric | Without Normalization | With Normalization | Difference |
|--------|----------------------|--------------------|-----------:|
| Trades | 881 | 881 | 0 |
| Hit Rate | 55.5% | 55.5% | 0.0% |
| Total P&L | 29.45 | 29.45 | 0.00 |
| **Prediction Match** | - | - | **100.0%** |
| Confidence Diff | - | - | 0.000000 |

## What This Proves

Your intuition was 100% correct! With threshold=0:
- Normalization (dividing by IQR) doesn't change the sign
- The classification boundary (zero) remains the same
- When using the SAME features, results are IDENTICAL

## Why Results Differed Before

The 20% difference in predictions with feature selection enabled was caused by:

1. **Feature Selection Stage**: Uses `RandomForestRegressor` on continuous target values
   - Normalization changes variance patterns in the continuous targets
   - This changes MDI (Mean Decrease in Impurity) scores
   - Different features get selected

2. **Evidence from Previous Tests**:
   - Without normalization: Ret_16-32hr selected 93.8% of time
   - With normalization: Ret_0-1hr selected 100% of time
   - Different features → Different models → Different predictions

## Conclusion

This definitively proves that:
- The normalization itself does NOT change the classification (as you correctly reasoned)
- The differences come entirely from feature selection operating on continuous values
- With fixed features, normalized and non-normalized models are IDENTICAL

Your logical thinking was spot-on - the classification at threshold=0 should be the same, and it is! The feature selection mechanism was the hidden factor causing the differences.