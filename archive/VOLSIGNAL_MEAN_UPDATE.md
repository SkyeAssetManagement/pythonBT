# VolSignal Calculation Updated: MAX → MEAN

## Change Summary
Changed the VolSignal engineered feature calculation from taking the **MAX** to taking the **MEAN** of percentile ranks across all features.

## Technical Change
```python
# OLD: Take MAX across all features
max_vol_signal = vol_signal_df.max(axis=1)

# NEW: Take MEAN across all features  
mean_vol_signal = vol_signal_df.mean(axis=1)
```

## Impact Comparison

### Statistics Comparison
| Metric | MAX Version | MEAN Version | Change |
|--------|------------|--------------|--------|
| Mean | 84.87 | 49.28 | -42% |
| Median | 88.15 | 48.15 | -45% |
| Std Dev | 12.52 | 14.16 | +13% |
| Min | 28.15 | 13.67 | -51% |
| Max | 99.30 | 97.59 | -2% |

### Distribution Changes
| Percentile Range | MAX Version | MEAN Version |
|-----------------|-------------|--------------|
| > 90 | 44.1% | 0.4% |
| > 80 | 70.9% | 2.6% |
| > 70 | 86.5% | 8.4% |
| < 30 | 0.0% | 7.6% |

### Predictive Power
| Version | Correlation with \|Returns\| | High Vol Ratio |
|---------|------------------------------|----------------|
| MAX | 0.21 | 1.55x |
| MEAN | 0.34 | 2.06x |

## Key Improvements

### 1. Better Correlation
- MEAN version: 0.34 correlation (62% improvement)
- Stronger relationship with actual volatility

### 2. More Balanced Distribution
- MAX was heavily skewed high (median 88)
- MEAN is centered (median 48)
- Better use of 0-100 scale

### 3. Higher Predictive Power
- 2.06x ratio vs 1.55x for identifying high volatility periods
- Better discrimination between regimes

### 4. More Stable Signal
- Less prone to single-feature spikes
- Represents overall market volatility regime
- Reduces noise from individual timeframe extremes

## Interpretation

### MAX Approach (Previous)
- Captured ANY extreme volatility in ANY timeframe
- Very sensitive to outliers
- Often saturated near 100

### MEAN Approach (New)
- Captures OVERALL volatility across ALL timeframes
- Balanced view of market conditions
- Better centered distribution for model learning

## Files Updated
- `OMtree_preprocessing.py` - Changed calculation to mean
- `OMtree_validation.py` - Updated feature name
- `OMtree_gui_v3.py` - Updated display text
- Feature renamed: `VolSignal_Max250d` → `VolSignal_Mean250d`

## Usage Remains Same
- Still immune to normalization
- Still calculated from raw data
- Still configurable in GUI
- Still auto-included when enabled

The MEAN approach provides a more balanced, stable, and predictive volatility signal that better represents overall market conditions rather than just extremes.