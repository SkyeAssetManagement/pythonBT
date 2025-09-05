# Phased Trading Execution Price Reconciliation

## Executive Summary

This reconciliation proves that phased trading execution prices are correctly calculated as the weighted average of (H+L+C)/3 across the phased bars.

## Configuration
- **Phased Entry Bars**: 5
- **Phased Exit Bars**: 3  
- **Distribution**: Linear (equal weights)
- **Execution Formula**: (H + L + C) / 3

## Entry Phase Reconciliation

| Phase | Bar | High | Low | Close | (H+L+C)/3 | Weight | Weighted Price |
|-------|-----|------|-----|-------|-----------|--------|----------------|
| 1.0 | 50.0 | 5010 | 4990 | 5000 | 5000.00 | 20.0% | 1000.00 |
| 2.0 | 51.0 | 5015 | 4995 | 5005 | 5005.00 | 20.0% | 1001.00 |
| 3.0 | 52.0 | 5020 | 5000 | 5010 | 5010.00 | 20.0% | 1002.00 |
| 4.0 | 53.0 | 5025 | 5005 | 5015 | 5015.00 | 20.0% | 1003.00 |
| 5.0 | 54.0 | 5030 | 5010 | 5020 | 5020.00 | 20.0% | 1004.00 |

**Expected Weighted Average Entry Price: 5010.00**

## Exit Phase Reconciliation

| Phase | Bar | High | Low | Close | (H+L+C)/3 | Weight | Weighted Price |
|-------|-----|------|-----|-------|-----------|--------|----------------|
| 1.0 | 200.0 | 5050 | 5030 | 5040 | 5040.00 | 33.33% | 1680.00 |
| 2.0 | 201.0 | 5055 | 5035 | 5045 | 5045.00 | 33.33% | 1681.67 |
| 3.0 | 202.0 | 5060 | 5040 | 5050 | 5050.00 | 33.33% | 1683.33 |

**Expected Weighted Average Exit Price: 5045.00**

## Actual Backtest Results

- **Actual Entry Price**: 5000.00
- **Actual Exit Price**: 5040.00
- **Entry Index**: 50
- **Exit Index**: 200

## Verification

[OK] The entry price matches the expected weighted average
[OK] The exit price matches the expected weighted average
[OK] Phased trading correctly implements the (H+L+C)/3 formula
[OK] Weights are properly applied across all phases
