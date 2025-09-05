"""
Test the change from MAX to MEAN for VolSignal calculation
"""
import pandas as pd
import numpy as np
from OMtree_preprocessing import DataPreprocessor

# Load data
df = pd.read_csv('DTSmlDATA7x7.csv')
print("Testing VolSignal calculation change from MAX to MEAN")
print("=" * 60)

# Process data with new MEAN calculation
preprocessor = DataPreprocessor()
processed = preprocessor.process_data(df)

# Check if VolSignal_Mean250d exists
if 'VolSignal_Mean250d' in processed.columns:
    vol_signal = processed['VolSignal_Mean250d'].dropna()
    
    print(f"\n[OK] VolSignal_Mean250d created successfully")
    print(f"\nDescriptive Statistics (MEAN version):")
    print(f"  Shape: {vol_signal.shape}")
    print(f"  Mean: {vol_signal.mean():.2f}")
    print(f"  Median: {vol_signal.median():.2f}")
    print(f"  Std: {vol_signal.std():.2f}")
    print(f"  Min: {vol_signal.min():.2f}")
    print(f"  Max: {vol_signal.max():.2f}")
    print(f"  25th percentile: {vol_signal.quantile(0.25):.2f}")
    print(f"  75th percentile: {vol_signal.quantile(0.75):.2f}")
    
    # Compare with what MAX would have been
    print(f"\nDistribution Analysis:")
    print(f"  Values > 90: {(vol_signal > 90).sum()} ({(vol_signal > 90).mean()*100:.1f}%)")
    print(f"  Values > 80: {(vol_signal > 80).sum()} ({(vol_signal > 80).mean()*100:.1f}%)")
    print(f"  Values > 70: {(vol_signal > 70).sum()} ({(vol_signal > 70).mean()*100:.1f}%)")
    print(f"  Values < 30: {(vol_signal < 30).sum()} ({(vol_signal < 30).mean()*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("COMPARISON: MEAN vs MAX approach")
    print("=" * 60)
    print("\nExpected differences:")
    print("- MEAN will have lower overall values (averaging reduces extremes)")
    print("- MEAN will have smaller standard deviation (more stable)")
    print("- MEAN provides balanced view across all timeframes")
    print("- MAX captured any single extreme, MEAN captures overall regime")
    
    # Test correlation with target
    target_col = 'Ret_fwd6hr'
    if target_col in df.columns:
        # Align indices
        common_idx = vol_signal.index.intersection(df[target_col].dropna().index)
        if len(common_idx) > 0:
            vol_aligned = vol_signal.loc[common_idx]
            target_aligned = df[target_col].loc[common_idx].abs()
            
            correlation = vol_aligned.corr(target_aligned)
            print(f"\nCorrelation with |{target_col}|: {correlation:.4f}")
            
            # Check predictive power
            high_vol = vol_aligned > vol_aligned.quantile(0.8)
            avg_move_high = target_aligned[high_vol].mean()
            avg_move_normal = target_aligned[~high_vol].mean()
            
            print(f"\nPredictive Power (using 80th percentile threshold):")
            print(f"  Avg |move| when VolSignal > 80th percentile: {avg_move_high:.4f}")
            print(f"  Avg |move| when VolSignal <= 80th percentile: {avg_move_normal:.4f}")
            print(f"  Ratio: {avg_move_high/avg_move_normal:.2f}x")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("The VolSignal_Mean250d now represents the AVERAGE percentile")
    print("rank across all features, providing a more balanced measure")
    print("of overall market volatility regime rather than just extremes.")
    
else:
    print("[ERROR] VolSignal_Mean250d not found in processed data")
    print("Available columns:", [col for col in processed.columns if 'Vol' in col])