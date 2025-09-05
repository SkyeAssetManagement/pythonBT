"""
Test and visualize the volatility signal feature
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from OMtree_preprocessing import DataPreprocessor

# Load data
df = pd.read_csv('DTSmlDATA7x7.csv')

# Initialize preprocessor  
preprocessor = DataPreprocessor()

# Process data (this will add VolSignal_Max250d)
processed_df = preprocessor.process_data(df)

# Get the volatility signal
vol_signal = processed_df['VolSignal_Max250d'].dropna()

print("VOLATILITY SIGNAL FEATURE ANALYSIS")
print("=" * 60)
print(f"Shape: {vol_signal.shape}")
print(f"Non-null values: {vol_signal.notna().sum()}")
print(f"\nDescriptive Statistics:")
print(f"  Mean: {vol_signal.mean():.2f}")
print(f"  Median: {vol_signal.median():.2f}")
print(f"  Std: {vol_signal.std():.2f}")
print(f"  Min: {vol_signal.min():.2f}")
print(f"  Max: {vol_signal.max():.2f}")
print(f"  25th percentile: {vol_signal.quantile(0.25):.2f}")
print(f"  75th percentile: {vol_signal.quantile(0.75):.2f}")

# Check distribution
print(f"\nDistribution:")
print(f"  Values > 90: {(vol_signal > 90).sum()} ({(vol_signal > 90).mean()*100:.1f}%)")
print(f"  Values > 80: {(vol_signal > 80).sum()} ({(vol_signal > 80).mean()*100:.1f}%)")
print(f"  Values > 70: {(vol_signal > 70).sum()} ({(vol_signal > 70).mean()*100:.1f}%)")
print(f"  Values < 30: {(vol_signal < 30).sum()} ({(vol_signal < 30).mean()*100:.1f}%)")

# Create visualization
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Plot 1: Time series of volatility signal
axes[0].plot(vol_signal.values, 'b-', linewidth=0.5, alpha=0.7)
axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Median (50th percentile)')
axes[0].axhline(y=90, color='red', linestyle='--', alpha=0.5, label='High vol (90th percentile)')
axes[0].axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Low vol (10th percentile)')
axes[0].set_title('Volatility Signal Over Time (Percentile Rank of Current vs 250-day History)')
axes[0].set_ylabel('Percentile Rank (0-100)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Distribution histogram
axes[1].hist(vol_signal, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
axes[1].axvline(x=50, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('Distribution of Volatility Signal Values')
axes[1].set_xlabel('Percentile Rank')
axes[1].set_ylabel('Density')
axes[1].grid(True, alpha=0.3)

# Plot 3: Compare with raw feature volatility
feature_cols = [col for col in df.columns if col.startswith('Ret_') and 'fwd' not in col]
if feature_cols:
    # Calculate simple volatility (abs value) for comparison
    raw_vol = df[feature_cols[0]].abs().rolling(20).mean()
    
    # Normalize both for comparison
    vol_signal_norm = (vol_signal - vol_signal.mean()) / vol_signal.std()
    raw_vol_norm = (raw_vol - raw_vol.mean()) / raw_vol.std()
    
    axes[2].plot(raw_vol_norm.values[:len(vol_signal)], 'gray', linewidth=0.5, alpha=0.5, 
                label=f'Raw {feature_cols[0]} volatility (20-day)')
    axes[2].plot(vol_signal_norm.values, 'b-', linewidth=1, alpha=0.7, 
                label='Vol Signal (percentile rank)')
    axes[2].set_title('Comparison: Volatility Signal vs Raw Feature Volatility (normalized)')
    axes[2].set_ylabel('Normalized Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

# Plot 4: Zoomed in view of volatile period
start_idx = 2000
end_idx = 2500
axes[3].plot(vol_signal.iloc[start_idx:end_idx].values, 'b-', linewidth=1)
axes[3].fill_between(range(end_idx-start_idx), 
                     vol_signal.iloc[start_idx:end_idx].values,
                     50, 
                     where=(vol_signal.iloc[start_idx:end_idx].values > 50),
                     color='red', alpha=0.3, label='Above median')
axes[3].fill_between(range(end_idx-start_idx), 
                     vol_signal.iloc[start_idx:end_idx].values,
                     50, 
                     where=(vol_signal.iloc[start_idx:end_idx].values <= 50),
                     color='green', alpha=0.3, label='Below median')
axes[3].axhline(y=50, color='gray', linestyle='-', linewidth=0.5)
axes[3].set_title(f'Zoomed View: Volatility Signal (observations {start_idx}-{end_idx})')
axes[3].set_ylabel('Percentile Rank')
axes[3].set_xlabel('Observation')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('volatility_signal_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nVisualization saved as 'volatility_signal_analysis.png'")

# Check correlation with target
target_col = 'Ret_fwd6hr'
if target_col in df.columns:
    # Align indices
    common_idx = vol_signal.index.intersection(df[target_col].dropna().index)
    if len(common_idx) > 0:
        vol_aligned = vol_signal.loc[common_idx]
        target_aligned = df[target_col].loc[common_idx].abs()  # Use absolute target for vol correlation
        
        correlation = vol_aligned.corr(target_aligned)
        print(f"\nCorrelation with |{target_col}|: {correlation:.4f}")
        
        # Check if high volatility predicts larger moves
        high_vol = vol_aligned > 80
        avg_move_high = target_aligned[high_vol].mean()
        avg_move_normal = target_aligned[~high_vol].mean()
        
        print(f"\nPredictive Power:")
        print(f"  Avg |move| when VolSignal > 80: {avg_move_high:.4f}")
        print(f"  Avg |move| when VolSignal <= 80: {avg_move_normal:.4f}")
        print(f"  Ratio: {avg_move_high/avg_move_normal:.2f}x")

print("\n" + "=" * 60)
print("INTERPRETATION:")
print("The VolSignal_Max250d feature represents the percentile rank")
print("of current volatility compared to the past 250 days.")
print("- Values near 100 = extreme high volatility (rare)")
print("- Values near 50 = typical volatility")
print("- Values near 0 = extreme low volatility (rare)")
print("\nThis feature is immune to normalization and provides")
print("the model with regime-aware volatility information.")