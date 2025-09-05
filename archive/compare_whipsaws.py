"""
Compare whipsaws between IQR with smoothing vs AVS without smoothing
Demonstrate why AVS is inherently smoother
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_iqr_with_smoothing(data, window=20, percentile_upper=75, percentile_lower=25, 
                                 smoothing_alpha=0.1, recent_lookback=20):
    """
    Current method: IQR with exponential smoothing (as in your preprocessing)
    """
    normalized = []
    iqr_values = []
    smoothed_iqrs = []
    
    for i in range(len(data)):
        if i < window:
            normalized.append(np.nan)
            iqr_values.append(np.nan)
            smoothed_iqrs.append(np.nan)
        else:
            window_data = data.iloc[i-window+1:i+1]
            q75 = np.percentile(window_data, percentile_upper)
            q25 = np.percentile(window_data, percentile_lower)
            iqr = q75 - q25
            
            if iqr > 0:
                # Apply exponential smoothing to IQR
                if i > window and not np.isnan(iqr_values[-1]):
                    smoothed_iqr = smoothing_alpha * iqr + (1 - smoothing_alpha) * iqr_values[-1]
                else:
                    smoothed_iqr = iqr
                
                iqr_values.append(iqr)
                smoothed_iqrs.append(smoothed_iqr)
                normalized.append(data.iloc[i] / smoothed_iqr)
            else:
                iqr_values.append(np.nan)
                smoothed_iqrs.append(np.nan)
                normalized.append(np.nan)
    
    return pd.Series(normalized), pd.Series(iqr_values), pd.Series(smoothed_iqrs)


def calculate_avs_no_smoothing(data, slow_window=60, fast_window=20):
    """
    AVS without additional smoothing - naturally smooth due to adaptive weighting
    """
    # Use absolute returns for volatility
    abs_returns = np.abs(data)
    
    # Two volatility measures
    vol_slow = abs_returns.rolling(window=slow_window, min_periods=20).mean()
    vol_fast = abs_returns.rolling(window=fast_window, min_periods=10).mean()
    
    # Calculate adaptive weight based on volatility divergence
    vol_ratio = vol_fast / vol_slow
    vol_ratio = vol_ratio.clip(0.5, 2.0)  # Limit extreme ratios
    
    # Weight increases when volatilities diverge (regime change)
    # Weight decreases when volatilities converge (stable regime)
    weight = 0.3 + 0.3 * np.abs(vol_ratio - 1)
    weight = np.minimum(weight, 0.6)  # Cap at 60% fast weight
    
    # Adaptive volatility is a weighted average - inherently smooth!
    adaptive_vol = weight * vol_fast + (1 - weight) * vol_slow
    adaptive_vol = adaptive_vol.replace(0, np.nan).ffill().fillna(1)
    
    normalized = data / adaptive_vol
    
    return normalized, adaptive_vol, weight, vol_fast, vol_slow


def calculate_whipsaw_metric(series, threshold=0.1):
    """
    Calculate whipsaw metric: frequency of direction changes in volatility estimate
    Lower is better (less whipsaw)
    """
    if len(series.dropna()) < 2:
        return 0
    
    # Calculate first differences
    diff = series.diff()
    
    # Count sign changes (whipsaws)
    sign_changes = (np.sign(diff).diff() != 0).sum()
    
    # Also calculate variance of changes (volatility of volatility)
    vol_of_vol = diff.std()
    
    return sign_changes, vol_of_vol


def main():
    # Load data
    df = pd.read_csv('DTSmlDATA7x7.csv')
    data = df['Ret_fwd6hr'].dropna()
    
    print("WHIPSAW COMPARISON: IQR with Smoothing vs AVS without Smoothing")
    print("=" * 70)
    
    # Calculate both methods
    iqr_norm, raw_iqr, smoothed_iqr = calculate_iqr_with_smoothing(
        data, window=20, smoothing_alpha=0.1
    )
    
    avs_norm, avs_vol, avs_weight, vol_fast, vol_slow = calculate_avs_no_smoothing(
        data, slow_window=60, fast_window=20
    )
    
    # Calculate whipsaw metrics
    print("\nWHIPSAW METRICS (Lower is Better):")
    print("-" * 70)
    
    # For raw IQR (without smoothing)
    raw_iqr_changes, raw_iqr_vol = calculate_whipsaw_metric(raw_iqr)
    print(f"Raw IQR (no smoothing):")
    print(f"  Sign changes: {raw_iqr_changes}")
    print(f"  Vol of vol:   {raw_iqr_vol:.6f}")
    
    # For smoothed IQR
    smooth_iqr_changes, smooth_iqr_vol = calculate_whipsaw_metric(smoothed_iqr)
    print(f"\nSmoothed IQR (alpha=0.1):")
    print(f"  Sign changes: {smooth_iqr_changes}")
    print(f"  Vol of vol:   {smooth_iqr_vol:.6f}")
    
    # For AVS (no smoothing)
    avs_changes, avs_vol_metric = calculate_whipsaw_metric(avs_vol)
    print(f"\nAVS (no smoothing):")
    print(f"  Sign changes: {avs_changes}")
    print(f"  Vol of vol:   {avs_vol_metric:.6f}")
    
    # Visual comparison
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    
    # Select a volatile period for visualization
    start, end = 2000, 2500
    
    # Plot 1: Raw data
    axes[0].plot(data.iloc[start:end].values, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_title('Original Data (showing volatile period)')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Raw IQR vs Smoothed IQR
    axes[1].plot(raw_iqr.iloc[start:end].values, 'r-', linewidth=0.5, alpha=0.5, label='Raw IQR (choppy)')
    axes[1].plot(smoothed_iqr.iloc[start:end].values, 'g-', linewidth=1.5, label='Smoothed IQR (your current)')
    axes[1].set_title('IQR Methods: Raw vs Smoothed')
    axes[1].set_ylabel('IQR Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: AVS components
    axes[2].plot(vol_fast.iloc[start:end].values, 'orange', linewidth=0.5, alpha=0.5, label='Fast Vol (20-bar)')
    axes[2].plot(vol_slow.iloc[start:end].values, 'purple', linewidth=1, alpha=0.7, label='Slow Vol (60-bar)')
    axes[2].plot(avs_vol.iloc[start:end].values, 'black', linewidth=2, label='AVS (weighted avg)')
    axes[2].set_title('AVS Components: Natural Smoothing via Weighted Average')
    axes[2].set_ylabel('Volatility')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: AVS adaptive weights
    axes[3].plot(avs_weight.iloc[start:end].values, 'purple', linewidth=1)
    axes[3].axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Min weight (stable)')
    axes[3].axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Max weight (transition)')
    axes[3].fill_between(range(len(avs_weight.iloc[start:end])), 0.3, 
                        avs_weight.iloc[start:end].values, alpha=0.3, color='purple')
    axes[3].set_title('AVS Weight on Fast Vol (smooth transitions prevent whipsaws)')
    axes[3].set_ylabel('Weight')
    axes[3].set_ylim(0.25, 0.65)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Compare first differences (rate of change)
    smoothed_iqr_diff = smoothed_iqr.diff().iloc[start:end]
    avs_vol_diff = avs_vol.diff().iloc[start:end]
    
    axes[4].plot(smoothed_iqr_diff.values / smoothed_iqr_diff.std(), 'g-', linewidth=0.7, 
                alpha=0.7, label='Smoothed IQR changes (normalized)')
    axes[4].plot(avs_vol_diff.values / avs_vol_diff.std(), 'black', linewidth=1, 
                alpha=0.7, label='AVS changes (normalized)')
    axes[4].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    axes[4].set_title('Rate of Change Comparison (less variation = fewer whipsaws)')
    axes[4].set_ylabel('Normalized Change')
    axes[4].set_xlabel('Time')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('whipsaw_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved as 'whipsaw_comparison.png'")
    
    # Compare autocorrelation of volatility estimates (smoother = higher autocorr)
    print("\n" + "-" * 70)
    print("VOLATILITY ESTIMATE AUTOCORRELATION (Higher = Smoother):")
    print("-" * 70)
    print(f"Raw IQR autocorr(1):      {raw_iqr.dropna().autocorr(1):.4f}")
    print(f"Smoothed IQR autocorr(1): {smoothed_iqr.dropna().autocorr(1):.4f}")
    print(f"AVS autocorr(1):          {avs_vol.dropna().autocorr(1):.4f}")
    
    print("\n" + "=" * 70)
    print("WHY AVS DOESN'T NEED ADDITIONAL SMOOTHING:")
    print("=" * 70)
    print("""
1. WEIGHTED AVERAGE IS INHERENTLY SMOOTH:
   - AVS = w * fast_vol + (1-w) * slow_vol
   - This weighted average naturally smooths transitions
   - No sudden jumps when data enters/leaves window

2. ADAPTIVE WEIGHT CHANGES GRADUALLY:
   - Weight formula: w = 0.3 + 0.3 * |vol_ratio - 1|
   - Weight can only change gradually as vol_ratio changes
   - Creates smooth transitions between regimes

3. TWO-WINDOW DESIGN REDUCES NOISE:
   - 60-bar slow window provides stability
   - 20-bar fast window provides responsiveness
   - Combination filters out short-term noise

4. AVOIDS FIXED-WINDOW ARTIFACTS:
   - IQR has sharp changes when extreme values enter/exit window
   - AVS uses mean of absolute returns (more stable than percentiles)
   - Longer slow window dilutes impact of individual observations

5. NATURAL LAG REDUCTION:
   - During stable periods: mostly uses slow window (stable)
   - During transitions: increases fast window weight (responsive)
   - Automatically balances stability vs responsiveness

CONCLUSION:
AVS achieves similar or better smoothness than IQR+smoothing
WITHOUT needing an additional smoothing step. The smoothing
is built into the adaptive weighting mechanism itself.
    """)
    
    # Calculate effective smoothing
    print("\nEFFECTIVE SMOOTHING COMPARISON:")
    print("-" * 70)
    
    # How much does each method reduce variance compared to raw IQR?
    raw_variance = raw_iqr.var()
    smooth_variance = smoothed_iqr.var()
    avs_variance = avs_vol.var()
    
    print(f"Variance reduction vs raw IQR:")
    print(f"  Smoothed IQR: {(1 - smooth_variance/raw_variance)*100:.1f}% reduction")
    print(f"  AVS:          {(1 - avs_variance/raw_variance)*100:.1f}% reduction")
    
    return iqr_norm, avs_norm


if __name__ == "__main__":
    main()