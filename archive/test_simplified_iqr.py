"""
Test the simplified IQR implementation without decay factor
"""

import numpy as np
import pandas as pd
from OMtree_preprocessing import DataPreprocessor
import time

print("Testing Simplified IQR Implementation")
print("="*60)

# Create test data with volatility spike
np.random.seed(42)
n_points = 300

# Normal period
normal = np.random.randn(200) * 0.5
# Spike
spike = np.random.randn(50) * 2.0
# Back to normal  
post_spike = np.random.randn(50) * 0.5

test_data = np.concatenate([normal, spike, post_spike])

df = pd.DataFrame({
    'Date': pd.date_range(start='2020-01-01', periods=n_points, freq='h'),
    'Time': pd.date_range(start='2020-01-01', periods=n_points, freq='h').strftime('%H:%M:%S'),
    'Ret_0-1hr': test_data,
    'Ret_1-2hr': test_data * 0.9,
    'Ret_2-4hr': test_data * 0.8,
    'Ret_fwd6hr': test_data * 0.7
})

try:
    # Load preprocessor with current config
    print("Loading preprocessor with current config...")
    preprocessor = DataPreprocessor('OMtree_config.ini')
    
    print(f"\nConfiguration:")
    print(f"  Normalization method: {preprocessor.normalization_method}")
    print(f"  Vol window: {preprocessor.vol_window}")
    print(f"  IQR lookback: {preprocessor.recent_iqr_lookback}")
    print(f"  Smoothing type: {preprocessor.smoothing_type}")
    print(f"  Alpha (calculated): {preprocessor.smoothing_alpha:.4f}")
    print(f"  Formula: 2/(n+1) = 2/({preprocessor.recent_iqr_lookback}+1) = {preprocessor.smoothing_alpha:.4f}")
    print(f"  Winsorization: {preprocessor.winsorize_enabled}")
    if preprocessor.winsorize_enabled:
        print(f"  Winsorize %: {preprocessor.winsorize_percentile}%")
    
    # Process data
    print("\nProcessing data...")
    start_time = time.time()
    processed, features, target = preprocessor.process_data(df.copy())
    elapsed = time.time() - start_time
    
    print(f"\n[SUCCESS] Processing completed in {elapsed:.3f} seconds")
    print(f"  Processed shape: {processed.shape}")
    print(f"  Feature columns: {features}")
    print(f"  Target column: {target}")
    
    # Analyze normalization quality
    if 'Ret_0-1hr_vol_adj' in processed.columns:
        normalized = processed['Ret_0-1hr_vol_adj'].values
        
        # Remove NaN values for statistics
        valid_norm = normalized[~np.isnan(normalized)]
        
        if len(valid_norm) > 0:
            print(f"\nNormalized data statistics:")
            print(f"  Mean: {np.mean(valid_norm):.3f}")
            print(f"  Std: {np.std(valid_norm):.3f}")
            print(f"  Min: {np.min(valid_norm):.3f}")
            print(f"  Max: {np.max(valid_norm):.3f}")
            
            # Check pre-spike, spike, and post-spike periods
            pre_spike = normalized[150:200]
            spike_period = normalized[200:250]
            post_spike_period = normalized[250:300]
            
            pre_clean = pre_spike[~np.isnan(pre_spike)]
            spike_clean = spike_period[~np.isnan(spike_period)]
            post_clean = post_spike_period[~np.isnan(post_spike_period)]
            
            if len(pre_clean) > 0 and len(spike_clean) > 0:
                print(f"\nVolatility normalization effect:")
                print(f"  Pre-spike std:  {np.std(pre_clean):.3f}")
                print(f"  Spike std:      {np.std(spike_clean):.3f}")
                print(f"  Post-spike std: {np.std(post_clean):.3f}")
                print(f"  Spike/Pre ratio: {np.std(spike_clean)/np.std(pre_clean):.2f}x")
                
                # Check smoothing effect
                if 'Ret_0-1hr_iqr' in processed.columns:
                    iqr_values = processed['Ret_0-1hr_iqr'].values
                    iqr_clean = iqr_values[~np.isnan(iqr_values)]
                    if len(iqr_clean) > 0:
                        # Calculate roughness (sum of absolute differences)
                        roughness = np.sum(np.abs(np.diff(iqr_clean)))
                        print(f"\nIQR smoothing quality:")
                        print(f"  IQR roughness: {roughness:.2f} (lower is smoother)")
                        print(f"  IQR std: {np.std(iqr_clean):.4f}")
        
except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test completed")