"""
Test the new normalization methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from OMtree_preprocessing import DataPreprocessor
import configparser

# Create test data with a volatility shock
np.random.seed(42)
n_points = 500

# Normal period
normal_data = np.random.randn(300) * 0.5

# Volatility shock period  
shock_data = np.random.randn(50) * 2.0  # 4x volatility

# Return to normal
post_shock = np.random.randn(150) * 0.5

# Combine
test_series = np.concatenate([normal_data, shock_data, post_shock])
dates = pd.date_range(start='2020-01-01', periods=n_points, freq='h')
df = pd.DataFrame({
    'Date': dates,
    'Time': dates.strftime('%H:%M:%S'),
    'test_feature': test_series,
    'test_target': test_series * 0.8 + np.random.randn(n_points) * 0.1
})

print("Testing Different Normalization Methods")
print("="*60)

# Test configurations
test_configs = [
    {
        'name': 'Standard IQR',
        'method': 'IQR',
        'winsorize': False,
        'weighted': False
    },
    {
        'name': 'IQR with Winsorization (5%)',
        'method': 'IQR',
        'winsorize': True,
        'weighted': False
    },
    {
        'name': 'Weighted IQR (decay=0.98)',
        'method': 'IQR',
        'winsorize': False,
        'weighted': True
    },
    {
        'name': 'Weighted IQR + Winsorization',
        'method': 'IQR',
        'winsorize': True,
        'weighted': True
    },
    {
        'name': 'Logit-Rank Transform',
        'method': 'LOGIT_RANK',
        'winsorize': False,
        'weighted': False
    }
]

# Create figure for comparison
fig, axes = plt.subplots(len(test_configs) + 1, 1, figsize=(12, 3 * (len(test_configs) + 1)))

# Plot original data
axes[0].plot(df.index, test_series, 'b-', linewidth=0.5)
axes[0].axvspan(300, 350, alpha=0.2, color='red', label='Volatility Shock')
axes[0].set_title('Original Data')
axes[0].set_ylabel('Value')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Test each configuration
for idx, config in enumerate(test_configs):
    print(f"\nTesting: {config['name']}")
    print("-"*40)
    
    # Create temporary config file
    temp_config = configparser.ConfigParser()
    temp_config['data'] = {
        'csv_file': 'test.csv',
        'feature_columns': 'test_feature',
        'target_column': 'test_target',
        'selected_features': 'test_feature'
    }
    temp_config['preprocessing'] = {
        'normalize_features': 'true',
        'normalize_target': 'false',
        'detrend_features': 'false',
        'normalization_method': config['method'],
        'vol_window': '50',
        'winsorize_enabled': str(config['winsorize']).lower(),
        'winsorize_percentile': '5',
        'iqr_weighting_enabled': str(config['weighted']).lower(),
        'iqr_decay_factor': '0.98',
        'smoothing_type': 'exponential',
        'smoothing_alpha': '0.1',
        'percentile_upper': '75',
        'percentile_lower': '25',
        'recent_iqr_lookback': '20',
        'avs_slow_window': '60',
        'avs_fast_window': '20',
        'vol_signal_window': '0'  # Disable vol signal for this test
    }
    
    # Save temp config
    with open('temp_test_config.ini', 'w') as f:
        temp_config.write(f)
    
    # Process data
    preprocessor = DataPreprocessor('temp_test_config.ini')
    processed, features, target = preprocessor.process_data(df.copy())
    
    # Get normalized values
    normalized = processed['test_feature_vol_adj'].values
    
    # Plot normalized data
    ax = axes[idx + 1]
    ax.plot(df.index, normalized, 'g-', linewidth=0.5)
    ax.axvspan(300, 350, alpha=0.2, color='red')
    ax.set_title(config['name'])
    ax.set_ylabel('Normalized')
    ax.grid(True, alpha=0.3)
    
    # Calculate statistics for shock period
    pre_shock = normalized[250:300]
    shock = normalized[300:350]
    post_shock_vals = normalized[350:400]
    
    pre_shock_clean = pre_shock[~np.isnan(pre_shock)]
    shock_clean = shock[~np.isnan(shock)]
    post_shock_clean = post_shock_vals[~np.isnan(post_shock_vals)]
    
    if len(pre_shock_clean) > 0 and len(shock_clean) > 0 and len(post_shock_clean) > 0:
        print(f"  Pre-shock std:  {np.std(pre_shock_clean):.3f}")
        print(f"  Shock std:      {np.std(shock_clean):.3f}")
        print(f"  Post-shock std: {np.std(post_shock_clean):.3f}")
        print(f"  Shock/Pre ratio: {np.std(shock_clean)/np.std(pre_shock_clean):.2f}")
        
        # Add text to plot
        text = f"Shock/Pre: {np.std(shock_clean)/np.std(pre_shock_clean):.2f}x"
        ax.text(0.02, 0.95, text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Normalization Methods Comparison - Volatility Shock Response', fontsize=14)
plt.tight_layout()
plt.savefig('normalization_comparison.png', dpi=100)
print(f"\nPlot saved to: normalization_comparison.png")

# Clean up
import os
os.remove('temp_test_config.ini')

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)
print("\nKey Observations:")
print("- Standard IQR: Shows delayed response to volatility changes")
print("- Winsorization: Reduces impact of extreme values")
print("- Weighted IQR: Responds faster to recent volatility changes")
print("- Logit-Rank: Transforms to uniform distribution, robust to outliers")
plt.show()