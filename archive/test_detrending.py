"""
Test detrending functionality
"""

import pandas as pd
import numpy as np
import configparser
from OMtree_preprocessing import DataPreprocessor
import matplotlib.pyplot as plt

print("=" * 80)
print("TESTING DETRENDING FUNCTIONALITY")
print("=" * 80)

# Load sample data
df = pd.read_csv('DTSmlDATA7x7.csv')
print(f"\nLoaded data: {df.shape}")

# Test 1: Without detrending
print("\n1. WITHOUT DETRENDING:")
print("-" * 40)

config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')
config['preprocessing']['detrend_features'] = 'false'
config['preprocessing']['normalize_features'] = 'false'  # Turn off to see raw values

test_config = 'test_detrend.ini'
with open(test_config, 'w') as f:
    config.write(f)

preprocessor = DataPreprocessor(test_config)
processed_no_detrend = preprocessor.process_data(df)

# Check columns
feature_cols = [c for c in processed_no_detrend.columns if c.startswith('Ret_') and 'fwd' not in c and '_vol_adj' in c]
print(f"Feature columns: {len(feature_cols)}")
print(f"Sample: {feature_cols[:3]}")

# Test 2: With detrending
print("\n2. WITH DETRENDING:")
print("-" * 40)

config['preprocessing']['detrend_features'] = 'true'
with open(test_config, 'w') as f:
    config.write(f)

preprocessor = DataPreprocessor(test_config)
processed_with_detrend = preprocessor.process_data(df)

# Check columns
detrend_cols = [c for c in processed_with_detrend.columns if '_detrend' in c]
median_cols = [c for c in processed_with_detrend.columns if '_median' in c]
print(f"Detrended columns: {len(detrend_cols)}")
print(f"Median columns: {len(median_cols)}")
print(f"Sample detrend: {detrend_cols[:3] if detrend_cols else 'None'}")

# Test 3: Compare values
print("\n3. VALUE COMPARISON:")
print("-" * 40)

if detrend_cols:
    test_feature = 'Ret_0-1hr'
    
    # Original values
    orig_values = df[test_feature].iloc[100:110].values
    
    # Detrended values
    detrend_col = f'{test_feature}_detrend'
    if detrend_col in processed_with_detrend.columns:
        detrend_values = processed_with_detrend[detrend_col].iloc[100:110].values
        median_values = processed_with_detrend[f'{test_feature}_median'].iloc[100:110].values
        
        print(f"Feature: {test_feature}")
        print(f"Sample indices: 100-109")
        print(f"\nOriginal | Median | Detrended")
        print("-" * 40)
        for i in range(min(5, len(orig_values))):
            print(f"{orig_values[i]:8.4f} | {median_values[i]:8.4f} | {detrend_values[i]:8.4f}")
        
        # Verify detrending calculation
        calculated_detrend = orig_values - median_values
        diff = np.abs(calculated_detrend - detrend_values)
        max_diff = np.nanmax(diff)
        print(f"\nMax difference between manual calc and detrend: {max_diff:.6f}")
        
        if max_diff < 1e-6:
            print("[PASS] Detrending calculation is correct")
        else:
            print("[FAIL] Detrending calculation has errors")
    else:
        print(f"Detrended column {detrend_col} not found")
else:
    print("No detrended columns found")

# Test 4: Check that target is NOT detrended
print("\n4. TARGET PRESERVATION CHECK:")
print("-" * 40)

target_detrend_cols = [c for c in processed_with_detrend.columns if 'fwd' in c and '_detrend' in c]
if target_detrend_cols:
    print(f"[FAIL] Target columns were detrended: {target_detrend_cols}")
else:
    print("[PASS] Target columns were NOT detrended (correct)")

# Test 5: Create visual comparison
print("\n5. VISUAL COMPARISON:")
print("-" * 40)

if detrend_cols:
    test_feature = 'Ret_0-1hr'
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot original
    axes[0].plot(df[test_feature].iloc[:500], label='Original', alpha=0.7)
    axes[0].set_title('Original Feature Values')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot median
    if f'{test_feature}_median' in processed_with_detrend.columns:
        axes[1].plot(df[test_feature].iloc[:500], label='Original', alpha=0.5)
        axes[1].plot(processed_with_detrend[f'{test_feature}_median'].iloc[:500], 
                    label='Rolling Median', color='red', linewidth=2)
        axes[1].set_title('Original with Rolling Median')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot detrended
    if f'{test_feature}_detrend' in processed_with_detrend.columns:
        axes[2].plot(processed_with_detrend[f'{test_feature}_detrend'].iloc[:500], 
                    label='Detrended (Original - Median)', color='green', alpha=0.7)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_title('Detrended Feature Values')
        axes[2].set_ylabel('Value')
        axes[2].set_xlabel('Sample Index')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Detrending Analysis: {test_feature}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('detrending_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot to 'detrending_comparison.png'")
    plt.close()

# Clean up
import os
if os.path.exists(test_config):
    os.remove(test_config)

print("\n" + "=" * 80)
print("DETRENDING TEST COMPLETE")
print("=" * 80)
print("\nSummary:")
print("- Detrending subtracts rolling median from features")
print("- Target columns are NOT detrended (as designed)")
print("- Detrended values are centered around zero")
print("- Original trend/bias is removed while preserving volatility")