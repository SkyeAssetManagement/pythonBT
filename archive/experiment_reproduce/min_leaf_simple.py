import sys
import os
sys.path.append('..')  # Add parent directory to path
import configparser
import pandas as pd
import numpy as np
from datetime import datetime

# Import the OMtree modules directly
from OMtree_validation import DirectionalValidator

print("="*80)
print("MIN_LEAF_FRACTION EXPERIMENT - SIMPLIFIED REPRODUCTION")
print("="*80)

# Read original experiment results for comparison
original_file = '../archive_old/experiment_min_leaf_fraction_results.csv'
if os.path.exists(original_file):
    original_df = pd.read_csv(original_file)
    print("\nOriginal experiment results (from archive):")
    print("-"*50)
    print("Min Leaf | Sharpe | Total PnL | Hit Rate | Trades")
    print("-"*50)
    for _, row in original_df.iterrows():
        if not pd.isna(row['sharpe_ratio']):
            print(f"{row['min_leaf_fraction']:8.2f} | {row['sharpe_ratio']:6.3f} | "
                  f"{row['total_pnl']:9.1f} | {row['hit_rate']:8.3f} | {row['total_trades']:6.0f}")

print("\n" + "="*80)
print("RUNNING CURRENT EXPERIMENT WITH OMTREE MODEL")
print("="*80)

# Test just 3 key values for min_leaf_fraction
test_values = [0.10, 0.20, 0.30]
results = []

# Save baseline config
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')
baseline_min_leaf = float(config['model']['min_leaf_fraction'])

# Set optimal parameters for testing
print("\nSetting optimal parameters:")
config['data']['selected_features'] = 'Overnight,3day'
config['preprocessing']['vol_window'] = '250'
config['model']['n_trees'] = '200'
config['model']['bootstrap_fraction'] = '0.6'
config['model']['target_threshold'] = '0.2'
config['model']['vote_threshold'] = '0.5'
print("  - Features: Overnight,3day")
print("  - Trees: 200, Bootstrap: 0.6")
print("  - Target: 0.2, Vote: 0.5")
print("  - Vol window: 250")

print(f"\nTesting {len(test_values)} min_leaf values: {test_values}")
print("-"*80)

for i, value in enumerate(test_values, 1):
    print(f"\n[{i}/{len(test_values)}] Testing min_leaf_fraction = {value}")
    
    # Update config
    config['model']['min_leaf_fraction'] = str(value)
    with open('OMtree_config.ini', 'w') as f:
        config.write(f)
    
    # Run validation directly
    print(f"  Running validation...")
    validator = DirectionalValidator(config_path='OMtree_config.ini')
    df = validator.run_validation(verbose=False)
    
    if len(df) > 0:
        # Filter to out-of-sample period
        df_filtered = validator.filter_by_date(df, '2010-01-01')
        metrics = validator.calculate_directional_metrics(df_filtered, filter_date=False)
        
        print(f"  Results: Sharpe={metrics['sharpe_ratio']:.3f}, "
              f"PnL={metrics['total_pnl']:.1f}, "
              f"Hit={metrics['hit_rate']:.1%}, "
              f"Edge={metrics['edge']:.1%}, "
              f"Trades={metrics['total_trades']}")
        
        results.append({
            'min_leaf_fraction': value,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_pnl': metrics['total_pnl'],
            'hit_rate': metrics['hit_rate'],
            'edge': metrics['edge'],
            'total_trades': metrics['total_trades'],
            'trading_frequency': metrics['trading_frequency']
        })
    else:
        print(f"  No valid predictions generated")
        results.append({
            'min_leaf_fraction': value,
            'sharpe_ratio': 0,
            'total_pnl': 0,
            'hit_rate': 0,
            'edge': 0,
            'total_trades': 0,
            'trading_frequency': 0
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('min_leaf_reproduction_results.csv', index=False)

print("\n" + "="*80)
print("EXPERIMENT RESULTS COMPARISON")
print("="*80)

print("\nReproduced Results (Current OMtree Model):")
print("-"*50)
print("Min Leaf | Sharpe | Total PnL | Hit Rate | Edge  | Trades")
print("-"*50)
for _, row in results_df.iterrows():
    print(f"{row['min_leaf_fraction']:8.2f} | {row['sharpe_ratio']:6.3f} | "
          f"{row['total_pnl']:9.1f} | {row['hit_rate']:8.1%} | "
          f"{row['edge']:5.1%} | {row['total_trades']:6.0f}")

# Find best configuration
if len(results_df) > 0 and results_df['sharpe_ratio'].max() > 0:
    best_idx = results_df['sharpe_ratio'].idxmax()
    best_row = results_df.loc[best_idx]
    
    print(f"\nBEST CONFIGURATION (Current):")
    print(f"  Min Leaf Fraction: {best_row['min_leaf_fraction']}")
    print(f"  Sharpe Ratio: {best_row['sharpe_ratio']:.3f}")
    print(f"  Edge: {best_row['edge']:.1%}")
    print(f"  Total Trades: {best_row['total_trades']:.0f}")

# Restore baseline
config['model']['min_leaf_fraction'] = str(baseline_min_leaf)
with open('OMtree_config.ini', 'w') as f:
    config.write(f)
print(f"\nRestored baseline min_leaf_fraction: {baseline_min_leaf}")

print("\nExperiment complete! Results saved to min_leaf_reproduction_results.csv")