import configparser
import subprocess
import pandas as pd
import numpy as np
from itertools import combinations
import time
import os

print("="*80)
print("PARAMETER OPTIMIZATION EXPERIMENTS")
print("="*80)

# Save baseline configuration
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config_longonly.ini')

baseline_config = {
    'selected_features': config['data']['selected_features'],
    'normalize_features': config['preprocessing']['normalize_features'],
    'normalize_target': config['preprocessing']['normalize_target'],
    'vol_window': config['preprocessing']['vol_window'],
    'smoothing_alpha': config['preprocessing']['smoothing_alpha'],
    'n_trees': config['model']['n_trees'],
    'max_depth': config['model']['max_depth'],
    'bootstrap_fraction': config['model']['bootstrap_fraction'],
    'min_leaf_fraction': config['model']['min_leaf_fraction'],
    'up_threshold': config['model']['up_threshold'],
    'vote_threshold': config['model']['vote_threshold']
}

print("\nBASELINE CONFIGURATION:")
for key, value in baseline_config.items():
    print(f"  {key}: {value}")

def run_walkforward():
    """Run walkforward and extract Sharpe ratio from log"""
    try:
        # Run the walkforward script
        result = subprocess.run(['python', 'walkforward_complete.py'], 
                              capture_output=True, text=True, timeout=300)
        
        # Read the last entry from performance log
        if os.path.exists('performance_log.csv'):
            df = pd.read_csv('performance_log.csv')
            if len(df) > 0:
                last_entry = df.iloc[-1]
                return float(last_entry['sharpe_ratio'])
    except Exception as e:
        print(f"Error running walkforward: {e}")
    return None

def update_config(param_section, param_name, value):
    """Update a single parameter in config file"""
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    config[param_section][param_name] = str(value)
    with open('config_longonly.ini', 'w') as f:
        config.write(f)

def restore_baseline():
    """Restore baseline configuration"""
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    
    config['data']['selected_features'] = baseline_config['selected_features']
    config['preprocessing']['normalize_features'] = baseline_config['normalize_features']
    config['preprocessing']['normalize_target'] = baseline_config['normalize_target']
    config['preprocessing']['vol_window'] = baseline_config['vol_window']
    config['preprocessing']['smoothing_alpha'] = baseline_config['smoothing_alpha']
    config['model']['n_trees'] = baseline_config['n_trees']
    config['model']['max_depth'] = baseline_config['max_depth']
    config['model']['bootstrap_fraction'] = baseline_config['bootstrap_fraction']
    config['model']['min_leaf_fraction'] = baseline_config['min_leaf_fraction']
    config['model']['up_threshold'] = baseline_config['up_threshold']
    config['model']['vote_threshold'] = baseline_config['vote_threshold']
    
    with open('config_longonly.ini', 'w') as f:
        config.write(f)

# Define experiments
experiments = []

# 1. Vol window experiments (30 to 250 in 5 steps)
vol_windows = np.linspace(30, 250, 5, dtype=int)
for val in vol_windows:
    experiments.append(('preprocessing', 'vol_window', val, f'vol_window_{val}'))

# 2. Up threshold experiments (0 to 0.5 in 0.05 steps)
up_thresholds = np.arange(0, 0.55, 0.05)
for val in up_thresholds:
    experiments.append(('model', 'up_threshold', f'{val:.2f}', f'up_threshold_{val:.2f}'))

# 3. Vote threshold experiments (0.50 to 0.80 in 0.05 steps)
vote_thresholds = np.arange(0.50, 0.85, 0.05)
for val in vote_thresholds:
    experiments.append(('model', 'vote_threshold', f'{val:.2f}', f'vote_threshold_{val:.2f}'))

# 4. Bootstrap fraction experiments (0.3 to 1.0 in 0.1 steps)
bootstrap_fractions = np.arange(0.3, 1.1, 0.1)
for val in bootstrap_fractions:
    experiments.append(('model', 'bootstrap_fraction', f'{val:.1f}', f'bootstrap_fraction_{val:.1f}'))

# 5. Min leaf fraction experiments (0.05 to 0.40 in 0.05 steps)
min_leaf_fractions = np.arange(0.05, 0.45, 0.05)
for val in min_leaf_fractions:
    experiments.append(('model', 'min_leaf_fraction', f'{val:.2f}', f'min_leaf_fraction_{val:.2f}'))

# 6. Max depth experiments (1 or 2)
max_depths = [1, 2]
for val in max_depths:
    experiments.append(('model', 'max_depth', str(val), f'max_depth_{val}'))

# 7. Feature combination experiments
all_features = ['Last10', 'First20', 'Overnight', '3day']
feature_pairs = list(combinations(all_features, 2))
for pair in feature_pairs:
    feature_str = ','.join(pair)
    experiments.append(('data', 'selected_features', feature_str, f'features_{pair[0]}_{pair[1]}'))

print(f"\nTOTAL EXPERIMENTS TO RUN: {len(experiments)}")
print(f"Estimated time: {len(experiments) * 0.5:.1f} to {len(experiments) * 1.5:.1f} minutes")

# Run experiments
results = []
print("\nRUNNING EXPERIMENTS:")
print("-" * 60)

for i, (section, param, value, label) in enumerate(experiments, 1):
    print(f"\nExperiment {i}/{len(experiments)}: {label}")
    print(f"  Setting {section}.{param} = {value}")
    
    # Update config
    update_config(section, param, value)
    
    # Run walkforward
    sharpe = run_walkforward()
    
    if sharpe is not None:
        results.append({
            'experiment': label,
            'section': section,
            'parameter': param,
            'value': value,
            'sharpe_ratio': sharpe
        })
        print(f"  Result: Sharpe = {sharpe:.3f}")
    else:
        print(f"  Result: FAILED")
    
    # Restore baseline
    restore_baseline()
    print(f"  Baseline restored")
    
    # Small delay to avoid overwhelming the system
    time.sleep(1)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('experiment_results.csv', index=False)

print("\n" + "="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)

# Group by parameter type and show best values
param_types = results_df['parameter'].unique()
for param in param_types:
    param_results = results_df[results_df['parameter'] == param]
    best = param_results.loc[param_results['sharpe_ratio'].idxmax()]
    print(f"\n{param.upper()}:")
    print(f"  Best value: {best['value']}")
    print(f"  Best Sharpe: {best['sharpe_ratio']:.3f}")
    print(f"  All results:")
    for _, row in param_results.sort_values('sharpe_ratio', ascending=False).head(5).iterrows():
        print(f"    {row['value']}: {row['sharpe_ratio']:.3f}")

# Overall best configuration
print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS:")
print("="*80)
for i, row in enumerate(results_df.nlargest(10, 'sharpe_ratio').iterrows(), 1):
    _, data = row
    print(f"{i}. {data['experiment']}: Sharpe = {data['sharpe_ratio']:.3f}")

print("\n" + "="*80)
print("EXPERIMENTS COMPLETE!")
print(f"Results saved to: experiment_results.csv")
print(f"Performance log updated: performance_log.csv")
print("="*80)