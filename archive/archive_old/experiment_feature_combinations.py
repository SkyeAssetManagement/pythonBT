import configparser
import subprocess
import pandas as pd
import numpy as np
import time
import os
from itertools import combinations

print("="*80)
print("EXPERIMENT: FEATURE COMBINATION SWEEP")
print("="*80)

# Save baseline configuration
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config_longonly.ini')

baseline_features = config['data']['selected_features']
print(f"\nBaseline features: {baseline_features}")

def run_walkforward():
    """Run walkforward and extract metrics from log"""
    try:
        # Run the walkforward script
        result = subprocess.run(['python', 'walkforward_complete.py'], 
                              capture_output=True, text=True, timeout=600)
        
        # Read the last entry from performance log
        if os.path.exists('performance_log.csv'):
            df = pd.read_csv('performance_log.csv')
            if len(df) > 0:
                last_entry = df.iloc[-1]
                return {
                    'sharpe_ratio': float(last_entry['sharpe_ratio']),
                    'total_pnl': float(last_entry['total_pnl']),
                    'hit_rate': float(last_entry['hit_rate']),
                    'trading_frequency': float(last_entry['trading_frequency']),
                    'total_trades': int(last_entry['total_trades'])
                }
    except Exception as e:
        print(f"Error running walkforward: {e}")
    return None

def update_features(feature_list):
    """Update selected_features in config file"""
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    config['data']['selected_features'] = ','.join(feature_list)
    with open('config_longonly.ini', 'w') as f:
        config.write(f)

def restore_baseline():
    """Restore baseline features"""
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    config['data']['selected_features'] = baseline_features
    with open('config_longonly.ini', 'w') as f:
        config.write(f)

# Available features from baseline
available_features = ['Last10', 'First20', 'Overnight', '3day']
print(f"Available features: {available_features}")

# Generate all combinations of 2 features
feature_combinations = list(combinations(available_features, 2))
print(f"\nTesting {len(feature_combinations)} feature combinations")
print(f"Estimated time: {len(feature_combinations) * 1} to {len(feature_combinations) * 2} minutes")

# Run experiments
results = []
print("\nRUNNING EXPERIMENTS:")
print("-" * 60)

for i, combo in enumerate(feature_combinations, 1):
    feature_list = list(combo)
    feature_str = ','.join(feature_list)
    
    print(f"\nTest {i}/{len(feature_combinations)}: {feature_str}")
    
    # Update config
    update_features(feature_list)
    
    # Run walkforward
    start_time = time.time()
    metrics = run_walkforward()
    elapsed = time.time() - start_time
    
    if metrics:
        results.append({
            'features': feature_str,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_pnl': metrics['total_pnl'],
            'hit_rate': metrics['hit_rate'],
            'trading_frequency': metrics['trading_frequency'],
            'total_trades': metrics['total_trades'],
            'runtime_seconds': elapsed
        })
        print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
        print(f"  P&L: {metrics['total_pnl']:.2f}")
        print(f"  Hit Rate: {metrics['hit_rate']:.3f}")
        print(f"  Trades: {metrics['total_trades']:,}")
        print(f"  Runtime: {elapsed:.1f}s")
    else:
        print(f"  Result: FAILED")

# Restore baseline
restore_baseline()
print(f"\nBaseline restored: {baseline_features}")

# Save and display results
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_feature_combinations_results.csv', index=False)
    
    print("\n" + "="*80)
    print("FEATURE COMBINATION EXPERIMENT RESULTS")
    print("="*80)
    
    print("\nAll Results (sorted by Sharpe):")
    print(results_df.sort_values('sharpe_ratio', ascending=False).to_string(index=False, float_format='%.3f'))
    
    best = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    print(f"\nBEST CONFIGURATION:")
    print(f"  Features: {best['features']}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"  Total P&L: {best['total_pnl']:.2f}")
    print(f"  Hit Rate: {best['hit_rate']:.3f}")
    print(f"  Total Trades: {best['total_trades']:,}")
    
    print(f"\nResults saved to: experiment_feature_combinations_results.csv")

print("\n" + "="*80)
print("FEATURE COMBINATION EXPERIMENT COMPLETE!")
print("="*80)