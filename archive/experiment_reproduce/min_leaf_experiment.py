import sys
import os
sys.path.append('..')
import configparser
import pandas as pd
import numpy as np
from datetime import datetime
import time
from OMtree_validation import DirectionalValidator

print("="*80)
print("MIN_LEAF_FRACTION OPTIMIZATION EXPERIMENT")
print("Testing values: 0.05 to 0.4 in 5 steps")
print("="*80)

# Load baseline config
baseline_config = configparser.ConfigParser(inline_comment_prefixes='#')
baseline_config.read('baseline_config.ini')

# Set optimal baseline parameters
baseline_config['data']['selected_features'] = 'Overnight,3day'
baseline_config['model']['n_trees'] = '200'
baseline_config['model']['max_depth'] = '1'

# Save as working config
with open('OMtree_config.ini', 'w') as f:
    baseline_config.write(f)

# Test values: 0.05 to 0.4 in 5 steps
min_leaf_values = [0.05, 0.14, 0.23, 0.31, 0.4]
results = []

for i, value in enumerate(min_leaf_values, 1):
    print(f"\n[{i}/{len(min_leaf_values)}] Testing min_leaf_fraction = {value}")
    
    # Update config
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('OMtree_config.ini')
    config['model']['min_leaf_fraction'] = str(value)
    with open('OMtree_config.ini', 'w') as f:
        config.write(f)
    
    # Run validation
    start_time = time.time()
    try:
        validator = DirectionalValidator(config_path='OMtree_config.ini')
        df = validator.run_validation(verbose=False)
        
        if len(df) > 0:
            df_filtered = validator.filter_by_date(df, '2010-01-01')
            metrics = validator.calculate_directional_metrics(df_filtered, filter_date=False)
            
            result = {
                'min_leaf_fraction': value,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_pnl': metrics['total_pnl'],
                'hit_rate': metrics['hit_rate'],
                'edge': metrics['edge'],
                'total_trades': metrics['total_trades'],
                'trading_frequency': metrics['trading_frequency'],
                'runtime': time.time() - start_time
            }
            
            print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}, PnL: {metrics['total_pnl']:.1f}, "
                  f"Edge: {metrics['edge']:.1%}, Trades: {metrics['total_trades']}, Time: {result['runtime']:.1f}s")
        else:
            result = {
                'min_leaf_fraction': value,
                'sharpe_ratio': 0,
                'total_pnl': 0,
                'hit_rate': 0,
                'edge': 0,
                'total_trades': 0,
                'trading_frequency': 0,
                'runtime': time.time() - start_time
            }
            print(f"  No trades generated")
    except Exception as e:
        print(f"  Error: {e}")
        result = {
            'min_leaf_fraction': value,
            'sharpe_ratio': 0,
            'total_pnl': 0,
            'hit_rate': 0,
            'edge': 0,
            'total_trades': 0,
            'trading_frequency': 0,
            'runtime': 0
        }
    
    results.append(result)
    
    # Restore baseline for next test
    with open('OMtree_config.ini', 'w') as f:
        baseline_config.write(f)

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('min_leaf_fraction_results.csv', index=False)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print("\nMin Leaf | Sharpe | Total PnL | Hit Rate | Edge  | Trades")
print("-"*60)
for _, row in df_results.iterrows():
    print(f"{row['min_leaf_fraction']:8.2f} | {row['sharpe_ratio']:6.3f} | "
          f"{row['total_pnl']:9.1f} | {row['hit_rate']:8.1%} | "
          f"{row['edge']:5.1%} | {row['total_trades']:6.0f}")

# Find best
if df_results['sharpe_ratio'].max() > 0:
    best_idx = df_results['sharpe_ratio'].idxmax()
    best = df_results.loc[best_idx]
    
    print(f"\nBEST MIN_LEAF_FRACTION:")
    print(f"  Value: {best['min_leaf_fraction']:.2f}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"  Total PnL: {best['total_pnl']:.1f}")
    print(f"  Edge: {best['edge']:.1%}")
    print(f"  Trades: {best['total_trades']:.0f}")

print("\nExperiment complete! Results saved to min_leaf_fraction_results.csv")