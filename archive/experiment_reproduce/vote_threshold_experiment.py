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
print("VOTE_THRESHOLD OPTIMIZATION EXPERIMENT")
print("Testing values: 0.5 to 0.9 in 6 steps")
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

# Test values: 0.5 to 0.9 in 6 steps
vote_values = [0.5, 0.58, 0.66, 0.74, 0.82, 0.9]
results = []

for i, value in enumerate(vote_values, 1):
    print(f"\n[{i}/{len(vote_values)}] Testing vote_threshold = {value}")
    
    # Update config
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('OMtree_config.ini')
    config['model']['vote_threshold'] = str(value)
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
                'vote_threshold': value,
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
                'vote_threshold': value,
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
            'vote_threshold': value,
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
df_results.to_csv('vote_threshold_results.csv', index=False)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print("\nVote     | Sharpe | Total PnL | Hit Rate | Edge  | Trades")
print("-"*60)
for _, row in df_results.iterrows():
    print(f"{row['vote_threshold']:8.2f} | {row['sharpe_ratio']:6.3f} | "
          f"{row['total_pnl']:9.1f} | {row['hit_rate']:8.1%} | "
          f"{row['edge']:5.1%} | {row['total_trades']:6.0f}")

# Find best
if df_results['sharpe_ratio'].max() > 0:
    best_idx = df_results['sharpe_ratio'].idxmax()
    best = df_results.loc[best_idx]
    
    print(f"\nBEST VOTE_THRESHOLD:")
    print(f"  Value: {best['vote_threshold']:.2f}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"  Total PnL: {best['total_pnl']:.1f}")
    print(f"  Edge: {best['edge']:.1%}")
    print(f"  Trades: {best['total_trades']:.0f}")

print("\nExperiment complete! Results saved to vote_threshold_results.csv")