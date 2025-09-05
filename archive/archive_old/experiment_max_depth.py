import configparser
import subprocess
import pandas as pd
import numpy as np
import time
import os

print("="*80)
print("EXPERIMENT: MAX_DEPTH PARAMETER SWEEP")
print("="*80)

# Save baseline configuration
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config_longonly.ini')

baseline_max_depth = config['model']['max_depth']
print(f"\nBaseline max_depth: {baseline_max_depth}")

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

def update_max_depth(value):
    """Update max_depth in config file"""
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    config['model']['max_depth'] = str(value)
    with open('config_longonly.ini', 'w') as f:
        config.write(f)

def restore_baseline():
    """Restore baseline max_depth"""
    update_max_depth(baseline_max_depth)

# Define test values (1 or 2)
max_depths = [1, 2]
print(f"\nTesting max_depth values: {max_depths}")
print(f"Estimated time: {len(max_depths) * 1} to {len(max_depths) * 2} minutes")

# Run experiments
results = []
print("\nRUNNING EXPERIMENTS:")
print("-" * 60)

for i, val in enumerate(max_depths, 1):
    print(f"\nTest {i}/{len(max_depths)}: max_depth = {val}")
    
    # Update config
    update_max_depth(str(val))
    
    # Run walkforward
    start_time = time.time()
    metrics = run_walkforward()
    elapsed = time.time() - start_time
    
    if metrics:
        results.append({
            'max_depth': val,
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
print(f"\nBaseline restored: max_depth = {baseline_max_depth}")

# Save and display results
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_max_depth_results.csv', index=False)
    
    print("\n" + "="*80)
    print("MAX_DEPTH EXPERIMENT RESULTS")
    print("="*80)
    
    print("\nAll Results (sorted by Sharpe):")
    print(results_df.sort_values('sharpe_ratio', ascending=False).to_string(index=False, float_format='%.3f'))
    
    best = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    print(f"\nBEST CONFIGURATION:")
    print(f"  max_depth: {int(best['max_depth'])}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"  Total P&L: {best['total_pnl']:.2f}")
    print(f"  Hit Rate: {best['hit_rate']:.3f}")
    print(f"  Total Trades: {best['total_trades']:,}")
    
    print(f"\nResults saved to: experiment_max_depth_results.csv")

print("\n" + "="*80)
print("MAX_DEPTH EXPERIMENT COMPLETE!")
print("="*80)