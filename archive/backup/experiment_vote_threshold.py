import configparser
import subprocess
import pandas as pd
import numpy as np
import time
import os

print("="*80)
print("EXPERIMENT: VOTE_THRESHOLD PARAMETER SWEEP")
print("="*80)

# Save baseline configuration
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config_longonly.ini')

baseline_vote_threshold = config['model']['vote_threshold']
print(f"\nBaseline vote_threshold: {baseline_vote_threshold}")

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

def update_vote_threshold(value):
    """Update vote_threshold in config file"""
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    config['model']['vote_threshold'] = str(value)
    with open('config_longonly.ini', 'w') as f:
        config.write(f)

def restore_baseline():
    """Restore baseline vote_threshold"""
    update_vote_threshold(baseline_vote_threshold)

# Define test values (0.50 to 0.80 in 0.05 steps)
vote_thresholds = np.arange(0.50, 0.85, 0.05)
print(f"\nTesting vote_threshold values: {[f'{x:.2f}' for x in vote_thresholds]}")
print(f"Estimated time: {len(vote_thresholds) * 1} to {len(vote_thresholds) * 2} minutes")

# Run experiments
results = []
print("\nRUNNING EXPERIMENTS:")
print("-" * 60)

for i, val in enumerate(vote_thresholds, 1):
    print(f"\nTest {i}/{len(vote_thresholds)}: vote_threshold = {val:.2f}")
    
    # Update config
    update_vote_threshold(f"{val:.2f}")
    
    # Run walkforward
    start_time = time.time()
    metrics = run_walkforward()
    elapsed = time.time() - start_time
    
    if metrics:
        results.append({
            'vote_threshold': f"{val:.2f}",
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
print(f"\nBaseline restored: vote_threshold = {baseline_vote_threshold}")

# Save and display results
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_vote_threshold_results.csv', index=False)
    
    print("\n" + "="*80)
    print("VOTE_THRESHOLD EXPERIMENT RESULTS")
    print("="*80)
    
    print("\nAll Results (sorted by Sharpe):")
    display_df = results_df.copy()
    display_df['vote_threshold'] = display_df['vote_threshold'].astype(float)
    print(display_df.sort_values('sharpe_ratio', ascending=False).to_string(index=False, float_format='%.3f'))
    
    best = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    print(f"\nBEST CONFIGURATION:")
    print(f"  vote_threshold: {best['vote_threshold']}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"  Total P&L: {best['total_pnl']:.2f}")
    print(f"  Hit Rate: {best['hit_rate']:.3f}")
    print(f"  Total Trades: {best['total_trades']:,}")
    
    print(f"\nResults saved to: experiment_vote_threshold_results.csv")

print("\n" + "="*80)
print("VOTE_THRESHOLD EXPERIMENT COMPLETE!")
print("="*80)