import configparser
import subprocess
import pandas as pd
import numpy as np
import time
import os

print("="*80)
print("EXPERIMENT: VOL_WINDOW PARAMETER SWEEP")
print("="*80)

# Save baseline configuration
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config_longonly.ini')

baseline_vol_window = config['preprocessing']['vol_window']
print(f"\nBaseline vol_window: {baseline_vol_window}")

def run_walkforward():
    """Run walkforward and extract Sharpe ratio from log"""
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
                    'trading_frequency': float(last_entry['trading_frequency'])
                }
    except Exception as e:
        print(f"Error running walkforward: {e}")
    return None

def update_vol_window(value):
    """Update vol_window in config file"""
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    config['preprocessing']['vol_window'] = str(value)
    with open('config_longonly.ini', 'w') as f:
        config.write(f)

def restore_baseline():
    """Restore baseline vol_window"""
    update_vol_window(baseline_vol_window)

# Define test values (30 to 250 in 5 steps)
vol_windows = np.linspace(30, 250, 5, dtype=int)
print(f"\nTesting vol_window values: {list(vol_windows)}")
print(f"Estimated time: {len(vol_windows) * 1} to {len(vol_windows) * 2} minutes")

# Run experiments
results = []
print("\nRUNNING EXPERIMENTS:")
print("-" * 60)

for i, val in enumerate(vol_windows, 1):
    print(f"\nTest {i}/{len(vol_windows)}: vol_window = {val}")
    
    # Update config
    update_vol_window(val)
    
    # Run walkforward
    start_time = time.time()
    metrics = run_walkforward()
    elapsed = time.time() - start_time
    
    if metrics:
        results.append({
            'vol_window': val,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_pnl': metrics['total_pnl'],
            'hit_rate': metrics['hit_rate'],
            'trading_frequency': metrics['trading_frequency'],
            'runtime_seconds': elapsed
        })
        print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
        print(f"  P&L: {metrics['total_pnl']:.2f}")
        print(f"  Hit Rate: {metrics['hit_rate']:.3f}")
        print(f"  Trade Freq: {metrics['trading_frequency']:.3f}")
        print(f"  Runtime: {elapsed:.1f}s")
    else:
        print(f"  Result: FAILED")

# Restore baseline
restore_baseline()
print(f"\nBaseline restored: vol_window = {baseline_vol_window}")

# Save and display results
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_vol_window_results.csv', index=False)
    
    print("\n" + "="*80)
    print("VOL_WINDOW EXPERIMENT RESULTS")
    print("="*80)
    
    print("\nAll Results (sorted by Sharpe):")
    print(results_df.sort_values('sharpe_ratio', ascending=False).to_string(index=False))
    
    best = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    print(f"\nBEST CONFIGURATION:")
    print(f"  vol_window: {int(best['vol_window'])}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"  Total P&L: {best['total_pnl']:.2f}")
    print(f"  Hit Rate: {best['hit_rate']:.3f}")
    
    print(f"\nResults saved to: experiment_vol_window_results.csv")

print("\n" + "="*80)
print("VOL_WINDOW EXPERIMENT COMPLETE!")
print("="*80)