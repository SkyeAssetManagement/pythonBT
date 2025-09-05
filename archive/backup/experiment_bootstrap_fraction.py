import configparser
import subprocess
import pandas as pd
import numpy as np
import time
import os

print("="*80)
print("EXPERIMENT: BOOTSTRAP_FRACTION PARAMETER SWEEP")
print("="*80)

# Save baseline configuration
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config_longonly.ini')

baseline_bootstrap_fraction = config['model']['bootstrap_fraction']
print(f"\nBaseline bootstrap_fraction: {baseline_bootstrap_fraction}")

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

def update_bootstrap_fraction(value):
    """Update bootstrap_fraction in config file"""
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    config['model']['bootstrap_fraction'] = str(value)
    with open('config_longonly.ini', 'w') as f:
        config.write(f)

def restore_baseline():
    """Restore baseline bootstrap_fraction"""
    update_bootstrap_fraction(baseline_bootstrap_fraction)

# Define test values (0.3 to 1.0 in 0.1 steps)
bootstrap_fractions = np.arange(0.3, 1.1, 0.1)
print(f"\nTesting bootstrap_fraction values: {[f'{x:.1f}' for x in bootstrap_fractions]}")
print(f"Estimated time: {len(bootstrap_fractions) * 1} to {len(bootstrap_fractions) * 2} minutes")

# Run experiments
results = []
print("\nRUNNING EXPERIMENTS:")
print("-" * 60)

for i, val in enumerate(bootstrap_fractions, 1):
    print(f"\nTest {i}/{len(bootstrap_fractions)}: bootstrap_fraction = {val:.1f}")
    
    # Update config
    update_bootstrap_fraction(f"{val:.1f}")
    
    # Run walkforward
    start_time = time.time()
    metrics = run_walkforward()
    elapsed = time.time() - start_time
    
    if metrics:
        results.append({
            'bootstrap_fraction': f"{val:.1f}",
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
print(f"\nBaseline restored: bootstrap_fraction = {baseline_bootstrap_fraction}")

# Save and display results
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_bootstrap_fraction_results.csv', index=False)
    
    print("\n" + "="*80)
    print("BOOTSTRAP_FRACTION EXPERIMENT RESULTS")
    print("="*80)
    
    print("\nAll Results (sorted by Sharpe):")
    display_df = results_df.copy()
    display_df['bootstrap_fraction'] = display_df['bootstrap_fraction'].astype(float)
    print(display_df.sort_values('sharpe_ratio', ascending=False).to_string(index=False, float_format='%.3f'))
    
    best = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    print(f"\nBEST CONFIGURATION:")
    print(f"  bootstrap_fraction: {best['bootstrap_fraction']}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"  Total P&L: {best['total_pnl']:.2f}")
    print(f"  Hit Rate: {best['hit_rate']:.3f}")
    print(f"  Total Trades: {best['total_trades']:,}")
    
    print(f"\nResults saved to: experiment_bootstrap_fraction_results.csv")

print("\n" + "="*80)
print("BOOTSTRAP_FRACTION EXPERIMENT COMPLETE!")
print("="*80)