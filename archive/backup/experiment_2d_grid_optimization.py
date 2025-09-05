import configparser
import subprocess
import pandas as pd
import numpy as np
import time
import os
from itertools import product

print("="*80)
print("EXPERIMENT: 2D GRID OPTIMIZATION - UP_THRESHOLD vs VOTE_THRESHOLD")
print("="*80)

# Save baseline configuration
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config_longonly.ini')

baseline_up_threshold = config['model']['up_threshold']
baseline_vote_threshold = config['model']['vote_threshold']
print(f"\nBaseline up_threshold: {baseline_up_threshold}")
print(f"Baseline vote_threshold: {baseline_vote_threshold}")

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

def update_parameters(up_thresh, vote_thresh):
    """Update both parameters in config file"""
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    config['model']['up_threshold'] = str(up_thresh)
    config['model']['vote_threshold'] = str(vote_thresh)
    with open('config_longonly.ini', 'w') as f:
        config.write(f)

def restore_baseline():
    """Restore baseline parameters"""
    update_parameters(baseline_up_threshold, baseline_vote_threshold)

# Define parameter grids
vote_thresholds = [0.50, 0.60, 0.70, 0.80]
up_thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

print(f"\nVote thresholds: {vote_thresholds}")
print(f"Up thresholds: {up_thresholds}")

# Generate all combinations
parameter_combinations = list(product(vote_thresholds, up_thresholds))
total_tests = len(parameter_combinations)

print(f"\nTotal combinations to test: {total_tests} (4 x 7 grid)")
print(f"Estimated time: {total_tests * 1} to {total_tests * 2} minutes")

# Create progress tracking
def show_progress(current, total, bar_length=50):
    """Display progress bar"""
    percent = current / total
    progress = int(bar_length * percent)
    bar = '=' * progress + '-' * (bar_length - progress)
    print(f"\rProgress: [{bar}] {current}/{total} ({percent*100:.1f}%)", end='', flush=True)

# Run experiments
results = []
print("\n\nRUNNING 2D GRID OPTIMIZATION:")
print("-" * 80)

for i, (vote_thresh, up_thresh) in enumerate(parameter_combinations, 1):
    print(f"\nTest {i:2d}/{total_tests}: vote_threshold={vote_thresh:.2f}, up_threshold={up_thresh:.2f}")
    
    # Update config
    update_parameters(up_thresh, vote_thresh)
    
    # Run walkforward
    start_time = time.time()
    metrics = run_walkforward()
    elapsed = time.time() - start_time
    
    if metrics:
        results.append({
            'vote_threshold': vote_thresh,
            'up_threshold': up_thresh,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_pnl': metrics['total_pnl'],
            'hit_rate': metrics['hit_rate'],
            'trading_frequency': metrics['trading_frequency'],
            'total_trades': metrics['total_trades'],
            'runtime_seconds': elapsed
        })
        print(f"    Sharpe: {metrics['sharpe_ratio']:.3f} | P&L: {metrics['total_pnl']:.1f} | Trades: {metrics['total_trades']:,} | Hit Rate: {metrics['hit_rate']:.3f}")
    else:
        print(f"    Result: FAILED")
    
    # Show progress
    show_progress(i, total_tests)

print("\n")

# Restore baseline
restore_baseline()
print(f"\nBaseline restored: up_threshold={baseline_up_threshold}, vote_threshold={baseline_vote_threshold}")

# Save and analyze results
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_2d_grid_results.csv', index=False)
    
    print("\n" + "="*80)
    print("2D GRID OPTIMIZATION RESULTS")
    print("="*80)
    
    # Find best overall
    best_idx = results_df['sharpe_ratio'].idxmax()
    best = results_df.iloc[best_idx]
    
    print(f"\nBEST OVERALL CONFIGURATION:")
    print(f"  vote_threshold: {best['vote_threshold']:.2f}")
    print(f"  up_threshold: {best['up_threshold']:.2f}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"  Total P&L: {best['total_pnl']:.2f}")
    print(f"  Hit Rate: {best['hit_rate']:.3f}")
    print(f"  Total Trades: {best['total_trades']:,}")
    print(f"  Trading Frequency: {best['trading_frequency']:.3f}")
    
    # Create pivot table for visualization
    pivot_sharpe = results_df.pivot(index='vote_threshold', columns='up_threshold', values='sharpe_ratio')
    pivot_trades = results_df.pivot(index='vote_threshold', columns='up_threshold', values='total_trades')
    pivot_hit_rate = results_df.pivot(index='vote_threshold', columns='up_threshold', values='hit_rate')
    
    print("\n" + "-"*80)
    print("SHARPE RATIO HEATMAP (rows=vote_threshold, cols=up_threshold):")
    print("-"*80)
    print(pivot_sharpe.round(3).to_string())
    
    print("\n" + "-"*80)
    print("TOTAL TRADES HEATMAP (rows=vote_threshold, cols=up_threshold):")
    print("-"*80)
    print(pivot_trades.astype(int).to_string())
    
    print("\n" + "-"*80)
    print("HIT RATE HEATMAP (rows=vote_threshold, cols=up_threshold):")
    print("-"*80)
    print(pivot_hit_rate.round(3).to_string())
    
    # Analyze trade-offs
    print("\n" + "="*80)
    print("TRADE-OFF ANALYSIS")
    print("="*80)
    
    # Sort by Sharpe ratio
    top_10 = results_df.nlargest(10, 'sharpe_ratio')[['vote_threshold', 'up_threshold', 'sharpe_ratio', 'total_trades', 'hit_rate', 'trading_frequency']]
    print("\nTOP 10 CONFIGURATIONS BY SHARPE RATIO:")
    print(top_10.round(3).to_string(index=False))
    
    # Correlation analysis
    print(f"\nCORRELATION ANALYSIS:")
    print(f"  Sharpe vs Total Trades: {results_df['sharpe_ratio'].corr(results_df['total_trades']):.3f}")
    print(f"  Sharpe vs Hit Rate: {results_df['sharpe_ratio'].corr(results_df['hit_rate']):.3f}")
    print(f"  Sharpe vs Trading Frequency: {results_df['sharpe_ratio'].corr(results_df['trading_frequency']):.3f}")
    print(f"  Total Trades vs Hit Rate: {results_df['total_trades'].corr(results_df['hit_rate']):.3f}")
    
    print(f"\nResults saved to: experiment_2d_grid_results.csv")

print("\n" + "="*80)
print("2D GRID OPTIMIZATION COMPLETE!")
print("="*80)