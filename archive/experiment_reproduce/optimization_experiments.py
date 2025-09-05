import sys
import os
sys.path.append('..')  # Add parent directory to path
import configparser
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Import the OMtree modules directly
from OMtree_validation import DirectionalValidator

class OptimizationExperiment:
    def __init__(self, baseline_config_path='baseline_config.ini'):
        """Initialize with baseline configuration"""
        self.baseline_config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.baseline_config.read(baseline_config_path)
        self.results_log = []
        
    def save_baseline(self):
        """Save current config as baseline"""
        with open('OMtree_config.ini', 'w') as f:
            self.baseline_config.write(f)
    
    def restore_baseline(self):
        """Restore baseline configuration"""
        self.save_baseline()
        
    def update_config(self, section, parameter, value):
        """Update a specific parameter in config"""
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read('OMtree_config.ini')
        config[section][parameter] = str(value)
        with open('OMtree_config.ini', 'w') as f:
            config.write(f)
    
    def run_single_test(self, param_name, param_value, section, config_param):
        """Run a single walkforward test with given parameter"""
        print(f"  Testing {param_name} = {param_value}")
        
        # Update config
        self.update_config(section, config_param, param_value)
        
        # Run validation
        start_time = time.time()
        validator = DirectionalValidator(config_path='OMtree_config.ini')
        df = validator.run_validation(verbose=False)
        
        if len(df) > 0:
            # Filter to out-of-sample period
            df_filtered = validator.filter_by_date(df, '2010-01-01')
            metrics = validator.calculate_directional_metrics(df_filtered, filter_date=False)
            
            runtime = time.time() - start_time
            
            result = {
                'parameter': param_name,
                'value': param_value,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_pnl': metrics['total_pnl'],
                'hit_rate': metrics['hit_rate'],
                'edge': metrics['edge'],
                'total_trades': metrics['total_trades'],
                'trading_frequency': metrics['trading_frequency'],
                'runtime_seconds': runtime
            }
            
            print(f"    Sharpe: {metrics['sharpe_ratio']:.3f}, PnL: {metrics['total_pnl']:.1f}, "
                  f"Edge: {metrics['edge']:.1%}, Trades: {metrics['total_trades']}, Time: {runtime:.1f}s")
        else:
            result = {
                'parameter': param_name,
                'value': param_value,
                'sharpe_ratio': 0,
                'total_pnl': 0,
                'hit_rate': 0,
                'edge': 0,
                'total_trades': 0,
                'trading_frequency': 0,
                'runtime_seconds': 0
            }
            print(f"    No valid predictions generated")
        
        return result
    
    def run_experiment(self, param_name, test_values, section, config_param, results_file):
        """Run experiment for a parameter across multiple values"""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {param_name.upper()}")
        print(f"Testing {len(test_values)} values: {test_values}")
        print(f"{'='*80}")
        
        # Restore baseline before experiment
        self.restore_baseline()
        
        results = []
        for i, value in enumerate(test_values, 1):
            print(f"\n[{i}/{len(test_values)}] {param_name} = {value}")
            result = self.run_single_test(param_name, value, section, config_param)
            results.append(result)
            self.results_log.append(result)
            
            # Save intermediate results
            pd.DataFrame(results).to_csv(results_file, index=False)
            
            # Restore baseline for next test
            self.restore_baseline()
        
        # Find best configuration
        results_df = pd.DataFrame(results)
        if not results_df.empty and results_df['sharpe_ratio'].max() > 0:
            best_idx = results_df['sharpe_ratio'].idxmax()
            best_row = results_df.loc[best_idx]
            
            print(f"\nBEST {param_name.upper()}:")
            print(f"  Value: {best_row['value']}")
            print(f"  Sharpe Ratio: {best_row['sharpe_ratio']:.3f}")
            print(f"  Total PnL: {best_row['total_pnl']:.1f}")
            print(f"  Edge: {best_row['edge']:.1%}")
            print(f"  Trades: {best_row['total_trades']:.0f}")
        
        return results_df

# Initialize experiment
exp = OptimizationExperiment('baseline_config.ini')

# Set optimal baseline parameters for consistent testing
print("\nSetting optimal baseline parameters...")
exp.update_config('data', 'selected_features', 'Overnight,3day')
exp.update_config('model', 'n_trees', '200')
exp.update_config('model', 'max_depth', '1')
exp.save_baseline()
print("  Features: Overnight,3day")
print("  Trees: 200")
print("  Max Depth: 1")

# Track all results
all_results = []

# EXPERIMENT 1: vol_window (50 to 300 in 5 steps)
print("\n" + "="*80)
print("STARTING OPTIMIZATION EXPERIMENTS")
print("Objective: Maximize Annualized Sharpe Ratio")
print("="*80)

vol_window_values = [50, 112, 175, 237, 300]
vol_results = exp.run_experiment('vol_window', vol_window_values, 'preprocessing', 'vol_window', 
                                 'vol_window_results.csv')
all_results.append(('vol_window', vol_results))

print("\n" + "="*80)
print("EXPERIMENT 1 COMPLETE: vol_window")
print("="*80)

# Save summary
summary = []
for param_name, results_df in all_results:
    if not results_df.empty:
        best_idx = results_df['sharpe_ratio'].idxmax()
        best_row = results_df.loc[best_idx]
        summary.append({
            'parameter': param_name,
            'best_value': best_row['value'],
            'best_sharpe': best_row['sharpe_ratio'],
            'best_pnl': best_row['total_pnl'],
            'best_edge': best_row['edge']
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv('optimization_summary.csv', index=False)

print("\nOptimization experiment 1 of 5 complete. Results saved to vol_window_results.csv")
print(f"Best vol_window: {vol_results.loc[vol_results['sharpe_ratio'].idxmax(), 'value']}")
print(f"Best Sharpe: {vol_results['sharpe_ratio'].max():.3f}")