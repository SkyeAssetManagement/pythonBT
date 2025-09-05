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
print("COMPREHENSIVE OPTIMIZATION EXPERIMENTS")
print("Objective: Maximize Annualized Sharpe Ratio")
print("="*80)

class OptimizationRunner:
    def __init__(self):
        # Save baseline config
        self.baseline_config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.baseline_config.read('../OMtree_config.ini')
        
        # Set optimal baseline parameters
        self.baseline_config['data']['selected_features'] = 'Overnight,3day'
        self.baseline_config['model']['n_trees'] = '200'
        self.baseline_config['model']['max_depth'] = '1'
        
        # Save as working config
        with open('OMtree_config.ini', 'w') as f:
            self.baseline_config.write(f)
    
    def run_parameter_sweep(self, param_name, values, section, config_key):
        """Run full walkforward for each parameter value"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {param_name}")
        print(f"Testing values: {values}")
        print(f"{'='*60}")
        
        for i, value in enumerate(values, 1):
            print(f"\n[{i}/{len(values)}] Testing {param_name} = {value}")
            
            # Update config
            config = configparser.ConfigParser(inline_comment_prefixes='#')
            config.read('OMtree_config.ini')
            config[section][config_key] = str(value)
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
                        'parameter': param_name,
                        'value': value,
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'total_pnl': metrics['total_pnl'],
                        'hit_rate': metrics['hit_rate'],
                        'edge': metrics['edge'],
                        'total_trades': metrics['total_trades'],
                        'runtime': time.time() - start_time
                    }
                    
                    print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}, PnL: {metrics['total_pnl']:.1f}, "
                          f"Edge: {metrics['edge']:.1%}, Trades: {metrics['total_trades']}")
                else:
                    result = {
                        'parameter': param_name,
                        'value': value,
                        'sharpe_ratio': -999,
                        'total_pnl': 0,
                        'hit_rate': 0,
                        'edge': 0,
                        'total_trades': 0,
                        'runtime': time.time() - start_time
                    }
                    print(f"  No trades generated")
            except Exception as e:
                print(f"  Error: {e}")
                result = {
                    'parameter': param_name,
                    'value': value,
                    'sharpe_ratio': -999,
                    'total_pnl': 0,
                    'hit_rate': 0,
                    'edge': 0,
                    'total_trades': 0,
                    'runtime': 0
                }
            
            results.append(result)
            
            # Restore baseline for next test
            with open('OMtree_config.ini', 'w') as f:
                self.baseline_config.write(f)
        
        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{param_name}_optimization.csv', index=False)
        
        # Find best
        if len(df_results[df_results['sharpe_ratio'] > -999]) > 0:
            best_idx = df_results[df_results['sharpe_ratio'] > -999]['sharpe_ratio'].idxmax()
            best = df_results.loc[best_idx]
            print(f"\nBest {param_name}: {best['value']} (Sharpe: {best['sharpe_ratio']:.3f})")
        
        return df_results

# Run all experiments
runner = OptimizationRunner()
all_results = {}

# 1. VOL_WINDOW: 50 to 300 in 5 steps
vol_values = [50, 112, 175, 237, 300]
all_results['vol_window'] = runner.run_parameter_sweep('vol_window', vol_values, 'preprocessing', 'vol_window')

# 2. BOOTSTRAP_FRACTION: 0.3 to 1.0 in 5 steps  
bootstrap_values = [0.3, 0.475, 0.65, 0.825, 1.0]
all_results['bootstrap_fraction'] = runner.run_parameter_sweep('bootstrap_fraction', bootstrap_values, 'model', 'bootstrap_fraction')

# 3. MIN_LEAF_FRACTION: 0.05 to 0.4 in 5 steps
min_leaf_values = [0.05, 0.14, 0.23, 0.31, 0.4]
all_results['min_leaf_fraction'] = runner.run_parameter_sweep('min_leaf_fraction', min_leaf_values, 'model', 'min_leaf_fraction')

# 4. TARGET_THRESHOLD: 0 to 0.3 in 6 steps
target_values = [0.0, 0.06, 0.12, 0.18, 0.24, 0.3]
all_results['target_threshold'] = runner.run_parameter_sweep('target_threshold', target_values, 'model', 'target_threshold')

# 5. VOTE_THRESHOLD: 0.5 to 0.9 in 6 steps
vote_values = [0.5, 0.58, 0.66, 0.74, 0.82, 0.9]
all_results['vote_threshold'] = runner.run_parameter_sweep('vote_threshold', vote_values, 'model', 'vote_threshold')

# Create summary
print("\n" + "="*80)
print("OPTIMIZATION COMPLETE - SUMMARY")
print("="*80)

summary = []
for param, df in all_results.items():
    if len(df[df['sharpe_ratio'] > -999]) > 0:
        valid_df = df[df['sharpe_ratio'] > -999]
        best_idx = valid_df['sharpe_ratio'].idxmax()
        best = valid_df.loc[best_idx]
        summary.append({
            'parameter': param,
            'best_value': best['value'],
            'best_sharpe': best['sharpe_ratio'],
            'best_pnl': best['total_pnl'],
            'best_edge': best['edge'],
            'best_trades': best['total_trades']
        })
        print(f"{param:20s}: Best={best['value']:8.3f}, Sharpe={best['sharpe_ratio']:6.3f}, Edge={best['edge']:6.1%}")

summary_df = pd.DataFrame(summary)
summary_df.to_csv('OPTIMIZATION_SUMMARY.csv', index=False)

print("\nAll results saved to individual CSV files and OPTIMIZATION_SUMMARY.csv")