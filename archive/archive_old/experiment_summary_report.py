import pandas as pd
import numpy as np

print("="*80)
print("PARAMETER OPTIMIZATION EXPERIMENT SUMMARY")
print("="*80)

# Load all experiment results
experiments = {}

# Vol window results
try:
    experiments['vol_window'] = pd.read_csv('experiment_vol_window_results.csv')
    print(f"\n[OK] Vol Window: {len(experiments['vol_window'])} tests completed")
except:
    print("\n[MISS] Vol Window: Results not found")

# Up threshold results  
try:
    experiments['up_threshold'] = pd.read_csv('experiment_up_threshold_results.csv')
    print(f"[OK] Up Threshold: {len(experiments['up_threshold'])} tests completed")
except:
    print("[MISS] Up Threshold: Results not found")

# Vote threshold results
try:
    experiments['vote_threshold'] = pd.read_csv('experiment_vote_threshold_results.csv')
    print(f"[OK] Vote Threshold: {len(experiments['vote_threshold'])} tests completed")
except:
    print("[MISS] Vote Threshold: Results not found")

# Bootstrap fraction results
try:
    experiments['bootstrap_fraction'] = pd.read_csv('experiment_bootstrap_fraction_results.csv')
    print(f"[OK] Bootstrap Fraction: {len(experiments['bootstrap_fraction'])} tests completed")
except:
    print("[MISS] Bootstrap Fraction: Results not found")

# Min leaf fraction results
try:
    experiments['min_leaf_fraction'] = pd.read_csv('experiment_min_leaf_fraction_results.csv')
    print(f"[OK] Min Leaf Fraction: {len(experiments['min_leaf_fraction'])} tests completed")
except:
    print("[MISS] Min Leaf Fraction: Results not found")

# Max depth results
try:
    experiments['max_depth'] = pd.read_csv('experiment_max_depth_results.csv')
    print(f"[OK] Max Depth: {len(experiments['max_depth'])} tests completed")
except:
    print("[MISS] Max Depth: Results not found")

# Feature combinations results
try:
    experiments['feature_combinations'] = pd.read_csv('experiment_feature_combinations_results.csv')
    print(f"[OK] Feature Combinations: {len(experiments['feature_combinations'])} tests completed")
except:
    print("[MISS] Feature Combinations: Results not found")

print("\n" + "="*80)
print("BEST CONFIGURATIONS BY PARAMETER")
print("="*80)

best_configs = {}

for param_name, df in experiments.items():
    if len(df) > 0:
        best_idx = df['sharpe_ratio'].idxmax()
        best_row = df.iloc[best_idx]
        best_configs[param_name] = best_row
        
        print(f"\n{param_name.upper().replace('_', ' ')}:")
        print(f"  Best Value: {best_row.iloc[0]}")
        print(f"  Sharpe Ratio: {best_row['sharpe_ratio']:.3f}")
        print(f"  Total P&L: {best_row['total_pnl']:.2f}")
        print(f"  Hit Rate: {best_row['hit_rate']:.3f}")
        if 'total_trades' in df.columns:
            print(f"  Total Trades: {best_row['total_trades']:,}")
        elif 'trading_frequency' in df.columns:
            print(f"  Trading Frequency: {best_row['trading_frequency']:.3f}")

print("\n" + "="*80)
print("BASELINE VS BEST CONFIGURATIONS")
print("="*80)

# Baseline configuration (typical values)
baseline_sharpe = 0.675  # From current config walkforward

print(f"\nBaseline Sharpe Ratio: {baseline_sharpe:.3f}")
print("\nImprovement Potential by Parameter:")
for param_name, best_row in best_configs.items():
    improvement = best_row['sharpe_ratio'] - baseline_sharpe
    pct_improvement = (improvement / baseline_sharpe) * 100
    print(f"  {param_name.replace('_', ' ').title()}: {improvement:+.3f} ({pct_improvement:+.1f}%)")

print("\n" + "="*80)
print("TOP PERFORMING INDIVIDUAL OPTIMIZATIONS")
print("="*80)

# Collect all best results
all_best = []
for param_name, best_row in best_configs.items():
    all_best.append({
        'parameter': param_name.replace('_', ' ').title(),
        'value': str(best_row.iloc[0]),
        'sharpe_ratio': best_row['sharpe_ratio'],
        'total_pnl': best_row['total_pnl'],
        'improvement_vs_baseline': best_row['sharpe_ratio'] - baseline_sharpe
    })

# Sort by Sharpe ratio
best_df = pd.DataFrame(all_best)
best_df = best_df.sort_values('sharpe_ratio', ascending=False)

print("\nRanked by Sharpe Ratio:")
print(best_df.to_string(index=False, float_format='%.3f'))

print("\n" + "="*80)
print("RECOMMENDED OPTIMIZATION STRATEGY")
print("="*80)

print("\nBased on the experiments, here are the recommendations:")

# Find top 3 improvements
top_3 = best_df.head(3)
print("\n1. HIGHEST IMPACT PARAMETERS (Top 3):")
for i, row in top_3.iterrows():
    print(f"   - {row['parameter']}: {row['value']} (Sharpe: {row['sharpe_ratio']:.3f})")

print("\n2. PARAMETER SETTINGS FOR OPTIMAL CONFIGURATION:")
print("   Based on individual parameter optimization:")
for param_name, best_row in best_configs.items():
    param_display = param_name.replace('_', ' ')
    print(f"   - {param_display}: {best_row.iloc[0]}")

print("\n3. NEXT STEPS:")
print("   - Test combinations of the top performing parameters")
print("   - Run full walkforward with optimized configuration")
print("   - Consider ensemble approaches using multiple configurations")

print("\n" + "="*80)
print("EXPERIMENT SUMMARY COMPLETE!")
print("="*80)