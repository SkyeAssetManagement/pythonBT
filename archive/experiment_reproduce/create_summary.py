import pandas as pd
import os

print("="*80)
print("OPTIMIZATION EXPERIMENTS SUMMARY")
print("Objective: Maximize Annualized Sharpe Ratio")
print("="*80)

# Collect all results
experiments = {
    'vol_window': 'vol_window_results.csv',
    'bootstrap_fraction': 'bootstrap_fraction_results.csv', 
    'min_leaf_fraction': 'min_leaf_fraction_results.csv',
    'target_threshold': 'target_threshold_results.csv',
    'vote_threshold': 'vote_threshold_results.csv'
}

summary = []
all_results = []

for param_name, filename in experiments.items():
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        
        # Find the parameter column name
        param_col = [col for col in df.columns if param_name in col][0]
        
        # Add experiment name to each row
        df['experiment'] = param_name
        all_results.append(df)
        
        # Find best configuration
        if len(df) > 0 and df['sharpe_ratio'].max() > -999:
            best_idx = df['sharpe_ratio'].idxmax()
            best_row = df.loc[best_idx]
            
            summary.append({
                'parameter': param_name,
                'best_value': best_row[param_col],
                'best_sharpe': best_row['sharpe_ratio'],
                'best_pnl': best_row['total_pnl'],
                'best_hit_rate': best_row['hit_rate'],
                'best_edge': best_row['edge'],
                'best_trades': best_row['total_trades']
            })
            
            print(f"\n{param_name.upper()}:")
            print(f"  Best Value: {best_row[param_col]:.3f}")
            print(f"  Sharpe Ratio: {best_row['sharpe_ratio']:.3f}")
            print(f"  Total PnL: {best_row['total_pnl']:.1f}")
            print(f"  Hit Rate: {best_row['hit_rate']:.1%}")
            print(f"  Edge: {best_row['edge']:.1%}")
            print(f"  Trades: {best_row['total_trades']:.0f}")

# Save summary
if summary:
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('FINAL_OPTIMIZATION_SUMMARY.csv', index=False)
    
    print("\n" + "="*80)
    print("BEST PARAMETERS SUMMARY")
    print("="*80)
    print("\nParameter           | Best Value | Sharpe | PnL    | Edge  | Trades")
    print("-"*70)
    for _, row in summary_df.iterrows():
        print(f"{row['parameter']:19s} | {row['best_value']:10.3f} | {row['best_sharpe']:6.3f} | "
              f"{row['best_pnl']:6.1f} | {row['best_edge']:5.1%} | {row['best_trades']:6.0f}")
    
    # Find overall best
    print("\n" + "="*80)
    print("OVERALL BEST CONFIGURATION")
    print("="*80)
    
    best_idx = summary_df['best_sharpe'].idxmax()
    best = summary_df.loc[best_idx]
    
    print(f"\nBest Parameter: {best['parameter']}")
    print(f"Best Value: {best['best_value']:.3f}")
    print(f"Sharpe Ratio: {best['best_sharpe']:.3f}")
    print(f"Total PnL: {best['best_pnl']:.1f}")
    print(f"Edge: {best['best_edge']:.1%}")
    print(f"Trades: {best['best_trades']:.0f}")
    
    # Combine all individual results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv('ALL_EXPERIMENT_RESULTS.csv', index=False)
        print(f"\nAll {len(combined_df)} experiment runs saved to ALL_EXPERIMENT_RESULTS.csv")
    
    print("\nSummary saved to FINAL_OPTIMIZATION_SUMMARY.csv")
else:
    print("\nNo experiment results found yet.")

print("\n" + "="*80)