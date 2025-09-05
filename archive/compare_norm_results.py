import pandas as pd
import numpy as np

# Load results
df_no_norm = pd.read_csv('OMtree_results_NO_NORM.csv')
df_with_norm = pd.read_csv('OMtree_results_WITH_NORM.csv')

# Filter to post-2013 for comparison
df_no_norm['date'] = pd.to_datetime(df_no_norm['date'])
df_with_norm['date'] = pd.to_datetime(df_with_norm['date'])

post_2013_no_norm = df_no_norm[df_no_norm['date'] >= '2013-01-01'].reset_index(drop=True)
post_2013_with_norm = df_with_norm[df_with_norm['date'] >= '2013-01-01'].reset_index(drop=True)

print("="*80)
print("COMPARISON OF POST-2013 RESULTS (WITH 250-SAMPLE OFFSET)")
print("="*80)

# Check if dates align
dates_match = (post_2013_no_norm['date'] == post_2013_with_norm['date']).all()
print(f"\nDates match: {dates_match}")
print(f"Number of observations: {len(post_2013_no_norm)} vs {len(post_2013_with_norm)}")

if len(post_2013_no_norm) == len(post_2013_with_norm):
    # Compare predictions
    pred_match = (post_2013_no_norm['prediction'] == post_2013_with_norm['prediction']).mean() * 100
    print(f"\nPrediction match rate: {pred_match:.1f}%")
    
    # Compare confidence scores (probability column)
    conf_diff = np.abs(post_2013_no_norm['probability'] - post_2013_with_norm['probability'])
    print(f"Average confidence difference: {conf_diff.mean():.4f}")
    print(f"Max confidence difference: {conf_diff.max():.4f}")
    
    # Performance comparison
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON (Post-2013)")
    print("="*50)
    
    # No normalization performance
    no_norm_signals = post_2013_no_norm['prediction'] == 1
    no_norm_trades = post_2013_no_norm[no_norm_signals]
    no_norm_hit_rate = (no_norm_trades['actual_profitable'] == 1).mean() * 100 if len(no_norm_trades) > 0 else 0
    no_norm_pnl = no_norm_trades['target_value'].sum() if len(no_norm_trades) > 0 else 0
    
    # With normalization performance
    norm_signals = post_2013_with_norm['prediction'] == 1
    norm_trades = post_2013_with_norm[norm_signals]
    norm_hit_rate = (norm_trades['actual_profitable'] == 1).mean() * 100 if len(norm_trades) > 0 else 0
    norm_pnl = norm_trades['target_value'].sum() if len(norm_trades) > 0 else 0
    
    print(f"\nWithout Normalization:")
    print(f"  Trades: {len(no_norm_trades)}")
    print(f"  Hit Rate: {no_norm_hit_rate:.1f}%")
    print(f"  Total P&L: {no_norm_pnl:.2f}")
    
    print(f"\nWith Target Normalization:")
    print(f"  Trades: {len(norm_trades)}")
    print(f"  Hit Rate: {norm_hit_rate:.1f}%")
    print(f"  Total P&L: {norm_pnl:.2f}")
    
    print(f"\nDifference:")
    print(f"  Trade Count Difference: {len(norm_trades) - len(no_norm_trades)}")
    print(f"  Hit Rate Difference: {norm_hit_rate - no_norm_hit_rate:.1f}%")
    print(f"  P&L Difference: {norm_pnl - no_norm_pnl:.2f}")
    
    # Check if predictions differ where they shouldn't with threshold=0
    diff_predictions = post_2013_no_norm['prediction'] != post_2013_with_norm['prediction']
    if diff_predictions.any():
        print(f"\nNumber of different predictions: {diff_predictions.sum()}")
        print(f"Percentage of different predictions: {diff_predictions.mean()*100:.1f}%")
        
        # Sample some differences
        diff_indices = np.where(diff_predictions)[0][:5]
        print("\nSample of different predictions (first 5):")
        for idx in diff_indices:
            print(f"  Date: {post_2013_no_norm.loc[idx, 'date']}")
            print(f"    No Norm: Pred={post_2013_no_norm.loc[idx, 'prediction']}, Conf={post_2013_no_norm.loc[idx, 'probability']:.3f}")
            print(f"    With Norm: Pred={post_2013_with_norm.loc[idx, 'prediction']}, Conf={post_2013_with_norm.loc[idx, 'probability']:.3f}")
else:
    print("\nERROR: Different number of observations in the two datasets!")
    print("This should not happen with the 250-sample offset fix.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nKEY FINDINGS:")
print("With the 250-sample offset fix, we now compare the SAME time periods.")
print("Differences in predictions are due to:")
print("1. Different feature selections during walk-forward")
print("2. Models trained on normalized vs non-normalized targets")
print("3. Decision boundaries at threshold=0 may differ slightly due to tree splits")