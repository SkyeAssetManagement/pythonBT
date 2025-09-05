#!/usr/bin/env python
"""
Test the debug CSV export functionality
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check if we have the walkforward results
if os.path.exists('results/walkforward_results_longonly.csv'):
    print("Found existing walkforward results")
    
    # Load the results
    df = pd.read_csv('results/walkforward_results_longonly.csv')
    print(f"Total rows: {len(df)}")
    print(f"Trades (prediction=1): {(df['prediction']==1).sum()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Filter to trades
    trades = df[df['prediction'] == 1].copy()
    
    print(f"\nTarget value statistics:")
    print(f"Sum: {trades['target_value'].sum():.2f}")
    print(f"Mean: {trades['target_value'].mean():.4f}")
    print(f"Min: {trades['target_value'].min():.4f}")
    print(f"Max: {trades['target_value'].max():.4f}")
    
    # Check if debug CSV exists
    if os.path.exists('results/returns_debug_longonly.csv'):
        print("\n" + "="*60)
        print("DEBUG CSV ALREADY EXISTS - Contents:")
        print("="*60)
        debug_df = pd.read_csv('results/returns_debug_longonly.csv')
        print(debug_df.head(10))
        print("\nLast 5 rows:")
        print(debug_df.tail(5))
        print(f"\nFinal cumsum_pnl: {debug_df['cumsum_pnl'].iloc[-1]:.2f}%")
        print(f"Final compound_return_pct: {debug_df['compound_return_pct'].iloc[-1]:.2f}%")
    else:
        print("\nDebug CSV not found - will be created on next walk-forward run")
        print("Run a walk-forward test to generate the debug CSV")
else:
    print("No walkforward results found.")
    print("Please run a walk-forward validation first.")

print("\n" + "="*60)
print("The debug CSV will contain these columns:")
print("- trade_num: Sequential trade number")
print("- date: Trade date")  
print("- target_value: Original return value from predictions")
print("- pnl: P&L (inverted for shorts)")
print("- cumsum_pnl: Simple cumulative sum (matches console)")
print("- pnl_decimal: P&L converted to decimal (divided by 100)")
print("- portfolio_value: Compounded portfolio value starting at 100")
print("- compound_return_pct: Compound return as percentage")
print("- drawdown: Drawdown from peak (in %)")
print("- actual_profitable: Win/loss flag")
print("- hit_rate_cumulative: Running hit rate")
print("="*60)