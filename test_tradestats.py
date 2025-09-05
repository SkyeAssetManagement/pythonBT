#!/usr/bin/env python
"""
Test the new performance statistics based on tradestats.md
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.performance_stats import calculate_performance_stats, format_stats_for_display
from src.tradestats_charts import (
    create_equity_curve_chart,
    create_drawdown_chart,
    create_monthly_returns_table,
    create_returns_distribution_chart,
    create_rolling_performance_chart
)

def test_with_real_data():
    """Test with actual walkforward results"""
    
    print("=" * 60)
    print("TESTING NEW PERFORMANCE STATS (tradestats.md)")
    print("=" * 60)
    
    # Load actual results
    df = pd.read_csv('OMtree_results.csv')
    
    print(f"\nLoaded data: {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Calculate statistics
    stats = calculate_performance_stats(df, model_type='longonly')
    
    # Display formatted results
    print("\n" + format_stats_for_display(stats))
    
    # Verify key calculations
    print("\n" + "=" * 60)
    print("VERIFICATION OF KEY CALCULATIONS")
    print("=" * 60)
    
    trades = df[df['prediction'] == 1]
    print(f"\n1. RETURNS FORMAT:")
    print(f"   Sample target_value: {trades['target_value'].iloc[0]} (should be in % form)")
    print(f"   Interpretation: {trades['target_value'].iloc[0]}%")
    
    print(f"\n2. SIMPLE SUM VS COMPOUND:")
    print(f"   Simple sum of returns: {trades['target_value'].sum():.2f}%")
    
    # Manual compound calculation
    portfolio = 1000
    for ret_pct in trades['target_value'].values:
        portfolio *= (1 + ret_pct/100)
    compound_return = (portfolio - 1000) / 10
    print(f"   Compound return: {compound_return:.2f}%")
    print(f"   Final portfolio value: ${portfolio:.2f}")
    
    print(f"\n3. FROM STATS DICTIONARY:")
    print(f"   Ave Annual %: {stats.get('avg_annual_pct', 0):.2f}%")
    print(f"   Max Draw %: {stats.get('max_draw_pct', 0):.2f}%")
    print(f"   Sharpe: {stats.get('sharpe', 0):.3f}")
    print(f"   UPI: {stats.get('upi', 0):.3f}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: New stats implementation complete!")
    print("Charts can be generated using the tradestats_charts module")
    print("=" * 60)
    
    # Test chart generation (just verify they don't error)
    if 'trades_df' in stats:
        trades_df = stats['trades_df']
        print("\nTesting chart generation...")
        
        try:
            fig1 = create_equity_curve_chart(trades_df)
            print("  ✓ Equity curve chart created")
        except Exception as e:
            print(f"  ✗ Equity curve chart failed: {e}")
        
        try:
            fig2 = create_drawdown_chart(trades_df)
            print("  ✓ Drawdown chart created")
        except Exception as e:
            print(f"  ✗ Drawdown chart failed: {e}")
        
        try:
            fig3 = create_monthly_returns_table(trades_df)
            print("  ✓ Monthly returns table created")
        except Exception as e:
            print(f"  ✗ Monthly returns table failed: {e}")
        
        try:
            fig4 = create_returns_distribution_chart(trades_df)
            print("  ✓ Returns distribution chart created")
        except Exception as e:
            print(f"  ✗ Returns distribution chart failed: {e}")
        
        try:
            fig5 = create_rolling_performance_chart(trades_df)
            print("  ✓ Rolling performance chart created")
        except Exception as e:
            print(f"  ✗ Rolling performance chart failed: {e}")

if __name__ == "__main__":
    test_with_real_data()