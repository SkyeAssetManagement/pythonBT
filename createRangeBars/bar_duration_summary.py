#!/usr/bin/env python3
"""
Bar Duration Summary
====================
Summary of actual bar durations (not processing time)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def get_bar_duration_stats(atr_period):
    """Get duration statistics for a specific ATR period"""
    
    # File path
    base_dir = Path(f"../parquetData/range-ATR-amibroker/ATR{atr_period}d")
    diff_file = base_dir / f"ES-DIFF-range-ATR{atr_period}d-amibroker.parquet"
    
    if not diff_file.exists():
        return None
    
    # Load data
    bars = pd.read_parquet(diff_file)
    
    # Calculate time between bars in minutes
    time_diffs = bars['timestamp'].diff().dt.total_seconds() / 60
    time_diffs = time_diffs[time_diffs.notna()]  # Remove NaN from first row
    
    # Remove extreme outliers (weekends) for better statistics
    time_diffs_filtered = time_diffs[time_diffs < 1440]  # Less than 24 hours
    
    return {
        'atr_period': atr_period,
        'bar_count': len(bars),
        'avg_mins': time_diffs.mean(),
        'median_mins': time_diffs.median(),
        'avg_mins_no_weekend': time_diffs_filtered.mean(),
        'median_mins_no_weekend': time_diffs_filtered.median(),
        'p10': time_diffs.quantile(0.10),
        'p25': time_diffs.quantile(0.25),
        'p75': time_diffs.quantile(0.75),
        'p90': time_diffs.quantile(0.90),
        'p95': time_diffs.quantile(0.95),
        'p99': time_diffs.quantile(0.99),
        'min': time_diffs.min(),
        'max': time_diffs.max()
    }

def main():
    print("\n" + "="*100)
    print("BAR DURATION SUMMARY (Actual Data Time, Not Processing Time)")
    print("="*100)
    
    # Check all ATR periods
    results = []
    for atr_period in [14, 30, 90]:
        stats = get_bar_duration_stats(atr_period)
        if stats:
            results.append(stats)
            print(f"\n[OK] ATR-{atr_period}: {stats['bar_count']:,} bars found")
        else:
            print(f"\n[X] ATR-{atr_period}: No data found")
    
    if not results:
        print("\nNo completed datasets found!")
        return
    
    # Summary table
    print("\n" + "="*100)
    print("MINUTES PER BAR SUMMARY")
    print("="*100)
    
    # Main statistics table
    print("\n" + "-"*80)
    print(f"{'ATR':<8} {'Bars':<12} {'Avg (mins)':<12} {'Median (mins)':<14} {'Status'}")
    print("-"*80)
    
    for atr in [14, 30, 90]:
        found = False
        for r in results:
            if r['atr_period'] == atr:
                print(f"{r['atr_period']:<8} {r['bar_count']:<12,} {r['avg_mins']:<12.2f} {r['median_mins']:<14.2f} Complete")
                found = True
                break
        if not found:
            print(f"{atr:<8} {'N/A':<12} {'N/A':<12} {'N/A':<14} Not processed")
    
    # Detailed statistics for completed ATRs
    print("\n" + "="*100)
    print("DETAILED DURATION STATISTICS (Minutes per Bar)")
    print("="*100)
    
    for r in results:
        print(f"\nATR-{r['atr_period']}:")
        print(f"  Average: {r['avg_mins']:.2f} minutes")
        print(f"  Average (excluding weekends): {r['avg_mins_no_weekend']:.2f} minutes")
        print(f"  Median: {r['median_mins']:.2f} minutes")
        print(f"  Median (excluding weekends): {r['median_mins_no_weekend']:.2f} minutes")
        print(f"\n  Percentiles:")
        print(f"    10%: {r['p10']:.2f} mins (90% of bars take longer than this)")
        print(f"    25%: {r['p25']:.2f} mins (75% of bars take longer than this)")
        print(f"    50%: {r['median_mins']:.2f} mins (median)")
        print(f"    75%: {r['p75']:.2f} mins (25% of bars take longer than this)")
        print(f"    90%: {r['p90']:.2f} mins (10% of bars take longer than this)")
        print(f"    95%: {r['p95']:.2f} mins (5% of bars take longer than this)")
        print(f"    99%: {r['p99']:.2f} mins (1% of bars take longer than this)")
        print(f"\n  Range:")
        print(f"    Min: {r['min']:.3f} minutes")
        print(f"    Max: {r['max']:.1f} minutes ({r['max']/60:.1f} hours)")
    
    # Completion status
    print("\n" + "="*100)
    print("PROCESSING STATUS")
    print("="*100)
    
    completed = len(results)
    total = 3
    
    print(f"\nCompleted: {completed}/{total} ATR periods")
    print(f"Remaining: ATR-30, ATR-90")
    print(f"\nNote: ATR-30 and ATR-90 failed due to memory constraints during final bar assembly.")
    print(f"      The chunked+Numba processing worked, but combining 600M ticks used 80GB RAM.")

if __name__ == "__main__":
    main()