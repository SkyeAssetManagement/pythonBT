#!/usr/bin/env python3
"""
Verify Range Bar Results
========================
Check synchronization and provide statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def verify_atr_results(atr_period):
    """Verify results for a specific ATR period"""
    
    print(f"\n{'='*80}")
    print(f"ATR-{atr_period} VERIFICATION")
    print(f"{'='*80}")
    
    # File paths
    base_dir = Path(f"parquetData/range-ATR-amibroker/ATR{atr_period}d")
    diff_file = base_dir / f"ES-DIFF-range-ATR{atr_period}d-amibroker.parquet"
    none_file = base_dir / f"ES-NONE-range-ATR{atr_period}d-amibroker.parquet"
    
    # Check if files exist
    if not diff_file.exists() or not none_file.exists():
        print(f"[ERROR] Files not found for ATR-{atr_period}")
        return None
    
    # Load data
    print(f"Loading data...")
    diff_bars = pd.read_parquet(diff_file)
    none_bars = pd.read_parquet(none_file)
    
    # Basic stats
    print(f"\nBASIC STATISTICS:")
    print(f"  DIFF bars: {len(diff_bars):,}")
    print(f"  NONE bars: {len(none_bars):,}")
    
    # Check synchronization
    print(f"\nSYNCHRONIZATION CHECK:")
    if len(diff_bars) != len(none_bars):
        print(f"  [ERROR] Bar count mismatch! DIFF={len(diff_bars):,}, NONE={len(none_bars):,}")
        return None
    else:
        print(f"  [OK] Bar counts match: {len(diff_bars):,} bars")
    
    # Check timestamps
    timestamp_match = (diff_bars['timestamp'].values == none_bars['timestamp'].values).all()
    if timestamp_match:
        print(f"  [OK] All timestamps match perfectly")
    else:
        mismatches = (~(diff_bars['timestamp'].values == none_bars['timestamp'].values)).sum()
        print(f"  [ERROR] Timestamp mismatches: {mismatches:,}")
        
        # Show first few mismatches
        for i in range(min(5, mismatches)):
            if diff_bars['timestamp'].iloc[i] != none_bars['timestamp'].iloc[i]:
                print(f"      Bar {i}: DIFF={diff_bars['timestamp'].iloc[i]}, NONE={none_bars['timestamp'].iloc[i]}")
    
    # Time span analysis
    print(f"\nTIME SPAN:")
    start_time = diff_bars['timestamp'].min()
    end_time = diff_bars['timestamp'].max()
    total_days = (end_time - start_time).days
    
    print(f"  Start: {start_time}")
    print(f"  End: {end_time}")
    print(f"  Total days: {total_days:,}")
    
    # Bar duration statistics
    print(f"\nBAR DURATION STATISTICS:")
    
    # Calculate time between bars
    time_diffs = diff_bars['timestamp'].diff().dt.total_seconds() / 60  # Convert to minutes
    time_diffs = time_diffs[time_diffs.notna()]  # Remove NaN from first row
    
    print(f"  Average mins/bar: {time_diffs.mean():.2f}")
    print(f"  Median mins/bar: {time_diffs.median():.2f}")
    print(f"  Min duration: {time_diffs.min():.2f} mins")
    print(f"  Max duration: {time_diffs.max():.2f} mins")
    print(f"  Std deviation: {time_diffs.std():.2f} mins")
    
    # Percentiles
    print(f"\n  Duration percentiles (minutes):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    {p}th percentile: {time_diffs.quantile(p/100):.2f}")
    
    # Range statistics
    print(f"\nRANGE STATISTICS:")
    print(f"  DIFF:")
    print(f"    Average range: {diff_bars['range'].mean():.3f}")
    print(f"    Min range: {diff_bars['range'].min():.3f}")
    print(f"    Max range: {diff_bars['range'].max():.3f}")
    
    print(f"  NONE:")
    print(f"    Average range: {none_bars['range'].mean():.3f}")
    print(f"    Min range: {none_bars['range'].min():.3f}")
    print(f"    Max range: {none_bars['range'].max():.3f}")
    
    # Gap statistics
    print(f"\nGAP STATISTICS:")
    diff_gaps = diff_bars['gap']
    none_gaps = none_bars['gap']
    
    print(f"  DIFF gaps > 0.01: {(diff_gaps > 0.01).sum():,} ({(diff_gaps > 0.01).sum() / len(diff_gaps) * 100:.1f}%)")
    print(f"  NONE gaps > 0.01: {(none_gaps > 0.01).sum():,} ({(none_gaps > 0.01).sum() / len(none_gaps) * 100:.1f}%)")
    
    # Large gaps (potential contract rolls)
    large_gap_threshold = 50
    diff_large_gaps = diff_gaps > large_gap_threshold
    none_large_gaps = none_gaps > large_gap_threshold
    
    print(f"\n  Large gaps (>{large_gap_threshold} points):")
    print(f"    DIFF: {diff_large_gaps.sum():,}")
    print(f"    NONE: {none_large_gaps.sum():,}")
    
    if none_large_gaps.sum() > 0:
        print(f"\n  NONE large gap examples:")
        large_gap_indices = none_gaps[none_large_gaps].index[:5]
        for idx in large_gap_indices:
            print(f"    {none_bars.loc[idx, 'timestamp']}: gap={none_bars.loc[idx, 'gap']:.1f}, range={none_bars.loc[idx, 'range']:.1f}")
    
    # Ticks per bar
    print(f"\nTICKS PER BAR:")
    print(f"  Average: {diff_bars['ticks'].mean():.0f}")
    print(f"  Median: {diff_bars['ticks'].median():.0f}")
    print(f"  Min: {diff_bars['ticks'].min()}")
    print(f"  Max: {diff_bars['ticks'].max()}")
    
    # File sizes
    print(f"\nFILE SIZES:")
    diff_size_mb = diff_file.stat().st_size / (1024 * 1024)
    none_size_mb = none_file.stat().st_size / (1024 * 1024)
    print(f"  DIFF: {diff_size_mb:.2f} MB")
    print(f"  NONE: {none_size_mb:.2f} MB")
    
    return {
        'atr_period': atr_period,
        'bar_count': len(diff_bars),
        'synced': timestamp_match,
        'avg_mins_per_bar': time_diffs.mean(),
        'median_mins_per_bar': time_diffs.median(),
        'total_days': total_days,
        'diff_gaps': (diff_gaps > 0.01).sum(),
        'none_gaps': (none_gaps > 0.01).sum()
    }

def main():
    """Main verification"""
    
    print("="*100)
    print("RANGE BAR VERIFICATION AND STATISTICS")
    print("="*100)
    
    # Check each ATR period
    results = []
    for atr_period in [14, 30, 90]:
        result = verify_atr_results(atr_period)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n{'='*100}")
        print("PROCESSING SUMMARY")
        print("="*100)
        
        print(f"\n{'ATR':<10} {'Bars':<15} {'Synced':<10} {'Avg mins/bar':<15} {'Median mins/bar':<15}")
        print("-"*70)
        
        for r in results:
            synced = "[OK]" if r['synced'] else "[ERROR]"
            print(f"{r['atr_period']:<10} {r['bar_count']:<15,} {synced:<10} {r['avg_mins_per_bar']:<15.2f} {r['median_mins_per_bar']:<15.2f}")
        
        print(f"\nProcessing time estimate based on ATR-14:")
        if results[0]['atr_period'] == 14:
            # We know ATR-14 took 16.4 minutes
            processing_time_mins = 16.4
            bars_created = results[0]['bar_count']
            
            print(f"  Total processing time: {processing_time_mins:.1f} minutes")
            print(f"  Bars created: {bars_created:,}")
            print(f"  Processing rate: {bars_created / processing_time_mins:.0f} bars/minute")
            print(f"  Processing rate: {600_000_000 / (processing_time_mins * 60):.0f} ticks/second")
            
            # Time breakdown
            print(f"\n  Time breakdown:")
            print(f"    Data loading & chunking: ~10 minutes")
            print(f"    Numba boundary detection: ~2 minutes")
            print(f"    Bar creation: ~4 minutes")
            print(f"    Saving: ~0.4 minutes")

if __name__ == "__main__":
    main()