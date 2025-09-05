#!/usr/bin/env python3
"""
Range Bar Time Analysis
======================
Generate charts showing average bar count by 30-minute time buckets (0:00-23:30)
for each range bar series (ATR-14, ATR-30, ATR-90 × DIFF/CURR)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_csv_data(csv_file_path):
    """Load CSV range bar data and prepare for analysis"""
    try:
        logger.info(f"Loading: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Convert DateTime to pandas datetime
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Extract time components
        df['Date'] = df['DateTime'].dt.date
        df['Time'] = df['DateTime'].dt.time
        df['Hour'] = df['DateTime'].dt.hour
        df['Minute'] = df['DateTime'].dt.minute
        
        # Create 30-minute time buckets (0:00, 0:30, 1:00, 1:30, etc.)
        df['TimeBucket'] = df['Hour'] + (df['Minute'] // 30) * 0.5
        
        # Create time bucket labels
        df['TimeBucketLabel'] = df['TimeBucket'].apply(lambda x: f"{int(x):02d}:{int((x % 1) * 60):02d}")
        
        logger.info(f"Loaded {len(df):,} bars spanning {df['Date'].nunique()} days")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load {csv_file_path}: {e}")
        return None

def analyze_bar_counts_by_time(df, series_name):
    """Analyze bar counts by 30-minute time buckets"""
    
    # Group by date and time bucket to count bars per day per bucket
    daily_counts = df.groupby(['Date', 'TimeBucket']).size().reset_index(name='BarCount')
    
    # Calculate average bars per time bucket across all days
    avg_by_time = daily_counts.groupby('TimeBucket')['BarCount'].agg(['mean', 'std', 'count']).reset_index()
    avg_by_time['TimeBucketLabel'] = avg_by_time['TimeBucket'].apply(lambda x: f"{int(x):02d}:{int((x % 1) * 60):02d}")
    
    logger.info(f"{series_name}: Analyzed {avg_by_time['count'].iloc[0]} days of data")
    
    return avg_by_time

def create_time_bucket_chart(avg_data, series_name, output_path):
    """Create bar chart showing average bar count by time of day"""
    
    plt.figure(figsize=(16, 8))
    
    # Create the bar chart
    bars = plt.bar(range(len(avg_data)), avg_data['mean'], 
                   alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.5)
    
    # Add error bars if we have std data
    if 'std' in avg_data.columns and not avg_data['std'].isna().all():
        plt.errorbar(range(len(avg_data)), avg_data['mean'], yerr=avg_data['std'], 
                    fmt='none', color='red', alpha=0.6, capsize=2)
    
    # Customize the chart
    plt.title(f'{series_name} - Average Range Bars per 30-Minute Period', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time of Day (30-minute buckets)', fontsize=12, fontweight='bold')
    plt.ylabel('Average Number of Range Bars', fontsize=12, fontweight='bold')
    
    # Set x-axis labels (show every 4th label to avoid crowding)
    tick_positions = range(0, len(avg_data), 4)  # Every 2 hours
    tick_labels = [avg_data.iloc[i]['TimeBucketLabel'] for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars for peak periods
    max_val = avg_data['mean'].max()
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > max_val * 0.8:  # Show labels for high values
            plt.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.01,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Add summary statistics text box
    total_avg = avg_data['mean'].sum()
    peak_time = avg_data.loc[avg_data['mean'].idxmax(), 'TimeBucketLabel']
    peak_value = avg_data['mean'].max()
    quiet_time = avg_data.loc[avg_data['mean'].idxmin(), 'TimeBucketLabel']
    quiet_value = avg_data['mean'].min()
    
    stats_text = f'''Statistics:
Daily Total Avg: {total_avg:.0f} bars
Peak: {peak_value:.1f} bars at {peak_time}
Quiet: {quiet_value:.1f} bars at {quiet_time}
Peak/Quiet Ratio: {peak_value/quiet_value:.1f}x'''
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved chart: {output_path}")
    
    # Show some key statistics
    logger.info(f"  Peak activity: {peak_value:.1f} bars at {peak_time}")
    logger.info(f"  Quiet period: {quiet_value:.1f} bars at {quiet_time}")
    logger.info(f"  Daily average: {total_avg:.0f} bars total")
    
    plt.close()
    
    return {
        'series': series_name,
        'peak_time': peak_time,
        'peak_bars': peak_value,
        'quiet_time': quiet_time,
        'quiet_bars': quiet_value,
        'daily_total': total_avg,
        'peak_quiet_ratio': peak_value/quiet_value
    }

def create_comparison_chart(all_data, output_path):
    """Create comparison chart showing all series together"""
    
    plt.figure(figsize=(20, 12))
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Range Bar Activity by Time of Day - All Series Comparison', 
                 fontsize=18, fontweight='bold')
    
    colors = ['steelblue', 'lightcoral', 'mediumseagreen', 'gold', 'mediumpurple', 'darkorange']
    
    for idx, (series_name, avg_data) in enumerate(all_data.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Create bar chart
        bars = ax.bar(range(len(avg_data)), avg_data['mean'], 
                     alpha=0.7, color=colors[idx], edgecolor='black', linewidth=0.3)
        
        # Customize subplot
        ax.set_title(series_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time of Day', fontsize=10)
        ax.set_ylabel('Avg Bars', fontsize=10)
        
        # Set x-axis labels (every 4 hours)
        tick_positions = range(0, len(avg_data), 8)  # Every 4 hours
        tick_labels = [avg_data.iloc[i]['TimeBucketLabel'] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add peak value annotation
        max_idx = avg_data['mean'].idxmax()
        max_val = avg_data['mean'].max()
        max_time = avg_data.iloc[max_idx]['TimeBucketLabel']
        ax.annotate(f'Peak: {max_val:.1f}\n{max_time}', 
                   xy=(max_idx, max_val), xytext=(max_idx, max_val * 1.1),
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison chart: {output_path}")
    plt.close()

def discover_csv_files():
    """Discover all range bar CSV files"""
    csv_files = []
    base_path = Path("dataRaw")
    
    if not base_path.exists():
        logger.error("dataRaw directory not found!")
        return []
    
    # Find all CSV files
    for csv_file in base_path.rglob("*.csv"):
        if "range-ATR" in str(csv_file):
            # Extract metadata from path
            path_parts = csv_file.parts
            
            atr_period = None
            dataset_type = None
            
            for part in path_parts:
                if "range-ATR" in part:
                    if "ATR14" in part:
                        atr_period = "ATR-14"
                    elif "ATR30" in part:
                        atr_period = "ATR-30"
                    elif "ATR90" in part:
                        atr_period = "ATR-90"
                
                if part in ["Current"]:
                    dataset_type = "CURR"
                elif part in ["diffAdjusted"]:
                    dataset_type = "DIFF"
            
            if atr_period and dataset_type:
                series_name = f"{atr_period} {dataset_type}"
                csv_files.append({
                    'file_path': csv_file,
                    'series_name': series_name,
                    'atr_period': atr_period,
                    'dataset_type': dataset_type
                })
    
    logger.info(f"Found {len(csv_files)} CSV files to analyze")
    return csv_files

def main():
    """Main analysis function"""
    logger.info("📊 RANGE BAR TIME ANALYSIS - 30-MINUTE BUCKETS")
    logger.info("="*70)
    
    # Create output directory
    output_dir = Path("charts")
    output_dir.mkdir(exist_ok=True)
    
    # Discover CSV files
    csv_files = discover_csv_files()
    
    if not csv_files:
        logger.error("No CSV files found for analysis!")
        return
    
    all_data = {}
    all_stats = []
    
    # Process each series
    for file_info in csv_files:
        logger.info(f"\n📈 Analyzing {file_info['series_name']}")
        logger.info("-" * 50)
        
        # Load and analyze data
        df = load_csv_data(file_info['file_path'])
        if df is None:
            continue
        
        # Analyze by time buckets
        avg_data = analyze_bar_counts_by_time(df, file_info['series_name'])
        all_data[file_info['series_name']] = avg_data
        
        # Create individual chart
        chart_name = f"{file_info['atr_period']}_{file_info['dataset_type']}_time_analysis.png"
        chart_path = output_dir / chart_name
        
        stats = create_time_bucket_chart(avg_data, file_info['series_name'], chart_path)
        all_stats.append(stats)
    
    # Create comparison chart
    if len(all_data) > 1:
        logger.info(f"\n📊 Creating comparison chart")
        comparison_path = output_dir / "all_series_time_comparison.png"
        create_comparison_chart(all_data, comparison_path)
    
    # Summary statistics table
    logger.info(f"\n📋 SUMMARY STATISTICS")
    logger.info("="*70)
    logger.info(f"{'Series':<12} {'Peak Time':<10} {'Peak Bars':<10} {'Quiet Time':<12} {'Daily Total':<12}")
    logger.info("-" * 70)
    
    for stat in sorted(all_stats, key=lambda x: x['series']):
        logger.info(f"{stat['series']:<12} {stat['peak_time']:<10} {stat['peak_bars']:<10.1f} {stat['quiet_time']:<12} {stat['daily_total']:<12.0f}")
    
    logger.info("="*70)
    logger.info(f"📁 All charts saved in: {output_dir.absolute()}")
    logger.info("🎯 Analysis complete!")

if __name__ == "__main__":
    main()