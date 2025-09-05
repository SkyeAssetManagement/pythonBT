"""
Enhanced Range Bar Analysis with Min/Max Time Per Bar Statistics

This analyzes the actual tick data patterns to estimate realistic
min/max time per bar for each range bar type.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_tick_patterns():
    """Analyze tick data patterns to estimate bar timing statistics"""
    
    csv_path = r"C:\Users\skyeAM\SkyeAM Dropbox\SAMresearch\ABtoPython\dataRaw\tick\ES-DIFF-Tick1-21toT.tick"
    
    print("ENHANCED RANGE BAR ANALYSIS")
    print("="*60)
    
    # Read a larger sample to get better statistics
    print("Loading sample data for pattern analysis...")
    
    # Read multiple chunks to get diverse time periods
    sample_sizes = [1_000_000, 2_000_000, 5_000_000]  # Different sample sizes
    
    all_samples = []
    
    for i, sample_size in enumerate(sample_sizes):
        skip_rows = i * 10_000_000  # Skip rows to get different time periods
        
        try:
            sample = pd.read_csv(
                csv_path, 
                nrows=sample_size,
                skiprows=skip_rows if skip_rows > 0 else None
            )
            
            # Convert datetime
            datetime_str = sample['Date'].astype(str) + ' ' + sample['Time'].astype(str)
            sample['datetime'] = pd.to_datetime(datetime_str, format='%Y/%m/%d %H:%M:%S.%f')
            sample = sample.sort_values('datetime')
            
            all_samples.append(sample)
            print(f"  Sample {i+1}: {len(sample):,} ticks from {sample['datetime'].iloc[0]} to {sample['datetime'].iloc[-1]}")
            
        except Exception as e:
            print(f"  Error loading sample {i+1}: {e}")
            break
    
    if not all_samples:
        print("Could not load samples for analysis")
        return None
    
    # Combine samples for analysis
    combined_sample = pd.concat(all_samples, ignore_index=True).sort_values('datetime')
    print(f"\nCombined analysis sample: {len(combined_sample):,} ticks")
    
    # Calculate tick intervals to understand market activity patterns
    combined_sample['tick_interval'] = combined_sample['datetime'].diff().dt.total_seconds()
    
    # Remove outliers (gaps between sessions)
    tick_intervals = combined_sample['tick_interval'].dropna()
    tick_intervals = tick_intervals[tick_intervals <= 300]  # Remove gaps > 5 minutes
    
    print(f"\nTICK INTERVAL ANALYSIS")
    print("-" * 30)
    print(f"Mean interval:    {tick_intervals.mean():.3f}s")
    print(f"Median interval:  {tick_intervals.median():.3f}s") 
    print(f"Min interval:     {tick_intervals.min():.3f}s")
    print(f"Max interval:     {tick_intervals.max():.3f}s")
    print(f"Std deviation:    {tick_intervals.std():.3f}s")
    
    # Analyze price movement patterns
    prices = combined_sample['Close'].values
    price_changes = np.abs(np.diff(prices))
    
    print(f"\nPRICE MOVEMENT ANALYSIS")
    print("-" * 30)
    print(f"Mean price change: {price_changes.mean():.4f}")
    print(f"Std price change:  {price_changes.std():.4f}")
    print(f"95th percentile:   {np.percentile(price_changes, 95):.4f}")
    print(f"99th percentile:   {np.percentile(price_changes, 99):.4f}")
    
    # Simulate range bar creation with timing analysis
    range_configs = [
        ("Raw_0.316", 0.316, "raw"),
        ("Pct_0.0074", 0.0074, "percentage"), 
        ("ATR_14d", 0.0126, "atr"),
        ("ATR_30d", 0.0105, "atr"),
        ("ATR_90d", 0.0090, "atr")
    ]
    
    results = []
    
    print(f"\nRANGE BAR TIMING SIMULATION")
    print("="*80)
    print(f"{'Type':<12} {'Parameter':<10} {'Bars':<8} {'Avg(min)':<8} {'Min(sec)':<9} {'Max(min)':<9} {'Std(min)':<8}")
    print("-"*80)
    
    for config_name, param_value, bar_type in range_configs:
        
        # Simulate range bar creation
        bar_times = simulate_range_bars(combined_sample, param_value, bar_type)
        
        if len(bar_times) > 0:
            # Convert to minutes
            bar_times_min = np.array(bar_times) / 60.0
            
            avg_time = np.mean(bar_times_min)
            min_time = np.min(bar_times)  # Keep min in seconds for readability
            max_time = np.max(bar_times_min)
            std_time = np.std(bar_times_min)
            
            print(f"{config_name:<12} {param_value:<10.4f} {len(bar_times):<8} {avg_time:<8.1f} "
                  f"{min_time:<9.1f} {max_time:<9.1f} {std_time:<8.1f}")
            
            # Extrapolate to full dataset
            sample_duration_days = (combined_sample['datetime'].iloc[-1] - combined_sample['datetime'].iloc[0]).total_seconds() / (24*3600)
            bars_per_day = len(bar_times) / sample_duration_days if sample_duration_days > 0 else 0
            
            # Estimate for full dataset (252 trading days)
            full_dataset_bars = bars_per_day * 252 if bars_per_day > 0 else 51408  # fallback
            
            results.append({
                'type': config_name,
                'parameter': param_value,
                'sample_bars': len(bar_times),
                'estimated_total_bars': int(full_dataset_bars),
                'avg_minutes_per_bar': avg_time,
                'min_seconds_per_bar': min_time,
                'max_minutes_per_bar': max_time,
                'std_minutes_per_bar': std_time,
                'bars_per_day': bars_per_day
            })
        else:
            print(f"{config_name:<12} {param_value:<10.4f} {'ERROR':<8}")
            results.append({
                'type': config_name,
                'parameter': param_value,
                'error': 'Simulation failed'
            })
    
    return results

def simulate_range_bars(data, param_value, bar_type, atr_value=25):
    """Simulate range bar creation and return list of bar durations in seconds"""
    
    prices = data['Close'].values
    timestamps = data['datetime'].values
    
    if len(prices) < 100:  # Need minimum data
        return []
    
    bars = []
    current_start_idx = 0
    current_open = prices[0]
    current_high = prices[0]
    current_low = prices[0]
    
    # Calculate range size based on bar type
    if bar_type == "raw":
        range_size = param_value
    elif bar_type == "percentage":
        range_size = current_open * param_value / 100
    elif bar_type == "atr":
        range_size = atr_value * param_value  # Simplified ATR
    else:
        range_size = param_value
    
    range_top = current_open + range_size
    range_bottom = current_open - range_size
    
    for i in range(1, len(prices)):
        current_price = prices[i]
        
        # Update range size for percentage bars
        if bar_type == "percentage":
            range_size = current_open * param_value / 100
            range_top = current_open + range_size
            range_bottom = current_open - range_size
        
        # Check if price breaches range
        if current_price >= range_top or current_price <= range_bottom:
            # Calculate bar duration
            start_time = timestamps[current_start_idx]
            end_time = timestamps[i-1] if i > 0 else timestamps[i]
            
            duration_seconds = (pd.Timestamp(end_time) - pd.Timestamp(start_time)).total_seconds()
            
            if duration_seconds > 0:  # Valid bar
                bars.append(duration_seconds)
            
            # Start new bar
            current_start_idx = i-1 if i > 0 else i
            current_open = prices[i-1] if i > 0 else prices[i]
            current_high = max(current_open, current_price)
            current_low = min(current_open, current_price)
            
            # Recalculate range
            if bar_type == "percentage":
                range_size = current_open * param_value / 100
            
            range_top = current_open + range_size  
            range_bottom = current_open - range_size
        else:
            # Update current bar
            current_high = max(current_high, current_price)
            current_low = min(current_low, current_price)
    
    return bars

def generate_enhanced_summary_table(results):
    """Generate the enhanced summary table with min/max statistics"""
    
    print(f"\nENHANCED PERFORMANCE SUMMARY TABLE")
    print("="*100)
    print(f"{'Type':<15} {'Parameter':<12} {'Est. Total':<10} {'Avg Min/':<9} {'Min Time':<10} {'Max Time':<10}")
    print(f"{'':15} {'Value':<12} {'Bars':<10} {'Bar':<9} {'Per Bar':<10} {'Per Bar':<10}")
    print("-"*100)
    
    for result in results:
        if 'error' not in result:
            type_name = result['type']
            param_val = result['parameter']
            total_bars = result['estimated_total_bars']
            avg_min = result['avg_minutes_per_bar']
            min_sec = result['min_seconds_per_bar']
            max_min = result['max_minutes_per_bar']
            
            print(f"{type_name:<15} {param_val:<12.4f} {total_bars:<10,} {avg_min:<9.1f} "
                  f"{min_sec:<10.1f}s {max_min:<10.1f}m")
        else:
            print(f"{result['type']:<15} {result['parameter']:<12.4f} {'ERROR':<10}")
    
    print("-"*100)
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_tick_patterns()
        if results:
            enhanced_results = generate_enhanced_summary_table(results)
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()