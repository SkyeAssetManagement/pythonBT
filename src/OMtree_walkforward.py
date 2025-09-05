import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from src.OMtree_validation import DirectionalValidator
import configparser

print("OMtree WALK-FORWARD VALIDATION & ANALYSIS")
print("=" * 80)
print("This script runs full validation for directional trading strategies")
print("=" * 80)

# Step 1: Run the validation
print("STEP 1: Running OMtree walk-forward validation...")
validator = DirectionalValidator()
df = validator.run_validation(verbose=True)

if len(df) == 0:
    print("No valid predictions generated. Check data and configuration.")
    exit(1)

print(f"[OK] Validation completed! Generated {len(df):,} total observations")

# Save all results
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')
results_file = config['output']['results_file']
df.to_csv(results_file, index=False)
print(f"[OK] Results saved to: {results_file}")

# Get model type and configuration info
model_type = config['model']['model_type']
target_threshold = float(config['model']['target_threshold'])
signal_name = "LONG" if model_type == 'longonly' else "SHORT"
direction_name = "UP" if model_type == 'longonly' else "DOWN"

# Get validation start date for out-of-sample filtering
validation_start_date = config['validation'].get('validation_start_date', None)
if validation_start_date:
    df_filtered = validator.filter_by_date(df, validation_start_date)
    print(f"\nOut-of-sample period: {validation_start_date} onwards")
    print(f"  Total observations: {len(df):,} -> Out-of-sample: {len(df_filtered):,}")
    
    # Calculate metrics for out-of-sample period only
    metrics = validator.calculate_directional_metrics(df_filtered, filter_date=False)
    print(f"[OK] Out-of-sample Results: {metrics['hit_rate']:.1%} hit rate, {metrics['edge']:+.1%} edge")
    
    # Use filtered data for all subsequent analysis
    df = df_filtered
else:
    # No filtering, use all data
    metrics = validator.calculate_directional_metrics(df, filter_date=False)
    print(f"[OK] Full Results: {metrics['hit_rate']:.1%} hit rate, {metrics['edge']:+.1%} edge")

# Step 2: Prepare data for analysis
print(f"\nSTEP 2: Preparing data for comprehensive {model_type} analysis...")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# Filter to TRADE signals only
trades = df[df['prediction'] == 1].copy()
trades = trades.sort_values('date').reset_index(drop=True)

# Calculate P&L based on model type
if model_type == 'longonly':
    trades['trade_pnl'] = trades['target_value']  # Long trades: profit from positive returns
else:  # shortonly
    trades['trade_pnl'] = -trades['target_value']  # Short trades: profit from negative returns

# Calculate rolling metrics
trades['cumulative_pnl'] = trades['trade_pnl'].cumsum()
rolling_window_short = int(config['analysis']['rolling_window_short'])
rolling_window_long = int(config['analysis']['rolling_window_long'])

trades['rolling_hit_rate_short'] = trades['actual_profitable'].rolling(window=rolling_window_short, min_periods=10).mean()
trades['rolling_hit_rate_long'] = trades['actual_profitable'].rolling(window=rolling_window_long, min_periods=20).mean()

print(f"[OK] Data prepared: {len(trades):,} {signal_name} trades for analysis")

# Step 3: Generate comprehensive 6-panel chart
print(f"\nSTEP 3: Generating comprehensive 6-panel {model_type} analysis chart...")

fig = plt.figure(figsize=(20, 15))

# Panel 1: Cumulative P&L
ax1 = plt.subplot(3, 2, 1)
plt.plot(trades['date'], trades['cumulative_pnl'], 'b-', linewidth=3, label='Cumulative P&L')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title(f'Walk-Forward: Cumulative P&L Evolution ({model_type.title()})', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Return')
plt.grid(True, alpha=0.3)
plt.legend()

# Panel 2: Rolling hit rate over time
ax2 = plt.subplot(3, 2, 2)
plt.plot(trades['date'], trades['rolling_hit_rate_long'], 'g-', linewidth=2, 
         label=f'{rolling_window_long}-trade rolling')
plt.plot(trades['date'], trades['rolling_hit_rate_short'], 'orange', linewidth=1.5, alpha=0.7, 
         label=f'{rolling_window_short}-trade rolling')
base_rate = float(config['validation']['base_rate'])
plt.axhline(y=base_rate, color='red', linestyle='--', alpha=0.7, label=f'Base Rate ({base_rate:.0%})')
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50% Target')
plt.axhline(y=trades['actual_profitable'].mean(), color='blue', linestyle='--', alpha=0.7, label='Overall Hit Rate')
plt.title(f'Walk-Forward: Rolling Hit Rate Performance ({signal_name})', fontsize=14, fontweight='bold')
plt.ylabel('Hit Rate')
plt.ylim(0.25, 0.8)
plt.grid(True, alpha=0.3)
plt.legend()

# Panel 3: Trade frequency over time (monthly)
ax3 = plt.subplot(3, 2, 3)
df_monthly = df.copy()
df_monthly['year_month'] = df_monthly['date'].dt.to_period('M')
monthly_stats = df_monthly.groupby('year_month').agg({
    'prediction': 'sum',
    'date': 'count'
}).reset_index()
monthly_stats['trade_rate'] = monthly_stats['prediction'] / monthly_stats['date']
monthly_stats['year_month'] = monthly_stats['year_month'].dt.to_timestamp()

plt.plot(monthly_stats['year_month'], monthly_stats['trade_rate'] * 100, 'purple', linewidth=2)
overall_trade_rate = len(trades)/len(df) * 100
plt.axhline(y=overall_trade_rate, color='red', linestyle='--', alpha=0.7, 
           label=f'Overall Rate ({overall_trade_rate:.1f}%)')
plt.title(f'Walk-Forward: Monthly Trading Frequency ({signal_name})', fontsize=14, fontweight='bold')
plt.ylabel(f'% of Days with {signal_name} Signal')
plt.grid(True, alpha=0.3)
plt.legend()

# Panel 4: Individual trade returns scatter
ax4 = plt.subplot(3, 2, 4)
plt.scatter(trades['date'], trades['trade_pnl'], 
           c=trades['actual_profitable'], cmap='RdYlGn', alpha=0.7, s=30)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title(f'Walk-Forward: Individual Trade P&L ({signal_name})', fontsize=14, fontweight='bold')
plt.ylabel('P&L per Trade')
plt.colorbar(label='Profitable (0=No, 1=Yes)')
plt.grid(True, alpha=0.3)

# Panel 5: Rolling edge over time
ax5 = plt.subplot(3, 2, 5)
window_size = rolling_window_long
rolling_edge = []
rolling_dates = []

for i in range(window_size, len(trades)):
    window_data = trades.iloc[i-window_size:i]
    hit_rate = window_data['actual_profitable'].mean()
    edge = hit_rate - base_rate
    rolling_edge.append(edge)
    rolling_dates.append(window_data['date'].iloc[-1])

plt.plot(rolling_dates, np.array(rolling_edge) * 100, 'darkred', linewidth=2, 
         label=f'{window_size}-trade rolling edge')
overall_edge = (trades['actual_profitable'].mean() - base_rate) * 100
plt.axhline(y=overall_edge, color='blue', linestyle='--', alpha=0.7, 
           label=f'Overall Edge ({overall_edge:.1f}%)')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title(f'Walk-Forward: Rolling Edge vs Base Rate ({signal_name})', fontsize=14, fontweight='bold')
plt.ylabel('Edge (%)')
plt.grid(True, alpha=0.3)
plt.legend()

# Panel 6: Model confidence distribution
ax6 = plt.subplot(3, 2, 6)
plt.plot(trades['date'], trades['probability'], 'darkgreen', alpha=0.7, linewidth=1, 
         label=f'{direction_name} Probability')
vote_threshold = float(config['model']['vote_threshold'])
plt.axhline(y=vote_threshold, color='black', linestyle='--', alpha=0.7, label=f'Vote Threshold ({vote_threshold:.0%})')
plt.axhline(y=trades['probability'].mean(), color='red', linestyle='--', alpha=0.7, 
           label=f'Avg Confidence ({trades["probability"].mean():.3f})')
plt.title(f'Walk-Forward: Model Confidence Evolution ({signal_name})', fontsize=14, fontweight='bold')
plt.ylabel(f'{direction_name} Probability')
plt.ylim(0.4, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
chart_dpi = int(config['output']['chart_dpi'])
os.makedirs('results', exist_ok=True)
plt.savefig(f'results/OMtree_comprehensive_{model_type}.png', dpi=chart_dpi, bbox_inches='tight')
plt.close()
print(f"[OK] Comprehensive chart saved: results/OMtree_comprehensive_{model_type}.png")

# Step 4: Generate yearly progression analysis
print(f"\nSTEP 4: Generating yearly progression analysis ({model_type})...")

# Check if there are any trades
if len(trades) == 0:
    print(f"[WARNING] No {signal_name} trades generated. Model may be too restrictive.")
    print(f"  Consider adjusting vote_threshold or target_threshold in config.")
    print("\n" + "=" * 80)
    print(f"OMtree WALK-FORWARD ANALYSIS ({model_type.upper()}) FINISHED WITH WARNINGS!")
    print("=" * 80)
    exit(0)

# Analyze performance by year
yearly_stats = []
for year in sorted(df['year'].unique()):
    year_data = df[df['year'] == year]
    trades_year = year_data[year_data['prediction'] == 1]
    
    if len(trades_year) > 0:
        if model_type == 'longonly':
            year_pnl = trades_year['target_value'].sum()
        else:  # shortonly
            year_pnl = -trades_year['target_value'].sum()
            
        yearly_stats.append({
            'year': year,
            'total_obs': len(year_data),
            'trade_signals': len(trades_year),
            'trade_rate': len(trades_year) / len(year_data),
            'hit_rate': trades_year['actual_profitable'].mean(),
            'avg_return': year_pnl / len(trades_year) if len(trades_year) > 0 else 0,
            'total_pnl': year_pnl,
            'edge': trades_year['actual_profitable'].mean() - year_data['actual_profitable'].mean()
        })

yearly_df = pd.DataFrame(yearly_stats)

# Create yearly progression charts
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Annual edge performance
ax1.bar(yearly_df['year'], yearly_df['edge'] * 100, alpha=0.7, color='darkblue')
ax1.axhline(y=overall_edge, color='red', linestyle='--', alpha=0.7, 
           label=f'Overall Edge ({overall_edge:.1f}%)')
ax1.set_title(f'Annual Edge Performance ({model_type.title()})', fontsize=14, fontweight='bold')
ax1.set_ylabel('Edge vs Base Rate (%)')
ax1.set_xlabel('Year')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Annual trading frequency
ax2.bar(yearly_df['year'], yearly_df['trade_rate'] * 100, alpha=0.7, color='darkgreen')
ax2.axhline(y=overall_trade_rate, color='red', linestyle='--', alpha=0.7, 
           label=f'Overall Rate ({overall_trade_rate:.1f}%)')
ax2.set_title(f'Annual Trading Frequency ({signal_name})', fontsize=14, fontweight='bold')
ax2.set_ylabel('Trading Frequency (%)')
ax2.set_xlabel('Year')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Cumulative P&L progression
trades_sorted = df[df['prediction'] == 1].sort_values('date')
if model_type == 'longonly':
    cumulative_pnl = trades_sorted['target_value'].cumsum()
else:  # shortonly
    cumulative_pnl = (-trades_sorted['target_value']).cumsum()

ax3.plot(trades_sorted['date'], cumulative_pnl, 'b-', linewidth=3, label='Cumulative P&L')
ax3.fill_between(trades_sorted['date'], 0, cumulative_pnl, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.set_title(f'Cumulative P&L Progression ({signal_name})', fontsize=14, fontweight='bold')
ax3.set_ylabel('Cumulative Return')
ax3.set_xlabel('Date')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Hit rate stability over time
window_size = 30
rolling_hit_rates = []
rolling_dates = []

for i in range(window_size, len(trades)):
    window_data = trades.iloc[i-window_size:i]
    hit_rate = window_data['actual_profitable'].mean()
    rolling_hit_rates.append(hit_rate)
    rolling_dates.append(window_data['date'].iloc[-1])

ax4.plot(rolling_dates, np.array(rolling_hit_rates) * 100, 'darkred', linewidth=2, 
         label='30-trade rolling hit rate')
ax4.axhline(y=trades['actual_profitable'].mean() * 100, color='blue', linestyle='--', alpha=0.7, 
           label=f'Overall Hit Rate ({trades["actual_profitable"].mean():.1%})')
ax4.axhline(y=base_rate * 100, color='red', linestyle='--', alpha=0.7, 
           label=f'Base Rate ({base_rate:.0%})')
ax4.fill_between(rolling_dates, base_rate * 100, np.array(rolling_hit_rates) * 100, 
                 alpha=0.3, color='green', label='Edge Region')
ax4.set_title(f'Hit Rate Stability Over Time ({signal_name})', fontsize=14, fontweight='bold')
ax4.set_ylabel('Hit Rate (%)')
ax4.set_xlabel('Date')
ax4.set_ylim(25, 75)
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig(f'results/OMtree_progression_{model_type}.png', dpi=chart_dpi, bbox_inches='tight')
plt.close()
print(f"[OK] Yearly progression chart saved: results/OMtree_progression_{model_type}.png")

# Step 5: Generate comprehensive performance report
print(f"\nSTEP 5: Generating comprehensive performance report ({model_type})...")

print("\n" + "=" * 80)
print(f"DIRECTIONAL WALK-FORWARD VALIDATION RESULTS ({model_type.upper()})")
print("=" * 80)

print(f"\nMODEL CONFIGURATION:")
print(f"  Model Type: {model_type}")
print(f"  Features: {config['data']['selected_features']}")
print(f"  Trees: {config['model']['n_trees']}")
print(f"  Target Threshold: {target_threshold}")
print(f"  Min Leaf Fraction: {float(config['model']['min_leaf_fraction']):.0%}")
print(f"  Step Size: {config['validation']['step_size']} days")
print(f"  Vote Threshold: {float(config['model']['vote_threshold']):.0%}")

print(f"\nVALIDATION PERIOD:")
if validation_start_date:
    print(f"  Out-of-Sample Start: {validation_start_date} (configured for fair comparison)")
else:
    print(f"  Start Date: {df['date'].min().strftime('%Y-%m-%d')}")
print(f"  End Date: {df['date'].max().strftime('%Y-%m-%d')}")
print(f"  Duration: {(df['date'].max() - df['date'].min()).days:,} days ({(df['date'].max() - df['date'].min()).days/365.25:.1f} years)")
print(f"  Total Observations: {len(df):,}")
print(f"  Walk-Forward Models: ~{len(df)//int(config['validation']['test_size']):,}")

print(f"\nTRADING PERFORMANCE:")
print(f"  Total {signal_name} Signals: {len(trades):,}")
print(f"  Trading Frequency: {len(trades)/len(df):.1%} of days")
print(f"  Hit Rate: {trades['actual_profitable'].mean():.1%}")
print(f"  Base Rate: {base_rate:.1%}")
print(f"  Edge vs Base Rate: {(trades['actual_profitable'].mean() - base_rate):.1%}")
print(f"  Average Return/Trade: {trades['trade_pnl'].mean():+.4f}")
print(f"  Total Cumulative P&L: {trades['trade_pnl'].sum():.2f}")

# Calculate monthly P&L statistics using the comprehensive metrics
print(f"\nMONTHLY P&L STATISTICS:")
print(f"  Average Monthly P&L: {metrics['avg_monthly_pnl']:+.2f}")
print(f"  StdDev Monthly P&L: {metrics['std_monthly_pnl']:.2f}")
print(f"  Annualized Sharpe (0%): {metrics['sharpe_ratio']:.3f}")

print(f"\nMONTHLY PERFORMANCE:")
print(f"  Positive Months: {metrics['positive_months_pct']:.1%}")
print(f"  Best Month: {metrics['best_month']:+.2f}")
print(f"  Worst Month: {metrics['worst_month']:+.2f}")

print(f"\nMODEL STABILITY:")
print(f"  Best {rolling_window_long}-trade Period: {trades['rolling_hit_rate_long'].max():.1%}")
print(f"  Worst {rolling_window_long}-trade Period: {trades['rolling_hit_rate_long'].min():.1%}")
print(f"  Average Confidence: {trades['probability'].mean():.3f}")
print(f"  Confidence Range: {trades['probability'].min():.3f} - {trades['probability'].max():.3f}")

print(f"\nYEARLY BREAKDOWN:")
print(f"{'Year':<6} {'Obs':<6} {'Trades':<7} {'Freq%':<6} {'Hit%':<6} {'Edge%':<7} {'P&L':<8}")
print("-" * 60)
for _, row in yearly_df.iterrows():
    print(f"{int(row['year']):<6} {int(row['total_obs']):<6} {int(row['trade_signals']):<7} "
          f"{row['trade_rate']*100:<6.1f} {row['hit_rate']*100:<6.1f} {row['edge']*100:<7.1f} "
          f"{row['total_pnl']:<8.2f}")

print(f"\nMODEL CONSISTENCY:")
positive_years = (yearly_df['edge'] > 0).sum()
strong_years = (yearly_df['edge'] > 0.05).sum()
excellent_years = (yearly_df['edge'] > 0.10).sum()

print(f"  Years with Positive Edge: {positive_years}/{len(yearly_df)} ({positive_years/len(yearly_df):.1%})")
print(f"  Years with Strong Edge (>5%): {strong_years}/{len(yearly_df)} ({strong_years/len(yearly_df):.1%})")
print(f"  Years with Excellent Edge (>10%): {excellent_years}/{len(yearly_df)} ({excellent_years/len(yearly_df):.1%})")

print(f"\nEDGE STATISTICS:")
print(f"  Mean Annual Edge: {yearly_df['edge'].mean():.1%}")
print(f"  Median Annual Edge: {yearly_df['edge'].median():.1%}")
print(f"  Edge Standard Deviation: {yearly_df['edge'].std():.1%}")
print(f"  Best Year: {yearly_df.loc[yearly_df['edge'].idxmax(), 'year']:.0f} ({yearly_df['edge'].max():.1%})")
print(f"  Worst Year: {yearly_df.loc[yearly_df['edge'].idxmin(), 'year']:.0f} ({yearly_df['edge'].min():.1%})")

print(f"\nFILES GENERATED:")
print(f"  [OK] {results_file} - Complete validation results")
print(f"  [OK] results/OMtree_comprehensive_{model_type}.png - 6-panel detailed analysis")
print(f"  [OK] results/OMtree_progression_{model_type}.png - 4-panel yearly progression")

# Log performance to CSV
log_file = 'results/OMtree_performance.csv'
import os
from datetime import datetime

# Prepare log entry
log_entry = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    # Data settings
    'features': config['data']['selected_features'],
    # Preprocessing settings
    'normalize_features': config['preprocessing']['normalize_features'],
    'normalize_target': config['preprocessing']['normalize_target'],
    'vol_window': config['preprocessing']['vol_window'],
    'smoothing_type': config['preprocessing']['smoothing_type'],
    'recent_iqr_lookback': config['preprocessing']['recent_iqr_lookback'],
    # Model settings
    'n_trees': config['model']['n_trees'],
    'max_depth': config['model']['max_depth'],
    'min_leaf_fraction': config['model']['min_leaf_fraction'],
    'target_threshold': config['model']['target_threshold'],
    'vote_threshold': config['model']['vote_threshold'],
    # Validation settings
    'train_size': config['validation']['train_size'],
    'test_size': config['validation']['test_size'],
    'step_size': config['validation']['step_size'],
    'validation_start_date': config['validation'].get('validation_start_date', 'None'),
    # Performance metrics
    'total_observations': len(df),
    'total_trades': len(trades),
    'trading_frequency': f"{len(trades)/len(df):.3f}",
    'hit_rate': f"{trades['actual_profitable'].mean():.3f}",
    'edge': f"{(trades['actual_profitable'].mean() - base_rate):.3f}",
    'total_pnl': f"{trades['trade_pnl'].sum():.2f}",
    'avg_monthly_pnl': f"{metrics['avg_monthly_pnl']:.2f}",
    'std_monthly_pnl': f"{metrics['std_monthly_pnl']:.2f}",
    'sharpe_ratio': f"{metrics['sharpe_ratio']:.3f}",
    'positive_months_pct': f"{metrics['positive_months_pct']:.3f}",
    'best_month': f"{metrics['best_month']:.2f}",
    'worst_month': f"{metrics['worst_month']:.2f}",
    'positive_years_pct': f"{positive_years/len(yearly_df):.3f}"
}

# Create or append to log file
log_df = pd.DataFrame([log_entry])
if os.path.exists(log_file):
    # Append to existing log
    existing_log = pd.read_csv(log_file)
    combined_log = pd.concat([existing_log, log_df], ignore_index=True)
    combined_log.to_csv(log_file, index=False)
    print(f"  [OK] {log_file} - Performance metrics logged (entry #{len(combined_log)})")
else:
    # Create new log file
    log_df.to_csv(log_file, index=False)
    print(f"  [OK] {log_file} - Performance log created")

print("\n" + "=" * 80)
print(f"DIRECTIONAL WALK-FORWARD ANALYSIS ({model_type.upper()}) FINISHED SUCCESSFULLY!")
print("=" * 80)