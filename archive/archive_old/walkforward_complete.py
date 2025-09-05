import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from validation_directional import DirectionalValidator
import configparser

print("COMPLETE WALK-FORWARD VALIDATION & ANALYSIS")
print("=" * 80)
print("This script runs full validation and generates all charts and analysis")
print("=" * 80)

# Step 1: Run the validation
print("STEP 1: Running walk-forward validation...")
validator = DirectionalValidator()
df = validator.run_validation(verbose=True)

if len(df) == 0:
    print("No valid predictions generated. Check data and configuration.")
    exit(1)

print(f"[OK] Validation completed! Generated {len(df):,} total observations")

# Save all results
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config_longonly.ini')
results_file = config['output']['results_file']
df.to_csv(results_file, index=False)
print(f"[OK] Results saved to: {results_file}")

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
print("\nSTEP 2: Preparing data for comprehensive analysis...")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# Get model type for proper P&L calculation
model_type = config['model']['model_type']
signal_name = "LONG" if model_type == 'longonly' else "SHORT"

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
print("\nSTEP 3: Generating comprehensive 6-panel analysis chart...")

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
direction_name = "UP" if model_type == 'longonly' else "DOWN"
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
plt.savefig('walkforward_comprehensive.png', dpi=chart_dpi, bbox_inches='tight')
plt.close()
print(f"[OK] Comprehensive chart saved: walkforward_comprehensive.png")

# Step 4: Generate yearly progression analysis
print("\nSTEP 4: Generating yearly progression analysis...")

# Analyze performance by year
yearly_stats = []
for year in sorted(df['year'].unique()):
    year_data = df[df['year'] == year]
    trades_year = year_data[year_data['prediction'] == 1]
    
    if len(trades_year) > 0:
        # Calculate P&L based on model type
        if model_type == 'longonly':
            year_pnl = trades_year['target_value'].sum()
            avg_return = trades_year['target_value'].mean()
        else:  # shortonly
            year_pnl = -trades_year['target_value'].sum()
            avg_return = -trades_year['target_value'].mean()
            
        yearly_stats.append({
            'year': year,
            'total_obs': len(year_data),
            'trade_signals': len(trades_year),
            'trade_rate': len(trades_year) / len(year_data),
            'hit_rate': trades_year['actual_profitable'].mean(),
            'avg_return': avg_return,
            'total_pnl': year_pnl,
            'edge': trades_year['actual_profitable'].mean() - year_data['actual_profitable'].mean()
        })

yearly_df = pd.DataFrame(yearly_stats)

# Check if we have any yearly data
if len(yearly_df) == 0:
    print("[WARNING] No trades found in any year - skipping yearly analysis charts")
    print("This may be due to very restrictive trading parameters.")
    
    # Create empty chart as placeholder
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    ax1.text(0.5, 0.5, 'No Trades Found\nAdjust Parameters', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Annual Edge Performance (No Data)', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.5, 'No Trades Found\nAdjust Parameters', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Annual Trading Frequency (No Data)', fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.5, 'No Trades Found\nAdjust Parameters', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Cumulative P&L Progression (No Data)', fontsize=14, fontweight='bold')
    ax4.text(0.5, 0.5, 'No Trades Found\nAdjust Parameters', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Hit Rate Stability (No Data)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('walkforward_progression.png', dpi=chart_dpi, bbox_inches='tight')
    plt.close()
    print(f"[OK] Placeholder progression chart saved: walkforward_progression.png")
    
    # Set up empty values for the rest of the script
    positive_years = 0
    strong_years = 0  
    excellent_years = 0
else:
    # Create yearly progression charts
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Annual edge performance
ax1.bar(yearly_df['year'], yearly_df['edge'] * 100, alpha=0.7, color='darkblue')
ax1.axhline(y=overall_edge, color='red', linestyle='--', alpha=0.7, 
           label=f'Overall Edge ({overall_edge:.1f}%)')
ax1.set_title('Annual Edge Performance', fontsize=14, fontweight='bold')
ax1.set_ylabel('Edge vs Base Rate (%)')
ax1.set_xlabel('Year')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Annual trading frequency
ax2.bar(yearly_df['year'], yearly_df['trade_rate'] * 100, alpha=0.7, color='darkgreen')
ax2.axhline(y=overall_trade_rate, color='red', linestyle='--', alpha=0.7, 
           label=f'Overall Rate ({overall_trade_rate:.1f}%)')
ax2.set_title('Annual Trading Frequency', fontsize=14, fontweight='bold')
ax2.set_ylabel('Trading Frequency (%)')
ax2.set_xlabel('Year')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Cumulative P&L progression
long_trades_sorted = df[df['prediction'] == 1].sort_values('date')
cumulative_pnl = long_trades_sorted['target_value'].cumsum()
ax3.plot(long_trades_sorted['date'], cumulative_pnl, 'b-', linewidth=3, label='Cumulative P&L')
ax3.fill_between(long_trades_sorted['date'], 0, cumulative_pnl, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.set_title('Cumulative P&L Progression', fontsize=14, fontweight='bold')
ax3.set_ylabel('Cumulative Return')
ax3.set_xlabel('Date')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Hit rate stability over time
window_size = 30
rolling_hit_rates = []
rolling_dates = []

for i in range(window_size, len(long_trades)):
    window_data = long_trades.iloc[i-window_size:i]
    hit_rate = window_data['actual_up'].mean()
    rolling_hit_rates.append(hit_rate)
    rolling_dates.append(window_data['date'].iloc[-1])

ax4.plot(rolling_dates, np.array(rolling_hit_rates) * 100, 'darkred', linewidth=2, 
         label='30-trade rolling hit rate')
ax4.axhline(y=long_trades['actual_up'].mean() * 100, color='blue', linestyle='--', alpha=0.7, 
           label=f'Overall Hit Rate ({long_trades["actual_up"].mean():.1%})')
ax4.axhline(y=base_rate * 100, color='red', linestyle='--', alpha=0.7, 
           label=f'Base Rate ({base_rate:.0%})')
ax4.fill_between(rolling_dates, base_rate * 100, np.array(rolling_hit_rates) * 100, 
                 alpha=0.3, color='green', label='Edge Region')
ax4.set_title('Hit Rate Stability Over Time', fontsize=14, fontweight='bold')
ax4.set_ylabel('Hit Rate (%)')
ax4.set_xlabel('Date')
ax4.set_ylim(25, 75)
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('walkforward_progression.png', dpi=chart_dpi, bbox_inches='tight')
plt.close()
print(f"[OK] Yearly progression chart saved: walkforward_progression.png")

# Step 5: Generate comprehensive performance report
print("\nSTEP 5: Generating comprehensive performance report...")

print("\n" + "=" * 80)
print("COMPLETE WALK-FORWARD VALIDATION RESULTS")
print("=" * 80)

print(f"\nMODEL CONFIGURATION:")
print(f"  Features: {config['data']['selected_features']}")
print(f"  Trees: {config['model']['n_trees']}")
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
print(f"  Total LONG Signals: {len(long_trades):,}")
print(f"  Trading Frequency: {len(long_trades)/len(df):.1%} of days")
print(f"  Hit Rate: {long_trades['actual_up'].mean():.1%}")
print(f"  Base Rate: {base_rate:.1%}")
print(f"  Edge vs Base Rate: {(long_trades['actual_up'].mean() - base_rate):.1%}")
print(f"  Average Return/Trade: {long_trades['target_value'].mean():+.4f}")
print(f"  Total Cumulative P&L: {long_trades['target_value'].sum():.2f}")

# Calculate monthly P&L statistics
print(f"\nMONTHLY P&L STATISTICS:")
# Group by year-month for monthly P&L
long_trades['year_month'] = pd.to_datetime(long_trades['date']).dt.to_period('M')
monthly_pnl = long_trades.groupby('year_month')['target_value'].sum()
avg_monthly_pnl = monthly_pnl.mean()
std_monthly_pnl = monthly_pnl.std()
print(f"  Average Monthly P&L: {avg_monthly_pnl:+.2f}")
print(f"  StdDev Monthly P&L: {std_monthly_pnl:.2f}")

# Calculate annualized Sharpe ratio (0% risk-free rate)
# Sharpe = (Annual Return - Risk Free Rate) / Annual Volatility
# With 0% risk-free rate: Sharpe = Annual Return / Annual Volatility
annual_return = avg_monthly_pnl * 12
annual_volatility = std_monthly_pnl * np.sqrt(12)
sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
print(f"  Annualized Sharpe (0%): {sharpe_ratio:.3f}")
print(f"    Annual Return: {annual_return:+.2f}")
print(f"    Annual Volatility: {annual_volatility:.2f}")

# Show monthly P&L distribution
positive_months = (monthly_pnl > 0).sum()
total_months = len(monthly_pnl)
print(f"\nMONTHLY PERFORMANCE:")
print(f"  Positive Months: {positive_months}/{total_months} ({positive_months/total_months:.1%})")
print(f"  Best Month: {monthly_pnl.max():+.2f}")
print(f"  Worst Month: {monthly_pnl.min():+.2f}")
print(f"  Median Monthly P&L: {monthly_pnl.median():+.2f}")

print(f"\nMODEL STABILITY:")
print(f"  Best {rolling_window_long}-trade Period: {long_trades['rolling_hit_rate_long'].max():.1%}")
print(f"  Worst {rolling_window_long}-trade Period: {long_trades['rolling_hit_rate_long'].min():.1%}")
print(f"  Average Confidence: {long_trades['up_probability'].mean():.3f}")
print(f"  Confidence Range: {long_trades['up_probability'].min():.3f} - {long_trades['up_probability'].max():.3f}")

print(f"\nYEARLY BREAKDOWN:")
print(f"{'Year':<6} {'Obs':<6} {'Trades':<7} {'Freq%':<6} {'Hit%':<6} {'Edge%':<7} {'P&L':<8}")
print("-" * 60)
for _, row in yearly_df.iterrows():
    print(f"{int(row['year']):<6} {int(row['total_obs']):<6} {int(row['long_signals']):<7} "
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
print(f"  [OK] walkforward_comprehensive.png - 6-panel detailed analysis")
print(f"  [OK] walkforward_progression.png - 4-panel yearly progression")

# Log performance to CSV
log_file = 'performance_log.csv'
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
    'smoothing_alpha': config['preprocessing']['smoothing_alpha'],
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
    'avg_monthly_pnl': f"{avg_monthly_pnl:.2f}",
    'std_monthly_pnl': f"{std_monthly_pnl:.2f}",
    'sharpe_ratio': f"{sharpe_ratio:.3f}",
    'positive_months_pct': f"{positive_months/total_months:.3f}",
    'best_month': f"{monthly_pnl.max():.2f}",
    'worst_month': f"{monthly_pnl.min():.2f}",
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
print("COMPLETE WALK-FORWARD ANALYSIS FINISHED SUCCESSFULLY!")
print("=" * 80)