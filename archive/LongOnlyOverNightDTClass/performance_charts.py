import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def create_performance_charts():
    """
    Create comprehensive performance charts for the long-only model.
    """
    # Load the validation results
    try:
        df = pd.read_csv('longonly_validation_results.csv')
        df['date'] = pd.to_datetime(df['date'])
    except:
        print("Error: longonly_validation_results.csv not found. Run the model first.")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 20))
    
    # 1. Cumulative Returns Chart
    ax1 = plt.subplot(4, 2, 1)
    
    # Calculate cumulative returns for different strategies
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    # Strategy 1: Long-only signals
    long_trades = df_sorted[df_sorted['prediction'] == 1].copy()
    long_trades['cumulative_return'] = long_trades['target_value'].cumsum()
    
    # Strategy 2: Buy and hold (all observations)
    df_sorted['buy_hold_cumulative'] = df_sorted['target_value'].cumsum()
    
    # Strategy 3: Only UP moves (oracle)
    up_moves = df_sorted[df_sorted['actual_up'] == 1].copy()
    up_moves['oracle_cumulative'] = up_moves['target_value'].cumsum()
    
    plt.plot(df_sorted['date'], df_sorted['buy_hold_cumulative'], 
             label='Buy & Hold', alpha=0.7, linewidth=1)
    plt.plot(long_trades['date'], long_trades['cumulative_return'], 
             label='Long-Only Model', linewidth=2)
    plt.plot(up_moves['date'], up_moves['oracle_cumulative'], 
             label='Oracle (All UP moves)', alpha=0.6, linestyle='--')
    
    plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Hit Rate Over Time
    ax2 = plt.subplot(4, 2, 2)
    
    # Calculate rolling hit rate for long signals
    long_trades_sorted = long_trades.sort_values('date').reset_index(drop=True)
    window_size = 100
    
    if len(long_trades_sorted) > window_size:
        hit_rates = []
        dates = []
        
        for i in range(window_size, len(long_trades_sorted)):
            window_data = long_trades_sorted.iloc[i-window_size:i]
            hit_rate = (window_data['actual_up'] == 1).mean()
            hit_rates.append(hit_rate)
            dates.append(window_data['date'].iloc[-1])
        
        plt.plot(dates, hit_rates, label='Rolling Hit Rate (100 trades)', linewidth=2)
        plt.axhline(y=df['actual_up'].mean(), color='red', linestyle='--', 
                   label=f'Base Rate ({df["actual_up"].mean():.3f})')
        
    plt.title('Hit Rate Over Time (Long Signals Only)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Hit Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Signal Frequency Over Time
    ax3 = plt.subplot(4, 2, 3)
    
    # Group by month and count signals
    df_sorted['year_month'] = df_sorted['date'].dt.to_period('M')
    monthly_stats = df_sorted.groupby('year_month').agg({
        'prediction': ['count', 'sum'],
        'actual_up': 'mean'
    }).reset_index()
    
    monthly_stats.columns = ['year_month', 'total_obs', 'long_signals', 'base_rate']
    monthly_stats['signal_rate'] = monthly_stats['long_signals'] / monthly_stats['total_obs']
    monthly_stats['year_month_dt'] = monthly_stats['year_month'].dt.to_timestamp()
    
    plt.plot(monthly_stats['year_month_dt'], monthly_stats['signal_rate'], 
             marker='o', linewidth=2, markersize=3)
    
    plt.title('Long Signal Frequency Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Fraction of Days with Long Signal')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    
    # 4. Return Distribution
    ax4 = plt.subplot(4, 2, 4)
    
    # Plot return distributions
    all_returns = df['target_value']
    long_returns = df[df['prediction'] == 1]['target_value']
    no_trade_returns = df[df['prediction'] == 0]['target_value']
    
    plt.hist(all_returns, bins=50, alpha=0.5, label='All Returns', density=True)
    plt.hist(long_returns, bins=30, alpha=0.7, label='Long Signal Returns', density=True)
    plt.hist(no_trade_returns, bins=50, alpha=0.4, label='No Trade Returns', density=True)
    
    plt.axvline(all_returns.mean(), color='blue', linestyle='--', alpha=0.7, label='All Mean')
    plt.axvline(long_returns.mean(), color='orange', linestyle='--', alpha=0.7, label='Long Mean')
    
    plt.title('Return Distributions', fontsize=14, fontweight='bold')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Prediction Probability vs Actual Performance
    ax5 = plt.subplot(4, 2, 5)
    
    # Bin by prediction probability
    prob_bins = np.arange(0, 1.1, 0.1)
    df['prob_bin'] = pd.cut(df['up_probability'], prob_bins)
    
    prob_performance = df.groupby('prob_bin').agg({
        'actual_up': 'mean',
        'target_value': 'mean',
        'prediction': 'count'
    }).reset_index()
    
    prob_centers = [interval.mid for interval in prob_performance['prob_bin']]
    
    bars = plt.bar(prob_centers, prob_performance['actual_up'], 
                   width=0.08, alpha=0.7, label='Actual UP Rate')
    plt.axhline(y=df['actual_up'].mean(), color='red', linestyle='--', 
               label=f'Base Rate ({df["actual_up"].mean():.3f})')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, prob_performance['prediction'])):
        if not np.isnan(count):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{int(count)}', ha='center', va='bottom', fontsize=8)
    
    plt.title('Model Calibration: Probability vs Actual Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted UP Probability')
    plt.ylabel('Actual UP Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Feature Value vs Performance
    ax6 = plt.subplot(4, 2, 6)
    
    # Bin by feature value (vol-adjusted overnight gap)
    feature_bins = np.percentile(df['feature_value'], np.arange(0, 101, 10))
    df['feature_bin'] = pd.cut(df['feature_value'], feature_bins)
    
    feature_performance = df.groupby('feature_bin').agg({
        'actual_up': 'mean',
        'prediction': ['count', 'mean'],
        'feature_value': 'mean'
    }).reset_index()
    
    feature_performance.columns = ['feature_bin', 'actual_up_rate', 'count', 'signal_rate', 'avg_feature']
    
    # Plot actual UP rate vs feature value
    plt.plot(feature_performance['avg_feature'], feature_performance['actual_up_rate'], 
             'o-', linewidth=2, markersize=5, label='Actual UP Rate')
    plt.plot(feature_performance['avg_feature'], feature_performance['signal_rate'], 
             's-', linewidth=2, markersize=4, label='Model Signal Rate')
    
    plt.axhline(y=df['actual_up'].mean(), color='red', linestyle='--', 
               label=f'Base Rate ({df["actual_up"].mean():.3f})')
    
    plt.title('Feature Value vs Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Vol-Adjusted Overnight Gap (Deciles)')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Drawdown Analysis
    ax7 = plt.subplot(4, 2, 7)
    
    # Calculate running maximum and drawdown for long-only strategy
    long_cumret = long_trades['cumulative_return'].values
    running_max = np.maximum.accumulate(long_cumret)
    drawdown = (long_cumret - running_max) / (running_max + 1e-8)  # Avoid division by zero
    
    plt.fill_between(long_trades['date'], drawdown, 0, alpha=0.7, color='red')
    plt.plot(long_trades['date'], drawdown, color='darkred', linewidth=1)
    
    plt.title('Strategy Drawdown', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)
    
    # 8. Summary Statistics
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    
    # Calculate key statistics
    total_obs = len(df)
    long_signals = (df['prediction'] == 1).sum()
    long_hit_rate = df[df['prediction'] == 1]['actual_up'].mean()
    base_rate = df['actual_up'].mean()
    long_avg_return = df[df['prediction'] == 1]['target_value'].mean()
    overall_avg_return = df['target_value'].mean()
    
    # Sharpe-like metrics
    long_returns_std = df[df['prediction'] == 1]['target_value'].std()
    overall_std = df['target_value'].std()
    
    stats_text = f"""
    PERFORMANCE SUMMARY
    ═══════════════════════════════
    
    Total Observations: {total_obs:,}
    Long Signals: {long_signals:,} ({long_signals/total_obs:.1%})
    
    HIT RATES:
    • Long Signal Hit Rate: {long_hit_rate:.3f}
    • Base Rate: {base_rate:.3f}
    • Edge: +{long_hit_rate - base_rate:.3f}
    
    RETURNS:
    • Long Avg Return: {long_avg_return:+.4f}
    • Overall Avg Return: {overall_avg_return:+.4f}
    • Return Edge: {long_avg_return - overall_avg_return:+.4f}
    
    RISK METRICS:
    • Long Signal Std: {long_returns_std:.4f}
    • Overall Std: {overall_std:.4f}
    • Risk-Adj Edge: {(long_avg_return - overall_avg_return)/long_returns_std:.4f}
    
    VALIDATION PERIOD:
    • Start: {df['date'].min().strftime('%Y-%m-%d')}
    • End: {df['date'].max().strftime('%Y-%m-%d')}
    • Duration: {(df['date'].max() - df['date'].min()).days} days
    """
    
    plt.text(0.1, 0.95, stats_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('longonly_performance_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance charts saved as 'longonly_performance_charts.png'")
    
    # Additional detailed analysis
    print("\nDETAILED PERFORMANCE ANALYSIS:")
    print("=" * 50)
    
    # Monthly performance summary
    monthly_perf = df.groupby(df['date'].dt.to_period('M')).agg({
        'prediction': ['count', 'sum'],
        'actual_up': 'mean',
        'target_value': 'mean'
    })
    
    print(f"\nAverage monthly signal rate: {monthly_stats['signal_rate'].mean():.3f}")
    print(f"Signal rate std dev: {monthly_stats['signal_rate'].std():.3f}")
    print(f"Max monthly signal rate: {monthly_stats['signal_rate'].max():.3f}")
    print(f"Min monthly signal rate: {monthly_stats['signal_rate'].min():.3f}")

if __name__ == "__main__":
    create_performance_charts()