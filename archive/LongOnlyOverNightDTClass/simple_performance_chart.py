import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def create_simple_performance_charts():
    """
    Create focused performance charts showing key metrics.
    """
    # Load the validation results
    try:
        df = pd.read_csv('longonly_validation_results.csv')
        df['date'] = pd.to_datetime(df['date'])
    except:
        print("Error: longonly_validation_results.csv not found. Run the model first.")
        return
    
    # Create a 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Cumulative Returns Chart
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    # Long-only strategy returns
    long_trades = df_sorted[df_sorted['prediction'] == 1].copy()
    if len(long_trades) > 0:
        long_trades['cumulative_return'] = long_trades['target_value'].cumsum()
        ax1.plot(long_trades['date'], long_trades['cumulative_return'], 
                label='Long-Only Strategy', linewidth=2, color='blue')
    
    # Buy and hold baseline
    df_sorted['buy_hold_cumulative'] = df_sorted['target_value'].cumsum()
    ax1.plot(df_sorted['date'], df_sorted['buy_hold_cumulative'], 
             label='Buy & Hold', alpha=0.7, linewidth=1.5, color='gray')
    
    ax1.set_title('Cumulative Returns Comparison', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Signal Frequency Over Time (Quarterly)
    df_sorted['quarter'] = df_sorted['date'].dt.to_period('Q')
    quarterly_stats = df_sorted.groupby('quarter').agg({
        'prediction': ['count', 'sum']
    }).reset_index()
    
    quarterly_stats.columns = ['quarter', 'total_obs', 'long_signals']
    quarterly_stats['signal_rate'] = quarterly_stats['long_signals'] / quarterly_stats['total_obs']
    quarterly_stats['quarter_dt'] = quarterly_stats['quarter'].dt.to_timestamp()
    
    ax2.bar(range(len(quarterly_stats)), quarterly_stats['signal_rate'], 
            color='orange', alpha=0.7)
    ax2.set_title('Long Signal Frequency by Quarter', fontweight='bold')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Fraction of Days with Long Signal')
    ax2.set_xticks(range(0, len(quarterly_stats), 4))
    ax2.set_xticklabels([quarterly_stats['quarter'].iloc[i].strftime('%Y-Q%q') 
                        for i in range(0, len(quarterly_stats), 4)], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Return Distributions
    all_returns = df['target_value']
    long_returns = df[df['prediction'] == 1]['target_value']
    
    ax3.hist(all_returns, bins=50, alpha=0.5, label='All Returns', 
             color='lightblue', density=True)
    ax3.hist(long_returns, bins=30, alpha=0.8, label='Long Signal Returns', 
             color='red', density=True)
    
    ax3.axvline(all_returns.mean(), color='blue', linestyle='--', 
               label=f'All Mean: {all_returns.mean():.3f}')
    ax3.axvline(long_returns.mean(), color='red', linestyle='--', 
               label=f'Long Mean: {long_returns.mean():.3f}')
    
    ax3.set_title('Return Distributions', fontweight='bold')
    ax3.set_xlabel('Return')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Table
    ax4.axis('off')
    
    # Calculate key statistics
    total_obs = len(df)
    long_signals = (df['prediction'] == 1).sum()
    long_hit_rate = df[df['prediction'] == 1]['actual_up'].mean()
    base_rate = df['actual_up'].mean()
    long_avg_return = df[df['prediction'] == 1]['target_value'].mean()
    overall_avg_return = df['target_value'].mean()
    
    # Win/Loss breakdown for long signals
    long_wins = ((df['prediction'] == 1) & (df['actual_up'] == 1)).sum()
    long_losses = ((df['prediction'] == 1) & (df['actual_up'] == 0)).sum()
    
    metrics_text = f"""
    PERFORMANCE SUMMARY
    ═══════════════════════════════
    
    Dataset Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
    Total Trading Days: {total_obs:,}
    
    SIGNAL STATISTICS
    ─────────────────────────────────
    Long Signals: {long_signals:,} ({long_signals/total_obs:.1%})
    No Trade Days: {total_obs-long_signals:,} ({(total_obs-long_signals)/total_obs:.1%})
    
    ACCURACY METRICS
    ─────────────────────────────────
    Long Hit Rate: {long_hit_rate:.3f} ({long_hit_rate:.1%})
    Base Rate: {base_rate:.3f} ({base_rate:.1%})
    Edge over Base: +{long_hit_rate - base_rate:.3f} (+{(long_hit_rate - base_rate):.1%})
    
    Win/Loss Breakdown:
    • Wins: {long_wins:,}
    • Losses: {long_losses:,}
    • Win/Loss Ratio: {long_wins/max(long_losses,1):.2f}
    
    RETURN METRICS
    ─────────────────────────────────
    Long Avg Return: {long_avg_return:+.4f}
    Overall Avg Return: {overall_avg_return:+.4f}
    Return Edge: {long_avg_return - overall_avg_return:+.4f}
    
    Total Long P&L: {long_returns.sum():+.3f}
    Total Buy&Hold P&L: {all_returns.sum():+.3f}
    Strategy Outperformance: {long_returns.sum() - all_returns.sum():+.3f}
    """
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('simple_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Simple performance charts saved as 'simple_performance_summary.png'")
    
    # Print some additional insights
    print("\nKEY INSIGHTS:")
    print("=" * 50)
    print(f"• Model trades {long_signals/total_obs:.1%} of the time (selective approach)")
    print(f"• When it does trade, it beats the base rate by {(long_hit_rate - base_rate)*100:.1f} percentage points")
    print(f"• Average return per long signal: {long_avg_return:.4f}")
    print(f"• Strategy generates {long_returns.sum():.3f} total return vs {all_returns.sum():.3f} buy-and-hold")
    
    if long_signals > 0:
        annual_trades = long_signals / (len(df) / 252)  # Approximate annual trades
        print(f"• Approximately {annual_trades:.0f} trades per year")

if __name__ == "__main__":
    create_simple_performance_charts()