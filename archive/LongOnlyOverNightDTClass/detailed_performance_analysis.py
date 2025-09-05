import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def create_detailed_analysis():
    """
    Create detailed performance analysis focusing on time series patterns.
    """
    df = pd.read_csv('longonly_validation_results.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a comprehensive analysis
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Equity Curve with Drawdowns
    ax1 = plt.subplot(3, 2, 1)
    
    # Calculate cumulative performance
    df_sorted = df.sort_values('date').reset_index(drop=True)
    long_trades = df_sorted[df_sorted['prediction'] == 1].copy()
    
    if len(long_trades) > 0:
        long_trades['cumulative_return'] = long_trades['target_value'].cumsum()
        long_trades['running_max'] = long_trades['cumulative_return'].expanding().max()
        long_trades['drawdown'] = (long_trades['cumulative_return'] - long_trades['running_max']) / (long_trades['running_max'] + 1e-8)
        
        # Plot equity curve
        ax1.plot(long_trades['date'], long_trades['cumulative_return'], 
                linewidth=2, color='blue', label='Strategy P&L')
        ax1.fill_between(long_trades['date'], 
                        long_trades['cumulative_return'] + long_trades['drawdown'] * long_trades['running_max'],
                        long_trades['cumulative_return'], 
                        alpha=0.3, color='red', label='Drawdown')
    
    # Buy and hold comparison
    df_sorted['buy_hold_cumulative'] = df_sorted['target_value'].cumsum()
    ax1.plot(df_sorted['date'], df_sorted['buy_hold_cumulative'], 
             alpha=0.6, color='gray', linewidth=1.5, label='Buy & Hold')
    
    ax1.set_title('Strategy Performance vs Buy & Hold', fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rolling Performance Metrics
    ax2 = plt.subplot(3, 2, 2)
    
    # Calculate 6-month rolling metrics
    if len(long_trades) > 50:
        window = 50
        rolling_hit_rates = []
        rolling_returns = []
        rolling_dates = []
        
        for i in range(window, len(long_trades)):
            window_data = long_trades.iloc[i-window:i]
            hit_rate = (window_data['actual_up'] == 1).mean()
            avg_return = window_data['target_value'].mean()
            
            rolling_hit_rates.append(hit_rate)
            rolling_returns.append(avg_return)
            rolling_dates.append(window_data['date'].iloc[-1])
        
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(rolling_dates, rolling_hit_rates, 'b-', linewidth=2, label='Hit Rate (50-trade window)')
        line2 = ax2_twin.plot(rolling_dates, rolling_returns, 'r-', linewidth=2, label='Avg Return (50-trade window)')
        
        ax2.axhline(df['actual_up'].mean(), color='blue', linestyle='--', alpha=0.7, label='Base Rate')
        ax2_twin.axhline(0, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_ylabel('Hit Rate', color='blue')
        ax2_twin.set_ylabel('Average Return', color='red')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2.set_title('Rolling Performance Metrics', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Signal Timing Analysis
    ax3 = plt.subplot(3, 2, 3)
    
    # Monthly signal frequency
    df_sorted['year_month'] = df_sorted['date'].dt.to_period('M')
    monthly_signals = df_sorted.groupby('year_month').agg({
        'prediction': ['count', 'sum'],
        'actual_up': 'mean'
    }).reset_index()
    
    monthly_signals.columns = ['year_month', 'total_obs', 'long_signals', 'base_rate']
    monthly_signals['signal_rate'] = monthly_signals['long_signals'] / monthly_signals['total_obs']
    monthly_signals['year_month_dt'] = monthly_signals['year_month'].dt.to_timestamp()
    
    bars = ax3.bar(range(len(monthly_signals)), monthly_signals['signal_rate'], 
                   alpha=0.7, color='green')
    ax3.set_title('Monthly Signal Frequency', fontweight='bold')
    ax3.set_ylabel('Signal Rate')
    ax3.set_xticks(range(0, len(monthly_signals), 12))
    ax3.set_xticklabels([monthly_signals.iloc[i]['year_month'].strftime('%Y') 
                        for i in range(0, len(monthly_signals), 12)], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature Value Distribution and Performance
    ax4 = plt.subplot(3, 2, 4)
    
    # Bin feature values and show performance by bin
    feature_percentiles = np.percentile(df['feature_value'], np.arange(0, 101, 10))
    df['feature_decile'] = pd.cut(df['feature_value'], feature_percentiles, labels=False)
    
    decile_performance = df.groupby('feature_decile').agg({
        'actual_up': 'mean',
        'prediction': ['count', 'mean'],
        'feature_value': 'mean'
    }).reset_index()
    
    decile_performance.columns = ['decile', 'actual_up_rate', 'count', 'signal_rate', 'avg_feature']
    decile_performance = decile_performance.dropna()
    
    ax4_twin = ax4.twinx()
    bars1 = ax4.bar(decile_performance['decile'], decile_performance['actual_up_rate'], 
                    alpha=0.6, color='blue', label='Actual UP Rate')
    line1 = ax4_twin.plot(decile_performance['decile'], decile_performance['signal_rate'], 
                         'ro-', linewidth=2, markersize=4, label='Model Signal Rate')
    
    ax4.axhline(df['actual_up'].mean(), color='blue', linestyle='--', alpha=0.7)
    ax4.set_title('Performance by Feature Value Decile', fontweight='bold')
    ax4.set_xlabel('Feature Value Decile (0=lowest, 9=highest)')
    ax4.set_ylabel('Actual UP Rate', color='blue')
    ax4_twin.set_ylabel('Model Signal Rate', color='red')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Walk-Forward Model Stability
    ax5 = plt.subplot(3, 2, 5)
    
    # Group by test_end_idx to see model behavior over walk-forward
    model_performance = df.groupby('test_end_idx').agg({
        'prediction': ['count', 'sum'],
        'actual_up': 'mean',
        'target_value': 'mean',
        'date': 'max'
    }).reset_index()
    
    model_performance.columns = ['test_end_idx', 'total_obs', 'long_signals', 'base_rate', 'avg_return', 'date']
    model_performance['signal_rate'] = model_performance['long_signals'] / model_performance['total_obs']
    
    # Plot signal rate over time
    ax5.plot(model_performance['date'], model_performance['signal_rate'], 
             linewidth=1, alpha=0.7, color='orange')
    ax5.set_title('Model Signal Rate Over Walk-Forward Windows', fontweight='bold')
    ax5.set_ylabel('Signal Rate per Test Window')
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Summary Statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Calculate comprehensive statistics
    total_obs = len(df)
    long_signals = (df['prediction'] == 1).sum()
    long_hit_rate = df[df['prediction'] == 1]['actual_up'].mean()
    base_rate = df['actual_up'].mean()
    long_avg_return = df[df['prediction'] == 1]['target_value'].mean()
    overall_avg_return = df['target_value'].mean()
    
    # Risk metrics
    long_returns = df[df['prediction'] == 1]['target_value']
    long_std = long_returns.std()
    long_sharpe = (long_avg_return - overall_avg_return) / long_std if long_std > 0 else 0
    
    # Win/Loss analysis
    long_wins = ((df['prediction'] == 1) & (df['actual_up'] == 1)).sum()
    long_losses = long_signals - long_wins
    
    # Strategy total return
    strategy_total = long_returns.sum()
    buy_hold_total = df['target_value'].sum()
    
    # Calculate max drawdown
    if len(long_trades) > 0:
        max_dd = long_trades['drawdown'].min()
        max_dd_pct = f"{max_dd:.1%}"
    else:
        max_dd_pct = "N/A"
    
    stats_text = f"""
    COMPREHENSIVE PERFORMANCE ANALYSIS
    ═══════════════════════════════════════════
    
    DATASET OVERVIEW
    ─────────────────────────────────────────────
    Total Observations: {total_obs:,}
    Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
    Duration: {(df['date'].max() - df['date'].min()).days:,} days
    
    SIGNAL STATISTICS
    ─────────────────────────────────────────────
    Long Signals: {long_signals:,} ({long_signals/total_obs:.1%})
    Approx. Annual Trades: {long_signals / (total_obs/252):.0f}
    Signal Frequency: {252 * long_signals / total_obs:.1f} days/year
    
    ACCURACY METRICS
    ─────────────────────────────────────────────
    Hit Rate (Long): {long_hit_rate:.3f} ({long_hit_rate:.1%})
    Base Rate: {base_rate:.3f} ({base_rate:.1%})
    Edge: +{long_hit_rate - base_rate:.3f} (+{(long_hit_rate - base_rate)*100:.1f} pp)
    
    Win/Loss: {long_wins:,} / {long_losses:,} (Ratio: {long_wins/max(long_losses,1):.2f})
    
    RETURN METRICS
    ─────────────────────────────────────────────
    Avg Return/Trade: {long_avg_return:+.4f}
    Overall Avg Return: {overall_avg_return:+.4f}
    Return Edge: {long_avg_return - overall_avg_return:+.4f}
    
    Strategy Total: {strategy_total:+.2f}
    Buy&Hold Total: {buy_hold_total:+.2f}
    Outperformance: {strategy_total - buy_hold_total:+.2f}
    
    RISK METRICS
    ─────────────────────────────────────────────
    Return Std Dev: {long_std:.4f}
    Risk-Adj Edge: {long_sharpe:.3f}
    Max Drawdown: {max_dd_pct}
    
    WALK-FORWARD DETAILS
    ─────────────────────────────────────────────
    Training Window: 1,000 observations
    Test Window: 100 observations  
    Step Size: 1 observation (overlapping)
    Total Models: {len(model_performance):,}
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('detailed_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Detailed performance analysis saved as 'detailed_performance_analysis.png'")
    
    # Print key insights about the walk-forward validation
    print("\nWALK-FORWARD VALIDATION INSIGHTS:")
    print("=" * 50)
    print(f"• Step size: 1 observation (continuous retraining)")
    print(f"• Training window: 1,000 observations (~4 years of data)")
    print(f"• Test window: 100 observations (~4 months)")
    print(f"• Total models trained: {len(model_performance):,}")
    print(f"• Overlapping training sets: 99.9% overlap between consecutive models")
    print(f"• This mimics real-world deployment with daily model updates")

if __name__ == "__main__":
    create_detailed_analysis()