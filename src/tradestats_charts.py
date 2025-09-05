"""
Chart generation module based on tradestats.md specifications
Creates all required charts for performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
from datetime import datetime
import calendar


def create_equity_curve_chart(trades_df, initial_equity=1000):
    """
    Create compound equity curve chart
    Daily resolution with x-axis as time (not trade count)
    
    Parameters:
    trades_df: DataFrame with date and pnl_decimal columns
    initial_equity: Starting value (default 1000)
    """
    
    # Create daily DataFrame with all dates
    if 'date' not in trades_df.columns:
        return None
    
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    
    # Create date range from min to max
    date_range = pd.date_range(start=trades_df['date'].min(), 
                              end=trades_df['date'].max(), 
                              freq='D')
    
    # Create daily returns series (0 for non-trading days)
    daily_df = pd.DataFrame(index=date_range)
    daily_df['return'] = 0.0
    
    # Fill in actual trade returns
    for _, trade in trades_df.iterrows():
        if trade['date'] in daily_df.index:
            daily_df.loc[trade['date'], 'return'] = trade['pnl_decimal']
    
    # Calculate compound equity curve
    equity = initial_equity
    equity_values = []
    
    for ret in daily_df['return'].values:
        equity = equity * (1 + ret)
        equity_values.append(equity)
    
    daily_df['equity'] = equity_values
    
    # Create figure
    fig = plt.Figure(figsize=(14, 6), facecolor='white', tight_layout=True)
    ax = fig.add_subplot(111)
    
    # Plot equity curve
    ax.plot(daily_df.index, daily_df['equity'], 'b-', linewidth=2, label='Portfolio Value')
    ax.axhline(y=initial_equity, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Initial: ${initial_equity}')
    
    # Formatting with larger fonts
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax.set_title('Compound Equity Curve (Daily Resolution)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    # Increase tick label size
    ax.tick_params(axis='both', labelsize=10)
    
    # Add final value annotation with larger font
    final_value = daily_df['equity'].iloc[-1]
    total_return = (final_value - initial_equity) / initial_equity * 100
    ax.text(0.02, 0.98, f'Final: ${final_value:,.0f}\nReturn: {total_return:.1f}%',
            transform=ax.transAxes, fontsize=12, va='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    return fig


def create_drawdown_chart(trades_df, initial_equity=1000):
    """
    Create drawdown chart showing cumulative retreat from peak
    Line or area chart
    """
    
    if 'date' not in trades_df.columns:
        return None
    
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    
    # Create daily DataFrame
    date_range = pd.date_range(start=trades_df['date'].min(), 
                              end=trades_df['date'].max(), 
                              freq='D')
    
    daily_df = pd.DataFrame(index=date_range)
    daily_df['return'] = 0.0
    
    # Fill in actual trade returns
    for _, trade in trades_df.iterrows():
        if trade['date'] in daily_df.index:
            daily_df.loc[trade['date'], 'return'] = trade['pnl_decimal']
    
    # Calculate equity and drawdown
    equity = initial_equity
    peak = initial_equity
    equity_values = []
    drawdown_values = []
    
    for ret in daily_df['return'].values:
        equity = equity * (1 + ret)
        if equity > peak:
            peak = equity
        drawdown = (equity - peak)  # Absolute drawdown in dollars
        drawdown_pct = (drawdown / peak) * 100  # Percentage drawdown
        
        equity_values.append(equity)
        drawdown_values.append(drawdown)
    
    daily_df['drawdown'] = drawdown_values
    
    # Create figure
    fig = plt.Figure(figsize=(14, 5), facecolor='white', tight_layout=True)
    ax = fig.add_subplot(111)
    
    # Plot drawdown as area chart
    ax.fill_between(daily_df.index, daily_df['drawdown'], 0, 
                    color='red', alpha=0.3, label='Drawdown')
    ax.plot(daily_df.index, daily_df['drawdown'], 'r-', linewidth=2)
    
    # Formatting with larger fonts
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown ($)', fontsize=12, fontweight='bold')
    ax.set_title('Drawdown Chart (Cumulative Retreat from Peak)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Increase tick label size
    ax.tick_params(axis='both', labelsize=10)
    
    # Add max drawdown annotation
    max_dd = min(drawdown_values)
    max_dd_pct = (max_dd / peak) * 100
    max_dd_date = daily_df.index[drawdown_values.index(max_dd)]
    
    ax.annotate(f'Max DD: ${abs(max_dd):.0f} ({abs(max_dd_pct):.1f}%)', 
               xy=(max_dd_date, max_dd),
               xytext=(max_dd_date, max_dd * 0.5),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=11, color='red', fontweight='bold')
    
    return fig


def create_monthly_returns_table(trades_df):
    """
    Create table with monthly returns, one row per year
    Monthly returns calculated as exp(MonthlySum(ln(1+returns))) - 1
    With color coding: green for positive, red for negative
    """
    
    if 'date' not in trades_df.columns or 'pnl_decimal' not in trades_df.columns:
        return None
    
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    trades_df['year'] = trades_df['date'].dt.year
    trades_df['month'] = trades_df['date'].dt.month
    
    # Calculate ln(1+returns)
    trades_df['ln_1_plus_ret'] = np.log1p(trades_df['pnl_decimal'])
    
    # Calculate monthly sums of ln(1+returns)
    monthly_ln_sums = trades_df.groupby(['year', 'month'])['ln_1_plus_ret'].sum()
    
    # Convert back to percentage returns
    monthly_returns = (np.exp(monthly_ln_sums) - 1) * 100
    
    # Create pivot table (years as rows, months as columns)
    monthly_table = monthly_returns.unstack(fill_value=0)
    
    # Add annual returns column
    annual_returns = []
    for year in monthly_table.index:
        year_data = trades_df[trades_df['year'] == year]
        if len(year_data) > 0:
            annual_ln_sum = year_data['ln_1_plus_ret'].sum()
            annual_return = (np.exp(annual_ln_sum) - 1) * 100
        else:
            annual_return = 0
        annual_returns.append(annual_return)
    
    monthly_table['Annual'] = annual_returns
    
    # Create figure with improved layout
    fig = plt.Figure(figsize=(16, 10), facecolor='white', tight_layout=True)
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Create headers
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]
    headers = ['Year'] + month_names + ['Annual']
    
    # Prepare table data with color coding
    table_data = []
    cell_colors = []
    
    for year in monthly_table.index:
        row = [str(year)]
        colors = ['lightgray']  # Year column color
        
        for month in range(1, 13):
            if month in monthly_table.columns:
                val = monthly_table.loc[year, month]
                if val != 0:
                    row.append(f'{val:+.2f}%')
                    # Color based on positive/negative
                    if val > 0:
                        colors.append('#90EE90')  # Light green
                    else:
                        colors.append('#FFB6C1')  # Light red
                else:
                    row.append('-')
                    colors.append('white')
            else:
                row.append('-')
                colors.append('white')
        
        # Add annual return with color
        annual_val = monthly_table.loc[year, 'Annual']
        row.append(f'{annual_val:+.2f}%')
        if annual_val > 0:
            colors.append('#90EE90')  # Light green
        else:
            colors.append('#FFB6C1')  # Light red
            
        table_data.append(row)
        cell_colors.append(colors)
    
    # Add totals row
    total_row = ['Total']
    total_colors = ['lightgray']
    
    for month in range(1, 13):
        if month in monthly_table.columns:
            month_total = monthly_table[month].sum()
            total_row.append(f'{month_total:+.2f}%')
            if month_total > 0:
                total_colors.append('#90EE90')
            else:
                total_colors.append('#FFB6C1')
        else:
            total_row.append('-')
            total_colors.append('white')
    
    # Overall total
    overall_total = monthly_table['Annual'].sum()
    total_row.append(f'{overall_total:+.2f}%')
    if overall_total > 0:
        total_colors.append('#90EE90')
    else:
        total_colors.append('#FFB6C1')
    
    table_data.append(total_row)
    cell_colors.append(total_colors)
    
    # Create table with larger font
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    cellColours=cell_colors,
                    colColours=['lightblue']*len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Title
    ax.text(0.5, 0.95, 'Monthly Returns Table (%)', 
            transform=ax.transAxes, ha='center',
            fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#90EE90', label='Positive Returns'),
        mpatches.Patch(color='#FFB6C1', label='Negative Returns')
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(0.95, 0.92), fontsize=10)
    
    return fig


def create_all_charts(stats_dict):
    """
    Create all tradestats charts in a single figure with improved layout
    
    Parameters:
    stats_dict: Dictionary from calculate_performance_stats containing 'trades_df' and 'equity_curve'
    
    Returns:
    matplotlib figure with all charts
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Extract trades data
    if 'trades_df' not in stats_dict:
        raise ValueError("stats_dict must contain 'trades_df'")
    
    trades_df = stats_dict['trades_df']
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Performance Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # Create grid with better spacing
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35, 
                 top=0.94, bottom=0.06, left=0.06, right=0.94)
    
    # 1. Equity Curve (top left, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, 0:2])
    if 'equity_curve' in stats_dict and len(stats_dict['equity_curve']) > 0:
        ax1.plot(stats_dict['equity_curve'], 'b-', linewidth=2.5)
        ax1.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.set_title('Compound Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trade Number', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=10)
        
        # Add statistics annotation
        final_value = stats_dict['equity_curve'][-1] if len(stats_dict['equity_curve']) > 0 else 1000
        total_return = (final_value - 1000) / 1000 * 100
        ax1.text(0.02, 0.98, f'Final: ${final_value:,.0f}\nReturn: {total_return:.1f}%',
                transform=ax1.transAxes, fontsize=11, va='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 2. Performance Stats Text (top right) - Using tradestats.md metrics
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # Import tradestats calculator
    try:
        from src.tradestats_calculator import calculate_tradestats_metrics
        
        # Calculate tradestats metrics
        if 'trades_df' in stats_dict:
            tradestats = calculate_tradestats_metrics(stats_dict['trades_df'])
        else:
            tradestats = {}
    except:
        tradestats = {}
    
    # Create performance summary text with larger font and tradestats metrics
    perf_text = "TRADESTATS METRICS\n" + "="*22 + "\n"
    perf_text += "Count:\n"
    perf_text += f"Observations: {tradestats.get('num_observations', 0)}\n"
    perf_text += f"Years: {tradestats.get('years_of_data', 0):.3f}\n"
    perf_text += f"Trades: {tradestats.get('num_trades', 0)}\n"
    perf_text += f"Trade Freq: {tradestats.get('trade_frequency_pct', 0):.2f}\n"
    perf_text += f"Trades P.A.: {tradestats.get('avg_trades_pa', 0):.1f}\n\n"
    
    perf_text += "Trade:\n"
    perf_text += f"Win%: {tradestats.get('win_pct', 0):.2f}\n"
    perf_text += f"Avg Loss: {tradestats.get('avg_loss_pct', 0):.2f}\n"
    perf_text += f"Avg Profit: {tradestats.get('avg_profit_pct', 0):.2f}\n"
    perf_text += f"Expectancy: {tradestats.get('expectancy', 0):.2f}\n\n"
    
    perf_text += "Model:\n"
    perf_text += f"Annual%: {tradestats.get('avg_annual_pct', 0):.2f}\n"
    perf_text += f"Max DD%: {tradestats.get('max_draw_pct', 0):.2f}\n"
    perf_text += f"Sharpe: {tradestats.get('sharpe', 0):.3f}\n"
    perf_text += f"Profit/DD: {tradestats.get('profit_dd_ratio', 0):.2f}\n"
    perf_text += f"UPI: {tradestats.get('upi', 0):.3f}\n"
    
    ax2.text(0.05, 0.95, perf_text, transform=ax2.transAxes, 
            fontsize=11, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 3. Monthly Returns Table (middle row, spanning all columns)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Calculate monthly returns with color coding
    if 'date' in trades_df.columns:
        try:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df['year'] = trades_df['date'].dt.year
            trades_df['month'] = trades_df['date'].dt.month
            
            # Create pivot table
            monthly_pivot = trades_df.pivot_table(
                values='pnl_pct',
                index='year',
                columns='month',
                aggfunc='sum'
            )
            
            # Create table with color coding
            cell_text = []
            cell_colors = []
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
            
            for year in monthly_pivot.index:
                row = [str(year)]
                colors = ['lightgray']
                annual_return = 0
                
                for month in range(1, 13):
                    if month in monthly_pivot.columns:
                        val = monthly_pivot.loc[year, month]
                        if pd.notna(val) and val != 0:
                            row.append(f'{val:+.1f}')
                            annual_return += val
                            # Color coding
                            if val > 0:
                                colors.append('#90EE90')  # Light green
                            else:
                                colors.append('#FFB6C1')  # Light red
                        else:
                            row.append('-')
                            colors.append('white')
                    else:
                        row.append('-')
                        colors.append('white')
                
                # Annual return
                row.append(f'{annual_return:+.1f}')
                if annual_return > 0:
                    colors.append('#90EE90')
                else:
                    colors.append('#FFB6C1')
                    
                cell_text.append(row)
                cell_colors.append(colors)
            
            # Create table with improved styling and reduced spacing
            table = ax3.table(cellText=cell_text,
                            colLabels=['Year'] + months,
                            cellLoc='center',
                            loc='center',
                            cellColours=cell_colors,
                            colColours=['#4CAF50']*14)
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(0.9, 1.2)  # Reduced width and height scaling for tighter spacing
            
            # Style header row
            for i in range(14):
                cell = table[(0, i)]
                cell.set_text_props(weight='bold', color='white')
            
            ax3.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold', pad=20)
            
            # Add color legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#90EE90', label='Positive'),
                Patch(facecolor='#FFB6C1', label='Negative')
            ]
            ax3.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(0.98, 0.95), fontsize=10)
            
        except Exception as e:
            ax3.text(0.5, 0.5, f'Error creating monthly table: {e}', 
                    ha='center', va='center', fontsize=11)
    
    # 4. Returns Distribution (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    if 'pnl_pct' in trades_df.columns:
        returns = trades_df['pnl_pct'].values
        positive = returns[returns > 0]
        negative = returns[returns < 0]
        
        # Create histogram with separate colors
        n_bins = min(50, max(10, len(returns) // 10))
        ax4.hist(positive, bins=n_bins, color='green', alpha=0.6, label=f'Wins ({len(positive)})', edgecolor='black')
        ax4.hist(negative, bins=n_bins, color='red', alpha=0.6, label=f'Losses ({len(negative)})', edgecolor='black')
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax4.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Return (%)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', labelsize=10)
    
    # 5. Drawdown Chart (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])
    if 'drawdown_pct' in trades_df.columns:
        dd_values = -trades_df['drawdown_pct'].values
        ax5.fill_between(range(len(trades_df)), 0, dd_values, 
                        where=(dd_values<0), color='red', alpha=0.4, interpolate=True)
        ax5.plot(dd_values, 'r-', linewidth=2)
        ax5.set_title('Drawdown %', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Trade Number', fontsize=12)
        ax5.set_ylabel('Drawdown (%)', fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='both', labelsize=10)
        
        # Add max drawdown annotation
        max_dd = min(dd_values)
        max_dd_idx = np.argmin(dd_values)
        ax5.annotate(f'Max: {abs(max_dd):.1f}%', 
                    xy=(max_dd_idx, max_dd),
                    xytext=(max_dd_idx + len(trades_df)*0.1, max_dd),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, color='red', fontweight='bold')
    
    # 6. Hit Rate Over Time (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    if 'pnl_pct' in trades_df.columns:
        # Calculate rolling hit rate
        window = min(50, max(20, len(trades_df) // 20))
        trades_df['is_win'] = trades_df['pnl_pct'] > 0
        rolling_hit_rate = trades_df['is_win'].rolling(window=window, min_periods=10).mean() * 100
        
        ax6.plot(rolling_hit_rate.values, 'g-', linewidth=2)
        ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax6.fill_between(range(len(rolling_hit_rate)), 50, rolling_hit_rate.values,
                        where=(rolling_hit_rate.values > 50), color='green', alpha=0.2, interpolate=True)
        ax6.fill_between(range(len(rolling_hit_rate)), rolling_hit_rate.values, 50,
                        where=(rolling_hit_rate.values < 50), color='red', alpha=0.2, interpolate=True)
        ax6.set_title(f'Rolling Hit Rate ({window} trades)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Trade Number', fontsize=12)
        ax6.set_ylabel('Hit Rate (%)', fontsize=12)
        ax6.set_ylim(0, 100)
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    return fig