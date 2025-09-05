"""
Performance Statistics Calculator for OMtree - Version 2
Based on tradestats.md specifications
Calculates trading metrics with proper handling of percentage returns
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_performance_stats(df, annual_trading_days=252, model_type='longonly'):
    """
    Calculate comprehensive performance statistics from results DataFrame
    Based on tradestats.md specifications
    
    Parameters:
    df: DataFrame with columns including 'date', 'prediction', 'target_value', 'actual_profitable'
    annual_trading_days: Number of trading days per year (default 252)
    model_type: 'longonly' or 'shortonly' for correct P&L calculation
    
    IMPORTANT: Assumes target_value is in percentage form (e.g., 0.664 = 0.664%)
    """
    
    stats = {}
    
    # === COUNT METRICS ===
    # Total observations (days) in the out-of-sample walk forward period
    stats['total_observations'] = len(df)
    
    # Filter to trades only
    if 'prediction' in df.columns:
        trades = df[df['prediction'] == 1].copy()
        stats['total_trades'] = len(trades)
    else:
        trades = df.copy()
        stats['total_trades'] = len(trades)
    
    # Years of data (observations / 252)
    stats['years_of_data'] = stats['total_observations'] / annual_trading_days
    
    # Trade frequency
    stats['trade_frequency_pct'] = (stats['total_trades'] / stats['total_observations'] * 100) if stats['total_observations'] > 0 else 0
    
    # Average trades per annum
    stats['avg_trades_pa'] = stats['total_trades'] / stats['years_of_data'] if stats['years_of_data'] > 0 else 0
    
    # Average trades per month
    stats['avg_trades_pm'] = stats['avg_trades_pa'] / 12
    
    if len(trades) == 0:
        return stats
    
    # Calculate P&L based on model type
    if 'target_value' in trades.columns:
        # CRITICAL: target_value is already in percentage form
        # A value of 0.664 means 0.664%, NOT 66.4%
        if model_type == 'shortonly':
            trades['pnl_pct'] = -trades['target_value']  # Shorts profit from negative returns
        else:
            trades['pnl_pct'] = trades['target_value']  # Longs profit from positive returns
        
        # Convert percentage to decimal for calculations (divide by 100)
        trades['pnl_decimal'] = trades['pnl_pct'] / 100.0
        
        # === TRADE METRICS ===
        # Win rate
        if 'actual_profitable' in trades.columns:
            stats['win_pct'] = trades['actual_profitable'].mean() * 100
        else:
            stats['win_pct'] = (trades['pnl_pct'] > 0).mean() * 100
        
        # Separate winners and losers
        winning_trades = trades[trades['pnl_pct'] > 0]
        losing_trades = trades[trades['pnl_pct'] < 0]
        
        # Average profit % (for winning trades)
        stats['avg_profit_pct'] = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        
        # Average loss % (for losing trades)
        stats['avg_loss_pct'] = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        # Average P&L % (all trades)
        stats['avg_pnl_pct'] = trades['pnl_pct'].mean()
        
        # Expectancy calculation
        # (Win % * Ave Profit %) - ((1 - Win %) * abs(Ave Loss %))
        win_rate_decimal = stats['win_pct'] / 100
        stats['expectancy'] = (win_rate_decimal * stats['avg_profit_pct']) - ((1 - win_rate_decimal) * abs(stats['avg_loss_pct']))
        
        # Best and worst days
        stats['best_day_pct'] = trades['pnl_pct'].max()
        stats['worst_day_pct'] = trades['pnl_pct'].min()
        
        # === EQUITY CURVE CALCULATIONS ===
        # Build compound equity curve starting at $1000
        # Equity[t] = Equity[t-1] × (1 + Return[t])
        initial_equity = 1000
        equity_curve = [initial_equity]
        
        for ret_decimal in trades['pnl_decimal'].values:
            new_equity = equity_curve[-1] * (1 + ret_decimal)
            equity_curve.append(new_equity)
        
        # Remove initial value
        equity_curve = equity_curve[1:]
        trades['equity'] = equity_curve
        
        # === MODEL METRICS (using compound equity curve) ===
        # Average Annual % (CAGR)
        # (Ending Value / Beginning Value)^(1 / Number of Years) - 1
        ending_value = equity_curve[-1]
        beginning_value = initial_equity
        
        if stats['years_of_data'] > 0 and ending_value > 0:
            stats['avg_annual_pct'] = ((ending_value / beginning_value) ** (1 / stats['years_of_data']) - 1) * 100
        else:
            stats['avg_annual_pct'] = 0
        
        # Maximum Drawdown %
        # (Peak Value - Current Value) / Peak Value × 100
        peak = initial_equity
        max_dd_pct = 0
        drawdowns = []
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd_pct = ((peak - equity) / peak) * 100
            drawdowns.append(dd_pct)
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
        
        stats['max_draw_pct'] = max_dd_pct
        trades['drawdown_pct'] = drawdowns
        
        # Profit DD Ratio (Ave Annual % / Max Draw %)
        if stats['max_draw_pct'] > 0:
            stats['profit_dd_ratio'] = stats['avg_annual_pct'] / stats['max_draw_pct']
        else:
            stats['profit_dd_ratio'] = 0
        
        # Sharpe Ratio
        # (Average daily return * 252) / (standard deviation daily returns * sqrt(252))
        # Using percentage returns
        daily_returns_pct = trades['pnl_pct'].values
        avg_daily_return = np.mean(daily_returns_pct)
        std_daily_return = np.std(daily_returns_pct)
        
        if std_daily_return > 0:
            annualized_return = avg_daily_return * annual_trading_days
            annualized_std = std_daily_return * np.sqrt(annual_trading_days)
            stats['sharpe'] = annualized_return / annualized_std
        else:
            stats['sharpe'] = 0
        
        # UPI (Ulcer Performance Index)
        # Ave Annual %^ / √(Mean(((Peak - Value) / Peak)² × 100²))
        # Where Peak is the highest value up to that point
        if len(drawdowns) > 0 and max(drawdowns) > 0:
            # Calculate ulcer index (root mean square of drawdowns)
            dd_squared = [(dd/100) ** 2 for dd in drawdowns]  # Convert to decimal for calculation
            ulcer_index = np.sqrt(np.mean(dd_squared)) * 100  # Convert back to percentage
            
            if ulcer_index > 0:
                stats['upi'] = stats['avg_annual_pct'] / ulcer_index
            else:
                stats['upi'] = 0
        else:
            stats['upi'] = 0
        
        # === DATE-BASED CALCULATIONS ===
        if 'date' in trades.columns:
            try:
                trades['date'] = pd.to_datetime(trades['date'])
                
                # Monthly returns calculation
                # Monthly returns = exp(MonthlySum(ln(1 + returns))) - 1
                trades['ln_1_plus_ret'] = np.log(1 + trades['pnl_decimal'])
                monthly_ln_sums = trades.set_index('date')['ln_1_plus_ret'].resample('ME').sum()
                monthly_returns_pct = (np.exp(monthly_ln_sums) - 1) * 100
                
                stats['best_month_pct'] = monthly_returns_pct.max()
                stats['worst_month_pct'] = monthly_returns_pct.min()
                stats['positive_months'] = (monthly_returns_pct > 0).sum()
                stats['negative_months'] = (monthly_returns_pct <= 0).sum()
                stats['monthly_win_rate_pct'] = (stats['positive_months'] / len(monthly_returns_pct) * 100) if len(monthly_returns_pct) > 0 else 0
                
            except Exception as e:
                print(f"Error calculating date-based metrics: {e}")
        
        # Store additional data for charts
        stats['equity_curve'] = equity_curve
        stats['trades_df'] = trades
        
    return stats


def format_stats_for_display(stats):
    """
    Format statistics dictionary for display according to tradestats.md
    
    Returns formatted string with sections
    """
    
    output = []
    output.append("=" * 60)
    output.append("PERFORMANCE STATISTICS (tradestats.md spec)")
    output.append("=" * 60)
    
    # Count Section
    output.append("\nCOUNT")
    output.append("-" * 40)
    if 'total_observations' in stats:
        output.append(f"# Observations: {stats['total_observations']:,}")
    if 'years_of_data' in stats:
        output.append(f"Years of Data: {stats['years_of_data']:.2f}")
    if 'total_trades' in stats:
        output.append(f"# Trades: {stats['total_trades']:,}")
    if 'trade_frequency_pct' in stats:
        output.append(f"Trade Frequency %: {stats['trade_frequency_pct']:.2f}%")
    if 'avg_trades_pa' in stats:
        output.append(f"Ave Trades P.A.: {stats['avg_trades_pa']:.0f}")
    if 'avg_trades_pm' in stats:
        output.append(f"Ave Trades P.M.: {stats['avg_trades_pm']:.1f}")
    
    # Trades Section
    output.append("\nTRADES")
    output.append("-" * 40)
    if 'win_pct' in stats:
        output.append(f"Win %: {stats['win_pct']:.1f}%")
    if 'avg_loss_pct' in stats:
        output.append(f"Ave Loss %: {stats['avg_loss_pct']:.3f}%")
    if 'avg_profit_pct' in stats:
        output.append(f"Ave Profit %: {stats['avg_profit_pct']:.3f}%")
    if 'avg_pnl_pct' in stats:
        output.append(f"Ave PnL %: {stats['avg_pnl_pct']:.3f}%")
    if 'expectancy' in stats:
        output.append(f"Expectancy: {stats['expectancy']:.3f}%")
    if 'best_day_pct' in stats:
        output.append(f"Best Day %: {stats['best_day_pct']:.3f}%")
    if 'worst_day_pct' in stats:
        output.append(f"Worst Day %: {stats['worst_day_pct']:.3f}%")
    
    # Model Section
    output.append("\nMODEL")
    output.append("-" * 40)
    if 'avg_annual_pct' in stats:
        output.append(f"Ave Annual %: {stats['avg_annual_pct']:.2f}%")
    if 'max_draw_pct' in stats:
        output.append(f"Max Draw %: {stats['max_draw_pct']:.2f}%")
    if 'sharpe' in stats:
        output.append(f"Sharpe: {stats['sharpe']:.3f}")
    if 'profit_dd_ratio' in stats:
        output.append(f"Profit DD Ratio: {stats['profit_dd_ratio']:.3f}")
    if 'upi' in stats:
        output.append(f"UPI: {stats['upi']:.3f}")
    
    # Monthly Performance (if available)
    if 'best_month_pct' in stats:
        output.append("\nMONTHLY PERFORMANCE")
        output.append("-" * 40)
        output.append(f"Best Month %: {stats['best_month_pct']:.2f}%")
        output.append(f"Worst Month %: {stats['worst_month_pct']:.2f}%")
        output.append(f"Positive Months: {stats.get('positive_months', 0)}")
        output.append(f"Negative Months: {stats.get('negative_months', 0)}")
        output.append(f"Monthly Win Rate %: {stats.get('monthly_win_rate_pct', 0):.1f}%")
    
    output.append("\n" + "=" * 60)
    output.append("Note: Returns are in percentage form (0.664 = 0.664%)")
    output.append("Equity curve starts at $1000")
    output.append("=" * 60)
    
    return "\n".join(output)


def calculate_samupi(returns_array, lookback=50):
    """
    Calculate samUPI - a trade-count based version of UPI
    Based on AFL code from tradestats.md
    
    Parameters:
    returns_array: Array of returns (non-zero values represent trades)
    lookback: Number of trades to look back
    
    Returns: samUPI value
    """
    # Filter to only trades (non-zero returns)
    trade_returns = returns_array[returns_array != 0]
    
    if len(trade_returns) < lookback:
        return 0
    
    samupi_values = []
    
    for i in range(lookback, len(trade_returns) + 1):
        window = trade_returns[i-lookback:i]
        
        # Calculate cumulative sum and drawdowns
        sum_rtn = 0
        max_sum_rtn = -1000
        sum_dd_sqd = 0
        
        for ret in window:
            sum_rtn += ret
            max_sum_rtn = max(sum_rtn, max_sum_rtn)
            dd = sum_rtn - max_sum_rtn
            sum_dd_sqd += dd ** 2
        
        # Calculate UPI
        ulcer = np.sqrt(sum_dd_sqd) + 0.00001
        upi = sum_rtn / ulcer
        upi = upi * lookback  # Normalize by lookback
        
        samupi_values.append(upi)
    
    return samupi_values