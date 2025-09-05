"""
Calculate all statistics as specified in tradestats.md
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def calculate_tradestats_metrics(trades_df: pd.DataFrame, initial_equity: float = 1000.0) -> Dict[str, Any]:
    """
    Calculate all metrics specified in tradestats.md
    
    Parameters:
    trades_df: DataFrame with columns including 'date', 'pnl_pct', 'pnl_decimal', etc.
    initial_equity: Starting equity value (default 1000)
    
    Returns:
    Dictionary containing all calculated metrics
    """
    metrics = {}
    
    # Ensure we have the necessary columns
    if 'date' in trades_df.columns:
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        # Create date range for all days (not just trading days)
        date_range = pd.date_range(start=trades_df['date'].min(), 
                                  end=trades_df['date'].max(), freq='D')
        
        # Create daily returns series (0 for non-trading days)
        daily_df = pd.DataFrame(index=date_range)
        daily_df['return'] = 0.0
        daily_df['is_trade'] = False
        
        # Fill in actual trade returns
        for _, trade in trades_df.iterrows():
            if trade['date'] in daily_df.index:
                # Convert percentage to decimal if needed
                if 'pnl_decimal' in trade:
                    daily_df.loc[trade['date'], 'return'] = trade['pnl_decimal']
                elif 'pnl_pct' in trade:
                    # Assume pnl_pct is in percentage form (e.g., 1.0 = 1%)
                    daily_df.loc[trade['date'], 'return'] = trade['pnl_pct'] / 100.0
                daily_df.loc[trade['date'], 'is_trade'] = True
    else:
        # If no date column, assume each row is a trading day
        daily_df = pd.DataFrame()
        daily_df['return'] = trades_df.get('pnl_decimal', trades_df.get('pnl_pct', 0) / 100.0)
        daily_df['is_trade'] = daily_df['return'] != 0
    
    # ===== COUNT METRICS =====
    metrics['num_observations'] = len(daily_df)  # Total days in period
    metrics['years_of_data'] = metrics['num_observations'] / 252.0
    metrics['num_trades'] = daily_df['is_trade'].sum()
    metrics['trade_frequency_pct'] = (metrics['num_trades'] / metrics['num_observations']) * 100 if metrics['num_observations'] > 0 else 0
    metrics['avg_trades_pa'] = metrics['num_trades'] / metrics['years_of_data'] if metrics['years_of_data'] > 0 else 0
    metrics['avg_trades_pm'] = metrics['avg_trades_pa'] / 12.0
    
    # ===== TRADE METRICS =====
    trade_returns = daily_df[daily_df['is_trade']]['return']
    
    if len(trade_returns) > 0:
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        
        metrics['win_pct'] = (len(winning_trades) / len(trade_returns)) * 100
        metrics['avg_loss_pct'] = losing_trades.mean() * 100 if len(losing_trades) > 0 else 0
        metrics['avg_profit_pct'] = winning_trades.mean() * 100 if len(winning_trades) > 0 else 0
        metrics['avg_pnl_pct'] = trade_returns.mean() * 100
        
        # Expectancy = (Win % * Ave Profit %) - ((1 - Win %) * abs(Ave Loss %))
        win_rate = metrics['win_pct'] / 100
        metrics['expectancy'] = (win_rate * metrics['avg_profit_pct']) - ((1 - win_rate) * abs(metrics['avg_loss_pct']))
        
        metrics['best_day_pct'] = trade_returns.max() * 100
        metrics['worst_day_pct'] = trade_returns.min() * 100
    else:
        metrics.update({
            'win_pct': 0, 'avg_loss_pct': 0, 'avg_profit_pct': 0,
            'avg_pnl_pct': 0, 'expectancy': 0, 'best_day_pct': 0, 'worst_day_pct': 0
        })
    
    # ===== MODEL METRICS (using compound equity curve) =====
    # Calculate compound equity curve
    equity_curve = [initial_equity]
    for ret in daily_df['return'].values:
        new_equity = equity_curve[-1] * (1 + ret)
        equity_curve.append(new_equity)
    equity_curve = equity_curve[1:]  # Remove initial value
    
    # Annual return: (Ending Value / Beginning Value)^(1 / Number of Years) - 1
    if metrics['years_of_data'] > 0 and len(equity_curve) > 0:
        ending_value = equity_curve[-1]
        beginning_value = initial_equity
        metrics['avg_annual_pct'] = ((ending_value / beginning_value) ** (1 / metrics['years_of_data']) - 1) * 100
    else:
        metrics['avg_annual_pct'] = 0
    
    # Maximum Drawdown
    peak = initial_equity
    max_dd_pct = 0
    drawdowns = []
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd_pct = ((peak - equity) / peak) * 100
        drawdowns.append(dd_pct)
        max_dd_pct = max(max_dd_pct, dd_pct)
    
    metrics['max_draw_pct'] = max_dd_pct
    
    # Sharpe Ratio: (Average daily return * 252) / (standard deviation daily returns * sqrt(252))
    if len(daily_df) > 1:
        daily_returns = daily_df['return']
        avg_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        if std_daily_return > 0:
            metrics['sharpe'] = (avg_daily_return * 252) / (std_daily_return * np.sqrt(252))
        else:
            metrics['sharpe'] = 0
    else:
        metrics['sharpe'] = 0
    
    # Profit to DD Ratio: Ave Annual % / Max Draw %
    if metrics['max_draw_pct'] > 0:
        metrics['profit_dd_ratio'] = metrics['avg_annual_pct'] / metrics['max_draw_pct']
    else:
        metrics['profit_dd_ratio'] = 0 if metrics['avg_annual_pct'] <= 0 else float('inf')
    
    # UPI: Ave Annual % / √(Mean(((Peak - Value) / Peak)² × 100²))
    # Calculate the denominator: root mean square of drawdowns
    if len(drawdowns) > 0:
        # Drawdowns are already in percentage form
        rms_dd = np.sqrt(np.mean([dd**2 for dd in drawdowns]))
        if rms_dd > 0:
            metrics['upi'] = metrics['avg_annual_pct'] / rms_dd
        else:
            metrics['upi'] = 0
    else:
        metrics['upi'] = 0
    
    # Store equity curve and drawdowns for charts
    metrics['equity_curve'] = equity_curve
    metrics['drawdowns'] = drawdowns
    metrics['daily_df'] = daily_df
    
    # Format all numeric values to reasonable precision
    for key, value in metrics.items():
        if isinstance(value, (float, np.float64, np.float32)):
            if 'pct' in key or key in ['sharpe', 'profit_dd_ratio', 'upi', 'expectancy']:
                metrics[key] = round(value, 2)
            elif key in ['years_of_data']:
                metrics[key] = round(value, 3)
            elif key in ['avg_trades_pa', 'avg_trades_pm']:
                metrics[key] = round(value, 1)
            else:
                metrics[key] = round(value, 4)
    
    return metrics


def format_tradestats_for_display(metrics: Dict[str, Any]) -> str:
    """
    Format tradestats metrics for text display
    
    Parameters:
    metrics: Dictionary of calculated metrics
    
    Returns:
    Formatted string for display
    """
    output = []
    output.append("=" * 50)
    output.append("TRADESTATS PERFORMANCE METRICS")
    output.append("=" * 50)
    output.append("")
    
    # Count Section
    output.append("COUNT METRICS")
    output.append("-" * 30)
    output.append(f"# Observations:      {metrics.get('num_observations', 0)}")
    output.append(f"Years of Data:       {metrics.get('years_of_data', 0):.3f}")
    output.append(f"# Trades:            {metrics.get('num_trades', 0)}")
    output.append(f"Trade Frequency:     {metrics.get('trade_frequency_pct', 0):.2f}")
    output.append(f"Avg Trades P.A.:     {metrics.get('avg_trades_pa', 0):.1f}")
    output.append(f"Avg Trades P.M.:     {metrics.get('avg_trades_pm', 0):.1f}")
    output.append("")
    
    # Trade Metrics
    output.append("TRADE METRICS")
    output.append("-" * 30)
    output.append(f"Win %:               {metrics.get('win_pct', 0):.2f}")
    output.append(f"Avg Loss:            {metrics.get('avg_loss_pct', 0):.2f}")
    output.append(f"Avg Profit:          {metrics.get('avg_profit_pct', 0):.2f}")
    output.append(f"Avg PnL:             {metrics.get('avg_pnl_pct', 0):.2f}")
    output.append(f"Expectancy:          {metrics.get('expectancy', 0):.2f}")
    output.append(f"Best Day:            {metrics.get('best_day_pct', 0):.2f}")
    output.append(f"Worst Day:           {metrics.get('worst_day_pct', 0):.2f}")
    output.append("")
    
    # Model Metrics
    output.append("MODEL METRICS")
    output.append("-" * 30)
    output.append(f"Avg Annual:          {metrics.get('avg_annual_pct', 0):.2f}")
    output.append(f"Max Draw:            {metrics.get('max_draw_pct', 0):.2f}")
    output.append(f"Sharpe:              {metrics.get('sharpe', 0):.3f}")
    output.append(f"Profit/DD Ratio:     {metrics.get('profit_dd_ratio', 0):.2f}")
    output.append(f"UPI:                 {metrics.get('upi', 0):.3f}")
    output.append("")
    
    return "\n".join(output)