"""
Performance calculation utilities for OMtree
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (252 for daily, 365*24 for hourly)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * mean_return / std_return

def calculate_max_drawdown(cumulative_returns: pd.Series) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown
    
    Args:
        cumulative_returns: Series of cumulative returns
    
    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx)
    """
    if len(cumulative_returns) == 0:
        return 0.0, 0, 0
    
    cummax = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cummax) / (cummax + 1e-10)
    
    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin() if not drawdown.empty else 0
    
    # Find the peak before the trough
    peak_idx = cumulative_returns[:trough_idx].idxmax() if trough_idx > 0 else 0
    
    return max_dd, peak_idx, trough_idx

def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of positive returns)
    
    Args:
        returns: Series of returns
    
    Returns:
        Win rate as a percentage
    """
    if len(returns) == 0:
        return 0.0
    
    return (returns > 0).sum() / len(returns)

def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (gross profits / gross losses)
    
    Args:
        returns: Series of returns
    
    Returns:
        Profit factor
    """
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf') if profits > 0 else 0.0
    
    return profits / losses

def calculate_performance_stats(
    returns: pd.Series,
    signals: Optional[pd.Series] = None,
    periods_per_year: int = 8760  # Hourly data default
) -> Dict:
    """
    Calculate comprehensive performance statistics
    
    Args:
        returns: Series of returns
        signals: Optional series of trade signals
        periods_per_year: Number of periods in a year
    
    Returns:
        Dictionary of performance statistics
    """
    stats = {}
    
    # Basic statistics
    stats['total_return'] = returns.sum()
    stats['mean_return'] = returns.mean()
    stats['std_return'] = returns.std()
    stats['num_periods'] = len(returns)
    
    # Risk-adjusted returns
    stats['sharpe_ratio'] = calculate_sharpe_ratio(returns, periods_per_year)
    
    # Drawdown analysis
    cumulative = returns.cumsum()
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(cumulative)
    stats['max_drawdown'] = max_dd
    stats['max_drawdown_duration'] = trough_idx - peak_idx if trough_idx > peak_idx else 0
    
    # Win/loss analysis
    stats['win_rate'] = calculate_win_rate(returns)
    stats['profit_factor'] = calculate_profit_factor(returns)
    
    # Trade analysis if signals provided
    if signals is not None:
        trade_returns = returns[signals == 1]
        stats['num_trades'] = signals.sum()
        stats['trade_rate'] = signals.mean()
        
        if len(trade_returns) > 0:
            stats['avg_trade_return'] = trade_returns.mean()
            stats['trade_win_rate'] = calculate_win_rate(trade_returns)
            stats['trade_profit_factor'] = calculate_profit_factor(trade_returns)
        else:
            stats['avg_trade_return'] = 0.0
            stats['trade_win_rate'] = 0.0
            stats['trade_profit_factor'] = 0.0
    
    # Additional metrics
    stats['skewness'] = returns.skew()
    stats['kurtosis'] = returns.kurtosis()
    stats['var_95'] = returns.quantile(0.05)  # Value at Risk (95%)
    stats['cvar_95'] = returns[returns <= stats['var_95']].mean()  # Conditional VaR
    
    return stats

def format_stats_for_display(stats: Dict) -> str:
    """
    Format performance statistics for display
    
    Args:
        stats: Dictionary of statistics
    
    Returns:
        Formatted string for display
    """
    lines = []
    lines.append("Performance Statistics")
    lines.append("=" * 40)
    
    # Returns
    lines.append(f"Total Return: {stats.get('total_return', 0):.4f}")
    lines.append(f"Mean Return: {stats.get('mean_return', 0):.6f}")
    lines.append(f"Std Return: {stats.get('std_return', 0):.6f}")
    
    # Risk metrics
    lines.append(f"\nRisk Metrics:")
    lines.append(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.3f}")
    lines.append(f"Max Drawdown: {stats.get('max_drawdown', 0):.2%}")
    lines.append(f"VaR (95%): {stats.get('var_95', 0):.4f}")
    lines.append(f"CVaR (95%): {stats.get('cvar_95', 0):.4f}")
    
    # Trade metrics
    if 'num_trades' in stats:
        lines.append(f"\nTrade Metrics:")
        lines.append(f"Total Trades: {stats.get('num_trades', 0)}")
        lines.append(f"Trade Rate: {stats.get('trade_rate', 0):.1%}")
        lines.append(f"Avg Trade Return: {stats.get('avg_trade_return', 0):.6f}")
        lines.append(f"Trade Win Rate: {stats.get('trade_win_rate', 0):.1%}")
        lines.append(f"Profit Factor: {stats.get('trade_profit_factor', 0):.2f}")
    
    # Win/loss
    lines.append(f"\nWin/Loss Analysis:")
    lines.append(f"Win Rate: {stats.get('win_rate', 0):.1%}")
    lines.append(f"Profit Factor: {stats.get('profit_factor', 0):.2f}")
    
    return "\n".join(lines)

def calculate_rolling_stats(
    returns: pd.Series,
    window: int = 100,
    min_periods: int = 30
) -> pd.DataFrame:
    """
    Calculate rolling performance statistics
    
    Args:
        returns: Series of returns
        window: Rolling window size
        min_periods: Minimum periods required for calculation
    
    Returns:
        DataFrame with rolling statistics
    """
    rolling_stats = pd.DataFrame(index=returns.index)
    
    # Rolling returns
    rolling_stats['rolling_return'] = returns.rolling(window, min_periods=min_periods).sum()
    rolling_stats['rolling_mean'] = returns.rolling(window, min_periods=min_periods).mean()
    rolling_stats['rolling_std'] = returns.rolling(window, min_periods=min_periods).std()
    
    # Rolling Sharpe (simplified)
    rolling_stats['rolling_sharpe'] = (
        rolling_stats['rolling_mean'] / rolling_stats['rolling_std']
    ) * np.sqrt(window)
    
    # Rolling win rate
    rolling_stats['rolling_win_rate'] = (
        (returns > 0).rolling(window, min_periods=min_periods).mean()
    )
    
    return rolling_stats