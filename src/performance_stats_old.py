"""
Performance Statistics Calculator for OMtree
Redirects to new implementation based on tradestats.md specifications
"""

# Import the new implementation
from .performance_stats_v2 import calculate_performance_stats, format_stats_for_display, calculate_samupi

# For backward compatibility, also import the originals in case needed
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_performance_stats_old(df, annual_trading_days=252, model_type='longonly'):
    """
    Calculate comprehensive performance statistics from results DataFrame
    
    Parameters:
    df: DataFrame with columns including 'date', 'prediction', 'target_value', 'actual_profitable'
    annual_trading_days: Number of trading days per year (default 252)
    model_type: 'longonly' or 'shortonly' for correct P&L calculation
    """
    
    stats = {}
    
    # Basic counts
    stats['total_observations'] = len(df)
    
    # Filter to trades only
    if 'prediction' in df.columns:
        trades = df[df['prediction'] == 1].copy()
        stats['total_trades'] = len(trades)
        stats['trade_frequency'] = len(trades) / len(df) * 100 if len(df) > 0 else 0
    else:
        trades = df.copy()
        stats['total_trades'] = len(trades)
        stats['trade_frequency'] = 100
    
    if len(trades) == 0:
        return stats
    
    # Calculate P&L based on model type
    if 'target_value' in trades.columns:
        if model_type == 'shortonly':
            # Shorts profit from negative returns (inverted)
            trades['pnl'] = -trades['target_value']
        else:
            # Longs profit from positive returns
            trades['pnl'] = trades['target_value']
        
        # IMPORTANT: target_value is already in percentage form (e.g., 0.664 = 0.664%)
        # Need to convert to decimal for compounding (divide by 100)
        trades['pnl_decimal'] = trades['pnl'] / 100.0
        
        # Calculate portfolio value using compound returns
        # Start with initial capital of 1 (100%)
        portfolio_value = [1.0]
        for pnl_dec in trades['pnl_decimal'].values:
            # Each return is applied to current portfolio value
            # pnl_dec is now the decimal return (e.g., 0.00664 for 0.664%)
            new_value = portfolio_value[-1] * (1 + pnl_dec)
            portfolio_value.append(new_value)
        
        # Remove the initial value since we have one extra
        portfolio_value = portfolio_value[1:]
        trades['portfolio_value'] = portfolio_value
        
        # Calculate cumulative return (percentage gain from initial)
        trades['cumulative_return'] = (trades['portfolio_value'] - 1) * 100
        
        # Keep simple cumulative P&L (sum of percentage returns)
        trades['cumulative_pnl'] = trades['pnl'].cumsum()
        
        # Basic return metrics (pnl is in percentage form)
        stats['total_return'] = trades['pnl'].sum()  # Sum of percentage returns
        stats['average_return'] = trades['pnl'].mean()  # Average percentage return per trade
        stats['return_std'] = trades['pnl'].std()  # Std of percentage returns
        
        # Win/Loss metrics
        if 'actual_profitable' in trades.columns:
            stats['hit_rate'] = trades['actual_profitable'].mean() * 100
            stats['win_count'] = trades['actual_profitable'].sum()
            stats['loss_count'] = len(trades) - stats['win_count']
        else:
            stats['hit_rate'] = (trades['pnl'] > 0).mean() * 100
            stats['win_count'] = (trades['pnl'] > 0).sum()
            stats['loss_count'] = (trades['pnl'] <= 0).sum()
        
        # Win/Loss magnitudes (in percentage terms)
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] <= 0]
        
        stats['avg_win'] = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        stats['avg_loss'] = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        stats['win_loss_ratio'] = abs(stats['avg_win'] / stats['avg_loss']) if stats['avg_loss'] != 0 else np.inf
        
        # Maximum Drawdown (using portfolio values for proper percentage calculation)
        if 'portfolio_value' in trades.columns:
            portfolio_values = trades['portfolio_value'].values
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - running_max) / running_max  # Percentage drawdown
            
            stats['max_drawdown_pct'] = drawdown.min() * 100  # Convert to percentage
            stats['max_drawdown'] = drawdown.min()  # Keep as decimal for ratios
        else:
            # Fallback to cumulative P&L if portfolio_value not available
            cumulative = trades['cumulative_pnl'].values
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            
            stats['max_drawdown'] = drawdown.min()
            stats['max_drawdown_pct'] = (drawdown.min() / running_max[np.argmin(drawdown)] * 100) if running_max[np.argmin(drawdown)] != 0 else 0
        
        # Recovery metrics
        if stats['max_drawdown'] < 0:
            dd_idx = np.argmin(drawdown)
            if dd_idx < len(drawdown) - 1:
                if 'portfolio_value' in trades.columns:
                    recovery_values = portfolio_values[dd_idx:]
                    recovery_point = running_max[dd_idx]
                else:
                    recovery_values = cumulative[dd_idx:]
                    recovery_point = running_max[dd_idx]
                recovered = np.where(recovery_values >= recovery_point)[0]
                if len(recovered) > 0:
                    stats['drawdown_recovery_trades'] = recovered[0]
                else:
                    stats['drawdown_recovery_trades'] = None
        
        # Date-based calculations
        if 'date' in trades.columns:
            try:
                trades['date'] = pd.to_datetime(trades['date'])
                date_range = (trades['date'].max() - trades['date'].min()).days
                years = date_range / 365.25
                
                if years > 0:
                    # Annualized metrics
                    stats['years_of_data'] = years
                    stats['avg_trades_per_year'] = len(trades) / years
                    
                    # Calculate annualized return using compound annual growth rate (CAGR)
                    if 'portfolio_value' in trades.columns:
                        # Use actual portfolio values for more accurate CAGR
                        initial_value = 1.0  # We start with 1
                        final_value = trades['portfolio_value'].iloc[-1]
                        if final_value > 0:
                            cagr = (final_value / initial_value) ** (1/years) - 1
                            stats['annual_return_pct'] = cagr * 100
                        else:
                            stats['annual_return_pct'] = -100.0  # Total loss
                    else:
                        # Fallback to total_return calculation
                        final_value = 1 + stats['total_return']
                        if final_value > 0:
                            cagr = (final_value ** (1/years)) - 1
                            stats['annual_return_pct'] = cagr * 100
                        else:
                            stats['annual_return_pct'] = (stats['total_return'] / years) * 100
                    
                    # Annualized Sharpe Ratio (assuming 0 risk-free rate)
                    if stats['return_std'] > 0:
                        # Use percentage returns for daily calculations
                        daily_returns = trades.set_index('date')['pnl'].resample('D').sum().fillna(0)
                        daily_std = daily_returns.std()
                        annualized_std = daily_std * np.sqrt(annual_trading_days)
                        # Convert annual return to percentage for Sharpe calculation
                        annual_return_pct = stats['annual_return_pct']
                        stats['sharpe_ratio'] = annual_return_pct / annualized_std if annualized_std > 0 else 0
                    else:
                        stats['sharpe_ratio'] = 0
                    
                    # Profit to Max DD Ratio
                    # This is the ratio of annual return percentage to max drawdown percentage
                    if stats['max_drawdown_pct'] != 0:
                        stats['profit_to_maxdd_ratio'] = abs(stats['annual_return_pct'] / stats['max_drawdown_pct'])
                    else:
                        stats['profit_to_maxdd_ratio'] = np.inf if stats['annual_return_pct'] > 0 else 0
                    
                    # Calmar Ratio (annualized return / max drawdown)
                    if stats['max_drawdown_pct'] != 0:
                        stats['calmar_ratio'] = stats['annual_return_pct'] / abs(stats['max_drawdown_pct'])
                    else:
                        stats['calmar_ratio'] = np.inf if stats['annual_return_pct'] > 0 else 0
                    
                    # Monthly statistics (summing percentage returns)
                    monthly_returns = trades.set_index('date')['pnl'].resample('ME').sum()
                    stats['best_month'] = monthly_returns.max()  # Best month in percentage
                    stats['worst_month'] = monthly_returns.min()  # Worst month in percentage
                    stats['positive_months'] = (monthly_returns > 0).sum()
                    stats['negative_months'] = (monthly_returns <= 0).sum()
                    stats['monthly_win_rate'] = stats['positive_months'] / len(monthly_returns) * 100 if len(monthly_returns) > 0 else 0
                    
            except Exception as e:
                print(f"Error calculating date-based metrics: {e}")
        
        # Risk metrics
        if stats['return_std'] > 0:
            stats['return_skewness'] = trades['pnl'].skew()
            stats['return_kurtosis'] = trades['pnl'].kurtosis()
            
            # Value at Risk (95% confidence)
            stats['var_95'] = np.percentile(trades['pnl'], 5)
            
            # Conditional Value at Risk (Expected Shortfall)
            var_threshold = stats['var_95']
            tail_losses = trades['pnl'][trades['pnl'] <= var_threshold]
            stats['cvar_95'] = tail_losses.mean() if len(tail_losses) > 0 else var_threshold
        
        # Consistency metrics
        if len(trades) > 20:
            # Rolling 20-trade hit rate
            trades['rolling_hit_rate'] = trades['actual_profitable'].rolling(20).mean() * 100 if 'actual_profitable' in trades.columns else (trades['pnl'] > 0).rolling(20).mean() * 100
            stats['hit_rate_stability'] = trades['rolling_hit_rate'].std()
        
        # Edge calculation
        base_rate = 0.5  # Assuming 50% base rate
        stats['edge'] = stats['hit_rate'] / 100 - base_rate
        stats['edge_pct'] = stats['edge'] * 100
    
    return stats

def format_stats_for_display_old(stats):
    """
    Format statistics dictionary for display in GUI
    
    Returns formatted string with sections
    """
    
    output = []
    output.append("=" * 60)
    output.append("COMPREHENSIVE PERFORMANCE STATISTICS")
    output.append("=" * 60)
    
    # Overview Section
    output.append("\nOVERVIEW")
    output.append("-" * 40)
    if 'total_observations' in stats:
        output.append(f"Total Observations: {stats['total_observations']:,}")
    if 'total_trades' in stats:
        output.append(f"Total Trades: {stats['total_trades']:,}")
    if 'trade_frequency' in stats:
        output.append(f"Trade Frequency: {stats['trade_frequency']:.1f}%")
    if 'years_of_data' in stats:
        output.append(f"Years of Data: {stats['years_of_data']:.2f}")
    if 'avg_trades_per_year' in stats:
        output.append(f"Avg Trades per Year: {stats['avg_trades_per_year']:.0f}")
    
    # Return Metrics
    output.append("\nRETURN METRICS")
    output.append("-" * 40)
    if 'total_return' in stats:
        output.append(f"Total Return: {stats['total_return']:.2f}%")
    if 'annual_return_pct' in stats:
        output.append(f"Annual Return: {stats['annual_return_pct']:.2f}%")
    if 'average_return' in stats:
        output.append(f"Average Trade Return: {stats['average_return']:.4f}%")
    if 'return_std' in stats:
        output.append(f"Return Std Dev: {stats['return_std']:.4f}%")
    
    # Win/Loss Analysis
    output.append("\nWIN/LOSS ANALYSIS")
    output.append("-" * 40)
    if 'hit_rate' in stats:
        output.append(f"Hit Rate: {stats['hit_rate']:.1f}%")
    if 'win_count' in stats and 'loss_count' in stats:
        output.append(f"Wins/Losses: {stats['win_count']}/{stats['loss_count']}")
    if 'avg_win' in stats:
        output.append(f"Average Win: {stats['avg_win']:.4f}%")
    if 'avg_loss' in stats:
        output.append(f"Average Loss: {stats['avg_loss']:.4f}%")
    if 'win_loss_ratio' in stats:
        output.append(f"Win/Loss Ratio: {stats['win_loss_ratio']:.2f}")
    if 'edge_pct' in stats:
        output.append(f"Edge: {stats['edge_pct']:+.2f}%")
    
    # Risk Metrics
    output.append("\nRISK METRICS")
    output.append("-" * 40)
    if 'max_drawdown_pct' in stats:
        output.append(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
    if 'max_drawdown' in stats and stats['max_drawdown'] != stats.get('max_drawdown_pct', 0) / 100:
        output.append(f"Max Drawdown (raw): {stats['max_drawdown']:.4f}")
    if 'drawdown_recovery_trades' in stats:
        recovery = stats['drawdown_recovery_trades']
        if recovery is not None:
            output.append(f"Drawdown Recovery: {recovery} trades")
        else:
            output.append(f"Drawdown Recovery: Not recovered")
    if 'var_95' in stats:
        output.append(f"Value at Risk (95%): {stats['var_95']:.4f}")
    if 'cvar_95' in stats:
        output.append(f"Conditional VaR (95%): {stats['cvar_95']:.4f}")
    
    # Performance Ratios
    output.append("\nPERFORMANCE RATIOS")
    output.append("-" * 40)
    if 'sharpe_ratio' in stats:
        output.append(f"Sharpe Ratio (Annualized): {stats['sharpe_ratio']:.2f}")
    if 'profit_to_maxdd_ratio' in stats:
        output.append(f"Profit to Max DD Ratio: {stats['profit_to_maxdd_ratio']:.2f}")
    if 'calmar_ratio' in stats:
        output.append(f"Calmar Ratio: {stats['calmar_ratio']:.2f}")
    
    # Distribution Metrics
    output.append("\nDISTRIBUTION METRICS")
    output.append("-" * 40)
    if 'return_skewness' in stats:
        output.append(f"Return Skewness: {stats['return_skewness']:.2f}")
    if 'return_kurtosis' in stats:
        output.append(f"Return Kurtosis: {stats['return_kurtosis']:.2f}")
    if 'hit_rate_stability' in stats:
        output.append(f"Hit Rate Stability (std): {stats['hit_rate_stability']:.2f}%")
    
    # Monthly Performance
    if 'best_month' in stats:
        output.append("\nMONTHLY PERFORMANCE")
        output.append("-" * 40)
        output.append(f"Best Month: {stats['best_month']:.2f}%")
        output.append(f"Worst Month: {stats['worst_month']:.2f}%")
        output.append(f"Positive Months: {stats['positive_months']}")
        output.append(f"Negative Months: {stats['negative_months']}")
        output.append(f"Monthly Win Rate: {stats['monthly_win_rate']:.1f}%")
    
    output.append("\n" + "=" * 60)
    
    return "\n".join(output)