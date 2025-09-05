import numpy as np
from numba import jit
from typing import Dict, Tuple


@jit(nopython=True)
def calculate_returns_vectorized(equity_curve: np.ndarray) -> np.ndarray:
    """
    Calculate returns from equity curve using vectorized operations.
    """
    if len(equity_curve) < 2:
        return np.array([0.0])
        
    returns = np.zeros(len(equity_curve))
    returns[1:] = (equity_curve[1:] - equity_curve[:-1]) / equity_curve[:-1]
    
    return returns


@jit(nopython=True)
def calculate_drawdown_vectorized(equity_curve: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Calculate drawdown series and maximum drawdown.
    
    Returns:
        (drawdown_series, max_drawdown, max_duration)
    """
    n = len(equity_curve)
    running_max = np.zeros(n)
    drawdown = np.zeros(n)
    
    # Calculate running maximum
    running_max[0] = equity_curve[0]
    for i in range(1, n):
        running_max[i] = max(running_max[i-1], equity_curve[i])
        
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = np.min(drawdown)
    
    # Calculate maximum duration
    max_duration = 0
    current_duration = 0
    
    for i in range(n):
        if drawdown[i] < 0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
            
    return drawdown, abs(max_dd), max_duration


@jit(nopython=True)
def calculate_sharpe_ratio_vectorized(returns: np.ndarray, 
                                    risk_free_rate: float = 0.0,
                                    periods_per_year: int = 252 * 78) -> float:
    """
    Calculate Sharpe ratio using vectorized operations.
    Assumes 5-minute bars (78 per day).
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns) == 0:
        return 0.0
        
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


@jit(nopython=True)
def calculate_sortino_ratio_vectorized(returns: np.ndarray,
                                     target_return: float = 0.0,
                                     periods_per_year: int = 252 * 78) -> float:
    """
    Calculate Sortino ratio using only downside deviation.
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - target_return / periods_per_year
    
    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
        
    downside_dev = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_dev == 0:
        return 0.0
        
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_dev


class PerformanceArrays:
    """Vectorized performance calculations for trading systems."""
    
    @staticmethod
    def calculate_rolling_metrics(equity_curve: np.ndarray, window: int = 252 * 78) -> Dict:
        """
        Calculate rolling performance metrics using vectorized operations.
        
        Args:
            equity_curve: Equity curve array
            window: Rolling window size (default 1 year for 5-min bars)
            
        Returns:
            Dictionary of rolling metrics arrays
        """
        n = len(equity_curve)
        returns = calculate_returns_vectorized(equity_curve)
        
        # Pre-allocate arrays
        rolling_return = np.zeros(n)
        rolling_volatility = np.zeros(n)
        rolling_sharpe = np.zeros(n)
        rolling_max_dd = np.zeros(n)
        
        # Calculate rolling metrics
        for i in range(window, n):
            window_returns = returns[i-window+1:i+1]
            window_equity = equity_curve[i-window:i+1]
            
            # Annualized return
            total_return = (equity_curve[i] - equity_curve[i-window]) / equity_curve[i-window]
            rolling_return[i] = total_return * (252 * 78) / window
            
            # Volatility
            rolling_volatility[i] = np.std(window_returns) * np.sqrt(252 * 78)
            
            # Sharpe ratio
            rolling_sharpe[i] = calculate_sharpe_ratio_vectorized(window_returns)
            
            # Maximum drawdown
            _, max_dd, _ = calculate_drawdown_vectorized(window_equity)
            rolling_max_dd[i] = max_dd
            
        return {
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_volatility,
            'rolling_sharpe': rolling_sharpe,
            'rolling_max_dd': rolling_max_dd
        }
    
    @staticmethod
    @jit(nopython=True)
    def calculate_trade_statistics_vectorized(entry_prices: np.ndarray,
                                            exit_prices: np.ndarray,
                                            position_sizes: np.ndarray) -> Dict:
        """
        Calculate trade statistics using vectorized operations.
        
        Returns:
            Dictionary of trade statistics
        """
        n_trades = len(entry_prices)
        
        if n_trades == 0:
            return {
                'n_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }
            
        # Calculate returns
        returns = (exit_prices - entry_prices) / entry_prices * position_sizes
        
        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        n_wins = len(wins)
        n_losses = len(losses)
        
        # Calculate statistics
        win_rate = n_wins / n_trades
        avg_win = np.mean(wins) if n_wins > 0 else 0.0
        avg_loss = np.mean(losses) if n_losses > 0 else 0.0
        
        # Profit factor
        total_wins = np.sum(wins) if n_wins > 0 else 0.0
        total_losses = abs(np.sum(losses)) if n_losses > 0 else 1.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Expectancy
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        return {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
    
    @staticmethod
    def calculate_parameter_heatmap(results_matrix: np.ndarray,
                                  param1_values: np.ndarray,
                                  param2_values: np.ndarray) -> Dict:
        """
        Prepare data for parameter optimization heatmap.
        
        Args:
            results_matrix: 2D array of performance metrics
            param1_values: Values for first parameter
            param2_values: Values for second parameter
            
        Returns:
            Dictionary ready for visualization
        """
        # Find optimal parameters
        best_idx = np.unravel_index(np.argmax(results_matrix), results_matrix.shape)
        best_param1 = param1_values[best_idx[0]]
        best_param2 = param2_values[best_idx[1]]
        best_value = results_matrix[best_idx]
        
        # Calculate stability (low variance around optimum)
        window = 2
        i, j = best_idx
        i_start = max(0, i - window)
        i_end = min(len(param1_values), i + window + 1)
        j_start = max(0, j - window)
        j_end = min(len(param2_values), j + window + 1)
        
        local_region = results_matrix[i_start:i_end, j_start:j_end]
        stability = 1 / (1 + np.std(local_region))
        
        return {
            'matrix': results_matrix,
            'param1_values': param1_values,
            'param2_values': param2_values,
            'optimal_param1': best_param1,
            'optimal_param2': best_param2,
            'optimal_value': best_value,
            'stability_score': stability
        }
    
    @staticmethod
    @jit(nopython=True)
    def calculate_monthly_returns(daily_returns: np.ndarray,
                                days_per_month: int = 21) -> np.ndarray:
        """
        Aggregate daily returns to monthly using vectorized operations.
        """
        n_days = len(daily_returns)
        n_months = n_days // days_per_month
        
        monthly_returns = np.zeros(n_months)
        
        for i in range(n_months):
            start = i * days_per_month
            end = min(start + days_per_month, n_days)
            
            # Compound daily returns
            month_return = 1.0
            for j in range(start, end):
                month_return *= (1 + daily_returns[j])
                
            monthly_returns[i] = month_return - 1
            
        return monthly_returns