"""
Ulcer Performance Index (UPI) Calculator with Array Processing

Provides vectorized UPI calculations with adjustable lookback periods.
UPI measures risk-adjusted returns using drawdown-based risk instead of volatility.

UPI = Annualized Return / Ulcer Index
UPI_adj = UPI * sqrt(lookback_period)

The lookback can be based on either:
- Number of trading days 
- Number of trades
The function uses whichever provides the longer time period.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class UPICalculator:
    """Vectorized Ulcer Performance Index calculator"""
    
    @staticmethod
    def calculate_upi_arrays(
        equity_curve: np.ndarray,
        timestamps: np.ndarray,
        trade_indices: Optional[np.ndarray] = None,
        lookback_period: int = 30,
        trading_days_per_year: int = 252
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate UPI and UPI_adj arrays over time using rolling lookbacks.
        
        Args:
            equity_curve: Array of portfolio equity values over time
            timestamps: Array of timestamps (nanoseconds) corresponding to equity values
            trade_indices: Array of bar indices where trades occurred (optional)
            lookback_period: Lookback period (will use longer of N trades or N trading days)
            trading_days_per_year: Number of trading days per year for annualization
            
        Returns:
            Tuple of (UPI array, UPI_adj array)
        """
        n_bars = len(equity_curve)
        
        if n_bars < lookback_period:
            logger.warning(f"Insufficient data: {n_bars} bars < {lookback_period} lookback")
            return np.full(n_bars, np.nan), np.full(n_bars, np.nan)
        
        # Convert timestamps to days for trading day calculations
        timestamps_days = UPICalculator._timestamps_to_trading_days(timestamps)
        
        # Initialize output arrays
        upi_array = np.full(n_bars, np.nan)
        upi_adj_array = np.full(n_bars, np.nan)
        
        # Calculate UPI for each point using rolling window
        for i in range(lookback_period - 1, n_bars):
            try:
                # Determine actual lookback window (longer of trades or trading days)
                lookback_bars = UPICalculator._determine_lookback_bars(
                    i, lookback_period, timestamps_days, trade_indices
                )
                
                if lookback_bars < 2:  # Need at least 2 points
                    continue
                    
                start_idx = max(0, i - lookback_bars + 1)
                
                # Extract lookback window data
                window_equity = equity_curve[start_idx:i+1]
                window_timestamps = timestamps_days[start_idx:i+1]
                
                # Calculate UPI for this window
                upi_val, upi_adj_val = UPICalculator._calculate_window_upi(
                    window_equity, window_timestamps, lookback_period, 
                    trading_days_per_year
                )
                
                upi_array[i] = upi_val
                upi_adj_array[i] = upi_adj_val
                
            except Exception as e:
                logger.debug(f"Error calculating UPI at index {i}: {e}")
                continue
        
        return upi_array, upi_adj_array
    
    @staticmethod
    def _timestamps_to_trading_days(timestamps: np.ndarray) -> np.ndarray:
        """Convert nanosecond timestamps to trading day numbers"""
        # Convert to days since epoch
        days_since_epoch = timestamps / (1e9 * 60 * 60 * 24)
        
        # Convert to pandas datetime for business day handling
        timestamps_sec = timestamps / 1e9
        dates = pd.to_datetime(timestamps_sec, unit='s')
        
        # Map to business days (approximation for trading days)
        business_days = np.array([d.toordinal() for d in dates])
        return business_days
    
    @staticmethod
    def _determine_lookback_bars(
        current_idx: int, 
        lookback_period: int,
        timestamps_days: np.ndarray,
        trade_indices: Optional[np.ndarray]
    ) -> int:
        """
        Determine actual lookback window in bars.
        Uses the longer of N trades or N trading days.
        """
        # Option 1: Lookback by trading days
        current_day = timestamps_days[current_idx]
        target_day = current_day - lookback_period
        
        # Find bars that are at least lookback_period trading days back
        day_mask = timestamps_days >= target_day
        day_lookback_bars = np.sum(day_mask[:current_idx + 1])
        
        # Option 2: Lookback by number of trades (if trade data available)
        trade_lookback_bars = lookback_period  # Default fallback
        
        if trade_indices is not None and len(trade_indices) > 0:
            # Find trades up to current index
            trades_up_to_current = trade_indices[trade_indices <= current_idx]
            
            if len(trades_up_to_current) >= lookback_period:
                # Find the bar index of the Nth trade back
                nth_trade_back_idx = trades_up_to_current[-(lookback_period)]
                trade_lookback_bars = current_idx - nth_trade_back_idx + 1
        
        # Use the longer of the two lookback periods
        return max(day_lookback_bars, trade_lookback_bars)
    
    @staticmethod
    def _calculate_window_upi(
        equity_window: np.ndarray,
        timestamps_days: np.ndarray,
        lookback_period: int,
        trading_days_per_year: int
    ) -> Tuple[float, float]:
        """
        Calculate UPI and UPI_adj for a specific window.
        
        Returns:
            Tuple of (UPI, UPI_adj)
        """
        if len(equity_window) < 2:
            return np.nan, np.nan
        
        # Calculate period length in years for annualization
        period_days = timestamps_days[-1] - timestamps_days[0] + 1
        period_years = max(period_days / trading_days_per_year, 1/trading_days_per_year)  # Min 1 day
        
        # Calculate total return
        start_equity = equity_window[0]
        end_equity = equity_window[-1]
        
        if start_equity <= 0:
            return np.nan, np.nan
            
        total_return = (end_equity / start_equity) - 1.0
        
        # Annualize the return
        if period_years > 0:
            annualized_return = (1 + total_return) ** (1 / period_years) - 1
        else:
            annualized_return = total_return
        
        # Calculate Ulcer Index (drawdown-based risk measure)
        ulcer_index = UPICalculator._calculate_ulcer_index(equity_window)
        
        if ulcer_index <= 0:
            return np.nan, np.nan
        
        # Calculate UPI
        upi = annualized_return / ulcer_index
        
        # Calculate UPI_adj 
        upi_adj = upi * np.sqrt(lookback_period)
        
        return upi, upi_adj
    
    @staticmethod
    def _calculate_ulcer_index(equity_curve: np.ndarray) -> float:
        """
        Calculate Ulcer Index - measures depth and duration of drawdowns.
        
        Ulcer Index = sqrt(mean(drawdown_squared))
        where drawdown is the percentage decline from peak
        """
        if len(equity_curve) < 2:
            return np.nan
        
        # Calculate running maximum (peaks)
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdowns as percentage decline from peak
        drawdowns = (equity_curve - running_max) / running_max
        
        # Square the drawdowns
        drawdowns_squared = drawdowns ** 2
        
        # Calculate Ulcer Index
        mean_drawdown_squared = np.mean(drawdowns_squared)
        ulcer_index = np.sqrt(mean_drawdown_squared)
        
        return ulcer_index
    
    @staticmethod
    def calculate_upi_metrics(
        equity_curve: np.ndarray,
        timestamps: np.ndarray,
        trade_indices: Optional[np.ndarray] = None,
        trading_days_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Calculate UPI metrics for performance summary.
        
        Returns:
            Dictionary with UPI_30, UPI_50, UPI_30_max, UPI_50_max
        """
        metrics = {}
        
        # Calculate UPI arrays for both 30 and 50 period lookbacks
        for period in [30, 50]:
            try:
                upi_array, upi_adj_array = UPICalculator.calculate_upi_arrays(
                    equity_curve, timestamps, trade_indices, period, trading_days_per_year
                )
                
                # Get final values (endpoint)
                final_upi = upi_array[-1] if len(upi_array) > 0 else np.nan
                final_upi_adj = upi_adj_array[-1] if len(upi_adj_array) > 0 else np.nan
                
                # Get maximum values over the entire period
                max_upi = np.nanmax(upi_array) if len(upi_array) > 0 else np.nan
                max_upi_adj = np.nanmax(upi_adj_array) if len(upi_adj_array) > 0 else np.nan
                
                # Store in metrics dictionary
                metrics[f'UPI_{period}'] = final_upi
                metrics[f'UPI_{period}_adj'] = final_upi_adj
                metrics[f'UPI_{period}_max'] = max_upi
                metrics[f'UPI_{period}_adj_max'] = max_upi_adj
                
                logger.info(f"UPI_{period}: {final_upi:.4f}, UPI_{period}_max: {max_upi:.4f}")
                
            except Exception as e:
                logger.warning(f"Error calculating UPI_{period}: {e}")
                metrics[f'UPI_{period}'] = np.nan
                metrics[f'UPI_{period}_adj'] = np.nan
                metrics[f'UPI_{period}_max'] = np.nan
                metrics[f'UPI_{period}_adj_max'] = np.nan
        
        return metrics


# Convenience functions for easy access
def calculate_upi_30_50(
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
    trade_indices: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Convenience function to calculate UPI_30 and UPI_50 metrics.
    
    Args:
        equity_curve: Portfolio equity values over time
        timestamps: Corresponding timestamps (nanoseconds)
        trade_indices: Optional array of trade bar indices
        
    Returns:
        Dictionary with UPI_30, UPI_50, UPI_30_max, UPI_50_max
    """
    return UPICalculator.calculate_upi_metrics(
        equity_curve, timestamps, trade_indices
    )


def get_upi_arrays(
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
    lookback_period: int = 30,
    trade_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get UPI and UPI_adj arrays for plotting or analysis.
    
    Args:
        equity_curve: Portfolio equity values
        timestamps: Corresponding timestamps  
        lookback_period: Lookback period for calculation
        trade_indices: Optional trade indices
        
    Returns:
        Tuple of (UPI array, UPI_adj array)
    """
    return UPICalculator.calculate_upi_arrays(
        equity_curve, timestamps, trade_indices, lookback_period
    )