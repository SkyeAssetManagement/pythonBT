"""
Centralized timezone handling for consistent EST display across all components
"""

import pandas as pd
from datetime import datetime
from typing import Union
import numpy as np


class TimezoneHandler:
    """Centralized timezone conversion for consistent EST display"""
    
    @staticmethod
    def timestamp_to_est_string(timestamp: Union[int, float, np.int64]) -> str:
        """
        Convert any timestamp to EST time string in 'YYYY-MM-DD HH:MM:SS' format.
        
        Args:
            timestamp: Timestamp in nanoseconds, microseconds, milliseconds, or seconds
            
        Returns:
            Formatted timestamp string in EST timezone
        """
        try:
            # Convert to float for consistent handling
            timestamp_val = float(timestamp)
            
            # Detect timestamp format and convert to seconds
            if timestamp_val > 1e18:  # Nanoseconds (19+ digits)
                timestamp_seconds = timestamp_val / 1e9
            elif timestamp_val > 1e15:  # Microseconds (16-18 digits)
                timestamp_seconds = timestamp_val / 1e6
            elif timestamp_val > 1e12:  # Milliseconds (13-15 digits)
                timestamp_seconds = timestamp_val / 1e3
            elif timestamp_val > 1e9:   # Already in seconds (10-12 digits)
                timestamp_seconds = timestamp_val
            else:
                # Very small values - treat as invalid
                raise ValueError(f"Timestamp too small: {timestamp_val}")
            
            # Convert UTC seconds to EST by adding 5 hours (EST = UTC-5, so to display EST we add 5)
            est_timestamp_seconds = timestamp_seconds + (5 * 3600)  # Add 5 hours for EST display
            
            # Create datetime object using UTC to avoid system timezone interference
            dt = datetime.utcfromtimestamp(est_timestamp_seconds)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
            
        except (ValueError, OverflowError, OSError) as e:
            return f"Invalid: {timestamp}"
    
    @staticmethod
    def timestamp_to_est_time_only(timestamp: Union[int, float, np.int64]) -> str:
        """
        Convert timestamp to EST time only in 'HH:MM:SS' format.
        
        Args:
            timestamp: Timestamp in various formats
            
        Returns:
            Time string in 'HH:MM:SS' format (EST)
        """
        full_datetime = TimezoneHandler.timestamp_to_est_string(timestamp)
        if full_datetime.startswith("Invalid"):
            return full_datetime
        
        # Extract time part (everything after the space)
        return full_datetime.split(' ')[1] if ' ' in full_datetime else full_datetime
    
    @staticmethod
    def pandas_timestamp_to_est(timestamp_series: pd.Series) -> pd.Series:
        """
        Convert pandas Series of timestamps to EST strings.
        
        Args:
            timestamp_series: Pandas Series containing timestamps
            
        Returns:
            Pandas Series with EST formatted strings
        """
        return timestamp_series.apply(TimezoneHandler.timestamp_to_est_string)
    
    @staticmethod
    def numpy_timestamps_to_est(timestamps: np.ndarray) -> np.ndarray:
        """
        Convert numpy array of timestamps to EST strings.
        
        Args:
            timestamps: NumPy array of timestamps
            
        Returns:
            NumPy array of EST formatted strings
        """
        return np.array([TimezoneHandler.timestamp_to_est_string(ts) for ts in timestamps])


# Convenience functions for backward compatibility
def format_timestamp_est(timestamp: Union[int, float, np.int64]) -> str:
    """Alias for timestamp_to_est_string"""
    return TimezoneHandler.timestamp_to_est_string(timestamp)


def format_time_est(timestamp: Union[int, float, np.int64]) -> str:
    """Alias for timestamp_to_est_time_only"""
    return TimezoneHandler.timestamp_to_est_time_only(timestamp)