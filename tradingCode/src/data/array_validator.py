import numpy as np
from numba import jit
from typing import Dict, Tuple


class ArrayValidator:
    """Vectorized validation for trading data arrays."""
    
    @staticmethod
    @jit(nopython=True)
    def check_price_consistency(open_arr: np.ndarray, high_arr: np.ndarray,
                               low_arr: np.ndarray, close_arr: np.ndarray) -> np.ndarray:
        """
        Vectorized check for OHLC consistency.
        Returns boolean array indicating valid bars.
        """
        n = len(open_arr)
        valid = np.ones(n, dtype=np.bool_)
        
        # Vectorized checks
        valid &= high_arr >= low_arr
        valid &= high_arr >= open_arr
        valid &= high_arr >= close_arr
        valid &= low_arr <= open_arr
        valid &= low_arr <= close_arr
        valid &= close_arr > 0
        
        return valid
    
    @staticmethod
    def detect_gaps(timestamps: np.ndarray, expected_interval: int,
                   tolerance: float = 0.1) -> np.ndarray:
        """
        Detect gaps in time series using vectorized operations.
        
        Args:
            timestamps: Unix timestamps
            expected_interval: Expected interval in seconds
            tolerance: Tolerance for interval variation (0.1 = 10%)
            
        Returns:
            Boolean array where True indicates a gap
        """
        if len(timestamps) < 2:
            return np.array([False])
            
        # Calculate time differences
        time_diffs = np.diff(timestamps)
        
        # Detect gaps (intervals significantly larger than expected)
        min_interval = expected_interval * (1 - tolerance)
        max_interval = expected_interval * (1 + tolerance)
        
        gaps = (time_diffs < min_interval) | (time_diffs > max_interval)
        
        # Prepend False for first element
        return np.concatenate((np.array([False]), gaps))
    
    @staticmethod
    def validate_data_quality(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Comprehensive data quality validation using vectorized operations.
        
        Returns dictionary of validation results:
        {
            'valid_bars': boolean array,
            'gaps': boolean array,
            'zero_volume': boolean array,
            'price_spikes': boolean array
        }
        """
        n = len(data['close'])
        
        # Price consistency check
        valid_bars = ArrayValidator.check_price_consistency(
            data['open'], data['high'], data['low'], data['close']
        )
        
        # Gap detection (5 minutes = 300 seconds)
        gaps = ArrayValidator.detect_gaps(data['datetime'], 300)
        
        # Zero volume detection (vectorized)
        zero_volume = data['volume'] == 0
        
        # Price spike detection (vectorized)
        returns = np.diff(data['close']) / data['close'][:-1]
        spike_threshold = 0.1  # 10% move
        price_spikes = np.concatenate(([False], np.abs(returns) > spike_threshold))
        
        return {
            'valid_bars': valid_bars,
            'gaps': gaps,
            'zero_volume': zero_volume,
            'price_spikes': price_spikes,
            'total_issues': ~valid_bars | gaps | zero_volume | price_spikes
        }
    
    @staticmethod
    @jit(nopython=True)
    def clean_data_arrays(open_arr: np.ndarray, high_arr: np.ndarray,
                         low_arr: np.ndarray, close_arr: np.ndarray,
                         volume_arr: np.ndarray, valid_mask: np.ndarray) -> tuple:
        """
        Remove invalid bars from arrays using boolean mask.
        Fully vectorized operation.
        """
        return (
            open_arr[valid_mask],
            high_arr[valid_mask],
            low_arr[valid_mask],
            close_arr[valid_mask],
            volume_arr[valid_mask]
        )