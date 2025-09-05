import numpy as np
from typing import Dict, List, Tuple, Union
from numba import jit


class ArrayUtils:
    """Utility functions for array manipulation and processing."""
    
    @staticmethod
    def validate_array_shapes(*arrays) -> bool:
        """
        Validate that all arrays have the same length.
        """
        if len(arrays) < 2:
            return True
            
        first_len = len(arrays[0])
        return all(len(arr) == first_len for arr in arrays)
    
    @staticmethod
    def stack_features(features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Stack feature arrays into a single matrix.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            2D feature matrix [samples x features]
        """
        # Validate shapes
        arrays = list(features.values())
        if not ArrayUtils.validate_array_shapes(*arrays):
            raise ValueError("All feature arrays must have the same length")
            
        # Stack arrays
        return np.column_stack(arrays)
    
    @staticmethod
    def create_lagged_features(arr: np.ndarray, lags: List[int]) -> np.ndarray:
        """
        Create lagged versions of array for time series features.
        
        Args:
            arr: Input array
            lags: List of lag values
            
        Returns:
            2D array with lagged features
        """
        n = len(arr)
        max_lag = max(lags)
        
        # Initialize result array
        result = np.zeros((n - max_lag, len(lags) + 1))
        
        # Current values
        result[:, 0] = arr[max_lag:]
        
        # Lagged values
        for i, lag in enumerate(lags):
            result[:, i + 1] = arr[max_lag - lag:-lag if lag > 0 else None]
            
        return result
    
    @staticmethod
    @jit(nopython=True)
    def fast_zscore(arr: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate rolling z-score using vectorized operations.
        """
        n = len(arr)
        result = np.zeros(n)
        
        for i in range(window - 1, n):
            window_data = arr[i - window + 1:i + 1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std > 0:
                result[i] = (arr[i] - mean) / std
            else:
                result[i] = 0.0
                
        return result
    
    @staticmethod
    def create_time_features(timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create time-based features from timestamps.
        
        Returns:
            Dictionary of time features
        """
        # Convert to datetime for feature extraction
        from datetime import datetime
        
        n = len(timestamps)
        features = {
            'hour': np.zeros(n),
            'day_of_week': np.zeros(n),
            'day_of_month': np.zeros(n),
            'month': np.zeros(n),
            'is_month_start': np.zeros(n, dtype=bool),
            'is_month_end': np.zeros(n, dtype=bool)
        }
        
        for i, ts in enumerate(timestamps):
            dt = datetime.fromtimestamp(ts)
            features['hour'][i] = dt.hour
            features['day_of_week'][i] = dt.weekday()
            features['day_of_month'][i] = dt.day
            features['month'][i] = dt.month
            features['is_month_start'][i] = dt.day <= 5
            features['is_month_end'][i] = dt.day >= 25
            
        return features
    
    @staticmethod
    def normalize_features(features: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
        """
        Normalize feature matrix.
        
        Args:
            features: 2D feature matrix
            method: 'minmax' or 'zscore'
            
        Returns:
            (normalized_features, scaler_params)
        """
        if method == 'minmax':
            min_vals = np.min(features, axis=0)
            max_vals = np.max(features, axis=0)
            range_vals = max_vals - min_vals
            
            # Avoid division by zero
            range_vals[range_vals == 0] = 1.0
            
            normalized = (features - min_vals) / range_vals
            
            scaler_params = {
                'method': 'minmax',
                'min': min_vals,
                'max': max_vals
            }
            
        elif method == 'zscore':
            means = np.mean(features, axis=0)
            stds = np.std(features, axis=0)
            
            # Avoid division by zero
            stds[stds == 0] = 1.0
            
            normalized = (features - means) / stds
            
            scaler_params = {
                'method': 'zscore',
                'mean': means,
                'std': stds
            }
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return normalized, scaler_params
    
    @staticmethod
    def apply_scaler(features: np.ndarray, scaler_params: Dict) -> np.ndarray:
        """
        Apply saved scaler parameters to new data.
        """
        if scaler_params['method'] == 'minmax':
            min_vals = scaler_params['min']
            max_vals = scaler_params['max']
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0
            
            return (features - min_vals) / range_vals
            
        elif scaler_params['method'] == 'zscore':
            means = scaler_params['mean']
            stds = scaler_params['std']
            stds[stds == 0] = 1.0
            
            return (features - means) / stds
            
        else:
            raise ValueError(f"Unknown scaler method: {scaler_params['method']}")
    
    @staticmethod
    @jit(nopython=True)
    def fast_clip(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
        """
        Fast array clipping using Numba.
        """
        result = arr.copy()
        for i in range(len(result)):
            if result[i] < lower:
                result[i] = lower
            elif result[i] > upper:
                result[i] = upper
                
        return result