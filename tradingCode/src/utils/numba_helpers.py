import numpy as np
from numba import jit, prange
from typing import Tuple


@jit(nopython=True, parallel=True)
def parallel_rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Parallel rolling sum calculation using Numba.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window-1] = np.nan
    
    # Parallel computation for large arrays
    for i in prange(window-1, n):
        result[i] = np.sum(arr[i-window+1:i+1])
        
    return result


@jit(nopython=True)
def fast_ewma(arr: np.ndarray, alpha: float) -> np.ndarray:
    """
    Fast exponentially weighted moving average.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    
    result[0] = arr[0]
    for i in range(1, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        
    return result


@jit(nopython=True)
def fast_percentile(arr: np.ndarray, percentile: float) -> float:
    """
    Fast percentile calculation for sorted arrays.
    """
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    
    if n == 0:
        return np.nan
        
    k = (n - 1) * percentile / 100.0
    f = np.floor(k)
    c = np.ceil(k)
    
    if f == c:
        return sorted_arr[int(k)]
        
    d0 = sorted_arr[int(f)] * (c - k)
    d1 = sorted_arr[int(c)] * (k - f)
    
    return d0 + d1


@jit(nopython=True, parallel=True)
def parallel_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Calculate correlation matrix using parallel processing.
    
    Args:
        returns: 2D array [time x assets]
        
    Returns:
        Correlation matrix
    """
    n_assets = returns.shape[1]
    corr_matrix = np.eye(n_assets)
    
    # Standardize returns
    means = np.mean(returns, axis=0)
    stds = np.std(returns, axis=0)
    standardized = (returns - means) / stds
    
    # Calculate correlations in parallel
    for i in prange(n_assets):
        for j in range(i+1, n_assets):
            corr = np.mean(standardized[:, i] * standardized[:, j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            
    return corr_matrix


@jit(nopython=True)
def find_peaks_valleys_vectorized(arr: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local peaks and valleys in array.
    
    Args:
        arr: Input array
        order: Number of points on each side to check
        
    Returns:
        (peak_indices, valley_indices)
    """
    n = len(arr)
    peaks = []
    valleys = []
    
    for i in range(order, n - order):
        # Check if peak
        is_peak = True
        for j in range(i - order, i + order + 1):
            if j != i and arr[j] >= arr[i]:
                is_peak = False
                break
                
        if is_peak:
            peaks.append(i)
            
        # Check if valley
        is_valley = True
        for j in range(i - order, i + order + 1):
            if j != i and arr[j] <= arr[i]:
                is_valley = False
                break
                
        if is_valley:
            valleys.append(i)
            
    return np.array(peaks), np.array(valleys)


@jit(nopython=True)
def fast_rank(arr: np.ndarray) -> np.ndarray:
    """
    Fast ranking of array elements.
    """
    n = len(arr)
    idx = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        ranks[idx[i]] = i + 1
        
    return ranks


@jit(nopython=True, parallel=True)
def parallel_apply_func(data: np.ndarray, window: int) -> np.ndarray:
    """
    Template for parallel window operations.
    """
    n_rows, n_cols = data.shape
    result = np.zeros((n_rows - window + 1, n_cols))
    
    for col in prange(n_cols):
        for row in range(window - 1, n_rows):
            # Custom calculation here
            window_data = data[row - window + 1:row + 1, col]
            result[row - window + 1, col] = np.mean(window_data)
            
    return result