"""
Advanced transformation methods for financial time series stationarity
Based on analysis showing heteroskedasticity, non-normality, and volatility clustering
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import erfinv
import warnings
warnings.filterwarnings('ignore')


class AdvancedTransforms:
    """
    Advanced transformation methods for making financial time series more stationary
    """
    
    @staticmethod
    def realized_volatility_normalization(data, lookback=60, min_periods=20, 
                                         vol_lookback=20, use_squared_returns=True):
        """
        Normalize by realized volatility using high-frequency return information
        Better captures actual volatility than simple rolling std
        
        Parameters:
        -----------
        lookback : int
            Window for calculating realized volatility
        vol_lookback : int
            Shorter window for recent volatility (for adaptive weighting)
        use_squared_returns : bool
            Use squared returns for RV calculation (more robust)
        """
        series = pd.Series(data)
        
        if use_squared_returns:
            # Realized volatility using squared returns
            squared_returns = series ** 2
            rv_long = np.sqrt(squared_returns.rolling(window=lookback, min_periods=min_periods).mean())
            rv_short = np.sqrt(squared_returns.rolling(window=vol_lookback, min_periods=min_periods).mean())
        else:
            # Standard deviation approach
            rv_long = series.rolling(window=lookback, min_periods=min_periods).std()
            rv_short = series.rolling(window=vol_lookback, min_periods=min_periods).std()
        
        # Adaptive weighting between long and short term volatility
        # More weight on recent volatility when it diverges from long-term
        vol_ratio = rv_short / rv_long
        vol_ratio = vol_ratio.clip(0.5, 2.0)  # Limit extreme adjustments
        
        # Weighted average (more weight on short-term when volatility is changing)
        weight = 0.3 + 0.4 * np.abs(vol_ratio - 1)  # Weight increases with divergence
        weight = np.minimum(weight, 0.7)  # Cap at 70% short-term weight
        
        realized_vol = weight * rv_short + (1 - weight) * rv_long
        realized_vol = realized_vol.replace(0, np.nan).ffill().fillna(1)
        
        return series / realized_vol
    
    @staticmethod
    def garch_style_normalization(data, alpha=0.1, beta=0.8, omega=0.1):
        """
        GARCH(1,1) style volatility normalization
        Captures volatility clustering common in financial data
        
        Parameters:
        -----------
        alpha : float
            Weight on squared return (volatility shock)
        beta : float
            Weight on previous volatility (persistence)
        omega : float
            Long-run variance weight
        """
        series = pd.Series(data)
        n = len(series)
        
        # Initialize
        variance = np.zeros(n)
        variance[0] = series.var()  # Start with unconditional variance
        
        # GARCH(1,1) recursion
        for t in range(1, n):
            variance[t] = omega + alpha * series.iloc[t-1]**2 + beta * variance[t-1]
        
        # Convert variance to volatility
        volatility = np.sqrt(variance)
        volatility[volatility == 0] = 1  # Avoid division by zero
        
        return series / volatility
    
    @staticmethod
    def rank_percentile_transform(data, window=252, min_periods=60):
        """
        Transform to percentile ranks within rolling window
        Extremely robust to outliers and fat tails
        Produces uniform distribution in [0, 1]
        
        Parameters:
        -----------
        window : int
            Rolling window for rank calculation (252 = 1 year of trading days)
        """
        series = pd.Series(data)
        
        def rank_to_percentile(x):
            if len(x.dropna()) < min_periods:
                return np.nan
            return stats.rankdata(x, nan_policy='omit')[-1] / len(x.dropna())
        
        # Rolling rank transformation
        ranks = series.rolling(window=window, min_periods=min_periods).apply(
            rank_to_percentile, raw=False
        )
        
        # Transform to normal quantiles if desired (inverse normal CDF)
        # This makes the distribution closer to normal
        # Comment out if uniform [0,1] is preferred
        # ranks = stats.norm.ppf(ranks.clip(0.001, 0.999))
        
        return ranks
    
    @staticmethod
    def modified_z_score(data, window=60, robust=True, winsorize_level=0.05):
        """
        Modified Z-score with robust statistics and winsorization
        Better than standard Z-score for fat-tailed distributions
        
        Parameters:
        -----------
        robust : bool
            Use median and MAD instead of mean and std
        winsorize_level : float
            Fraction of data to winsorize at each tail
        """
        series = pd.Series(data)
        
        if robust:
            # Use median and MAD (Median Absolute Deviation)
            rolling_median = series.rolling(window=window, min_periods=20).median()
            
            def mad(x):
                return np.median(np.abs(x - np.median(x))) * 1.4826  # Scale factor for consistency
            
            rolling_mad = series.rolling(window=window, min_periods=20).apply(mad, raw=True)
            rolling_mad = rolling_mad.replace(0, np.nan).ffill().fillna(1)
            
            z_scores = (series - rolling_median) / rolling_mad
        else:
            # Standard z-score
            rolling_mean = series.rolling(window=window, min_periods=20).mean()
            rolling_std = series.rolling(window=window, min_periods=20).std()
            rolling_std = rolling_std.replace(0, np.nan).ffill().fillna(1)
            
            z_scores = (series - rolling_mean) / rolling_std
        
        # Winsorize extreme values
        if winsorize_level > 0:
            lower = z_scores.quantile(winsorize_level)
            upper = z_scores.quantile(1 - winsorize_level)
            z_scores = z_scores.clip(lower, upper)
        
        return z_scores
    
    @staticmethod
    def yang_zhang_volatility_norm(high, low, close, open_price, window=30):
        """
        Yang-Zhang volatility estimator - most efficient for OHLC data
        Better than simple close-to-close volatility
        
        Note: Requires OHLC data
        """
        n = len(close)
        
        # Overnight (close-to-open) returns
        overnight = np.log(open_price / close.shift(1))
        
        # Open-to-close returns
        open_close = np.log(close / open_price)
        
        # Rogers-Satchell volatility component
        rs = np.log(high / close) * np.log(high / open_price) + \
             np.log(low / close) * np.log(low / open_price)
        
        # Calculate components
        overnight_var = overnight.rolling(window=window).var()
        open_close_var = open_close.rolling(window=window).var()
        rs_var = rs.rolling(window=window).mean()
        
        # Yang-Zhang volatility
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_variance = overnight_var + k * open_close_var + (1 - k) * rs_var
        yz_vol = np.sqrt(yz_variance)
        
        yz_vol = yz_vol.replace(0, np.nan).ffill().fillna(1)
        
        # Normalize close-to-close returns
        returns = close.pct_change()
        return returns / yz_vol
    
    @staticmethod
    def regime_based_normalization(data, n_regimes=3, lookback=100):
        """
        Normalize based on detected volatility regimes
        Useful when market has distinct volatility states
        
        Parameters:
        -----------
        n_regimes : int
            Number of volatility regimes to detect
        lookback : int
            Window for regime detection
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            # Fallback to simple volatility-based normalization
            print("sklearn not available, using simple volatility normalization")
            series = pd.Series(data)
            rolling_std = series.rolling(window=lookback, min_periods=20).std()
            rolling_std = rolling_std.replace(0, np.nan).ffill().fillna(1)
            return series / rolling_std
        
        series = pd.Series(data)
        
        # Calculate rolling volatility for regime detection
        rolling_vol = series.rolling(window=20, min_periods=10).std()
        rolling_vol = rolling_vol.bfill().ffill()
        
        # Prepare data for GMM
        vol_features = pd.DataFrame({
            'vol': rolling_vol,
            'vol_change': rolling_vol.pct_change(),
            'vol_ma': rolling_vol.rolling(10).mean()
        }).fillna(0)
        
        # Fit Gaussian Mixture Model for regime detection
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regimes = gmm.fit_predict(vol_features)
        
        # Calculate normalization factor for each regime
        normalized_data = series.copy()
        for regime in range(n_regimes):
            mask = regimes == regime
            if mask.sum() > 20:  # Need enough data points
                regime_std = series[mask].std()
                normalized_data[mask] = series[mask] / regime_std
        
        return normalized_data
    
    @staticmethod
    def fractional_difference(data, d=0.4, threshold=0.01):
        """
        Fractional differencing to achieve stationarity while preserving memory
        Optimal for prediction tasks where you want to keep some autocorrelation
        
        Parameters:
        -----------
        d : float
            Differencing parameter (0 < d < 1)
            Lower d preserves more memory, higher d increases stationarity
        threshold : float
            Threshold for weight cutoff
        """
        series = pd.Series(data)
        
        # Calculate weights using the binomial series
        def get_weights(d, size):
            w = [1.0]
            for k in range(1, size):
                w_k = -w[-1] * (d - k + 1) / k
                if abs(w_k) < threshold:
                    break
                w.append(w_k)
            return np.array(w)
        
        # Get weights
        weights = get_weights(d, len(series))
        
        # Apply fractional differencing
        frac_diff = pd.Series(index=series.index, dtype=float)
        for i in range(len(weights), len(series)):
            window_data = series.iloc[i-len(weights)+1:i+1].values
            if len(window_data) == len(weights):
                frac_diff.iloc[i] = np.dot(weights[::-1], window_data)
        
        return frac_diff
    
    @staticmethod
    def adaptive_iqr_normalization(data, base_window=60, vol_window=20, 
                                  iqr_percentiles=(25, 75), adaptive=True):
        """
        Enhanced IQR normalization with adaptive window sizing
        Improves on static IQR by adjusting to market conditions
        
        Parameters:
        -----------
        base_window : int
            Base window for IQR calculation
        vol_window : int
            Window for volatility assessment
        adaptive : bool
            Whether to adapt window size based on volatility
        """
        series = pd.Series(data)
        
        if adaptive:
            # Calculate recent volatility ratio
            recent_vol = series.rolling(vol_window).std()
            long_vol = series.rolling(base_window).std()
            vol_ratio = recent_vol / long_vol
            
            # Adaptive window: shrink in high vol, expand in low vol
            # High volatility -> smaller window (more responsive)
            # Low volatility -> larger window (more stable)
            adaptive_window = (base_window * (2 - vol_ratio.clip(0.5, 1.5)))
            adaptive_window = adaptive_window.fillna(base_window).clip(20, base_window * 2).astype(int)
            
            # Calculate adaptive IQR
            normalized = series.copy()
            for i in range(len(series)):
                if i < 20:
                    continue
                    
                window = adaptive_window.iloc[i] if i < len(adaptive_window) else base_window
                start_idx = max(0, i - window)
                data_slice = series.iloc[start_idx:i+1]
                
                q1 = data_slice.quantile(iqr_percentiles[0] / 100)
                q3 = data_slice.quantile(iqr_percentiles[1] / 100)
                iqr = q3 - q1
                
                if iqr > 0:
                    median = data_slice.median()
                    normalized.iloc[i] = (series.iloc[i] - median) / iqr
        else:
            # Standard rolling IQR
            q1 = series.rolling(base_window).quantile(iqr_percentiles[0] / 100)
            q3 = series.rolling(base_window).quantile(iqr_percentiles[1] / 100)
            iqr = q3 - q1
            median = series.rolling(base_window).median()
            
            iqr = iqr.replace(0, np.nan).ffill().fillna(1)
            normalized = (series - median) / iqr
        
        return normalized


def compare_transformations(data, column_name='Ret_fwd6hr'):
    """
    Compare different transformation methods for stationarity
    """
    import matplotlib.pyplot as plt
    try:
        from statsmodels.tsa.stattools import adfuller
        has_statsmodels = True
    except ImportError:
        has_statsmodels = False
        print("Note: statsmodels not installed, skipping ADF test")
    
    # Load data
    df = pd.read_csv('DTSmlDATA7x7.csv')
    series = df[column_name].dropna()
    
    # Apply transformations
    transforms = {
        'Original': series,
        'Current IQR': AdvancedTransforms.adaptive_iqr_normalization(series, adaptive=False),
        'Adaptive IQR': AdvancedTransforms.adaptive_iqr_normalization(series, adaptive=True),
        'Realized Vol': AdvancedTransforms.realized_volatility_normalization(series),
        'GARCH Style': AdvancedTransforms.garch_style_normalization(series),
        # 'Rank Percentile': AdvancedTransforms.rank_percentile_transform(series),  # Commented due to pandas warning
        'Modified Z-Score': AdvancedTransforms.modified_z_score(series, robust=True),
        'Fractional Diff': AdvancedTransforms.fractional_difference(series, d=0.3)
    }
    
    # Compare statistics
    print("Transformation Comparison")
    print("=" * 80)
    if has_statsmodels:
        print(f"{'Method':<20} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurt':>10} {'ADF p-val':>12}")
    else:
        print(f"{'Method':<20} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurt':>10} {'AutoCorr(1)':>12}")
    print("-" * 80)
    
    for name, transformed in transforms.items():
        if transformed is not None and len(transformed.dropna()) > 100:
            clean = transformed.dropna()
            
            if has_statsmodels:
                adf_pval = adfuller(clean, autolag='AIC')[1]
                print(f"{name:<20} {clean.mean():>10.4f} {clean.std():>10.4f} "
                      f"{stats.skew(clean):>10.4f} {stats.kurtosis(clean):>10.4f} "
                      f"{adf_pval:>12.6f}")
            else:
                # Use autocorrelation as a simple stationarity indicator
                autocorr = clean.autocorr(lag=1) if len(clean) > 1 else 0
                print(f"{name:<20} {clean.mean():>10.4f} {clean.std():>10.4f} "
                      f"{stats.skew(clean):>10.4f} {stats.kurtosis(clean):>10.4f} "
                      f"{autocorr:>12.4f}")
    
    print("\nNotes:")
    if has_statsmodels:
        print("- Lower ADF p-value indicates better stationarity (< 0.05 is stationary)")
    else:
        print("- Lower autocorrelation (closer to 0) indicates better stationarity")
    print("- Std closer to 1.0 indicates better normalization")
    print("- Skew closer to 0 indicates more symmetric distribution")
    print("- Lower kurtosis indicates fewer outliers")
    
    return transforms


if __name__ == "__main__":
    print("\nComparing transformation methods on financial time series data...")
    print("Current data shows: heteroskedasticity, fat tails, volatility clustering\n")
    
    transforms = compare_transformations(pd.Series([]), 'Ret_fwd6hr')
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS based on data characteristics:")
    print("=" * 80)
    print("""
1. **Realized Volatility Normalization** (RECOMMENDED)
   - Best for: Capturing actual market volatility vs simple rolling std
   - Advantages: Adapts to volatility clustering, uses squared returns
   - Your data shows volatility ranging 0.25-3.0, this handles it well

2. **Modified Z-Score with Robust Statistics** (GOOD ALTERNATIVE)
   - Best for: Handling outliers and fat tails
   - Uses median/MAD instead of mean/std
   - Your data is non-normal (p<0.001), this is more robust

3. **Adaptive IQR** (ENHANCEMENT OF CURRENT)
   - Improves your current IQR approach
   - Adjusts window size based on market volatility
   - Smaller windows in volatile periods, larger in calm periods

4. **Rank Percentile Transform** (FOR EXTREME OUTLIERS)
   - Completely robust to outliers
   - Produces uniform or normal distribution
   - Good if outliers are a major problem

5. **Fractional Differencing** (FOR PREDICTION)
   - Balances stationarity with memory preservation
   - Good for ML models that benefit from some autocorrelation
   - Your data shows -0.10 autocorr at lag 1, this could help

NOT recommended for your data:
- GARCH: Better for higher frequency data
- Yang-Zhang: Requires OHLC data (you have close-only)
    """)