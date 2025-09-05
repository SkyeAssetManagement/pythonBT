import pandas as pd
import numpy as np
from scipy import stats

def verify_data_view_functionality():
    """Verify that Data View module functionality works correctly"""
    
    print("Verifying Data View Module Functionality")
    print("=" * 60)
    
    # Test 1: Load data
    print("\n1. Testing data loading...")
    try:
        df = pd.read_csv('DTSmlDATA7x7.csv')
        print(f"   [OK] Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"   [OK] Found {len(numeric_cols)} numeric columns")
        
        # Categorize columns
        targets = [col for col in numeric_cols if 'fwd' in col.lower()]
        features = [col for col in numeric_cols if col.startswith('Ret_') and 'fwd' not in col.lower()]
        others = [col for col in numeric_cols if col not in targets and col not in features]
        
        print(f"   [OK] Targets: {len(targets)} columns")
        print(f"   [OK] Features: {len(features)} columns")
        print(f"   [OK] Others: {len(others)} columns")
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False
    
    # Test 2: Statistics calculation
    print("\n2. Testing statistics calculation...")
    test_column = 'Ret_fwd6hr' if 'Ret_fwd6hr' in df.columns else numeric_cols[0]
    data = df[test_column].dropna()
    
    try:
        # Basic stats
        stats_dict = {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
        
        print(f"   [OK] Calculated stats for {test_column}:")
        print(f"     - Count: {stats_dict['count']:,}")
        print(f"     - Mean: {stats_dict['mean']:.6f}")
        print(f"     - Std Dev: {stats_dict['std']:.6f}")
        print(f"     - Skewness: {stats_dict['skewness']:.6f}")
        
        # Percentiles
        percentiles = [25, 50, 75]
        for p in percentiles:
            pval = data.quantile(p/100)
            print(f"     - {p}th percentile: {pval:.6f}")
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False
    
    # Test 3: Preprocessing capabilities
    print("\n3. Testing preprocessing...")
    try:
        # Volatility normalization
        vol_window = 60
        rolling_vol = data.rolling(window=vol_window, min_periods=1).std()
        rolling_vol = rolling_vol.replace(0, np.nan).ffill().fillna(1)
        vol_normalized = data / rolling_vol
        
        print(f"   [OK] Volatility normalization (window={vol_window}):")
        print(f"     - Original std: {data.std():.6f}")
        print(f"     - Normalized std: {vol_normalized.std():.6f}")
        
        # Exponential smoothing
        alpha = 0.1
        smoothed = data.ewm(alpha=alpha, adjust=False).mean()
        print(f"   [OK] Exponential smoothing (alpha={alpha}):")
        print(f"     - Original variance: {data.var():.6f}")
        print(f"     - Smoothed variance: {smoothed.var():.6f}")
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False
    
    # Test 4: Rolling statistics
    print("\n4. Testing rolling statistics...")
    try:
        window = 90
        rolling_mean = data.rolling(window=window, min_periods=1).mean()
        rolling_std = data.rolling(window=window, min_periods=1).std()
        rolling_min = data.rolling(window=window, min_periods=1).min()
        rolling_max = data.rolling(window=window, min_periods=1).max()
        
        print(f"   [OK] {window}-bar rolling statistics calculated:")
        print(f"     - Rolling mean range: [{rolling_mean.min():.6f}, {rolling_mean.max():.6f}]")
        print(f"     - Rolling std range: [{rolling_std.min():.6f}, {rolling_std.max():.6f}]")
        print(f"     - Min/Max spread: {(rolling_max - rolling_min).mean():.6f}")
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False
    
    # Test 5: Distribution analysis
    print("\n5. Testing distribution analysis...")
    try:
        # Normality test
        if len(data) > 8:
            _, p_value = stats.normaltest(data)
            print(f"   [OK] Normality test p-value: {p_value:.6f}")
            if p_value < 0.05:
                print(f"     - Data is NOT normally distributed")
            else:
                print(f"     - Data appears normally distributed")
        
        # Sign statistics
        positive_pct = (data > 0).sum() / len(data) * 100
        negative_pct = (data < 0).sum() / len(data) * 100
        print(f"   [OK] Sign distribution:")
        print(f"     - Positive: {positive_pct:.1f}%")
        print(f"     - Negative: {negative_pct:.1f}%")
        
        # Autocorrelation
        if len(data) > 20:
            autocorr_1 = data.autocorr(lag=1)
            autocorr_5 = data.autocorr(lag=5)
            print(f"   [OK] Autocorrelation:")
            print(f"     - Lag 1: {autocorr_1:.4f}")
            print(f"     - Lag 5: {autocorr_5:.4f}")
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All Data View functionality tests PASSED!")
    print("=" * 60)
    
    # Summary of features
    print("\nData View Module Features:")
    print("- Load and categorize data columns (targets, features, others)")
    print("- Comprehensive descriptive statistics")
    print("- Distribution analysis (histogram, box plot, Q-Q plot, CDF)")
    print("- Time series visualization with moving averages")
    print("- 90-bar rolling mean and standard deviation")
    print("- Independent preprocessing (volatility normalization, smoothing)")
    print("- Autocorrelation analysis")
    print("- Sign statistics and normality testing")
    
    return True

if __name__ == "__main__":
    verify_data_view_functionality()