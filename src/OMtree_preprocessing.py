import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import logit
import configparser
from src.column_detector import ColumnDetector

class DataPreprocessor:
    def __init__(self, config_path='OMtree_config.ini'):
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.config.read(config_path)
        
        # Normalization toggles
        self.normalize_features = self.config['preprocessing'].getboolean('normalize_features', fallback=True)
        self.normalize_target = self.config['preprocessing'].getboolean('normalize_target', fallback=True)
        self.detrend_features = self.config['preprocessing'].getboolean('detrend_features', fallback=False)
        
        # Normalization method: IQR, AVS, or LOGIT_RANK
        self.normalization_method = self.config['preprocessing'].get('normalization_method', 'IQR')
        
        # IQR parameters
        self.vol_window = int(float(self.config['preprocessing']['vol_window']))
        self.smoothing_type = self.config['preprocessing']['smoothing_type']
        # Calculate smoothing alpha based on lookback window: alpha = 2/(n+1) for EMA
        self.recent_iqr_lookback = int(float(self.config['preprocessing']['recent_iqr_lookback']))
        self.smoothing_alpha = 2.0 / (self.recent_iqr_lookback + 1)
        self.percentile_upper = float(self.config['preprocessing']['percentile_upper'])
        self.percentile_lower = float(self.config['preprocessing']['percentile_lower'])
        
        # NEW: IQR weighting parameters (removed decay factor - using only alpha smoothing)
        
        # NEW: Winsorization parameters
        self.winsorize_enabled = self.config['preprocessing'].getboolean('winsorize_enabled', fallback=False)
        winsorize_str = self.config['preprocessing'].get('winsorize_percentile', '5.0')
        self.winsorize_percentile = float(winsorize_str) if winsorize_str.strip() else 5.0
        
        # AVS parameters
        avs_slow_str = self.config['preprocessing'].get('avs_slow_window', '60')
        self.avs_slow_window = int(float(avs_slow_str)) if avs_slow_str.strip() else 60
        avs_fast_str = self.config['preprocessing'].get('avs_fast_window', '20')
        self.avs_fast_window = int(float(avs_fast_str)) if avs_fast_str.strip() else 20
        
    def winsorize_data(self, window_data, percentile):
        """
        Winsorize data by clipping extreme values.
        
        Args:
            window_data: Array of values to winsorize
            percentile: Percentage for winsorization (e.g., 5 means clip at 5th and 95th percentiles)
        
        Returns:
            Winsorized data array
        """
        if len(window_data) < 3:  # Need at least 3 points for meaningful percentiles
            return window_data
        
        lower = np.percentile(window_data, percentile)
        upper = np.percentile(window_data, 100 - percentile)
        
        return np.clip(window_data, lower, upper)
    
    
    def logit_rank_transform(self, series, window_size):
        """
        Apply logit-rank transformation to a series using rolling windows.
        This ensures no look-ahead bias.
        
        Args:
            series: Pandas series to transform
            window_size: Size of rolling window for rank calculation
        
        Returns:
            Logit-transformed ranks
        """
        transformed = []
        
        for i in range(len(series)):
            if i < window_size:
                transformed.append(np.nan)
            else:
                # Get window data (not including current point to avoid look-ahead)
                window_data = series.iloc[i-window_size:i].values
                current_value = series.iloc[i]
                
                # Remove NaN values
                valid_data = window_data[~np.isnan(window_data)]
                
                if len(valid_data) < window_size * 0.8 or np.isnan(current_value):
                    transformed.append(np.nan)
                else:
                    # Calculate rank of current value within historical window
                    rank = np.sum(valid_data <= current_value)
                    percentile_rank = rank / (len(valid_data) + 1)
                    
                    # Clip to avoid infinities in logit (use 0.001 to 0.999)
                    percentile_rank = np.clip(percentile_rank, 0.001, 0.999)
                    
                    # Apply logit transformation
                    logit_value = logit(percentile_rank)
                    transformed.append(logit_value)
        
        return transformed
    
    def adaptive_volatility_scaling(self, series):
        """
        Apply Adaptive Volatility Scaling (AVS) to a series.
        Uses weighted average of fast and slow volatility windows.
        """
        # Use absolute values for volatility estimation
        abs_series = np.abs(series)
        
        # Calculate slow and fast volatility
        vol_slow = abs_series.rolling(window=self.avs_slow_window, min_periods=20).mean()
        vol_fast = abs_series.rolling(window=self.avs_fast_window, min_periods=10).mean()
        
        # Calculate adaptive weight based on volatility divergence
        vol_ratio = vol_fast / vol_slow
        vol_ratio = vol_ratio.clip(0.5, 2.0)  # Limit extreme ratios
        
        # Weight increases when volatilities diverge (regime change)
        weight = 0.3 + 0.3 * np.abs(vol_ratio - 1)
        weight = np.minimum(weight, 0.6)  # Cap at 60% fast weight
        
        # Adaptive volatility is weighted average
        adaptive_vol = weight * vol_fast + (1 - weight) * vol_slow
        adaptive_vol = adaptive_vol.replace(0, np.nan).ffill().fillna(1)
        
        return series / adaptive_vol
    
    def volatility_adjust(self, data, columns, skip_normalization=False, is_target=False):
        """
        Volatility adjust the data using IQR, AVS, or logit-rank normalization.
        If skip_normalization is True, returns original values with _vol_adj suffix.
        """
        adjusted_data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if skip_normalization:
                # If skipping normalization, just copy the original values
                adjusted_data[f'{col}_vol_adj'] = data[col].values
            elif self.normalization_method == 'LOGIT_RANK':
                # Apply logit-rank transformation
                transformed = self.logit_rank_transform(data[col], self.vol_window)
                adjusted_data[f'{col}_vol_adj'] = transformed
            elif self.normalization_method == 'AVS':
                # Apply Adaptive Volatility Scaling
                normalized = self.adaptive_volatility_scaling(data[col])
                adjusted_data[f'{col}_vol_adj'] = normalized
                # For consistency, store the volatility measure (similar to IQR storage)
                abs_series = np.abs(data[col])
                vol_slow = abs_series.rolling(window=self.avs_slow_window, min_periods=20).mean()
                vol_fast = abs_series.rolling(window=self.avs_fast_window, min_periods=10).mean()
                vol_ratio = vol_fast / vol_slow
                vol_ratio = vol_ratio.clip(0.5, 2.0)
                weight = 0.3 + 0.3 * np.abs(vol_ratio - 1)
                weight = np.minimum(weight, 0.6)
                adaptive_vol = weight * vol_fast + (1 - weight) * vol_slow
                adjusted_data[f'{col}_avs_vol'] = adaptive_vol
            else:
                # Default to IQR normalization (with optional weighting and winsorization)
                vol_adjusted = []
                iqr_values = []
                
                for i in range(len(data)):
                    if i < self.vol_window:
                        vol_adjusted.append(np.nan)
                        iqr_values.append(np.nan)
                    else:
                        # For targets: exclude current observation to avoid look-ahead
                        # For features: include observations up to current point
                        if is_target:
                            # Target: use strictly historical data (exclude point i)
                            window_data = data[col].iloc[i-self.vol_window:i].values
                        else:
                            # Feature: can use data up to and including point i
                            window_data = data[col].iloc[i-self.vol_window+1:i+1].values
                        
                        window_data = window_data[~np.isnan(window_data)]
                        
                        # Require at least 80% of window to have valid data
                        if len(window_data) < self.vol_window * 0.8:
                            vol_adjusted.append(np.nan)
                            iqr_values.append(np.nan)
                        else:
                            # Apply winsorization if enabled
                            if self.winsorize_enabled:
                                window_data = self.winsorize_data(window_data, self.winsorize_percentile)
                            
                            # Calculate standard IQR
                            q75 = np.percentile(window_data, self.percentile_upper)
                            q25 = np.percentile(window_data, self.percentile_lower)
                            iqr = q75 - q25
                            
                            if iqr > 0:
                                # Apply smoothing to IQR
                                if self.smoothing_type == 'exponential':
                                    if i > self.vol_window and not np.isnan(iqr_values[-1]):
                                        smoothed_iqr = self.smoothing_alpha * iqr + (1 - self.smoothing_alpha) * iqr_values[-1]
                                    else:
                                        smoothed_iqr = iqr
                                else:
                                    recent_iqrs = [x for x in iqr_values[-self.recent_iqr_lookback:] if not np.isnan(x)]
                                    if len(recent_iqrs) > 0:
                                        smoothed_iqr = np.mean(recent_iqrs + [iqr])
                                    else:
                                        smoothed_iqr = iqr
                                
                                iqr_values.append(smoothed_iqr)
                                normalized_value = data[col].iloc[i] / smoothed_iqr
                                vol_adjusted.append(normalized_value)
                            else:
                                vol_adjusted.append(np.nan)
                                iqr_values.append(np.nan)
                
                adjusted_data[f'{col}_vol_adj'] = vol_adjusted
                adjusted_data[f'{col}_iqr'] = iqr_values
        
        return adjusted_data
    
    def detrend_by_median(self, data, columns):
        """
        Detrend features by subtracting rolling median.
        Only applies to features, not targets.
        
        Args:
            data: DataFrame with data
            columns: List of columns to detrend
        
        Returns:
            DataFrame with detrended columns (adds _detrend suffix)
        """
        detrended_data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Calculate rolling median with same window as volatility
            rolling_median = data[col].rolling(window=self.vol_window, min_periods=1).median()
            
            # Subtract median to detrend
            detrended_values = data[col] - rolling_median
            
            # Store detrended values with suffix
            detrended_data[f'{col}_detrend'] = detrended_values
            
            # Also store the median for reference
            detrended_data[f'{col}_median'] = rolling_median
        
        return detrended_data
    
    def process_data(self, df):
        """
        Main preprocessing function that respects normalization toggles.
        Adds engineered volatility signal feature that is immune to normalization.
        """
        # Get configured columns
        config_features = [col.strip() for col in self.config['data']['feature_columns'].split(',')]
        config_target = self.config['data']['target_column']
        
        # Use column detector to find valid columns
        detected = ColumnDetector.detect_columns(df, config_features, [config_target])
        feature_columns = detected['features']
        
        # Handle target column - if config target doesn't exist, use first available target
        if config_target in df.columns:
            target_column = config_target
        elif detected['targets']:
            target_column = detected['targets'][0]
            print(f"Warning: Target '{config_target}' not found, using '{target_column}'")
        else:
            # Last resort - use last numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_column = numeric_cols[-1]
                print(f"Warning: No target found, using last numeric column '{target_column}'")
            else:
                raise ValueError("No numeric columns found in data")
        
        # Validate we have at least one feature
        if not feature_columns:
            print("Warning: No feature columns found from config, auto-detecting...")
            feature_columns = detected['features']
            if not feature_columns:
                raise ValueError("No feature columns could be detected")
        
        print(f"Preprocessing with normalization method: {self.normalization_method}")
        if self.normalization_method == 'IQR':
            print(f"  Smoothing: {self.smoothing_type} (alpha={self.smoothing_alpha:.3f})")
            print(f"  Winsorization: {self.winsorize_enabled}")
            if self.winsorize_enabled:
                print(f"  Winsorize percentile: {self.winsorize_percentile}%")
        
        print(f"Using features: {feature_columns}")
        print(f"Using target: {target_column}")
        
        # Add volatility signal feature before normalization (if applicable)
        vol_signal_window = int(self.config['preprocessing'].get('vol_signal_window', 20))
        if vol_signal_window > 0:
            # Create volatility signal - this is always calculated on raw data
            for col in feature_columns:
                if col in df.columns:
                    # Volatility signal is trailing standard deviation / mean of abs values
                    abs_values = df[col].abs()
                    vol = abs_values.rolling(window=vol_signal_window, min_periods=1).std()
                    mean_abs = abs_values.rolling(window=vol_signal_window, min_periods=1).mean()
                    vol_signal = vol / (mean_abs + 1e-10)  # Add small constant to avoid division by zero
                    df[f'{col}_vol_signal'] = vol_signal.fillna(0)
                    print(f"Added vol signal for {col}")
        
        processed = df.copy()
        
        # 1. Detrend features if enabled
        if self.detrend_features and self.normalize_features:
            print("Applying detrending to features...")
            processed = self.detrend_by_median(processed, feature_columns)
            # Update column names after detrending
            feature_columns = [f'{col}_detrend' for col in feature_columns]
        
        # 2. Normalize features if enabled
        if self.normalize_features:
            print(f"Normalizing features using {self.normalization_method} method...")
            processed = self.volatility_adjust(processed, feature_columns, skip_normalization=False)
            # Update column names after normalization
            feature_columns = [f'{col}_vol_adj' for col in feature_columns]
        else:
            print("Skipping feature normalization...")
            processed = self.volatility_adjust(processed, feature_columns, skip_normalization=True)
            feature_columns = [f'{col}_vol_adj' for col in feature_columns]
        
        # 3. Normalize target if enabled (never detrend target)
        if self.normalize_target:
            print(f"Normalizing target using {self.normalization_method} method...")
            processed = self.volatility_adjust(processed, [target_column], skip_normalization=False, is_target=True)
            target_column = f'{target_column}_vol_adj'
        else:
            print("Skipping target normalization...")
            processed = self.volatility_adjust(processed, [target_column], skip_normalization=True, is_target=True)
            target_column = f'{target_column}_vol_adj'
        
        # Return the processed DataFrame
        return processed, feature_columns, target_column