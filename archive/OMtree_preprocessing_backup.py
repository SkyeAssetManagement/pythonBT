import pandas as pd
import numpy as np
from scipy import stats
import configparser
from column_detector import ColumnDetector

class DataPreprocessor:
    def __init__(self, config_path='OMtree_config.ini'):
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.config.read(config_path)
        
        # Normalization toggles
        self.normalize_features = self.config['preprocessing'].getboolean('normalize_features', fallback=True)
        self.normalize_target = self.config['preprocessing'].getboolean('normalize_target', fallback=True)
        self.detrend_features = self.config['preprocessing'].getboolean('detrend_features', fallback=False)
        
        # Normalization method: IQR or AVS
        self.normalization_method = self.config['preprocessing'].get('normalization_method', 'IQR')
        
        # IQR parameters
        self.vol_window = int(float(self.config['preprocessing']['vol_window']))
        self.smoothing_type = self.config['preprocessing']['smoothing_type']
        self.smoothing_alpha = float(self.config['preprocessing']['smoothing_alpha'])
        self.percentile_upper = float(self.config['preprocessing']['percentile_upper'])
        self.percentile_lower = float(self.config['preprocessing']['percentile_lower'])
        self.recent_iqr_lookback = int(float(self.config['preprocessing']['recent_iqr_lookback']))
        
        # AVS parameters
        self.avs_slow_window = int(float(self.config['preprocessing'].get('avs_slow_window', 60)))
        self.avs_fast_window = int(float(self.config['preprocessing'].get('avs_fast_window', 20)))
        
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
    
    def volatility_adjust(self, data, columns, skip_normalization=False):
        """
        Volatility adjust the data using either IQR or AVS normalization.
        If skip_normalization is True, returns original values with _vol_adj suffix.
        """
        adjusted_data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if skip_normalization:
                # If skipping normalization, just copy the original values
                adjusted_data[f'{col}_vol_adj'] = data[col].values
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
                # Default to IQR normalization
                vol_adjusted = []
                iqr_values = []
                
                for i in range(len(data)):
                    if i < self.vol_window:
                        vol_adjusted.append(np.nan)
                        iqr_values.append(np.nan)
                    else:
                        window_data = data[col].iloc[i-self.vol_window:i].values
                        
                        window_data = window_data[~np.isnan(window_data)]
                        
                        # Require at least 80% of window to have valid data
                        if len(window_data) < self.vol_window * 0.8:
                            vol_adjusted.append(np.nan)
                            iqr_values.append(np.nan)
                        else:
                            q75 = np.percentile(window_data, self.percentile_upper)
                            q25 = np.percentile(window_data, self.percentile_lower)
                            iqr = q75 - q25
                            
                            if iqr > 0:
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
        
        print(f"Using {len(feature_columns)} features and target '{target_column}'")
        
        # Start with copy of raw data
        processed_df = df.copy()
        
        # STEP 1: Detrend features if enabled (before normalization, only for features)
        if self.detrend_features:
            print(f"Detrending features by subtracting rolling median (window={self.vol_window})")
            processed_df = self.detrend_by_median(processed_df, feature_columns)
            # Update feature columns to use detrended versions
            feature_columns_to_process = [f'{col}_detrend' for col in feature_columns]
        else:
            feature_columns_to_process = feature_columns
        
        # STEP 2: Process features (apply normalization if enabled)
        if self.normalize_features:
            processed_df = self.volatility_adjust(processed_df, feature_columns_to_process, skip_normalization=False)
        else:
            processed_df = self.volatility_adjust(processed_df, feature_columns_to_process, skip_normalization=True)
        
        # STEP 3: Process target (never detrended, only normalized if enabled)
        if self.normalize_target:
            processed_df = self.volatility_adjust(processed_df, [target_column], skip_normalization=False)
        else:
            processed_df = self.volatility_adjust(processed_df, [target_column], skip_normalization=True)
        
        return processed_df