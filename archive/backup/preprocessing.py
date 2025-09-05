import pandas as pd
import numpy as np
from scipy import stats
import configparser

class DataPreprocessor:
    def __init__(self, config_path='config_longonly.ini'):
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.config.read(config_path)
        
        # Normalization toggles
        self.normalize_features = self.config['preprocessing'].getboolean('normalize_features', fallback=True)
        self.normalize_target = self.config['preprocessing'].getboolean('normalize_target', fallback=True)
        
        self.vol_window = int(self.config['preprocessing']['vol_window'])
        self.smoothing_type = self.config['preprocessing']['smoothing_type']
        self.smoothing_alpha = float(self.config['preprocessing']['smoothing_alpha'])
        self.vol_threshold_min = float(self.config['preprocessing']['vol_threshold_min'])
        self.percentile_upper = float(self.config['preprocessing']['percentile_upper'])
        self.percentile_lower = float(self.config['preprocessing']['percentile_lower'])
        self.recent_iqr_lookback = int(self.config['preprocessing']['recent_iqr_lookback'])
        
    def volatility_adjust(self, data, columns, skip_normalization=False):
        """
        Volatility adjust the data using trailing IQR normalization.
        If skip_normalization is True, returns original values with _vol_adj suffix.
        """
        adjusted_data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if skip_normalization:
                # If skipping normalization, just copy the original values
                adjusted_data[f'{col}_vol_adj'] = data[col].values
            else:
                # Perform volatility normalization
                vol_adjusted = []
                iqr_values = []
                
                for i in range(len(data)):
                    if i < self.vol_window:
                        vol_adjusted.append(np.nan)
                        iqr_values.append(np.nan)
                    else:
                        window_data = data[col].iloc[i-self.vol_window:i].values
                        
                        window_data = window_data[~np.isnan(window_data)]
                        
                        if len(window_data) < self.vol_window * self.vol_threshold_min:
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
    
    def process_data(self, df):
        """
        Main preprocessing function that respects normalization toggles.
        """
        feature_columns = [col.strip() for col in self.config['data']['feature_columns'].split(',')]
        target_column = self.config['data']['target_column']
        
        # Process features and target separately based on toggles
        processed_df = df.copy()
        
        # Process features
        if self.normalize_features:
            processed_df = self.volatility_adjust(processed_df, feature_columns, skip_normalization=False)
        else:
            processed_df = self.volatility_adjust(processed_df, feature_columns, skip_normalization=True)
        
        # Process target
        if self.normalize_target:
            processed_df = self.volatility_adjust(processed_df, [target_column], skip_normalization=False)
        else:
            processed_df = self.volatility_adjust(processed_df, [target_column], skip_normalization=True)
        
        return processed_df