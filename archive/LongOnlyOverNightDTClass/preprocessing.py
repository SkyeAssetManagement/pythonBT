import pandas as pd
import numpy as np
from scipy import stats
import configparser

class DataPreprocessor:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        self.vol_window = int(self.config['preprocessing']['vol_window'])
        self.smoothing_type = self.config['preprocessing']['smoothing_type']
        self.smoothing_alpha = float(self.config['preprocessing']['smoothing_alpha'])
        
    def volatility_adjust(self, data, columns):
        """
        Volatility adjust the data using trailing IQR normalization.
        """
        adjusted_data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            vol_adjusted = []
            iqr_values = []
            
            for i in range(len(data)):
                if i < self.vol_window:
                    vol_adjusted.append(np.nan)
                    iqr_values.append(np.nan)
                else:
                    window_data = data[col].iloc[i-self.vol_window:i].values
                    
                    window_data = window_data[~np.isnan(window_data)]
                    
                    if len(window_data) < self.vol_window * 0.8:
                        vol_adjusted.append(np.nan)
                        iqr_values.append(np.nan)
                    else:
                        q75 = np.percentile(window_data, 75)
                        q25 = np.percentile(window_data, 25)
                        iqr = q75 - q25
                        
                        if iqr > 0:
                            if self.smoothing_type == 'exponential':
                                if i > self.vol_window and not np.isnan(iqr_values[-1]):
                                    smoothed_iqr = self.smoothing_alpha * iqr + (1 - self.smoothing_alpha) * iqr_values[-1]
                                else:
                                    smoothed_iqr = iqr
                            else:
                                recent_iqrs = [x for x in iqr_values[-20:] if not np.isnan(x)]
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
        Main preprocessing function.
        """
        feature_columns = [col.strip() for col in self.config['data']['feature_columns'].split(',')]
        target_column = self.config['data']['target_column']
        
        all_columns = feature_columns + [target_column]
        existing_columns = [col for col in all_columns if col in df.columns]
        
        processed_df = self.volatility_adjust(df, existing_columns)
        
        return processed_df