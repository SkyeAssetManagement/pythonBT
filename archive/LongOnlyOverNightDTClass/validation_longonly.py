import pandas as pd
import numpy as np
import configparser
from preprocessing import DataPreprocessor
from model_longonly import LongOnlyTreeEnsemble

class LongOnlyValidator:
    def __init__(self, config_path='config_longonly.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        self.train_size = int(self.config['validation']['train_size'])
        self.test_size = int(self.config['validation']['test_size'])
        self.min_initial_data = int(self.config['validation']['min_initial_data'])
        self.selected_feature = self.config['data']['selected_feature']
        self.target_column = self.config['data']['target_column']
        
        self.preprocessor = DataPreprocessor(config_path)
        
    def load_and_prepare_data(self):
        csv_file = self.config['data']['csv_file']
        df = pd.read_csv(csv_file)
        
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], dayfirst=True)
        df = df.sort_values('Date/Time').reset_index(drop=True)
        
        processed_df = self.preprocessor.process_data(df)
        
        return processed_df
    
    def get_train_test_split(self, data, test_end_idx):
        test_start_idx = test_end_idx - self.test_size + 1
        train_end_idx = test_start_idx - 1
        train_start_idx = train_end_idx - self.train_size + 1
        
        if train_start_idx < 0:
            return None, None, None, None
        
        feature_col = f'{self.selected_feature}_vol_adj'
        target_col = f'{self.target_column}_vol_adj'
        
        if feature_col not in data.columns:
            feature_col = self.selected_feature
        if target_col not in data.columns:
            target_col = self.target_column
        
        train_X = data[feature_col].iloc[train_start_idx:train_end_idx+1].values
        train_y = data[target_col].iloc[train_start_idx:train_end_idx+1].values
        
        test_X = data[feature_col].iloc[test_start_idx:test_end_idx+1].values
        test_y = data[target_col].iloc[test_start_idx:test_end_idx+1].values
        
        train_mask = ~(np.isnan(train_X) | np.isnan(train_y))
        test_mask = ~np.isnan(test_X)
        
        if np.sum(train_mask) < 100:
            return None, None, None, None
        
        return (train_X[train_mask], train_y[train_mask], 
                test_X[test_mask], test_y[test_mask])
    
    def run_validation(self):
        """
        Run walk-forward validation for long-only strategy.
        """
        data = self.load_and_prepare_data()
        
        min_start_idx = self.min_initial_data + self.train_size + self.test_size
        
        results = []
        
        for test_end_idx in range(min_start_idx, len(data)):
            train_X, train_y, test_X, test_y = self.get_train_test_split(data, test_end_idx)
            
            if train_X is None:
                continue
            
            try:
                model = LongOnlyTreeEnsemble()
                model.fit(train_X, train_y)
                
                predictions = model.predict(test_X)
                probabilities = model.predict_proba(test_X)
                
                test_period_start = test_end_idx - self.test_size + 1
                test_dates = data['Date/Time'].iloc[test_period_start:test_end_idx+1]
                
                for i in range(len(predictions)):
                    actual_up = 1 if test_y[i] > model.up_threshold else 0
                    
                    results.append({
                        'date': test_dates.iloc[i],
                        'prediction': predictions[i],
                        'up_probability': probabilities[i],
                        'actual_up': actual_up,
                        'target_value': test_y[i],
                        'feature_value': test_X[i],
                        'test_end_idx': test_end_idx
                    })
                
            except Exception as e:
                print(f"Error at test_end_idx {test_end_idx}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def calculate_longonly_metrics(self, df):
        """
        Calculate performance metrics for long-only strategy.
        """
        if len(df) == 0:
            return {}
        
        # Overall metrics
        total_observations = len(df)
        long_signals = (df['prediction'] == 1).sum()
        no_trade_signals = (df['prediction'] == 0).sum()
        
        # When model says LONG, what happens?
        long_trades = df[df['prediction'] == 1]
        if len(long_trades) > 0:
            long_hit_rate = (long_trades['actual_up'] == 1).mean()
            long_avg_return = long_trades['target_value'].mean()
        else:
            long_hit_rate = 0
            long_avg_return = 0
        
        # When model says NO TRADE, what would have happened?
        no_trade = df[df['prediction'] == 0]
        if len(no_trade) > 0:
            no_trade_would_be_up = (no_trade['actual_up'] == 1).mean()
            no_trade_avg_return = no_trade['target_value'].mean()
        else:
            no_trade_would_be_up = 0
            no_trade_avg_return = 0
        
        # Base rates (what happens randomly)
        overall_up_rate = (df['actual_up'] == 1).mean()
        overall_avg_return = df['target_value'].mean()
        
        return {
            'total_observations': total_observations,
            'long_signals': long_signals,
            'long_signal_rate': long_signals / total_observations,
            'no_trade_signals': no_trade_signals,
            'no_trade_rate': no_trade_signals / total_observations,
            
            'long_hit_rate': long_hit_rate,
            'long_avg_return': long_avg_return,
            
            'no_trade_would_be_up': no_trade_would_be_up,
            'no_trade_avg_return': no_trade_avg_return,
            
            'overall_up_rate': overall_up_rate,
            'overall_avg_return': overall_avg_return,
            
            'edge_vs_base': long_hit_rate - overall_up_rate,
            'return_edge': long_avg_return - overall_avg_return
        }