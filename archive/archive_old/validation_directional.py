import pandas as pd
import numpy as np
import configparser
from preprocessing import DataPreprocessor
from model_directional import DirectionalTreeEnsemble

class DirectionalValidator:
    def __init__(self, config_path='config_longonly.ini'):
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.config.read(config_path)
        
        self.train_size = int(self.config['validation']['train_size'])
        self.test_size = int(self.config['validation']['test_size'])
        self.step_size = int(self.config['validation'].get('step_size', 1))
        self.min_training_samples = int(self.config['validation'].get('min_training_samples', 100))
        if 'selected_features' in self.config['data']:
            self.selected_features = [f.strip() for f in self.config['data']['selected_features'].split(',')]
        else:
            self.selected_features = [self.config['data']['selected_feature']]
        self.target_column = self.config['data']['target_column']
        self.date_column = self.config['data'].get('date_column', 'Date/Time')
        
        # Get volatility window and normalization settings to calculate minimum required data
        self.vol_window = int(self.config['preprocessing']['vol_window'])
        self.normalize_features = self.config['preprocessing'].getboolean('normalize_features', fallback=True)
        self.normalize_target = self.config['preprocessing'].getboolean('normalize_target', fallback=True)
        
        # Get model type and threshold
        self.model_type = self.config['model']['model_type']
        self.target_threshold = float(self.config['model']['target_threshold'])
        
        self.preprocessor = DataPreprocessor(config_path)
        
    def load_and_prepare_data(self):
        csv_file = self.config['data']['csv_file']
        df = pd.read_csv(csv_file)
        
        df[self.date_column] = pd.to_datetime(df[self.date_column], dayfirst=True)
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        processed_df = self.preprocessor.process_data(df)
        
        return processed_df
    
    def get_train_test_split(self, data, test_end_idx):
        test_start_idx = test_end_idx - self.test_size + 1
        train_end_idx = test_start_idx - 1
        train_start_idx = train_end_idx - self.train_size + 1
        
        if train_start_idx < 0:
            return None, None, None, None, None
        
        # Handle multiple features
        feature_cols = []
        for feature in self.selected_features:
            vol_adj_col = f'{feature}_vol_adj'
            if vol_adj_col in data.columns:
                feature_cols.append(vol_adj_col)
            else:
                feature_cols.append(feature)
        
        target_col = f'{self.target_column}_vol_adj'
        if target_col not in data.columns:
            target_col = self.target_column
        
        train_X = data[feature_cols].iloc[train_start_idx:train_end_idx+1].values
        train_y = data[target_col].iloc[train_start_idx:train_end_idx+1].values
        
        test_X = data[feature_cols].iloc[test_start_idx:test_end_idx+1].values
        test_y = data[target_col].iloc[test_start_idx:test_end_idx+1].values
        
        # Also get raw target values for P&L calculation (always use raw returns)
        test_y_raw = data[self.target_column].iloc[test_start_idx:test_end_idx+1].values
        
        # Handle NaN masking for multiple features
        if train_X.ndim == 1:
            train_mask = ~(np.isnan(train_X) | np.isnan(train_y))
        else:
            train_mask = ~(np.isnan(train_X).any(axis=1) | np.isnan(train_y))
        
        if test_X.ndim == 1:
            test_mask = ~np.isnan(test_X)
        else:
            test_mask = ~np.isnan(test_X).any(axis=1)
        
        if np.sum(train_mask) < self.min_training_samples:
            return None, None, None, None, None
        
        return (train_X[train_mask], train_y[train_mask], 
                test_X[test_mask], test_y[test_mask], test_y_raw[test_mask])
    
    def run_validation(self, verbose=True):
        """
        Run walk-forward validation for directional strategy.
        Automatically starts at the earliest point where complete data is available.
        """
        data = self.load_and_prepare_data()
        
        # Calculate minimum start index to ensure complete data
        # If normalization is enabled, need vol_window for volatility calculation
        # Otherwise, can start immediately
        if self.normalize_features or self.normalize_target:
            min_start_idx = self.vol_window + self.train_size + self.test_size
        else:
            min_start_idx = self.train_size + self.test_size
        
        results = []
        
        # Calculate total iterations for progress bar
        total_iterations = len(range(min_start_idx, len(data), self.step_size))
        current_iteration = 0
        
        if verbose:
            if self.normalize_features or self.normalize_target:
                print(f"Volatility window: {self.vol_window} days (normalization enabled)")
            else:
                print(f"Volatility normalization: DISABLED")
            print(f"  Features normalization: {'ON' if self.normalize_features else 'OFF'}")
            print(f"  Target normalization: {'ON' if self.normalize_target else 'OFF'}")
            print(f"Model type: {self.model_type}")
            print(f"Target threshold: {self.target_threshold}")
            print(f"Training window: {self.train_size} days")
            print(f"Test window: {self.test_size} days")
            print(f"Starting validation at observation {min_start_idx} (earliest with complete data)")
            print(f"Running walk-forward validation with {total_iterations} steps...")
        
        for test_end_idx in range(min_start_idx, len(data), self.step_size):
            train_X, train_y, test_X, test_y, test_y_raw = self.get_train_test_split(data, test_end_idx)
            
            if train_X is None:
                continue
            
            try:
                model = DirectionalTreeEnsemble(verbose=False)
                model.fit(train_X, train_y)
                
                predictions = model.predict(test_X)
                probabilities = model.predict_proba(test_X)
                
                test_period_start = test_end_idx - self.test_size + 1
                test_dates = data[self.date_column].iloc[test_period_start:test_end_idx+1]
                
                for i in range(len(predictions)):
                    # Determine if the actual move was profitable for this strategy
                    if self.model_type == 'longonly':
                        actual_profitable = 1 if test_y[i] > self.target_threshold else 0
                    else:  # shortonly
                        actual_profitable = 1 if test_y[i] < -self.target_threshold else 0
                    
                    # Store multiple feature values
                    feature_dict = {}
                    for j, feature_name in enumerate(self.selected_features):
                        feature_dict[f'{feature_name}_value'] = test_X[i, j] if test_X.ndim > 1 else test_X[i]
                    
                    result_dict = {
                        'date': test_dates.iloc[i],
                        'prediction': predictions[i],
                        'probability': probabilities[i],
                        'actual_profitable': actual_profitable,
                        'target_value': test_y_raw[i],  # Use RAW returns for P&L
                        'target_value_normalized': test_y[i],  # Store normalized value too for reference
                        'test_end_idx': test_end_idx
                    }
                    result_dict.update(feature_dict)
                    
                    results.append(result_dict)
                
            except Exception as e:
                if verbose:
                    print(f"Error at test_end_idx {test_end_idx}: {e}")
                continue
            
            # Update progress
            current_iteration += 1
            if verbose:
                progress = current_iteration / total_iterations * 100
                bar_length = 30
                filled_length = int(bar_length * current_iteration // total_iterations)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                print(f'\rProgress: |{bar}| {progress:.1f}% ({current_iteration}/{total_iterations})', end='', flush=True)
        
        if verbose:
            print()  # New line after progress bar
        
        return pd.DataFrame(results)
    
    def filter_by_date(self, df, start_date=None):
        """
        Filter results to only include dates after start_date.
        If start_date is None, try to get it from config.
        """
        if start_date is None and 'validation_start_date' in self.config['validation']:
            start_date = self.config['validation']['validation_start_date']
        
        if start_date:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            start_date = pd.to_datetime(start_date)
            return df[df['date'] >= start_date]
        
        return df
    
    def calculate_directional_metrics(self, df, filter_date=True):
        """
        Calculate performance metrics for directional strategy.
        Works for both longonly and shortonly model types.
        Optionally filters to validation_start_date for out-of-sample comparison.
        """
        if filter_date:
            df = self.filter_by_date(df)
        
        if len(df) == 0:
            return {}
        
        # Overall metrics
        total_observations = len(df)
        trade_signals = (df['prediction'] == 1).sum()
        no_trade_signals = (df['prediction'] == 0).sum()
        
        # When model says TRADE, what happens?
        trades = df[df['prediction'] == 1]
        if len(trades) > 0:
            hit_rate = (trades['actual_profitable'] == 1).mean()
            
            # For P&L calculation, we need to consider the trading direction
            if self.model_type == 'longonly':
                # Long trades: profit = positive returns
                avg_return = trades['target_value'].mean()
                total_pnl = trades['target_value'].sum()
            else:  # shortonly
                # Short trades: profit = negative returns (multiply by -1)
                avg_return = -trades['target_value'].mean()
                total_pnl = -trades['target_value'].sum()
        else:
            hit_rate = 0
            avg_return = 0
            total_pnl = 0
        
        # When model says NO TRADE, what would have happened?
        no_trade = df[df['prediction'] == 0]
        if len(no_trade) > 0:
            no_trade_would_be_profitable = (no_trade['actual_profitable'] == 1).mean()
            if self.model_type == 'longonly':
                no_trade_avg_return = no_trade['target_value'].mean()
            else:  # shortonly
                no_trade_avg_return = -no_trade['target_value'].mean()
        else:
            no_trade_would_be_profitable = 0
            no_trade_avg_return = 0
        
        # Base rates (what happens randomly)
        overall_profitable_rate = (df['actual_profitable'] == 1).mean()
        if self.model_type == 'longonly':
            overall_avg_return = df['target_value'].mean()
        else:  # shortonly
            overall_avg_return = -df['target_value'].mean()
        
        # Calculate additional metrics
        trading_frequency = trade_signals / total_observations
        
        # Monthly P&L analysis
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['year_month'] = df_copy['date'].dt.to_period('M')
        
        monthly_pnl = []
        for month in df_copy['year_month'].unique():
            month_data = df_copy[df_copy['year_month'] == month]
            month_trades = month_data[month_data['prediction'] == 1]
            if len(month_trades) > 0:
                if self.model_type == 'longonly':
                    month_pnl = month_trades['target_value'].sum()
                else:  # shortonly
                    month_pnl = -month_trades['target_value'].sum()
                monthly_pnl.append(month_pnl)
            else:
                monthly_pnl.append(0)
        
        if len(monthly_pnl) > 0:
            avg_monthly_pnl = np.mean(monthly_pnl)
            std_monthly_pnl = np.std(monthly_pnl, ddof=1) if len(monthly_pnl) > 1 else 0
            sharpe_ratio = avg_monthly_pnl / std_monthly_pnl if std_monthly_pnl > 0 else 0
            positive_months_pct = np.mean(np.array(monthly_pnl) > 0)
            best_month = np.max(monthly_pnl)
            worst_month = np.min(monthly_pnl)
        else:
            avg_monthly_pnl = 0
            std_monthly_pnl = 0
            sharpe_ratio = 0
            positive_months_pct = 0
            best_month = 0
            worst_month = 0
        
        return {
            'total_observations': total_observations,
            'total_trades': trade_signals,
            'trading_frequency': trading_frequency,
            'no_trade_signals': no_trade_signals,
            'no_trade_rate': no_trade_signals / total_observations,
            
            'hit_rate': hit_rate,
            'avg_return': avg_return,
            'total_pnl': total_pnl,
            
            'no_trade_would_be_profitable': no_trade_would_be_profitable,
            'no_trade_avg_return': no_trade_avg_return,
            
            'overall_profitable_rate': overall_profitable_rate,
            'overall_avg_return': overall_avg_return,
            
            'edge': hit_rate - overall_profitable_rate,
            'return_edge': avg_return - overall_avg_return,
            
            'avg_monthly_pnl': avg_monthly_pnl,
            'std_monthly_pnl': std_monthly_pnl,
            'sharpe_ratio': sharpe_ratio,
            'positive_months_pct': positive_months_pct,
            'best_month': best_month,
            'worst_month': worst_month,
            
            'model_type': self.model_type,
            'target_threshold': self.target_threshold
        }