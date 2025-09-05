import pandas as pd
import numpy as np
import configparser
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from functools import partial
from src.OMtree_preprocessing import DataPreprocessor
from src.OMtree_model import DirectionalTreeEnsemble
from src.date_parser import FlexibleDateParser
from src.column_detector import ColumnDetector
from src.feature_selector import FeatureSelector

class DirectionalValidator:
    def __init__(self, config_path='OMtree_config.ini'):
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
        self.time_column = self.config['data'].get('time_column', 'Time')
        
        # Get volatility window and normalization settings to calculate minimum required data
        self.vol_window = int(float(self.config['preprocessing']['vol_window']))
        self.normalize_features = self.config['preprocessing'].getboolean('normalize_features', fallback=True)
        self.normalize_target = self.config['preprocessing'].getboolean('normalize_target', fallback=True)
        
        # Get model type and threshold
        self.model_type = self.config['model']['model_type']
        self.target_threshold = float(self.config['model']['target_threshold'])
        
        # Get feature selection configuration
        self.feature_selection_enabled = False
        self.feature_selector = None
        self.selection_history = []  # Store selected features at each step
        self.last_model = None  # Store the last trained model for saving
        
        # Get model training settings that will be used for RF feature selection
        self.n_trees_method = self.config['model'].get('n_trees_method', 'absolute')
        self.n_trees_base = int(float(self.config['model'].get('n_trees', 100)))
        self.model_max_depth = int(float(self.config['model'].get('max_depth', 2)))
        self.model_bootstrap_fraction = float(self.config['model'].get('bootstrap_fraction', 0.75))
        self.model_min_leaf_fraction = float(self.config['model'].get('min_leaf_fraction', 0.25))
        self.algorithm = self.config['model'].get('algorithm', 'decision_trees')
        self.balanced_bootstrap = self.config['model'].getboolean('balanced_bootstrap', fallback=False)
        self.n_jobs = int(self.config['model'].get('n_jobs', -1))
        self.regression_mode = self.config['model'].getboolean('regression_mode', fallback=False)
        
        # Initialize feature selection variables with defaults
        self.fs_min_features = 1
        self.fs_max_features = 8
        self.fs_importance_threshold = 0.0
        self.fs_threshold_mode = 'minimum'
        self.fs_cumulative_threshold = 0.95
        self.selection_lookback = 500
        
        if 'feature_selection' in self.config:
            self.feature_selection_enabled = self.config['feature_selection'].getboolean('enabled', fallback=False)
            if self.feature_selection_enabled:
                # Only get the feature selection specific settings
                self.fs_min_features = int(self.config['feature_selection'].get('min_features', 1))
                self.fs_max_features = int(self.config['feature_selection'].get('max_features', 8))
                self.fs_importance_threshold = float(self.config['feature_selection'].get('importance_threshold', 0.0))
                self.selection_lookback = int(self.config['feature_selection'].get('selection_lookback', 500))
                
                # Get threshold mode settings
                self.fs_threshold_mode = self.config['feature_selection'].get('threshold_mode', 'minimum')
                self.fs_cumulative_threshold = float(self.config['feature_selection'].get('cumulative_threshold', 0.95))
        
        self.preprocessor = DataPreprocessor(config_path)
        
    def load_and_prepare_data(self):
        csv_file = self.config['data']['csv_file']
        df = pd.read_csv(csv_file)
        
        # Apply ticker filter if specified
        if 'Ticker' in df.columns and 'ticker_filter' in self.config['data']:
            ticker_filter = self.config['data'].get('ticker_filter', 'All')
            if ticker_filter != 'All':
                df = df[df['Ticker'] == ticker_filter].copy()
                print(f"Filtered data to ticker {ticker_filter}: {len(df)} rows")
        
        # Apply hour filter if specified for hourly data
        if 'Hour' in df.columns and 'hour_filter' in self.config['data']:
            hour_filter = self.config['data'].get('hour_filter', 'All')
            if hour_filter != 'All':
                try:
                    hour_val = int(hour_filter)
                    df = df[df['Hour'] == hour_val].copy()
                    print(f"Filtered data to hour {hour_val}: {len(df)} rows")
                except:
                    pass
        
        # Check if dayfirst setting exists in config, default to False for YYYY-MM-DD format
        dayfirst = None  # Let parser auto-detect if not specified
        if 'validation' in self.config and 'date_format_dayfirst' in self.config['validation']:
            dayfirst_str = self.config['validation']['date_format_dayfirst'].lower()
            if dayfirst_str == 'true':
                dayfirst = True
            elif dayfirst_str == 'false':
                dayfirst = False
        
        # Use flexible date parser to handle multiple formats
        date_columns = FlexibleDateParser.get_date_columns(df)
        
        # Try to get date and time columns from config first
        date_col = self.config['data'].get('date_column', date_columns.get('date_column'))
        time_col = self.config['data'].get('time_column', date_columns.get('time_column'))
        datetime_col = date_columns.get('datetime_column')
        
        # Parse dates using the flexible parser
        try:
            parsed_dates = FlexibleDateParser.parse_dates(
                df, 
                date_column=date_col,
                time_column=time_col,
                datetime_column=datetime_col,
                dayfirst=dayfirst
            )
            
            # Add parsed dates to dataframe with a standard column name
            df['parsed_datetime'] = parsed_dates
            self.date_column = 'parsed_datetime'
            
        except Exception as e:
            print(f"Warning: Could not parse dates automatically: {e}")
            print("Attempting fallback parsing...")
            # Fallback to original date column if it exists
            if self.date_column in df.columns:
                df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
            else:
                raise ValueError(f"Could not find or parse date column. Available columns: {df.columns.tolist()}")
        
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        # Filter data up to validation_end_date if specified
        if 'validation' in self.config and 'validation_end_date' in self.config['validation']:
            validation_end_date = pd.to_datetime(self.config['validation']['validation_end_date'])
            original_len = len(df)
            df = df[df[self.date_column] <= validation_end_date].copy()
            filtered_len = len(df)
            if original_len > filtered_len:
                print(f"Data filtered to validation_end_date ({validation_end_date.strftime('%Y-%m-%d')})")
                print(f"  Using {filtered_len} of {original_len} rows ({original_len - filtered_len} rows reserved for out-of-sample)")
        
        processed_df, feature_cols, target_col = self.preprocessor.process_data(df)
        
        # Store the processed column names for later use
        self.processed_feature_columns = feature_cols
        self.processed_target_column = target_col
        
        return processed_df
    
    def get_train_test_split(self, data, test_end_idx):
        test_start_idx = test_end_idx - self.test_size + 1
        train_end_idx = test_start_idx - 1
        train_start_idx = train_end_idx - self.train_size + 1
        
        if train_start_idx < 0:
            return None, None, None, None, None
        
        # Generate random noise features if requested
        noise_features = [f for f in self.selected_features if f.startswith('RandomNoise_')]
        if noise_features:
            np.random.seed(42)  # Fixed seed for reproducibility
            for noise_col in noise_features:
                if noise_col not in data.columns:
                    # Generate Gaussian random data
                    data[noise_col] = np.random.randn(len(data))
                    # Add to both raw and vol_adj versions for consistency
                    data[f'{noise_col}_vol_adj'] = data[noise_col].copy()
                    print(f"Generated random noise feature: {noise_col}")
        
        # Handle multiple features - be flexible with missing columns
        feature_cols = []
        for feature in self.selected_features:
            vol_adj_col = f'{feature}_vol_adj'
            if vol_adj_col in data.columns:
                feature_cols.append(vol_adj_col)
            elif feature in data.columns:
                feature_cols.append(feature)
            else:
                # Try to find similar column (skip noise features as they're already handled)
                if not feature.startswith('RandomNoise_'):
                    similar = ColumnDetector.find_similar_columns(data, feature, threshold=0.5)
                    if similar:
                        print(f"Warning: Feature '{feature}' not found, using similar column '{similar[0]}'")
                        feature_cols.append(similar[0])
        
        # If no features found, auto-detect
        if not feature_cols:
            detected = ColumnDetector.auto_detect_columns(data)
            feature_cols = detected['features'][:5]  # Use top 5 features
            print(f"Warning: No configured features found, auto-detected: {feature_cols}")
        
        
        # Find the actual target column - check for vol_adj first, then original
        target_col = f'{self.target_column}_vol_adj'
        raw_target_col = self.target_column
        
        if target_col not in data.columns:
            if self.target_column in data.columns:
                target_col = self.target_column
                raw_target_col = self.target_column
            else:
                # Target doesn't exist, try to find a similar one
                detected = ColumnDetector.auto_detect_columns(data)
                if detected['targets']:
                    # Use first detected target
                    raw_target_col = detected['targets'][0]
                    target_col = f'{raw_target_col}_vol_adj'
                    if target_col not in data.columns:
                        target_col = raw_target_col
                    print(f"Warning: Target '{self.target_column}' not found, using '{raw_target_col}'")
                else:
                    raise ValueError(f"No target column found. Config target '{self.target_column}' not in data.")
        
        train_X = data[feature_cols].iloc[train_start_idx:train_end_idx+1].values
        train_y = data[target_col].iloc[train_start_idx:train_end_idx+1].values
        
        test_X = data[feature_cols].iloc[test_start_idx:test_end_idx+1].values
        test_y = data[target_col].iloc[test_start_idx:test_end_idx+1].values
        
        # Also get raw target values for P&L calculation (always use raw returns)
        test_y_raw = data[raw_target_col].iloc[test_start_idx:test_end_idx+1].values
        
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
        Always starts 250 observations in to ensure fair comparison between normalized
        and non-normalized runs, accommodating normalization windows up to 250 days.
        """
        data = self.load_and_prepare_data()
        
        # Always start 250 observations in to ensure fair comparison
        # This accommodates normalization windows up to 250 days
        min_start_idx = 250 + self.train_size + self.test_size
        
        results = []
        all_predictions_detailed = []  # Store point-in-time predictions with dates
        
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
            print(f"Mode: {'REGRESSION' if self.regression_mode else 'CLASSIFICATION'}")
            if self.feature_selection_enabled:
                print(f"Feature selection: ENABLED (Random Forest MDI)")
                print(f"  Min/Max features: {self.fs_min_features}/{self.fs_max_features}")
                print(f"  Importance threshold: {self.fs_importance_threshold}")
                print(f"  Using model settings for RF")
            else:
                print(f"Feature selection: DISABLED")
            print(f"Training window: {self.train_size} days")
            print(f"Test window: {self.test_size} days")
            print(f"Starting validation at observation {min_start_idx} (fixed 250 + train + test for fair comparison)")
            print(f"Running walk-forward validation with {total_iterations} steps...")
        
        for test_end_idx in range(min_start_idx, len(data), self.step_size):
            train_X, train_y, test_X, test_y, test_y_raw = self.get_train_test_split(data, test_end_idx)
            
            if train_X is None:
                continue
            
            try:
                # Apply feature selection if enabled
                if self.feature_selection_enabled:
                    # Calculate settings for feature selector
                    n_features = train_X.shape[1] if train_X.ndim > 1 else 1
                    if self.n_trees_method == 'per_feature':
                        # Use same logic as model: n_trees per feature, testing all features
                        rf_n_trees = self.n_trees_base * n_features
                    else:
                        # Absolute mode: use the configured number
                        rf_n_trees = self.n_trees_base
                    
                    # Convert min_leaf_fraction to min_samples_leaf
                    # Use bootstrap size for calculation, same as the actual model
                    bootstrap_size = int(self.selection_lookback * self.model_bootstrap_fraction)
                    min_samples_leaf = max(1, int(bootstrap_size * self.model_min_leaf_fraction))
                    # Ensure it's reasonable (at least 1)
                    min_samples_leaf = max(1, min_samples_leaf)
                    
                    # Check if we need to recreate the feature selector (settings changed or first time)
                    need_recreate = (
                        self.feature_selector is None or
                        not hasattr(self, '_last_fs_settings') or
                        self._last_fs_settings != (rf_n_trees, self.model_max_depth, min_samples_leaf, 
                                                   self.model_bootstrap_fraction, self.fs_importance_threshold,
                                                   self.fs_min_features, self.fs_max_features, self.target_threshold, 
                                                   self.regression_mode)
                    )
                    
                    if need_recreate:
                        # Store current settings to detect future changes
                        self._last_fs_settings = (rf_n_trees, self.model_max_depth, min_samples_leaf,
                                                 self.model_bootstrap_fraction, self.fs_importance_threshold,
                                                 self.fs_min_features, self.fs_max_features, self.target_threshold,
                                                 self.regression_mode)
                        
                        # Create feature selector with model settings
                        cumulative_mode = self.fs_threshold_mode == 'cumulative'
                        self.feature_selector = FeatureSelector(
                            n_features=n_features,  # Will select based on threshold/min/max
                            min_features=self.fs_min_features,
                            max_features=self.fs_max_features,
                            rf_n_estimators=rf_n_trees,
                            rf_max_depth=self.model_max_depth,
                            rf_min_samples_leaf=min_samples_leaf,
                            rf_bootstrap_fraction=self.model_bootstrap_fraction,
                            importance_threshold=self.fs_importance_threshold,
                            algorithm=self.algorithm,  # Use same algorithm as model
                            balanced_bootstrap=self.balanced_bootstrap,  # Use same bootstrap strategy
                            random_seed=42,  # Fixed seed for reproducibility
                            n_jobs=self.n_jobs,  # Use configured n_jobs
                            target_threshold=self.target_threshold,  # Use same threshold as model
                            regression_mode=self.regression_mode,  # Use same mode as model
                            model_type=self.model_type,  # Pass model type for correct threshold comparison
                            cumulative_importance_mode=cumulative_mode,
                            cumulative_importance_threshold=self.fs_cumulative_threshold
                        )
                        
                        if verbose:
                            if current_iteration == 1:
                                print(f"\nFeature selector RF settings (from model config):")
                            else:
                                print(f"\nFeature selector settings changed - recreating:")
                            print(f"  Trees: {rf_n_trees} ({'per_feature' if self.n_trees_method == 'per_feature' else 'absolute'} mode)")
                            print(f"  Max depth: {self.model_max_depth}")
                            print(f"  Min samples leaf: {min_samples_leaf} (from {self.model_min_leaf_fraction:.1%} min_leaf)")
                            print(f"  Bootstrap fraction: {self.model_bootstrap_fraction}")
                    
                    # Use recent data for feature selection (up to selection_lookback samples)
                    selection_data_size = min(self.selection_lookback, len(train_X))
                    X_selection = train_X[-selection_data_size:]
                    y_selection = train_y[-selection_data_size:]
                    
                    # Select features
                    selected_indices, selected_names = self.feature_selector.select_features(
                        X_selection, y_selection, 
                        feature_names=self.selected_features,
                        verbose=False
                    )
                    
                    # Store selection history
                    self.selection_history.append({
                        'test_end_idx': test_end_idx,
                        'selected_features': selected_names,
                        'selection_scores': self.feature_selector.selection_scores.copy()
                    })
                    
                    # Use only selected features for training and testing
                    train_X_selected = train_X[:, selected_indices]
                    test_X_selected = test_X[:, selected_indices]
                else:
                    # Use all features (current behavior)
                    train_X_selected = train_X
                    test_X_selected = test_X
                    selected_names = self.selected_features
                
                model = DirectionalTreeEnsemble(verbose=False)
                model.fit(train_X_selected, train_y)
                
                # Auto-calibrate threshold if enabled
                if model.auto_calibrate_threshold:
                    # Use the last calibration_lookback samples from training data
                    calibration_size = min(model.calibration_lookback, len(train_X_selected))
                    X_calib = train_X_selected[-calibration_size:]
                    y_calib = train_y[-calibration_size:]
                    
                    # Calibrate threshold to achieve target prediction rate
                    calibrated_threshold = model.calibrate_threshold(X_calib, y_calib)
                    
                    # Store calibration history if first time or if it changed significantly
                    if not hasattr(self, 'calibration_history'):
                        self.calibration_history = []
                    self.calibration_history.append({
                        'test_end_idx': test_end_idx,
                        'threshold': calibrated_threshold,
                        'type': 'vote' if model.convert_to_binary else 'trade'
                    })
                
                # Store the last model for saving later
                self.last_model = model
                
                # Collect feature importance with feature names
                importance = model.get_feature_importance()
                if importance is not None:
                    if not hasattr(self, 'feature_importances'):
                        self.feature_importances = []
                        self.feature_importance_names = []
                    self.feature_importances.append(importance)
                    self.feature_importance_names.append(selected_names)
                
                predictions = model.predict(test_X_selected)
                probabilities = model.predict_proba(test_X_selected)
                
                test_period_start = test_end_idx - self.test_size + 1
                test_dates = data[self.date_column].iloc[test_period_start:test_end_idx+1]
                test_times = data[self.time_column].iloc[test_period_start:test_end_idx+1] if self.time_column in data.columns else None
                
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
                    
                    # Store detailed prediction for visualizer
                    # For shortonly, we need to invert the PnL (we profit when price goes down)
                    if self.model_type == 'longonly':
                        pnl = test_y_raw[i] if predictions[i] == 1 else 0
                    else:  # shortonly
                        pnl = -test_y_raw[i] if predictions[i] == 1 else 0
                    
                    detailed_pred = {
                        'date': test_dates.iloc[i],
                        'time': test_times.iloc[i] if test_times is not None else '',
                        'prediction': probabilities[i] if self.regression_mode else predictions[i],
                        'actual': test_y_raw[i],  # Raw target value
                        'signal': predictions[i],
                        'pnl': pnl
                    }
                    all_predictions_detailed.append(detailed_pred)
                
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
        
        # Generate feature importance chart if data was collected
        if hasattr(self, 'feature_importances') and self.feature_importances:
            self.generate_feature_importance_chart()
        
        # Generate feature selection report if enabled
        if self.feature_selection_enabled and self.selection_history:
            self.generate_feature_selection_report()
            # Save selection history for visualization
            self.save_selection_history()
        
        # Save the final model for visualization
        if self.last_model is not None:
            self.save_final_model(self.last_model)
        
        # Save walk-forward predictions for visualization
        if all_predictions_detailed:
            pred_df = pd.DataFrame(all_predictions_detailed)
            os.makedirs('results', exist_ok=True)
            pred_df.to_csv(f'results/walkforward_results_{self.model_type}.csv', index=False)
            if verbose:
                print(f"Saved {len(pred_df)} predictions to results/walkforward_results_{self.model_type}.csv")
            
            # Export debug CSV with returns time series for equity curve debugging
            # Skip debug CSV export during permutation runs to avoid clutter
            if not os.environ.get('PERMUTATION_RUN'):
                self._export_returns_debug_csv(pred_df, verbose)
        
        # Save calibration history if auto-calibration was used
        if hasattr(self, 'calibration_history') and self.calibration_history:
            self.save_calibration_history()
            if verbose:
                print(f"Saved calibration history ({len(self.calibration_history)} calibrations)")
        
        return pd.DataFrame(results)
    
    def _export_returns_debug_csv(self, pred_df, verbose=True):
        """Export detailed returns time series for debugging equity curves"""
        try:
            # Check which columns we have
            if 'signal' in pred_df.columns:
                # New format: has 'signal' column for trades (1/0)
                trades_df = pred_df[pred_df['signal'] == 1].copy()
            elif 'prediction' in pred_df.columns and pred_df['prediction'].dtype == 'int64':
                # Standard format: prediction is 1/0
                trades_df = pred_df[pred_df['prediction'] == 1].copy()
            else:
                # Old format: prediction is probability, need to check threshold
                # For this format, all rows might be trades
                trades_df = pred_df.copy()
            
            if len(trades_df) == 0:
                return
            
            # Sort by date
            if 'date' in trades_df.columns:
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trades_df = trades_df.sort_values('date').reset_index(drop=True)
            
            # Create debug dataframe with all return calculations
            debug_df = pd.DataFrame()
            debug_df['trade_num'] = range(1, len(trades_df) + 1)
            
            if 'date' in trades_df.columns:
                debug_df['date'] = trades_df['date'].values
            
            # Get the return value based on available columns
            if 'target_value' in trades_df.columns:
                debug_df['target_value'] = trades_df['target_value'].values
            elif 'pnl' in trades_df.columns:
                debug_df['target_value'] = trades_df['pnl'].values
            elif 'actual' in trades_df.columns:
                debug_df['target_value'] = trades_df['actual'].values
            else:
                print("Warning: No return column found (target_value, pnl, or actual)")
                return
            
            # P&L calculation based on model type
            if self.model_type == 'shortonly':
                debug_df['pnl'] = -debug_df['target_value']
            else:
                debug_df['pnl'] = debug_df['target_value']
            
            # Simple cumulative sum (what console shows)
            debug_df['cumsum_pnl'] = debug_df['pnl'].cumsum()
            
            # Convert to decimal for compounding
            debug_df['pnl_decimal'] = debug_df['pnl'] / 100.0
            
            # Compound returns (starting from 100)
            portfolio_value = [100.0]
            for pnl_dec in debug_df['pnl_decimal'].values:
                new_value = portfolio_value[-1] * (1 + pnl_dec)
                portfolio_value.append(new_value)
            debug_df['portfolio_value'] = portfolio_value[1:]  # Remove initial 100
            
            # Compound return percentage
            debug_df['compound_return_pct'] = (debug_df['portfolio_value'] - 100)
            
            # Add actual profitable flag if available
            if 'actual_profitable' in trades_df.columns:
                debug_df['actual_profitable'] = trades_df['actual_profitable'].values
                debug_df['hit_rate_cumulative'] = debug_df['actual_profitable'].expanding().mean()
            elif 'actual' in trades_df.columns and 'signal' in trades_df.columns:
                # For old format, profitable means actual return was positive for longs
                if self.model_type == 'shortonly':
                    debug_df['actual_profitable'] = (trades_df['actual'].values < 0).astype(int)
                else:
                    debug_df['actual_profitable'] = (trades_df['actual'].values > 0).astype(int)
                debug_df['hit_rate_cumulative'] = debug_df['actual_profitable'].expanding().mean()
            
            # Add probability if available
            if 'probability' in trades_df.columns:
                debug_df['probability'] = trades_df['probability'].values
            
            # Calculate drawdown from peak
            running_max = debug_df['cumsum_pnl'].expanding().max()
            debug_df['drawdown'] = debug_df['cumsum_pnl'] - running_max
            debug_df['drawdown_pct'] = (debug_df['drawdown'] / running_max.replace(0, np.nan)) * 100
            
            # Save debug CSV
            debug_file = f'results/returns_debug_{self.model_type}.csv'
            debug_df.to_csv(debug_file, index=False)
            
            if verbose:
                print(f"\n=== RETURNS DEBUG INFO ===")
                print(f"Exported returns time series to: {debug_file}")
                print(f"Total trades: {len(debug_df)}")
                print(f"Sum of PnL (console result): {debug_df['cumsum_pnl'].iloc[-1]:.2f}%")
                print(f"Compound return: {debug_df['compound_return_pct'].iloc[-1]:.2f}%")
                print(f"Max drawdown: {debug_df['drawdown'].min():.2f}%")
                if 'actual_profitable' in debug_df.columns:
                    print(f"Overall hit rate: {debug_df['actual_profitable'].mean()*100:.1f}%")
                print("="*30)
                
        except Exception as e:
            print(f"Warning: Could not export returns debug CSV: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_feature_importance_chart(self):
        """Generate feature importance chart from walk-forward validation"""
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'feature_importances') or not self.feature_importances:
            return
        
        # Handle feature selection case where features may vary
        if hasattr(self, 'feature_importance_names') and self.feature_importance_names:
            # When using feature selection, aggregate importance by feature name
            feature_importance_dict = {}
            feature_count_dict = {}
            
            for importances, names in zip(self.feature_importances, self.feature_importance_names):
                for imp, name in zip(importances, names):
                    if name not in feature_importance_dict:
                        feature_importance_dict[name] = []
                    feature_importance_dict[name].append(imp)
                    feature_count_dict[name] = feature_count_dict.get(name, 0) + 1
            
            # Get all unique features and their average importance
            all_features = list(feature_importance_dict.keys())
            avg_importance = np.array([np.mean(feature_importance_dict[f]) for f in all_features])
            std_importance = np.array([np.std(feature_importance_dict[f]) if len(feature_importance_dict[f]) > 1 else 0 
                                      for f in all_features])
            feature_names = all_features
        else:
            # Standard case without feature selection
            avg_importance = np.mean(self.feature_importances, axis=0)
            std_importance = np.std(self.feature_importances, axis=0)
            
            # Ensure we have feature names
            if not hasattr(self, 'selected_features') or not self.selected_features:
                feature_names = [f'Feature_{i+1}' for i in range(len(avg_importance))]
            else:
                feature_names = self.selected_features
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart with error bars
        y_pos = np.arange(len(feature_names))
        ax1.barh(y_pos, avg_importance, xerr=std_importance if len(self.feature_importances) > 1 else None,
                 color='steelblue', alpha=0.8, capsize=5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feature_names)
        ax1.set_xlabel('Relative Importance')
        ax1.set_title(f'Feature Importance (Averaged over {len(self.feature_importances)} models)', 
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, v in enumerate(avg_importance):
            ax1.text(v + 0.001, i, f'{v*100:.1f}%', va='center')
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
        explode = [0.05 if imp > avg_importance.mean() else 0 for imp in avg_importance]
        
        ax2.pie(avg_importance, labels=feature_names, autopct='%1.1f%%',
                colors=colors, explode=explode, shadow=True, startangle=90)
        ax2.set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'Walk-Forward Feature Importance Analysis\n{self.model_type.title()} Model',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\nFeature importance chart updated: feature_importance.png")
        
        # Save importance scores to CSV
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_importance,
            'Std_Dev': std_importance if len(self.feature_importances) > 1 else [0] * len(avg_importance),
            'Percentage': avg_importance * 100
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv('feature_importance.csv', index=False)
        
        print("Feature importance scores saved: feature_importance.csv")
        print("\nTop Features:")
        print(importance_df.head())
    
    def run_validation_parallel(self, verbose=True, max_workers=None):
        """
        Run walk-forward validation in parallel using multiple CPU cores.
        This can significantly speed up validation when you have many cores available.
        
        Args:
            verbose: Print progress information
            max_workers: Maximum number of worker processes. If None, uses n_jobs setting.
                        If n_jobs is -1, uses all available cores.
        
        Returns:
            DataFrame with validation results
        """
        data = self.load_and_prepare_data()
        
        # Always start 250 observations in to ensure fair comparison
        min_start_idx = 250 + self.train_size + self.test_size
        
        # Calculate time steps for validation
        time_steps = list(range(min_start_idx, len(data), self.step_size))
        total_steps = len(time_steps)
        
        if verbose:
            print(f"\n=== PARALLEL WALK-FORWARD VALIDATION ===")
            print(f"Total time steps to process: {total_steps}")
            
            # Determine number of workers
            if max_workers is None:
                if self.n_jobs == -1:
                    max_workers = cpu_count()
                elif self.n_jobs == -2:
                    max_workers = max(1, cpu_count() - 1)
                else:
                    max_workers = max(1, self.n_jobs)
            
            print(f"Using {max_workers} parallel workers (cores)")
            print(f"Expected speedup: ~{min(max_workers, total_steps/max_workers):.1f}x")
            
            if self.normalize_features or self.normalize_target:
                print(f"Volatility window: {self.vol_window} days (normalization enabled)")
            else:
                print(f"Volatility normalization: DISABLED")
            print(f"  Features normalization: {'ON' if self.normalize_features else 'OFF'}")
            print(f"  Target normalization: {'ON' if self.normalize_target else 'OFF'}")
            print(f"Model type: {self.model_type}")
            print(f"Target threshold: {self.target_threshold}")
            print(f"Mode: {'REGRESSION' if self.regression_mode else 'CLASSIFICATION'}")
            if self.feature_selection_enabled:
                print(f"Feature selection: ENABLED (Random Forest MDI)")
                print(f"  Min/Max features: {self.fs_min_features}/{self.fs_max_features}")
                print(f"  Importance threshold: {self.fs_importance_threshold}")
            else:
                print(f"Feature selection: DISABLED")
            print(f"Training window: {self.train_size} days")
            print(f"Test window: {self.test_size} days")
            print(f"Processing...")
        
        # Prepare data serialization - convert to basic types for multiprocessing
        data_dict = {
            'data': data.to_dict('records'),
            'columns': data.columns.tolist(),
            'config_dict': dict(self.config._sections)  # Convert ConfigParser to dict
        }
        
        # Process in parallel
        all_results = []
        completed_count = 0
        
        # Use ProcessPoolExecutor for true parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_step = {}
            for step_idx in time_steps:
                future = executor.submit(
                    _process_validation_step,
                    data_dict,
                    step_idx,
                    self.train_size,
                    self.test_size,
                    self.selected_features,
                    self.target_column,
                    self.date_column
                )
                future_to_step[future] = step_idx
            
            # Collect results as they complete
            for future in as_completed(future_to_step):
                try:
                    result = future.result()
                    if result is not None:
                        all_results.extend(result)
                    
                    completed_count += 1
                    if verbose and completed_count % max(1, total_steps // 20) == 0:
                        progress = (completed_count / total_steps) * 100
                        print(f"Progress: {progress:.1f}% ({completed_count}/{total_steps} steps)")
                        
                except Exception as e:
                    step_idx = future_to_step[future]
                    if verbose:
                        print(f"Error in step {step_idx}: {e}")
        
        if verbose:
            print(f"Parallel processing completed! Processed {completed_count}/{total_steps} steps.")
            print(f"Generated {len(all_results)} total predictions.")
            if self.feature_selection_enabled:
                print("Note: Feature selection history not available in parallel mode.")
        
        # Convert results to DataFrame
        if all_results:
            df_results = pd.DataFrame(all_results)
            # Sort by date to maintain chronological order
            df_results['date'] = pd.to_datetime(df_results['date'])
            df_results = df_results.sort_values('date').reset_index(drop=True)
            return df_results
        else:
            return pd.DataFrame()
    
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
            # Calculate annualized Sharpe ratio (0% risk-free rate)
            # Sharpe = (Annual Return - Risk Free Rate) / Annual Volatility
            # With 0% risk-free rate: Sharpe = Annual Return / Annual Volatility
            annual_return = avg_monthly_pnl * 12
            annual_volatility = std_monthly_pnl * np.sqrt(12)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
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
    
    def generate_feature_selection_report(self, output_file='feature_selection_report.txt'):
        """Generate a report of feature selection history from walk-forward validation"""
        if not self.selection_history:
            print("No feature selection history available")
            return
        
        report = []
        report.append("="*80)
        report.append("FEATURE SELECTION REPORT - Random Forest MDI")
        report.append("="*80)
        report.append(f"Min/Max features: {self.fs_min_features}/{self.fs_max_features}")
        report.append(f"Selection lookback: {self.selection_lookback} samples")
        report.append(f"Importance threshold: {self.fs_importance_threshold}")
        report.append("")
        report.append("RF Configuration (from model settings):")
        if self.feature_selector:
            report.append(f"  Trees: {self.feature_selector.rf_n_estimators} ({self.n_trees_method} mode)")
            report.append(f"  Max depth: {self.feature_selector.rf_max_depth}")
            report.append(f"  Min samples leaf: {self.feature_selector.rf_min_samples_leaf}")
            report.append(f"  Bootstrap fraction: {self.feature_selector.rf_bootstrap_fraction}")
        else:
            report.append(f"  Trees: {self.n_trees_base} per feature" if self.n_trees_method == 'per_feature' else f"  Trees: {self.n_trees_base}")
            report.append(f"  Max depth: {self.model_max_depth}")
            report.append(f"  Min leaf fraction: {self.model_min_leaf_fraction}")
            report.append(f"  Bootstrap fraction: {self.model_bootstrap_fraction}")
        report.append("")
        
        # Analyze feature usage frequency
        feature_counts = {}
        feature_scores = {}
        
        for step in self.selection_history:
            for feat in step['selected_features']:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
                if feat not in feature_scores:
                    feature_scores[feat] = []
                if feat in step['selection_scores']:
                    feature_scores[feat].append(step['selection_scores'][feat])
        
        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        report.append("FEATURE USAGE FREQUENCY")
        report.append("-"*40)
        total_steps = len(self.selection_history)
        for feat, count in sorted_features:
            pct = (count / total_steps) * 100
            avg_score = np.mean(feature_scores[feat]) if feature_scores[feat] else 0
            report.append(f"{feat:30s} {count:4d}/{total_steps} ({pct:5.1f}%) | Avg Score: {avg_score:.4f}")
        
        report.append("")
        report.append("FEATURE SELECTION OVER TIME")
        report.append("-"*40)
        
        # Sample every N steps to avoid too much output
        sample_rate = max(1, len(self.selection_history) // 20)
        for i, step in enumerate(self.selection_history[::sample_rate]):
            report.append(f"Step {i*sample_rate}: {', '.join(step['selected_features'])}")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nFeature selection report saved to: {output_file}")
        
        # Print summary
        print("\nTop 5 Most Selected Features:")
        for feat, count in sorted_features[:5]:
            pct = (count / total_steps) * 100
            print(f"  {feat}: {pct:.1f}% of steps")
    
    def save_selection_history(self, filename='results/feature_selection_history.json'):
        """Save feature selection history for visualization"""
        import json
        
        if not self.selection_history:
            return
        
        # Get all unique features
        all_features = []
        for step in self.selection_history:
            for feat in step['selected_features']:
                if feat not in all_features:
                    all_features.append(feat)
        
        all_features.sort()
        
        data = {
            'history': self.selection_history,
            'all_features': all_features,
            'config': {
                'min_features': self.fs_min_features if hasattr(self, 'fs_min_features') else None,
                'max_features': self.fs_max_features if hasattr(self, 'fs_max_features') else None,
                'importance_threshold': self.fs_importance_threshold if hasattr(self, 'fs_importance_threshold') else None,
                'selection_lookback': self.selection_lookback,
                'model_type': self.model_type,
                'n_trees_method': self.n_trees_method,
                'n_trees_base': self.n_trees_base
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Feature selection history saved to: {filename}")
    
    def save_calibration_history(self, filename='results/calibration_history.json'):
        """Save threshold calibration history for visualization"""
        import json
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'calibration_history') or not self.calibration_history:
            return
        
        # Extract data for plotting
        test_indices = [c['test_end_idx'] for c in self.calibration_history]
        thresholds = [c['threshold'] for c in self.calibration_history]
        threshold_type = self.calibration_history[0]['type']
        
        # Create calibration chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_indices, thresholds, marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Walk-Forward Step (Test End Index)')
        ax.set_ylabel(f'{threshold_type.capitalize()} Threshold')
        ax.set_title(f'Auto-Calibrated Threshold Over Time (Target Rate: {self.last_model.target_prediction_rate:.1%})')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for original threshold
        original_threshold = self.last_model.config['model'].getfloat('vote_threshold' if threshold_type == 'vote' else 'trade_prediction_threshold')
        ax.axhline(y=original_threshold, color='red', linestyle='--', alpha=0.5, label=f'Original Threshold: {original_threshold:.3f}')
        
        # Add statistics
        avg_threshold = np.mean(thresholds)
        ax.axhline(y=avg_threshold, color='green', linestyle='--', alpha=0.5, label=f'Average: {avg_threshold:.3f}')
        ax.legend()
        
        plt.tight_layout()
        chart_filename = f'results/calibration_history_{self.model_type}.png'
        plt.savefig(chart_filename, dpi=100)
        plt.close()
        
        # Save to JSON
        data = {
            'calibration_history': self.calibration_history,
            'config': {
                'auto_calibrate': True,
                'target_prediction_rate': self.last_model.target_prediction_rate,
                'calibration_lookback': self.last_model.calibration_lookback,
                'model_type': self.model_type,
                'threshold_type': threshold_type
            },
            'statistics': {
                'avg_threshold': float(avg_threshold),
                'min_threshold': float(min(thresholds)),
                'max_threshold': float(max(thresholds)),
                'std_threshold': float(np.std(thresholds))
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Calibration history saved to: {filename} and {chart_filename}")
    
    def save_final_model(self, model, filename='final_model.pkl'):
        """Save the final trained model for visualization"""
        import pickle
        
        # Prepare model data for saving
        model_data = {
            'trees': model.trees if hasattr(model, 'trees') else [],
            'model': model.model if hasattr(model, 'model') else None,
            'feature_names': model.feature_names if hasattr(model, 'feature_names') else self.selected_features,
            'model_type': self.model_type,
            'signal_name': 'LONG' if self.model_type == 'longonly' else 'SHORT',
            'direction_name': 'UP' if self.model_type == 'longonly' else 'DOWN',
            'target_threshold': self.target_threshold,
            'effective_threshold': model.effective_threshold if hasattr(model, 'effective_threshold') else self.target_threshold,
            'algorithm': self.algorithm,
            'n_trees': self.n_trees_base,
            'max_depth': self.model_max_depth
        }
        
        # Save to pickle file
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Final model saved to: {filename}")


def _process_validation_step(data_dict, test_end_idx, train_size, test_size, selected_features, target_column, date_column):
    """
    Process a single validation step in parallel.
    This function must be at module level for multiprocessing to work.
    """
    try:
        # Reconstruct DataFrame from dict
        data = pd.DataFrame(data_dict['data'])
        data = data[data_dict['columns']]
        
        # Reconstruct config
        config = configparser.ConfigParser()
        for section_name, section_data in data_dict['config_dict'].items():
            if section_name != 'DEFAULT':
                config.add_section(section_name)
                for key, value in section_data.items():
                    config.set(section_name, key, str(value))
        
        # Create a temporary validator for this step
        temp_validator = DirectionalValidator()
        temp_validator.config = config
        
        # Set all the necessary attributes from the config
        temp_validator.train_size = train_size
        temp_validator.test_size = test_size
        temp_validator.selected_features = selected_features
        temp_validator.target_column = target_column
        temp_validator.date_column = date_column
        
        # Load all config values
        temp_validator.vol_window = int(float(config['preprocessing']['vol_window']))
        temp_validator.normalize_features = config['preprocessing'].getboolean('normalize_features', fallback=True)
        temp_validator.normalize_target = config['preprocessing'].getboolean('normalize_target', fallback=True)
        temp_validator.model_type = config['model']['model_type']
        temp_validator.target_threshold = float(config['model']['target_threshold'])
        temp_validator.n_jobs = int(config['model'].get('n_jobs', -1))  # Use configured n_jobs
        temp_validator.regression_mode = config['model'].getboolean('regression_mode', fallback=False)
        
        # Feature selection settings - match the sequential version exactly
        temp_validator.feature_selection_enabled = False
        if 'feature_selection' in config:
            temp_validator.feature_selection_enabled = config['feature_selection'].getboolean('enabled', fallback=False)
            if temp_validator.feature_selection_enabled:
                temp_validator.fs_min_features = int(config['feature_selection'].get('min_features', 1))
                temp_validator.fs_max_features = int(config['feature_selection'].get('max_features', 8))
                temp_validator.fs_importance_threshold = float(config['feature_selection'].get('importance_threshold', 0.0))
                temp_validator.selection_lookback = int(config['feature_selection'].get('selection_lookback', 500))
                temp_validator.fs_threshold_mode = config['feature_selection'].get('threshold_mode', 'minimum')
                temp_validator.fs_cumulative_threshold = float(config['feature_selection'].get('cumulative_threshold', 0.95))
        
        # Model training settings for feature selection
        temp_validator.n_trees_method = config['model'].get('n_trees_method', 'absolute')
        temp_validator.n_trees_base = int(float(config['model'].get('n_trees', 100)))
        temp_validator.model_max_depth = int(float(config['model'].get('max_depth', 2)))
        temp_validator.model_bootstrap_fraction = float(config['model'].get('bootstrap_fraction', 0.75))
        temp_validator.model_min_leaf_fraction = float(config['model'].get('min_leaf_fraction', 0.25))
        temp_validator.algorithm = config['model'].get('algorithm', 'decision_trees')
        temp_validator.balanced_bootstrap = config['model'].getboolean('balanced_bootstrap', fallback=False)
        
        # Initialize preprocessor with the config
        temp_validator.preprocessor = DataPreprocessor()
        temp_validator.preprocessor.config = config
        
        # Get train/test split
        train_X, train_y, test_X, test_y, test_y_raw = temp_validator.get_train_test_split(data, test_end_idx)
        
        if train_X is None:
            return None
        
        # Note: Feature selection is currently disabled in parallel mode to avoid complexity
        # Each worker would need to do its own feature selection which may lead to inconsistencies
        
        # Create and train model with consistent random seed
        model = DirectionalTreeEnsemble()
        model.config = config
        model.n_jobs = 1  # Single thread per worker to avoid nested parallelism
        
        # Set deterministic random seed based on test_end_idx for reproducibility
        # This ensures each time step always gets the same random seed
        model.random_seed = 42 + (test_end_idx % 1000)  # Deterministic but varying seed
        
        # Set model attributes
        model.model_type = temp_validator.model_type
        model.target_threshold = temp_validator.target_threshold
        model.algorithm = config['model'].get('algorithm', 'decision_trees')
        model.n_trees_method = config['model'].get('n_trees_method', 'absolute')
        model.n_trees_base = int(float(config['model'].get('n_trees', 100)))
        model.model_max_depth = int(float(config['model'].get('max_depth', 2)))
        model.model_bootstrap_fraction = float(config['model'].get('bootstrap_fraction', 0.75))
        model.model_min_leaf_fraction = float(config['model'].get('min_leaf_fraction', 0.25))
        model.balanced_bootstrap = config['model'].getboolean('balanced_bootstrap', fallback=False)
        model.regression_mode = temp_validator.regression_mode
        model.vote_threshold = float(config['model'].get('vote_threshold', 0.5))
        model.trade_prediction_threshold = float(config['model'].get('trade_prediction_threshold', 0.0))
        model.auto_calibrate_threshold = config['model'].getboolean('auto_calibrate_threshold', fallback=False)
        model.target_prediction_rate = float(config['model'].get('target_prediction_rate', 0.10))
        model.calibration_lookback = int(float(config['model'].get('calibration_lookback', 60)))
        
        # Train the model
        model.fit(train_X, train_y)
        
        # Make predictions
        predictions = model.predict(test_X)
        
        # Get test dates
        test_start_idx = test_end_idx - test_size
        test_dates = data.iloc[test_start_idx:test_end_idx][date_column]
        
        # Prepare results
        step_results = []
        for i, (pred, target_val, target_raw, date) in enumerate(zip(predictions, test_y, test_y_raw, test_dates)):
            # Determine profitability based on model type and target value
            if temp_validator.model_type == 'longonly':
                actual_profitable = target_raw > temp_validator.target_threshold
            else:  # shortonly
                actual_profitable = target_raw < -temp_validator.target_threshold
            
            result = {
                'date': date,
                'prediction': pred,
                'target_value': target_val,
                'target_value_raw': target_raw,
                'actual_profitable': actual_profitable,
                'step_idx': test_end_idx
            }
            step_results.append(result)
        
        return step_results
        
    except Exception as e:
        print(f"Error in validation step {test_end_idx}: {e}")
        return None