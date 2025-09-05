import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.utils import resample
import configparser
from src.column_detector import ColumnDetector

class DirectionalTreeEnsemble:
    def __init__(self, config_path='OMtree_config.ini', verbose=True):
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.config.read(config_path)
        
        # Tree count method
        self.n_trees_method = self.config['model'].get('n_trees_method', 'absolute')
        self.n_trees_base = int(float(self.config['model']['n_trees']))
        # Actual n_trees will be set in fit() based on number of features
        self.n_trees = self.n_trees_base  # Default for now
        self.max_depth = int(float(self.config['model']['max_depth']))
        self.bootstrap_fraction = float(self.config['model']['bootstrap_fraction'])
        self.min_leaf_fraction = float(self.config['model']['min_leaf_fraction'])
        self.target_threshold = float(self.config['model']['target_threshold'])
        self.vote_threshold = float(self.config['model']['vote_threshold'])
        self.random_seed = int(self.config['model']['random_seed'])
        self.model_type = self.config['model']['model_type']
        # New: algorithm type (decision_trees or extra_trees)
        self.algorithm = self.config['model'].get('algorithm', 'decision_trees')
        # New: probability aggregation method (mean or median)
        self.probability_aggregation = self.config['model'].get('probability_aggregation', 'mean')
        # New: convert tree predictions to binary option
        self.convert_to_binary = self.config['model'].get('convert_to_binary', 'true').lower() == 'true'
        # Trade prediction threshold for raw aggregation mode
        self.trade_prediction_threshold = float(self.config['model'].get('trade_prediction_threshold', '0.01'))
        # New: balanced bootstrap option
        self.balanced_bootstrap = self.config['model'].get('balanced_bootstrap', 'false').lower() == 'true'
        # New: n_jobs for parallel processing
        self.n_jobs = int(self.config['model'].get('n_jobs', -1))
        # New: regression mode option
        self.regression_mode = self.config['model'].get('regression_mode', 'false').lower() == 'true'
        # Auto-calibration settings
        self.auto_calibrate_threshold = self.config['model'].get('auto_calibrate_threshold', 'false').lower() == 'true'
        self.target_prediction_rate = float(self.config['model'].get('target_prediction_rate', '0.2'))
        self.calibration_lookback = int(self.config['model'].get('calibration_lookback', '90'))
        # Tree splitting criterion
        self.tree_criterion = self.config['model'].get('tree_criterion', 'default')
        self.verbose = verbose
        
        # Validate model_type
        if self.model_type not in ['longonly', 'shortonly']:
            raise ValueError(f"model_type must be 'longonly' or 'shortonly', got '{self.model_type}'")
        
        # For short trades, we look for negative returns (profitable shorts)
        # So we multiply the threshold by -1
        if self.model_type == 'shortonly':
            self.effective_threshold = -self.target_threshold
            self.direction_name = "DOWN"
            self.signal_name = "SHORT"
        else:
            self.effective_threshold = self.target_threshold
            self.direction_name = "UP"
            self.signal_name = "LONG"
        
        self.trees = []
        self.bootstrap_indices = []
        
    def create_directional_labels(self, y):
        """
        Convert continuous target to binary directional labels based on model_type.
        
        For longonly: 
        - PROFITABLE = 1 when return > target_threshold (positive returns)
        - NOT-PROFITABLE = 0 when return <= target_threshold
        
        For shortonly:
        - PROFITABLE = 1 when return < -target_threshold (negative returns, profitable for shorts)
        - NOT-PROFITABLE = 0 when return >= -target_threshold
        """
        labels = np.zeros(len(y))
        
        if self.model_type == 'longonly':
            # For longs: positive returns above threshold are profitable
            labels[y > self.effective_threshold] = 1
        else:  # shortonly
            # For shorts: negative returns below negative threshold are profitable
            labels[y < self.effective_threshold] = 1
            
        return labels
    
    def fit(self, X, y):
        """
        Train the ensemble using either Decision Trees or Extra Trees.
        In regression mode, uses continuous targets instead of binary labels.
        """
        if self.regression_mode:
            # In regression mode, use continuous values directly
            y_directional = y
        else:
            # In classification mode, create binary labels
            y_directional = self.create_directional_labels(y)
        
        # Handle masking for multiple features
        if X.ndim == 1:
            valid_mask = ~(np.isnan(X) | np.isnan(y_directional))
        else:
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_directional))
        X_valid = X[valid_mask]
        y_valid = y_directional[valid_mask]
        
        if len(X_valid) == 0:
            raise ValueError("No valid training samples")
        
        if self.verbose:
            print(f"Training data: {len(X_valid)} samples")
            if self.regression_mode:
                print(f"Regression mode: Target mean={np.mean(y_valid):.3f}, std={np.std(y_valid):.3f}")
            else:
                print(f"{self.direction_name} samples: {np.sum(y_valid == 1)} ({np.mean(y_valid):.3f})")
                print(f"NOT-{self.direction_name} samples: {np.sum(y_valid == 0)} ({1-np.mean(y_valid):.3f})")
            print(f"Model type: {self.model_type} (threshold: {self.effective_threshold:.3f})")
            print(f"Algorithm: {self.algorithm}")
            print(f"Regression mode: {self.regression_mode}")
        
        if X_valid.ndim == 1:
            X_valid = X_valid.reshape(-1, 1)
        
        # Calculate actual number of trees based on method
        n_features = X_valid.shape[1]
        if self.n_trees_method == 'per_feature':
            self.n_trees = self.n_trees_base * n_features
            if self.verbose:
                print(f"Trees: {self.n_trees} ({self.n_trees_base} per feature × {n_features} features)")
        else:  # absolute
            self.n_trees = self.n_trees_base
            if self.verbose:
                print(f"Trees: {self.n_trees} (absolute)")
        
        if self.algorithm == 'extra_trees':
            bootstrap_size = int(len(X_valid) * self.bootstrap_fraction)
            min_samples_leaf = max(1, int(bootstrap_size * self.min_leaf_fraction))
            
            if self.regression_mode:
                # Use Extra Trees Regressor for regression mode
                self.model = ExtraTreesRegressor(
                    n_estimators=self.n_trees,
                    max_depth=self.max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_samples=self.bootstrap_fraction,
                    bootstrap=True,
                    random_state=self.random_seed,
                    n_jobs=self.n_jobs
                )
            else:
                # Use Extra Trees Classifier for classification
                self.model = ExtraTreesClassifier(
                    n_estimators=self.n_trees,
                    max_depth=self.max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_samples=self.bootstrap_fraction,
                    bootstrap=True,
                    random_state=self.random_seed,
                    n_jobs=self.n_jobs
                )
            
            self.model.fit(X_valid, y_valid)
            self.trees = self.model.estimators_
            self.bootstrap_indices = None
            
        else:
            # Original Decision Trees implementation
            self.trees = []
            self.bootstrap_indices = []
            
            bootstrap_size = int(len(X_valid) * self.bootstrap_fraction)
            min_samples_leaf = max(1, int(bootstrap_size * self.min_leaf_fraction))
            
            for i in range(self.n_trees):
                if self.balanced_bootstrap and not self.regression_mode:
                    # Balanced bootstrap only makes sense for classification
                    pos_indices = np.where(y_valid == 1)[0]
                    neg_indices = np.where(y_valid == 0)[0]
                    
                    # Sample equal numbers from each class
                    n_samples_per_class = bootstrap_size // 2
                    
                    # Resample from each class
                    pos_sample = resample(pos_indices, 
                                        n_samples=min(n_samples_per_class, len(pos_indices)),
                                        random_state=self.random_seed + i)
                    neg_sample = resample(neg_indices,
                                        n_samples=min(n_samples_per_class, len(neg_indices)),
                                        random_state=self.random_seed + i + 1000)
                    
                    # Combine indices
                    indices = np.concatenate([pos_sample, neg_sample])
                    np.random.RandomState(self.random_seed + i).shuffle(indices)
                    
                    X_boot = X_valid[indices]
                    y_boot = y_valid[indices]
                else:
                    # Regular bootstrap sampling (for regression or non-balanced classification)
                    X_boot, y_boot, indices = resample(X_valid, y_valid, np.arange(len(X_valid)),
                                                      n_samples=bootstrap_size,
                                                      random_state=self.random_seed + i)
                
                # Use appropriate tree type
                if self.regression_mode:
                    # For regression: 'squared_error' (MSE), 'absolute_error' (MAE), 'friedman_mse', 'poisson'
                    if self.tree_criterion == 'mae':
                        reg_criterion = 'absolute_error'
                    elif self.tree_criterion == 'friedman':
                        reg_criterion = 'friedman_mse'
                    else:  # default or mse
                        reg_criterion = 'squared_error'
                    
                    tree = DecisionTreeRegressor(
                        criterion=reg_criterion,
                        max_depth=self.max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=self.random_seed + i
                    )
                else:
                    # For classification: 'gini' or 'entropy'
                    if self.tree_criterion == 'entropy':
                        class_criterion = 'entropy'
                    else:  # default or gini
                        class_criterion = 'gini'
                    
                    tree = DecisionTreeClassifier(
                        criterion=class_criterion,
                        max_depth=self.max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=self.random_seed + i
                    )
                
                if X_boot.ndim == 1:
                    X_boot = X_boot.reshape(-1, 1)
                
                tree.fit(X_boot, y_boot)
                self.trees.append(tree)
                self.bootstrap_indices.append(indices)
    
    def predict(self, X):
        """
        Make predictions using the ensemble with voting threshold.
        Returns: 1 for TRADE signal (LONG or SHORT based on model_type), 0 for NO TRADE
        """
        if self.algorithm == 'extra_trees':
            if not hasattr(self, 'model'):
                raise ValueError("Model must be fitted before making predictions")
            
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            # Get predictions from all trees
            predictions = []
            for tree in self.model.estimators_:
                pred = tree.predict(X)
                if self.regression_mode and self.convert_to_binary:
                    # Convert to binary for voting
                    if self.model_type == 'shortonly':
                        # For shorts: signal when prediction is below threshold
                        pred_binary = (pred < self.effective_threshold).astype(int)
                    else:
                        # For longs: signal when prediction is above threshold
                        pred_binary = (pred > self.effective_threshold).astype(int)
                    predictions.append(pred_binary)
                else:
                    # Keep raw predictions for aggregation
                    predictions.append(pred)
            
            predictions = np.array(predictions)
            
            final_predictions = []
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                
                if self.regression_mode and not self.convert_to_binary:
                    # RAW AGGREGATION MODE: Aggregate raw predictions, then apply trade threshold
                    if self.probability_aggregation == 'median':
                        aggregated_value = np.median(votes)
                    else:  # mean
                        aggregated_value = np.mean(votes)
                    
                    # Apply trade prediction threshold to aggregated value
                    if self.model_type == 'shortonly':
                        # For shorts: trade if aggregated prediction < negative threshold
                        signal = 1 if aggregated_value < -abs(self.trade_prediction_threshold) else 0
                    else:
                        # For longs: trade if aggregated prediction > threshold  
                        signal = 1 if aggregated_value > abs(self.trade_prediction_threshold) else 0
                    final_predictions.append(signal)
                else:
                    # BINARY VOTING MODE: Use vote threshold on binary votes
                    if self.probability_aggregation == 'median':
                        vote_probability = np.median(votes)
                    else:  # mean
                        profitable_votes = np.sum(votes == 1)
                        total_votes = len(votes)
                        vote_probability = profitable_votes / total_votes
                    
                    # Apply vote threshold
                    if vote_probability >= self.vote_threshold:
                        final_predictions.append(1)
                    else:
                        final_predictions.append(0)
            
            return np.array(final_predictions)
            
        else:
            # Original implementation for Decision Trees
            if len(self.trees) == 0:
                raise ValueError("Model must be fitted before making predictions")
            
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            predictions = []
            for tree in self.trees:
                pred = tree.predict(X)
                if self.regression_mode and self.convert_to_binary:
                    # Convert to binary for voting
                    if self.model_type == 'shortonly':
                        # For shorts: signal when prediction is below threshold
                        pred_binary = (pred < self.effective_threshold).astype(int)
                    else:
                        # For longs: signal when prediction is above threshold
                        pred_binary = (pred > self.effective_threshold).astype(int)
                    predictions.append(pred_binary)
                else:
                    # Keep raw predictions for aggregation
                    predictions.append(pred)
            
            predictions = np.array(predictions)
            
            final_predictions = []
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                
                if self.regression_mode and not self.convert_to_binary:
                    # RAW AGGREGATION MODE: Aggregate raw predictions, then apply trade threshold
                    if self.probability_aggregation == 'median':
                        aggregated_value = np.median(votes)
                    else:  # mean
                        aggregated_value = np.mean(votes)
                    
                    # Apply trade prediction threshold to aggregated value
                    if self.model_type == 'shortonly':
                        # For shorts: trade if aggregated prediction < negative threshold
                        signal = 1 if aggregated_value < -abs(self.trade_prediction_threshold) else 0
                    else:
                        # For longs: trade if aggregated prediction > threshold  
                        signal = 1 if aggregated_value > abs(self.trade_prediction_threshold) else 0
                    final_predictions.append(signal)
                else:
                    # BINARY VOTING MODE: Use vote threshold on binary votes
                    if self.probability_aggregation == 'median':
                        vote_probability = np.median(votes)
                    else:  # mean
                        profitable_votes = np.sum(votes == 1)
                        total_votes = len(votes)
                        vote_probability = profitable_votes / total_votes
                    
                    # Apply vote threshold
                    if vote_probability >= self.vote_threshold:
                        final_predictions.append(1)  # TRADE signal (LONG or SHORT based on model_type)
                    else:
                        final_predictions.append(0)  # NO TRADE
            
            return np.array(final_predictions)
    
    def predict_proba(self, X):
        """
        Get the probability of profitable direction prediction from the ensemble.
        """
        if self.algorithm == 'extra_trees':
            if not hasattr(self, 'model'):
                raise ValueError("Model must be fitted before making predictions")
            
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            # Get predictions from all trees
            predictions = []
            for tree in self.model.estimators_:
                pred = tree.predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            profitable_probabilities = []
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                
                # In regression mode, return aggregated raw values
                # In classification mode, return probability of class 1
                if self.regression_mode:
                    if self.probability_aggregation == 'median':
                        profitable_prob = np.median(votes)
                    else:  # mean
                        profitable_prob = np.mean(votes)
                else:
                    # Classification mode - calculate probability based on aggregation method
                    if self.probability_aggregation == 'median':
                        profitable_prob = np.median(votes)
                    else:  # mean
                        profitable_prob = np.mean(votes == 1)
                
                profitable_probabilities.append(profitable_prob)
            
            return np.array(profitable_probabilities)
            
        else:
            # Original implementation
            if len(self.trees) == 0:
                raise ValueError("Model must be fitted before making predictions")
            
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            predictions = []
            for tree in self.trees:
                pred = tree.predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            profitable_probabilities = []
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                
                # In regression mode, return aggregated raw values
                # In classification mode, return probability of class 1
                if self.regression_mode:
                    if self.probability_aggregation == 'median':
                        profitable_prob = np.median(votes)
                    else:  # mean
                        profitable_prob = np.mean(votes)
                else:
                    # Classification mode - calculate probability based on aggregation method
                    if self.probability_aggregation == 'median':
                        profitable_prob = np.median(votes)
                    else:  # mean
                        profitable_prob = np.mean(votes == 1)
                
                profitable_probabilities.append(profitable_prob)
            
            return np.array(profitable_probabilities)
    
    def calibrate_threshold(self, X, y):
        """
        Auto-calibrate the prediction threshold based on recent training data.
        Uses percentile method to achieve target prediction rate.
        
        Args:
            X: Recent training features (last calibration_lookback samples)
            y: Recent training targets
        
        Returns:
            Calibrated threshold value
        """
        if not self.auto_calibrate_threshold:
            # Return existing threshold if auto-calibration is disabled
            if self.convert_to_binary:
                return self.vote_threshold
            else:
                return self.trade_prediction_threshold
        
        # Get predictions on calibration window
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Get raw probabilities or predictions
        if self.regression_mode and not self.convert_to_binary:
            # For raw aggregation mode, get the aggregated predictions
            predictions = []
            for i in range(X.shape[0]):
                votes = []
                for tree in self.trees:
                    pred = tree.predict(X[i:i+1])
                    votes.append(pred[0])
                
                if self.probability_aggregation == 'median':
                    aggregated = np.median(votes)
                else:  # mean
                    aggregated = np.mean(votes)
                predictions.append(aggregated)
            predictions = np.array(predictions)
        else:
            # For binary voting mode, get vote probabilities
            predictions = self.predict_proba(X)
        
        # Calculate the percentile threshold
        # We want (target_prediction_rate * 100)% of predictions to be above threshold
        # So we need the (100 - target_prediction_rate*100)th percentile
        percentile = (1.0 - self.target_prediction_rate) * 100
        
        # Calculate threshold based on model type
        if self.model_type == 'shortonly':
            # For shorts, we want the lowest X% (most negative predictions)
            threshold = np.percentile(predictions, self.target_prediction_rate * 100)
        else:
            # For longs, we want the highest X% (most positive predictions)
            threshold = np.percentile(predictions, percentile)
        
        if self.verbose:
            print(f"Auto-calibrated threshold: {threshold:.4f} (target rate: {self.target_prediction_rate:.1%})")
            actual_rate = np.mean(predictions >= threshold) if self.model_type == 'longonly' else np.mean(predictions <= threshold)
            print(f"  Expected trade rate: {actual_rate:.1%}")
        
        # Update the appropriate threshold
        if self.convert_to_binary:
            self.vote_threshold = threshold
        else:
            # Store as absolute value since we apply sign in predict()
            self.trade_prediction_threshold = abs(threshold)
        
        return threshold
    
    def get_feature_importance(self):
        """
        Get average feature importance across all trees.
        """
        if len(self.trees) == 0:
            return None
        
        importances = []
        for tree in self.trees:
            importances.append(tree.feature_importances_)
        
        return np.mean(importances, axis=0)
    
    def get_model_info(self):
        """
        Get information about the model configuration.
        """
        return {
            'model_type': self.model_type,
            'target_threshold': self.target_threshold,
            'effective_threshold': self.effective_threshold,
            'direction_name': self.direction_name,
            'signal_name': self.signal_name,
            'vote_threshold': self.vote_threshold,
            'n_trees': self.n_trees
        }