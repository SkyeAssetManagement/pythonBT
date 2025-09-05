import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import configparser

class DirectionalTreeEnsemble:
    def __init__(self, config_path='config_longonly.ini', verbose=True):
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        self.config.read(config_path)
        
        self.n_trees = int(self.config['model']['n_trees'])
        self.max_depth = int(self.config['model']['max_depth'])
        self.bootstrap_fraction = float(self.config['model']['bootstrap_fraction'])
        self.min_leaf_fraction = float(self.config['model']['min_leaf_fraction'])
        self.target_threshold = float(self.config['model']['target_threshold'])
        self.vote_threshold = float(self.config['model']['vote_threshold'])
        self.random_seed = int(self.config['model']['random_seed'])
        self.model_type = self.config['model']['model_type']
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
        Train the ensemble of decision trees for directional strategy.
        """
        self.trees = []
        self.bootstrap_indices = []
        
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
            print(f"{self.direction_name} samples: {np.sum(y_valid == 1)} ({np.mean(y_valid):.3f})")
            print(f"NOT-{self.direction_name} samples: {np.sum(y_valid == 0)} ({1-np.mean(y_valid):.3f})")
            print(f"Model type: {self.model_type} (threshold: {self.effective_threshold:.3f})")
        
        bootstrap_size = int(len(X_valid) * self.bootstrap_fraction)
        min_samples_leaf = max(1, int(bootstrap_size * self.min_leaf_fraction))
        
        for i in range(self.n_trees):
            X_boot, y_boot, indices = resample(X_valid, y_valid, np.arange(len(X_valid)),
                                              n_samples=bootstrap_size,
                                              random_state=self.random_seed + i)
            
            tree = DecisionTreeClassifier(
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
        if len(self.trees) == 0:
            raise ValueError("Model must be fitted before making predictions")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        predictions = []
        for tree in self.trees:
            pred = tree.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            profitable_votes = np.sum(votes == 1)
            total_votes = len(votes)
            
            if profitable_votes / total_votes >= self.vote_threshold:
                final_predictions.append(1)  # TRADE signal (LONG or SHORT based on model_type)
            else:
                final_predictions.append(0)  # NO TRADE
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """
        Get the probability of profitable direction prediction from the ensemble.
        """
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
            profitable_prob = np.mean(votes == 1)
            profitable_probabilities.append(profitable_prob)
        
        return np.array(profitable_probabilities)
    
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