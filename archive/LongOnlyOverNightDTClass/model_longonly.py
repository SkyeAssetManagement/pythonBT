import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import configparser

class LongOnlyTreeEnsemble:
    def __init__(self, config_path='config_longonly.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        self.n_trees = int(self.config['model']['n_trees'])
        self.max_depth = int(self.config['model']['max_depth'])
        self.bootstrap_fraction = float(self.config['model']['bootstrap_fraction'])
        self.min_leaf_fraction = float(self.config['model']['min_leaf_fraction'])
        self.up_threshold = float(self.config['model']['up_threshold'])
        self.vote_threshold = float(self.config['model']['vote_threshold'])
        
        self.trees = []
        self.bootstrap_indices = []
        
    def create_longonly_labels(self, y):
        """
        Convert continuous target to binary UP/NOT-UP labels.
        UP = 1 when return > threshold
        NOT-UP = 0 when return <= threshold
        """
        labels = np.zeros(len(y))
        labels[y > self.up_threshold] = 1
        return labels
    
    def fit(self, X, y):
        """
        Train the ensemble of decision trees for long-only strategy.
        """
        self.trees = []
        self.bootstrap_indices = []
        
        y_longonly = self.create_longonly_labels(y)
        
        valid_mask = ~(np.isnan(X) | np.isnan(y))
        X_valid = X[valid_mask]
        y_valid = y_longonly[valid_mask]
        
        if len(X_valid) == 0:
            raise ValueError("No valid training samples")
        
        print(f"Training data: {len(X_valid)} samples")
        print(f"UP samples: {np.sum(y_valid == 1)} ({np.mean(y_valid):.3f})")
        print(f"NOT-UP samples: {np.sum(y_valid == 0)} ({1-np.mean(y_valid):.3f})")
        
        bootstrap_size = int(len(X_valid) * self.bootstrap_fraction)
        min_samples_leaf = max(1, int(bootstrap_size * self.min_leaf_fraction))
        
        for i in range(self.n_trees):
            X_boot, y_boot, indices = resample(X_valid, y_valid, np.arange(len(X_valid)),
                                              n_samples=bootstrap_size,
                                              random_state=42 + i)
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42 + i
            )
            
            if X_boot.ndim == 1:
                X_boot = X_boot.reshape(-1, 1)
            
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            self.bootstrap_indices.append(indices)
    
    def predict(self, X):
        """
        Make predictions using the ensemble with voting threshold.
        Returns: 1 for LONG signal, 0 for NO TRADE
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
            up_votes = np.sum(votes == 1)
            total_votes = len(votes)
            
            if up_votes / total_votes >= self.vote_threshold:
                final_predictions.append(1)  # LONG signal
            else:
                final_predictions.append(0)  # NO TRADE
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """
        Get the probability of UP prediction from the ensemble.
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
        
        up_probabilities = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            up_prob = np.mean(votes == 1)
            up_probabilities.append(up_prob)
        
        return np.array(up_probabilities)
    
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