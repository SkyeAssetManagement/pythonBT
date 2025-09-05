"""
Enhanced Feature Selection Module with RF Importance Controls
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureSelector:
    def __init__(self, method='rf_importance', n_features=5, min_features=1, max_features=None,
                 rf_n_estimators=50, rf_max_depth=3, rf_min_samples_split=20,
                 rf_bootstrap_fraction=0.8, importance_threshold=0.05):
        """
        Enhanced feature selector with RF importance controls.
        
        Parameters:
        -----------
        method : str
            Selection method (focus on 'rf_importance')
        n_features : int
            Target number of features to select
        min_features : int
            Minimum number of features to select
        max_features : int
            Maximum number of features to select
        rf_n_estimators : int
            Number of trees in the RF for importance calculation
        rf_max_depth : int
            Maximum depth of trees (deeper = more complex patterns)
        rf_min_samples_split : int
            Minimum samples to split a node
        rf_bootstrap_fraction : float
            Fraction of samples to use for each tree
        importance_threshold : float
            Minimum importance score to consider a feature (0-1)
        """
        self.method = method.lower()
        self.n_features = n_features
        self.min_features = max(1, min_features)
        self.max_features = max_features
        
        # RF-specific parameters
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_bootstrap_fraction = rf_bootstrap_fraction
        self.importance_threshold = importance_threshold
        
        self.selected_features = []
        self.selection_scores = {}
        self.rf_model = None  # Store the RF model for inspection
        
    def select_features(self, X, y, feature_names=None, verbose=False):
        """
        Select best features based on RF importance with threshold control.
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        n_features_available = X.shape[1]
        
        if self.method == 'rf_importance' or self.method == 'rf_importance_enhanced':
            selected_indices, scores = self._select_by_rf_importance_enhanced(X, y, verbose)
        else:
            # Fallback to original methods
            return self._select_original_method(X, y, feature_names, verbose)
        
        # Apply importance threshold filtering
        if self.importance_threshold > 0:
            # Only keep features above threshold
            valid_features = scores >= self.importance_threshold
            valid_indices = np.where(valid_features)[0]
            
            if verbose:
                print(f"Features above threshold ({self.importance_threshold}): {len(valid_indices)}/{n_features_available}")
            
            # Combine threshold filtering with n_features selection
            selected_from_valid = np.argsort(scores[valid_indices])[::-1]
            selected_indices = valid_indices[selected_from_valid]
        
        # Apply min/max constraints
        n_to_select = len(selected_indices)
        if n_to_select < self.min_features:
            # If too few pass threshold, take top min_features regardless
            selected_indices = np.argsort(scores)[::-1][:self.min_features]
            if verbose:
                print(f"Below min_features, selecting top {self.min_features}")
        elif n_to_select > self.n_features:
            # If too many pass threshold, take top n_features
            selected_indices = selected_indices[:self.n_features]
        
        # Ensure we don't exceed max_features
        if self.max_features:
            selected_indices = selected_indices[:self.max_features]
        
        # Store results
        selected_names = [feature_names[i] for i in selected_indices]
        self.selected_features = selected_names
        self.selection_scores = {name: scores[i] for i, name in zip(selected_indices, selected_names)}
        
        if verbose:
            print(f"\nRF Feature Selection Results:")
            print(f"Trees: {self.rf_n_estimators}, Max Depth: {self.rf_max_depth}")
            print(f"Selected {len(selected_indices)} features:")
            for name, score in sorted(self.selection_scores.items(), key=lambda x: x[1], reverse=True):
                status = "[PASS]" if score >= self.importance_threshold else "[BELOW THRESHOLD]"
                print(f"  {name}: {score:.4f} {status}")
            
            # Show model performance metrics
            if self.rf_model is not None:
                train_score = self.rf_model.score(X, y)
                print(f"\nRF Model R² on selection data: {train_score:.4f}")
        
        return selected_indices, selected_names
    
    def _select_by_rf_importance_enhanced(self, X, y, verbose=False):
        """
        Enhanced RF importance with configurable parameters.
        """
        # Calculate bootstrap sample size
        n_samples = X.shape[0]
        max_samples = int(n_samples * self.rf_bootstrap_fraction)
        
        # Build RF with custom parameters
        self.rf_model = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split,
            max_samples=max_samples,  # Bootstrap sample size
            random_state=42,
            n_jobs=-1  # Use all CPU cores for speed
        )
        
        # Fit and get importances
        self.rf_model.fit(X, y)
        importances = self.rf_model.feature_importances_
        
        if verbose:
            # Additional diagnostics
            print(f"\nRF Training Details:")
            print(f"  Samples used: {max_samples}/{n_samples} ({self.rf_bootstrap_fraction*100:.1f}%)")
            print(f"  Average tree depth: {np.mean([tree.get_depth() for tree in self.rf_model.estimators_]):.2f}")
            print(f"  Features always used: {np.sum(importances > 0.01)}")
        
        return np.arange(len(importances)), importances
    
    def _select_original_method(self, X, y, feature_names, verbose):
        """Fallback to original selection methods."""
        # Implementation of other methods...
        # (Keep original implementations for other methods)
        n_features_available = X.shape[1]
        n_to_select = min(self.n_features, n_features_available)
        if self.max_features:
            n_to_select = min(n_to_select, self.max_features)
        n_to_select = max(self.min_features, n_to_select)
        
        # Simple correlation-based selection as fallback
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        correlations = np.nan_to_num(correlations, nan=0.0)
        selected = np.argsort(correlations)[::-1][:n_to_select]
        
        selected_names = [feature_names[i] for i in selected]
        self.selected_features = selected_names
        self.selection_scores = {name: correlations[i] for i, name in zip(selected, selected_names)}
        
        return selected, selected_names
    
    def get_feature_importance_details(self):
        """
        Get detailed importance breakdown if RF was used.
        """
        if self.rf_model is None:
            return None
        
        details = {
            'importances': self.rf_model.feature_importances_,
            'n_trees': len(self.rf_model.estimators_),
            'avg_depth': np.mean([tree.get_depth() for tree in self.rf_model.estimators_]),
            'feature_usage': np.sum([tree.feature_importances_ > 0 for tree in self.rf_model.estimators_], axis=0)
        }
        
        return details
    
    def suggest_parameters(self, X, y, verbose=True):
        """
        Suggest optimal RF parameters based on data characteristics.
        """
        n_samples, n_features = X.shape
        
        suggestions = {}
        
        # Suggest n_estimators based on stability needs
        if n_samples < 500:
            suggestions['rf_n_estimators'] = 30  # Fewer trees for small data
        elif n_samples < 1000:
            suggestions['rf_n_estimators'] = 50
        else:
            suggestions['rf_n_estimators'] = 100  # More trees for stability
        
        # Suggest max_depth based on sample size
        if n_samples < 200:
            suggestions['rf_max_depth'] = 2  # Shallow to avoid overfitting
        elif n_samples < 1000:
            suggestions['rf_max_depth'] = 3
        else:
            suggestions['rf_max_depth'] = 4  # Can go deeper with more data
        
        # Suggest min_samples_split
        suggestions['rf_min_samples_split'] = max(20, int(n_samples * 0.02))
        
        # Suggest bootstrap fraction
        if n_samples < 500:
            suggestions['rf_bootstrap_fraction'] = 1.0  # Use all data
        else:
            suggestions['rf_bootstrap_fraction'] = 0.8  # Standard bootstrap
        
        # Suggest importance threshold based on number of features
        if n_features <= 5:
            suggestions['importance_threshold'] = 0.0  # Keep all
        elif n_features <= 10:
            suggestions['importance_threshold'] = 0.05  # Light filtering
        else:
            suggestions['importance_threshold'] = 0.10  # Stronger filtering
        
        if verbose:
            print(f"\nSuggested RF Parameters for {n_samples} samples, {n_features} features:")
            for param, value in suggestions.items():
                print(f"  {param}: {value}")
        
        return suggestions


# Test the enhanced selector
if __name__ == "__main__":
    import numpy as np
    
    # Create test data
    np.random.seed(42)
    n_samples, n_features = 1000, 8
    X = np.random.randn(n_samples, n_features)
    
    # Create target with known relationships
    y = 2*X[:, 0] + 0.5*X[:, 2] - X[:, 4] + 0.3*np.random.randn(n_samples)
    
    # Test enhanced RF selector
    print("="*60)
    print("Testing Enhanced RF Feature Selection")
    print("="*60)
    
    selector = EnhancedFeatureSelector(
        method='rf_importance',
        n_features=4,
        min_features=2,
        max_features=6,
        rf_n_estimators=100,
        rf_max_depth=3,
        rf_min_samples_split=20,
        rf_bootstrap_fraction=0.8,
        importance_threshold=0.08  # Only keep features with >8% importance
    )
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    selected_indices, selected_names = selector.select_features(X, y, feature_names, verbose=True)
    
    # Get detailed importance info
    details = selector.get_feature_importance_details()
    if details:
        print(f"\nFeature usage across trees:")
        for i, (name, usage) in enumerate(zip(feature_names, details['feature_usage'])):
            print(f"  {name}: used in {usage}/{details['n_trees']} trees")
    
    # Get parameter suggestions
    print("\n" + "="*60)
    suggestions = selector.suggest_parameters(X, y, verbose=True)