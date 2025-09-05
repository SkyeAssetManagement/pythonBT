"""
Random Forest Feature Selection Module
======================================
Supports both classification and regression modes for feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    def __init__(self, n_features=5, min_features=1, max_features=None,
                 rf_n_estimators=75, rf_max_depth=3, rf_min_samples_leaf=30,
                 rf_bootstrap_fraction=0.8, importance_threshold=0.05,
                 algorithm='decision_trees', balanced_bootstrap=False, random_seed=42, n_jobs=-1,
                 target_threshold=0.0, regression_mode=False, model_type='longonly',
                 cumulative_importance_mode=False, cumulative_importance_threshold=0.95):
        """
        Random Forest feature selector using Mean Decrease in Impurity (MDI).
        
        Parameters:
        -----------
        n_features : int
            Target number of features to select
        min_features : int
            Minimum number of features to always select
        max_features : int
            Maximum number of features allowed (None = no limit)
        rf_n_estimators : int
            Number of trees in the Random Forest
        rf_max_depth : int
            Maximum depth of each tree
        rf_min_samples_leaf : int
            Minimum samples required in each leaf node
            This affects decision stumps (depth=1) by controlling leaf size
        rf_bootstrap_fraction : float
            Fraction of samples to use for each tree (bootstrap)
        importance_threshold : float
            Minimum importance score (0-1) to consider a feature (standard mode)
        cumulative_importance_mode : bool
            If True, select features based on cumulative importance threshold
        cumulative_importance_threshold : float
            Cumulative importance threshold (0-1), e.g., 0.95 = top features explaining 95% of importance
        algorithm : str
            'decision_trees' or 'extra_trees' to match model training
        balanced_bootstrap : bool
            Whether to use balanced class sampling (for classification)
        random_seed : int
            Random seed for reproducibility
        n_jobs : int
            Number of CPU cores to use (-1 = all cores, 1 = single core, -2 = all but one)
        """
        self.n_features = n_features
        self.min_features = max(1, min_features)
        self.max_features = max_features
        
        # RF-specific parameters
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_bootstrap_fraction = rf_bootstrap_fraction
        self.importance_threshold = importance_threshold
        self.algorithm = algorithm
        self.balanced_bootstrap = balanced_bootstrap
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.target_threshold = target_threshold
        self.regression_mode = regression_mode
        self.model_type = model_type
        self.cumulative_importance_mode = cumulative_importance_mode
        self.cumulative_importance_threshold = cumulative_importance_threshold
        
        # Results storage
        self.selected_features = []
        self.selection_scores = {}
        self.rf_model = None
        self.all_importances = None
        
    def select_features(self, X, y, feature_names=None, verbose=False):
        """
        Select features using Random Forest importance with threshold filtering.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data features
        y : array-like, shape (n_samples,)
            Target variable
        feature_names : list
            Names of features (for reporting)
        verbose : bool
            Print selection details
            
        Returns:
        --------
        selected_indices : array
            Indices of selected features
        selected_names : list
            Names of selected features
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        n_features_available = X.shape[1]
        n_samples = X.shape[0]
        
        # Calculate bootstrap sample size
        max_samples = int(n_samples * self.rf_bootstrap_fraction)
        
        if self.regression_mode:
            # REGRESSION MODE: Use continuous target values
            if verbose:
                print(f"Feature selection using REGRESSION on continuous targets")
                print(f"  Target mean: {np.mean(y):.3f}, std: {np.std(y):.3f}")
            
            # Build Random Forest or Extra Trees REGRESSOR
            if self.algorithm == 'extra_trees':
                self.rf_model = ExtraTreesRegressor(
                    n_estimators=self.rf_n_estimators,
                    max_depth=self.rf_max_depth,
                    min_samples_leaf=self.rf_min_samples_leaf,
                    max_samples=max_samples,
                    bootstrap=True,
                    random_state=self.random_seed,
                    n_jobs=self.n_jobs
                )
            else:
                self.rf_model = RandomForestRegressor(
                    n_estimators=self.rf_n_estimators,
                    max_depth=self.rf_max_depth,
                    min_samples_leaf=self.rf_min_samples_leaf,
                    max_samples=max_samples,
                    bootstrap=True,
                    random_state=self.random_seed,
                    n_jobs=self.n_jobs
                )
            
            # Fit regressor on continuous values
            self.rf_model.fit(X, y)
            
        else:
            # CLASSIFICATION MODE: Apply threshold first
            if self.model_type == 'shortonly':
                # For shorts: profitable when returns are below (negative) threshold
                y_binary = (y < self.target_threshold).astype(int)
            else:
                # For longs: profitable when returns are above threshold
                y_binary = (y > self.target_threshold).astype(int)
            
            if verbose:
                print(f"Feature selection using CLASSIFICATION (threshold={self.target_threshold})")
                print(f"  Positive class: {np.sum(y_binary)} ({np.mean(y_binary)*100:.1f}%)")
                print(f"  Negative class: {np.sum(1-y_binary)} ({np.mean(1-y_binary)*100:.1f}%)")
            
            # Build Random Forest or Extra Trees CLASSIFIER
            if self.algorithm == 'extra_trees':
                self.rf_model = ExtraTreesClassifier(
                    n_estimators=self.rf_n_estimators,
                    max_depth=self.rf_max_depth,
                    min_samples_leaf=self.rf_min_samples_leaf,
                    max_samples=max_samples,
                    bootstrap=True,
                    random_state=self.random_seed,
                    n_jobs=self.n_jobs
                )
            else:
                self.rf_model = RandomForestClassifier(
                    n_estimators=self.rf_n_estimators,
                    max_depth=self.rf_max_depth,
                    min_samples_leaf=self.rf_min_samples_leaf,
                    max_samples=max_samples,
                    bootstrap=True,
                    random_state=self.random_seed,
                    n_jobs=self.n_jobs
                )
            
            # Fit classifier on binary labels
            self.rf_model.fit(X, y_binary)
        importances = self.rf_model.feature_importances_
        self.all_importances = importances
        
        # Apply threshold filtering based on mode
        if self.cumulative_importance_mode:
            # CUMULATIVE IMPORTANCE MODE
            # Sort features by importance (descending)
            sorted_indices = np.argsort(importances)[::-1]
            sorted_importances = importances[sorted_indices]
            
            # Calculate cumulative importance
            cumsum = np.cumsum(sorted_importances)
            total_importance = np.sum(sorted_importances)
            
            if total_importance > 0:
                cumsum_normalized = cumsum / total_importance
                
                # Find features needed to reach cumulative threshold
                n_features_for_threshold = np.argmax(cumsum_normalized >= self.cumulative_importance_threshold) + 1
                
                if verbose:
                    print(f"\nCumulative importance threshold ({self.cumulative_importance_threshold:.1%}): "
                          f"{n_features_for_threshold} features needed")
                    print(f"These {n_features_for_threshold} features explain "
                          f"{cumsum_normalized[n_features_for_threshold-1]:.1%} of total importance")
                
                # Apply min/max constraints
                n_to_select = max(self.min_features, n_features_for_threshold)
                if self.max_features:
                    n_to_select = min(n_to_select, self.max_features)
                
                # Limit by available features and n_features target
                n_to_select = min(n_to_select, len(sorted_indices), self.n_features)
                
                selected_indices = sorted_indices[:n_to_select]
                
                if verbose:
                    actual_cumsum = cumsum_normalized[n_to_select-1] if n_to_select > 0 else 0
                    print(f"Selected {n_to_select} features (covering {actual_cumsum:.1%} of importance)")
            else:
                # No importance, fall back to selecting min_features
                if verbose:
                    print(f"No feature importance found, selecting top {self.min_features}")
                selected_indices = np.argsort(importances)[::-1][:self.min_features]
                
        elif self.importance_threshold > 0:
            # Find features above threshold
            above_threshold = importances >= self.importance_threshold
            above_threshold_indices = np.where(above_threshold)[0]
            
            if verbose:
                n_above = len(above_threshold_indices)
                print(f"\nFeatures above threshold ({self.importance_threshold:.2f}): {n_above}/{n_features_available}")
            
            # Sort features above threshold by importance
            if len(above_threshold_indices) > 0:
                sorted_above = above_threshold_indices[np.argsort(importances[above_threshold_indices])[::-1]]
                
                # Determine how many to select
                n_to_select = len(sorted_above)
                
                # Apply min_features constraint
                if n_to_select < self.min_features:
                    # Need more features - take top ones regardless of threshold
                    all_sorted = np.argsort(importances)[::-1]
                    n_to_select = self.min_features
                    # Also apply max_features constraint even when adding more features
                    if self.max_features:
                        n_to_select = min(n_to_select, self.max_features)
                    selected_indices = all_sorted[:n_to_select]
                    if verbose:
                        print(f"Only {len(sorted_above)} above threshold, selecting top {n_to_select}")
                else:
                    # Apply max_features constraint first
                    if self.max_features:
                        n_to_select = min(n_to_select, self.max_features)
                    # Then apply n_features constraint
                    n_to_select = min(n_to_select, self.n_features)
                    selected_indices = sorted_above[:n_to_select]
                    if verbose:
                        if n_to_select < len(sorted_above):
                            print(f"Selecting top {n_to_select} from {len(sorted_above)} above threshold")
            else:
                # No features above threshold, select based on min_features but respect max_features
                n_to_select = self.min_features
                if self.max_features:
                    n_to_select = min(n_to_select, self.max_features)
                if verbose:
                    print(f"No features above threshold, selecting top {n_to_select}")
                selected_indices = np.argsort(importances)[::-1][:n_to_select]
        else:
            # No threshold, just take top n_features
            n_to_select = min(self.n_features, n_features_available)
            if self.max_features:
                n_to_select = min(n_to_select, self.max_features)
            n_to_select = max(self.min_features, n_to_select)
            selected_indices = np.argsort(importances)[::-1][:n_to_select]
        
        # Store results
        selected_names = [feature_names[i] for i in selected_indices]
        self.selected_features = selected_names
        self.selection_scores = {name: importances[i] for i, name in zip(selected_indices, selected_names)}
        
        if verbose:
            self._print_selection_summary(X.shape, max_samples)
        
        return selected_indices, selected_names
    
    def _print_selection_summary(self, data_shape, max_samples):
        """Print detailed selection summary."""
        n_samples, n_features = data_shape
        
        print("\n" + "="*60)
        print("Random Forest Feature Selection Summary")
        print("="*60)
        
        print(f"\nRF Configuration:")
        print(f"  Trees: {self.rf_n_estimators}")
        print(f"  Max depth: {self.rf_max_depth}")
        print(f"  Min samples leaf: {self.rf_min_samples_leaf}")
        print(f"  Bootstrap samples: {max_samples}/{n_samples} ({self.rf_bootstrap_fraction*100:.0f}%)")
        
        if self.rf_model:
            avg_depth = np.mean([tree.get_depth() for tree in self.rf_model.estimators_])
            print(f"  Actual avg depth: {avg_depth:.2f}")
        
        print(f"\nSelection Criteria:")
        print(f"  Target features: {self.n_features}")
        print(f"  Min features: {self.min_features}")
        print(f"  Max features: {self.max_features if self.max_features else 'unlimited'}")
        if self.cumulative_importance_mode:
            print(f"  Mode: Cumulative importance")
            print(f"  Cumulative threshold: {self.cumulative_importance_threshold:.1%}")
        else:
            print(f"  Mode: Minimum importance threshold")
            print(f"  Importance threshold: {self.importance_threshold:.3f}")
        
        print(f"\nSelected {len(self.selected_features)} features:")
        for name in self.selected_features:
            score = self.selection_scores[name]
            status = "PASS" if score >= self.importance_threshold else "BELOW"
            print(f"  {name:20s}: {score:.4f} [{status}]")
        
        # Show all feature importances if not too many
        if self.all_importances is not None and len(self.all_importances) <= 15:
            print(f"\nAll Feature Importances:")
            all_features = [f'Feature_{i}' for i in range(len(self.all_importances))]
            sorted_idx = np.argsort(self.all_importances)[::-1]
            for idx in sorted_idx:
                score = self.all_importances[idx]
                name = all_features[idx] if idx < len(all_features) else f'Feature_{idx}'
                selected = "* " if name in self.selected_features else "  "
                print(f"  {selected}{name:20s}: {score:.4f}")
    
    def get_feature_importance_details(self):
        """
        Get detailed importance information from the Random Forest.
        
        Returns:
        --------
        dict : Dictionary containing importance details
        """
        if self.rf_model is None:
            return None
        
        # Count how many trees use each feature
        feature_usage = np.zeros(len(self.all_importances))
        for tree in self.rf_model.estimators_:
            feature_usage += (tree.feature_importances_ > 0)
        
        details = {
            'importances': self.all_importances,
            'selected_features': self.selected_features,
            'selection_scores': self.selection_scores,
            'n_trees': len(self.rf_model.estimators_),
            'avg_tree_depth': np.mean([tree.get_depth() for tree in self.rf_model.estimators_]),
            'feature_usage_count': feature_usage,
            'feature_usage_pct': feature_usage / len(self.rf_model.estimators_) * 100
        }
        
        return details
    
    def get_selection_summary(self):
        """Get summary of selected features."""
        if not self.selected_features:
            return "No features selected yet"
        
        summary = f"Random Forest Feature Selection\n"
        summary += f"Selected {len(self.selected_features)} features:\n"
        for feat, score in sorted(self.selection_scores.items(), 
                                 key=lambda x: x[1], reverse=True):
            threshold_status = "✓" if score >= self.importance_threshold else "↓"
            summary += f"  {feat}: {score:.4f} [{threshold_status}]\n"
        return summary


# Test if run directly
if __name__ == "__main__":
    print("Testing RF-only Feature Selector")
    print("-"*40)
    
    # Create test data
    np.random.seed(42)
    n_samples, n_features = 500, 8
    X = np.random.randn(n_samples, n_features)
    
    # Create target with known relationships
    y = 2*X[:, 0] + 0.5*X[:, 2] - X[:, 4] + 0.3*np.random.randn(n_samples)
    
    # Test selector
    selector = FeatureSelector(
        n_features=4,
        min_features=2,
        max_features=6,
        rf_n_estimators=50,
        rf_max_depth=3,
        rf_min_samples_split=20,
        rf_bootstrap_fraction=0.8,
        importance_threshold=0.10  # 10% threshold
    )
    
    feature_names = [f'Ret_{i}hr' for i in [1, 2, 4, 8, 16, 32, 64, 128]]
    selected_indices, selected_names = selector.select_features(X, y, feature_names, verbose=True)
    
    # Get detailed info
    details = selector.get_feature_importance_details()
    if details:
        print(f"\nFeature usage across {details['n_trees']} trees:")
        for i, name in enumerate(feature_names):
            usage_pct = details['feature_usage_pct'][i]
            print(f"  {name}: {usage_pct:.1f}% of trees")