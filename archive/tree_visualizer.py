"""
Decision Tree Visualization Module for OMtree
Generates visual representations of trained decision trees
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import os
from sklearn.tree import _tree

class TreeVisualizer:
    """Visualizes decision trees with split rules and statistics"""
    
    def __init__(self, model_path='final_model.pkl'):
        """Initialize with path to saved model"""
        self.model_path = model_path
        self.model = None
        self.trees = []
        self.feature_names = []
        self.model_type = 'longonly'
        
    def load_model(self):
        """Load the saved model from pickle file"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        # Extract model components
        if isinstance(model_data, dict):
            self.trees = model_data.get('trees', [])
            self.feature_names = model_data.get('feature_names', [])
            self.model_type = model_data.get('model_type', 'longonly')
            # Get additional metadata
            self.target_threshold = model_data.get('target_threshold', 0)
            self.effective_threshold = model_data.get('effective_threshold', 0)
            self.direction_name = model_data.get('direction_name', 'UP')
            self.signal_name = model_data.get('signal_name', 'LONG')
        else:
            # Handle older model format
            self.trees = model_data.trees if hasattr(model_data, 'trees') else []
            self.feature_names = model_data.feature_names if hasattr(model_data, 'feature_names') else []
            
        return len(self.trees) > 0
    
    def extract_tree_structure(self, tree):
        """Extract structure from sklearn tree for visualization"""
        tree_ = tree.tree_
        
        def recurse(node, depth=0):
            indent = "  " * depth
            
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node (split)
                name = self.feature_names[tree_.feature[node]] if self.feature_names else f"X[{tree_.feature[node]}]"
                threshold = tree_.threshold[node]
                n_samples = tree_.n_samples[node] if hasattr(tree_, 'n_samples') else 0
                
                # Get the value (class distribution or mean for regression)
                if len(tree_.value[node].shape) == 3:
                    # Classification
                    value = tree_.value[node][0]
                else:
                    # Regression
                    value = tree_.value[node][0][0]
                
                return {
                    'type': 'split',
                    'feature': name,
                    'threshold': threshold,
                    'samples': n_samples,
                    'value': value,
                    'left': recurse(tree_.children_left[node], depth + 1),
                    'right': recurse(tree_.children_right[node], depth + 1),
                    'depth': depth
                }
            else:
                # Leaf node
                n_samples = tree_.n_samples[node] if hasattr(tree_, 'n_samples') else 0
                
                # Get the value
                if len(tree_.value[node].shape) == 3:
                    # Classification
                    value = tree_.value[node][0]
                    prediction = np.argmax(value)
                else:
                    # Regression
                    value = tree_.value[node][0][0]
                    prediction = value
                
                return {
                    'type': 'leaf',
                    'samples': n_samples,
                    'value': value,
                    'prediction': prediction,
                    'depth': depth
                }
        
        return recurse(0)
    
    def create_aggregate_rules(self, max_trees=10):
        """Create aggregate decision rules from multiple trees"""
        if not self.trees:
            return None
            
        # Analyze first N trees
        trees_to_analyze = self.trees[:min(max_trees, len(self.trees))]
        
        # Collect all split rules
        split_features = {}
        split_thresholds = {}
        
        for tree in trees_to_analyze:
            structure = self.extract_tree_structure(tree)
            self._collect_splits(structure, split_features, split_thresholds)
        
        # Create aggregate summary
        aggregate = {
            'n_trees_analyzed': len(trees_to_analyze),
            'total_trees': len(self.trees),
            'model_type': self.model_type,
            'signal_name': self.signal_name,
            'most_common_splits': [],
            'threshold_ranges': {}
        }
        
        # Sort features by frequency
        sorted_features = sorted(split_features.items(), key=lambda x: x[1], reverse=True)
        
        for feature, count in sorted_features[:5]:  # Top 5 features
            thresholds = split_thresholds.get(feature, [])
            if thresholds:
                aggregate['most_common_splits'].append({
                    'feature': feature,
                    'frequency': count,
                    'avg_threshold': np.mean(thresholds),
                    'min_threshold': np.min(thresholds),
                    'max_threshold': np.max(thresholds),
                    'std_threshold': np.std(thresholds)
                })
                aggregate['threshold_ranges'][feature] = {
                    'min': np.min(thresholds),
                    'max': np.max(thresholds),
                    'mean': np.mean(thresholds)
                }
        
        return aggregate
    
    def _collect_splits(self, node, split_features, split_thresholds):
        """Recursively collect split information from tree structure"""
        if node['type'] == 'split':
            feature = node['feature']
            threshold = node['threshold']
            
            # Count feature usage
            if feature not in split_features:
                split_features[feature] = 0
                split_thresholds[feature] = []
            
            split_features[feature] += 1
            split_thresholds[feature].append(threshold)
            
            # Recurse on children
            self._collect_splits(node['left'], split_features, split_thresholds)
            self._collect_splits(node['right'], split_features, split_thresholds)
    
    def plot_single_tree(self, tree_index=0, fig=None, ax=None):
        """Plot a single decision tree"""
        if tree_index >= len(self.trees):
            return None
            
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        tree = self.trees[tree_index]
        structure = self.extract_tree_structure(tree)
        
        # Clear the axis
        ax.clear()
        
        # Plot the tree
        self._plot_node(ax, structure, 0.5, 0.9, 0.5)
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'Decision Tree {tree_index + 1} of {len(self.trees)} ({self.signal_name} Model)', 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def _plot_node(self, ax, node, x, y, width):
        """Recursively plot tree nodes"""
        box_style = dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7)
        
        if node['type'] == 'split':
            # Plot split node
            text = f"{node['feature']}\n<= {node['threshold']:.3f}"
            if 'samples' in node and node['samples'] > 0:
                text += f"\nsamples: {node['samples']}"
            
            ax.text(x, y, text, ha='center', va='center', 
                   bbox=box_style, fontsize=9)
            
            # Plot edges to children
            child_width = width * 0.4
            left_x = x - width * 0.25
            right_x = x + width * 0.25
            child_y = y - 0.15
            
            # Left edge (True)
            ax.plot([x, left_x], [y - 0.03, child_y + 0.03], 'k-', alpha=0.5)
            ax.text((x + left_x) / 2, (y + child_y) / 2, 'True', 
                   fontsize=8, ha='center')
            
            # Right edge (False)
            ax.plot([x, right_x], [y - 0.03, child_y + 0.03], 'k-', alpha=0.5)
            ax.text((x + right_x) / 2, (y + child_y) / 2, 'False', 
                   fontsize=8, ha='center')
            
            # Recurse on children
            self._plot_node(ax, node['left'], left_x, child_y, child_width)
            self._plot_node(ax, node['right'], right_x, child_y, child_width)
            
        else:
            # Plot leaf node
            prediction = node['prediction']
            
            # Determine the signal
            if isinstance(prediction, (int, np.integer)):
                signal = self.signal_name if prediction == 1 else f"NO {self.signal_name}"
                color = 'lightgreen' if prediction == 1 else 'lightcoral'
            else:
                # For regression, check against threshold
                if self.model_type == 'shortonly':
                    is_signal = prediction < self.effective_threshold
                else:
                    is_signal = prediction > self.effective_threshold
                signal = self.signal_name if is_signal else f"NO {self.signal_name}"
                color = 'lightgreen' if is_signal else 'lightcoral'
                
            leaf_style = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7)
            
            text = f"{signal}"
            if isinstance(prediction, (float, np.floating)):
                text += f"\nvalue: {prediction:.3f}"
            if 'samples' in node and node['samples'] > 0:
                text += f"\nsamples: {node['samples']}"
            
            ax.text(x, y, text, ha='center', va='center',
                   bbox=leaf_style, fontsize=9, weight='bold')
    
    def plot_aggregate_summary(self, fig=None, ax=None):
        """Plot aggregate summary of all trees"""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        aggregate = self.create_aggregate_rules()
        
        if not aggregate:
            ax.text(0.5, 0.5, "No model loaded", ha='center', va='center', fontsize=14)
            return fig
        
        # Clear the axis
        ax.clear()
        ax.axis('off')
        
        # Title
        title = f"Aggregate Decision Rules - {aggregate['signal_name']} Model\n"
        title += f"(Analyzed {aggregate['n_trees_analyzed']} of {aggregate['total_trees']} trees)"
        ax.text(0.5, 0.95, title, ha='center', va='top', 
               fontsize=14, fontweight='bold')
        
        # Most common split features
        y_pos = 0.85
        ax.text(0.5, y_pos, "Most Frequently Used Features:", 
               ha='center', va='top', fontsize=12, fontweight='bold')
        
        y_pos -= 0.05
        for i, split_info in enumerate(aggregate['most_common_splits']):
            text = f"{i+1}. {split_info['feature']}: "
            text += f"Used {split_info['frequency']} times, "
            text += f"Avg threshold: {split_info['avg_threshold']:.3f} "
            text += f"(range: {split_info['min_threshold']:.3f} to {split_info['max_threshold']:.3f})"
            
            ax.text(0.1, y_pos, text, ha='left', va='top', fontsize=10)
            y_pos -= 0.08
        
        # Add a visual representation of split frequency
        if aggregate['most_common_splits']:
            y_pos -= 0.05
            ax.text(0.5, y_pos, "Feature Usage Frequency:", 
                   ha='center', va='top', fontsize=12, fontweight='bold')
            y_pos -= 0.05
            
            # Create bar chart
            features = [s['feature'] for s in aggregate['most_common_splits']]
            frequencies = [s['frequency'] for s in aggregate['most_common_splits']]
            
            # Normalize frequencies for display
            max_freq = max(frequencies)
            bar_width = 0.6
            bar_start = 0.2
            
            for i, (feature, freq) in enumerate(zip(features, frequencies)):
                bar_length = (freq / max_freq) * bar_width
                y_bar = y_pos - i * 0.06
                
                # Draw bar
                rect = Rectangle((bar_start, y_bar - 0.02), bar_length, 0.04,
                               facecolor='steelblue', alpha=0.7)
                ax.add_patch(rect)
                
                # Add label
                ax.text(bar_start - 0.02, y_bar, feature, 
                       ha='right', va='center', fontsize=9)
                ax.text(bar_start + bar_length + 0.02, y_bar, 
                       f"{freq}", ha='left', va='center', fontsize=9)
        
        return fig
    
    def get_model_info(self):
        """Get summary information about the model"""
        if not self.trees:
            return "No model loaded"
        
        info = []
        info.append(f"Model Type: {self.model_type}")
        info.append(f"Signal: {self.signal_name}")
        info.append(f"Direction: {self.direction_name}")
        info.append(f"Total Trees: {len(self.trees)}")
        info.append(f"Features: {', '.join(self.feature_names[:5])}")
        if len(self.feature_names) > 5:
            info.append(f"         ...and {len(self.feature_names) - 5} more")
        info.append(f"Target Threshold: {self.target_threshold:.3f}")
        
        # Get tree complexity
        if self.trees:
            tree = self.trees[0]
            if hasattr(tree, 'tree_'):
                max_depth = tree.tree_.max_depth
                n_leaves = tree.tree_.n_leaves if hasattr(tree.tree_, 'n_leaves') else 0
                info.append(f"Tree Max Depth: {max_depth}")
                info.append(f"Tree Leaves: {n_leaves}")
        
        return "\n".join(info)