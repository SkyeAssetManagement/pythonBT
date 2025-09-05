"""
Feature Selection Timeline Visualizer
Creates visual representation of which features are selected over time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os

class FeatureSelectionVisualizer:
    def __init__(self):
        self.selection_history = []
        self.all_features = []
        self.config = {}
        
    def load_selection_history(self, filename='feature_selection_history.json'):
        """Load feature selection history from JSON file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.selection_history = data.get('history', [])
                self.all_features = data.get('all_features', [])
                self.config = data.get('config', {})
                return True
        return False
    
    def set_selection_history(self, history, all_features):
        """Directly set selection history from validation results"""
        self.selection_history = history
        self.all_features = all_features
    
    def create_timeline_chart(self, parent_frame=None, fig_size=(14, 8)):
        """Create timeline chart showing feature selection over time
        
        Args:
            parent_frame: tkinter frame to embed chart (if None, returns figure)
            fig_size: tuple of (width, height) for figure size
        """
        if not self.selection_history:
            return None
            
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=fig_size, 
                                             gridspec_kw={'height_ratios': [3, 1.5, 1.5]})
        
        # Get all unique features if not provided
        if not self.all_features:
            for step in self.selection_history:
                for feat in step.get('selected_features', []):
                    if feat not in self.all_features:
                        self.all_features.append(feat)
        
        # Sort features for consistent display
        self.all_features.sort()
        
        # --- TOP CHART: Feature Usage Timeline ---
        self._create_timeline_grid(ax1)
        
        # --- MIDDLE CHART: Feature Count Over Time ---
        self._create_feature_count_chart(ax2)
        
        # --- BOTTOM CHART: Feature Frequency Bar Chart ---
        self._create_frequency_chart(ax3)
        
        plt.tight_layout()
        
        # If parent_frame provided, embed in tkinter
        if parent_frame:
            canvas = FigureCanvasTkAgg(fig, master=parent_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            return widget
        else:
            return fig
    
    def _create_timeline_grid(self, ax):
        """Create the main timeline grid showing feature selection"""
        n_steps = len(self.selection_history)
        n_features = len(self.all_features)
        
        # Create grid data
        grid = np.zeros((n_features, n_steps))
        importance_grid = np.zeros((n_features, n_steps))
        
        for i, step in enumerate(self.selection_history):
            selected = step.get('selected_features', [])
            scores = step.get('selection_scores', {})
            
            for feat in selected:
                if feat in self.all_features:
                    feat_idx = self.all_features.index(feat)
                    grid[feat_idx, i] = 1
                    importance_grid[feat_idx, i] = scores.get(feat, 0.5)
        
        # Create colored rectangles for selected features
        for i in range(n_features):
            for j in range(n_steps):
                if grid[i, j] == 1:
                    # Color intensity based on importance score
                    importance = importance_grid[i, j]
                    color_intensity = min(1.0, importance * 2)  # Scale for visibility
                    
                    rect = patches.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                                           linewidth=0.5,
                                           edgecolor='black',
                                           facecolor=plt.cm.RdYlGn(color_intensity),
                                           alpha=0.8)
                    ax.add_patch(rect)
        
        # Set axis properties
        ax.set_xlim(-0.5, n_steps-0.5)
        ax.set_ylim(-0.5, n_features-0.5)
        
        # Set ticks and labels
        ax.set_xticks(range(0, n_steps, max(1, n_steps//20)))  # Show max 20 x-labels
        ax.set_xticklabels([f'Step {i+1}' for i in range(0, n_steps, max(1, n_steps//20))], 
                          rotation=45, ha='right', fontsize=8)
        
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(self.all_features, fontsize=9)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Labels and title
        ax.set_xlabel('Walk-Forward Steps', fontsize=10)
        ax.set_ylabel('Features', fontsize=10)
        
        # Get model type from config if available
        model_type = self.config.get('model_type', 'unknown')
        ax.set_title(f'Feature Selection Timeline - {model_type.title()} Model (Green=High Importance, Red=Low)', 
                    fontsize=12, fontweight='bold')
        
        # Add colorbar for importance
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(vmin=0, vmax=0.5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01, fraction=0.03)
        cbar.set_label('Importance', rotation=270, labelpad=15, fontsize=9)
    
    def _create_feature_count_chart(self, ax):
        """Create line chart showing number of features selected over time"""
        n_features_per_step = []
        
        for step in self.selection_history:
            n_features_per_step.append(len(step.get('selected_features', [])))
        
        steps = range(1, len(n_features_per_step) + 1)
        ax.plot(steps, n_features_per_step, 'b-', linewidth=2, marker='o', 
                markersize=3, markerfacecolor='blue', alpha=0.7)
        
        # Add average line
        avg_features = np.mean(n_features_per_step)
        ax.axhline(y=avg_features, color='red', linestyle='--', alpha=0.5, 
                  label=f'Average: {avg_features:.1f}')
        
        ax.set_xlabel('Walk-Forward Step', fontsize=9)
        ax.set_ylabel('# Features', fontsize=9)
        ax.set_title('Number of Features Selected Over Time', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Set x-axis to match timeline
        ax.set_xlim(0.5, len(n_features_per_step) + 0.5)
    
    def _create_frequency_chart(self, ax):
        """Create horizontal bar chart showing feature selection frequency"""
        feature_counts = {}
        total_steps = len(self.selection_history)
        
        for step in self.selection_history:
            for feat in step.get('selected_features', []):
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_features:
            features, counts = zip(*sorted_features)
            percentages = [count/total_steps * 100 for count in counts]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(features))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
            
            bars = ax.barh(y_pos, percentages, color=colors, alpha=0.8)
            
            # Add percentage labels
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                       f'{pct:.1f}%', va='center', fontsize=8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('Selection Frequency (%)', fontsize=9)
            ax.set_title('Feature Selection Frequency', fontsize=10, fontweight='bold')
            ax.set_xlim(0, max(percentages) * 1.15)
            ax.grid(True, alpha=0.3, axis='x')
    
    def create_summary_stats(self):
        """Generate summary statistics from selection history"""
        if not self.selection_history:
            return {}
        
        total_steps = len(self.selection_history)
        feature_counts = {}
        feature_importances = {}
        features_per_step = []
        
        for step in self.selection_history:
            selected = step.get('selected_features', [])
            scores = step.get('selection_scores', {})
            
            features_per_step.append(len(selected))
            
            for feat in selected:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
                if feat not in feature_importances:
                    feature_importances[feat] = []
                if feat in scores:
                    feature_importances[feat].append(scores[feat])
        
        # Calculate statistics
        stats = {
            'total_steps': total_steps,
            'avg_features_per_step': np.mean(features_per_step) if features_per_step else 0,
            'std_features_per_step': np.std(features_per_step) if features_per_step else 0,
            'min_features': min(features_per_step) if features_per_step else 0,
            'max_features': max(features_per_step) if features_per_step else 0,
            'most_selected': max(feature_counts.items(), key=lambda x: x[1]) if feature_counts else None,
            'least_selected': min(feature_counts.items(), key=lambda x: x[1]) if feature_counts else None,
            'feature_stability': {},
            'avg_importances': {}
        }
        
        # Calculate stability (consecutive selections) and average importance
        for feat in feature_counts:
            consecutive_runs = []
            current_run = 0
            
            for step in self.selection_history:
                if feat in step.get('selected_features', []):
                    current_run += 1
                else:
                    if current_run > 0:
                        consecutive_runs.append(current_run)
                    current_run = 0
            
            if current_run > 0:
                consecutive_runs.append(current_run)
            
            stats['feature_stability'][feat] = {
                'selection_rate': feature_counts[feat] / total_steps,
                'avg_consecutive_steps': np.mean(consecutive_runs) if consecutive_runs else 0,
                'max_consecutive_steps': max(consecutive_runs) if consecutive_runs else 0
            }
            
            if feat in feature_importances and feature_importances[feat]:
                stats['avg_importances'][feat] = np.mean(feature_importances[feat])
        
        return stats


# Standalone test
if __name__ == "__main__":
    # Create sample data for testing
    sample_history = [
        {'selected_features': ['Ret_0-1hr', 'Ret_2-4hr', 'Ret_4-8hr'], 
         'selection_scores': {'Ret_0-1hr': 0.4, 'Ret_2-4hr': 0.3, 'Ret_4-8hr': 0.2}},
        {'selected_features': ['Ret_0-1hr', 'Ret_8-16hr'], 
         'selection_scores': {'Ret_0-1hr': 0.5, 'Ret_8-16hr': 0.35}},
        {'selected_features': ['Ret_2-4hr', 'Ret_16-32hr', 'Ret_8-16hr'], 
         'selection_scores': {'Ret_2-4hr': 0.25, 'Ret_16-32hr': 0.45, 'Ret_8-16hr': 0.3}},
        {'selected_features': ['Ret_16-32hr', 'Ret_32-64hr'], 
         'selection_scores': {'Ret_16-32hr': 0.6, 'Ret_32-64hr': 0.2}},
        {'selected_features': ['Ret_0-1hr', 'Ret_4-8hr'], 
         'selection_scores': {'Ret_0-1hr': 0.55, 'Ret_4-8hr': 0.25}},
    ]
    
    all_features = ['Ret_0-1hr', 'Ret_1-2hr', 'Ret_2-4hr', 'Ret_4-8hr', 
                   'Ret_8-16hr', 'Ret_16-32hr', 'Ret_32-64hr', 'Ret_64-128hr']
    
    # Create visualizer and test
    viz = FeatureSelectionVisualizer()
    viz.set_selection_history(sample_history, all_features)
    
    # Create chart
    fig = viz.create_timeline_chart()
    if fig:
        plt.show()
    
    # Print stats
    stats = viz.create_summary_stats()
    print("\nFeature Selection Statistics:")
    print(f"Average features per step: {stats['avg_features_per_step']:.2f}")
    print(f"Most selected: {stats['most_selected']}")
    print("\nFeature stability:")
    for feat, stability in stats['feature_stability'].items():
        print(f"  {feat}: {stability['selection_rate']*100:.1f}% selection rate")