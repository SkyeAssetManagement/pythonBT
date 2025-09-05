"""
Collect and aggregate feature importance from walk-forward validation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from glob import glob

def collect_walkforward_importance():
    """
    Collect feature importance from all models trained during walk-forward
    """
    
    # Look for saved models from walk-forward
    model_files = glob('walkforward_models/*.pkl')
    
    if not model_files:
        print("No walk-forward models found. Running separate training...")
        # Fall back to generating from scratch
        from generate_feature_importance import generate_feature_importance
        return generate_feature_importance()
    
    print(f"Found {len(model_files)} walk-forward models")
    
    # Aggregate importance across all models
    all_importances = []
    feature_names = None
    
    for model_file in model_files:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            
            if feature_names is None:
                feature_names = model_data.get('features', [])
            
            # Get importance from this model
            importance = model.get_feature_importance()
            if importance is not None:
                all_importances.append(importance)
    
    if not all_importances:
        print("No feature importance data found in models")
        return None
    
    # Average importance across all models
    avg_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)
    
    # Create enhanced visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Bar chart with error bars
    y_pos = np.arange(len(feature_names))
    ax1.barh(y_pos, avg_importance, xerr=std_importance, 
             color='steelblue', alpha=0.8, capsize=5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('Average Importance')
    ax1.set_title('Feature Importance Across Walk-Forward', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Box plot showing distribution
    ax2.boxplot(all_importances, vert=False, labels=feature_names)
    ax2.set_xlabel('Importance Distribution')
    ax2.set_title('Importance Variability', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Heatmap over time
    importance_matrix = np.array(all_importances).T
    im = ax3.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_yticks(range(len(feature_names)))
    ax3.set_yticklabels(feature_names)
    ax3.set_xlabel('Walk-Forward Step')
    ax3.set_title('Importance Evolution Over Time', fontweight='bold')
    plt.colorbar(im, ax=ax3)
    
    # 4. Cumulative importance
    sorted_idx = np.argsort(avg_importance)[::-1]
    cumsum = np.cumsum(avg_importance[sorted_idx])
    ax4.plot(range(1, len(cumsum)+1), cumsum, 'bo-')
    ax4.set_xlabel('Number of Features')
    ax4.set_ylabel('Cumulative Importance')
    ax4.set_title('Cumulative Feature Importance', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
    ax4.legend()
    
    plt.suptitle(f'Feature Importance Analysis from {len(model_files)} Walk-Forward Models', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('feature_importance_walkforward.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Walk-forward feature importance saved to: feature_importance_walkforward.png")
    
    # Save detailed statistics
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Importance': avg_importance,
        'Std_Importance': std_importance,
        'CV': std_importance / (avg_importance + 1e-10),  # Coefficient of variation
        'Rank': np.argsort(avg_importance)[::-1].argsort() + 1
    })
    importance_df = importance_df.sort_values('Mean_Importance', ascending=False)
    importance_df.to_csv('feature_importance_walkforward.csv', index=False)
    
    print("\nTop Features by Average Importance:")
    print(importance_df.head())
    
    return importance_df

if __name__ == "__main__":
    collect_walkforward_importance()