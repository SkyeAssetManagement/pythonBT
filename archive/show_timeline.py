"""
Quick display of saved feature selection timeline
"""
from feature_selection_visualizer import FeatureSelectionVisualizer
import matplotlib.pyplot as plt

# Create visualizer
viz = FeatureSelectionVisualizer()

# Load saved history
if viz.load_selection_history('feature_selection_history.json'):
    print("Feature Selection Timeline Loaded")
    print(f"Steps: {len(viz.selection_history)}")
    print(f"Features tracked: {viz.all_features}")
    
    # Create and show chart
    fig = viz.create_timeline_chart(fig_size=(14, 9))
    
    # Get statistics
    stats = viz.create_summary_stats()
    
    print("\n" + "="*50)
    print("FEATURE SELECTION STATISTICS")
    print("="*50)
    print(f"Total steps: {stats['total_steps']}")
    print(f"Avg features/step: {stats['avg_features_per_step']:.2f} ± {stats['std_features_per_step']:.2f}")
    print(f"Range: {stats['min_features']} to {stats['max_features']} features")
    
    if stats['most_selected']:
        feat, count = stats['most_selected']
        print(f"\nMost selected: {feat} ({count}/{stats['total_steps']} = {count/stats['total_steps']*100:.1f}%)")
    
    print("\nTop 3 Most Stable Features:")
    sorted_stability = sorted(stats['feature_stability'].items(), 
                            key=lambda x: x[1]['selection_rate'], 
                            reverse=True)
    for feat, stability in sorted_stability[:3]:
        print(f"  {feat}: {stability['selection_rate']*100:.1f}% selection rate")
    
    # Save high-res version
    fig.savefig('feature_timeline_hires.png', dpi=150, bbox_inches='tight')
    print("\nHigh-res chart saved: feature_timeline_hires.png")
    
    plt.show()
else:
    print("No feature selection history found. Run walk-forward validation first.")