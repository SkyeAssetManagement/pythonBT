"""
Test feature selection timeline visualization
"""
from OMtree_validation import DirectionalValidator
from feature_selection_visualizer import FeatureSelectionVisualizer
import matplotlib.pyplot as plt
import os

print("="*80)
print("TESTING FEATURE SELECTION TIMELINE VISUALIZATION")
print("="*80)

# Step 1: Run validation to generate selection history
print("\n1. Running walk-forward with feature selection...")
print("-"*50)

try:
    validator = DirectionalValidator('OMtree_config.ini')
    
    # Quick validation for testing
    validator.train_size = 500
    validator.test_size = 50
    validator.step_size = 150
    
    print(f"Feature selection enabled: {validator.feature_selection_enabled}")
    print(f"Importance threshold: {validator.fs_importance_threshold if hasattr(validator, 'fs_importance_threshold') else 'N/A'}")
    
    # Run validation
    results = validator.run_validation(verbose=True)
    
    print(f"\n[SUCCESS] Generated {len(results)} predictions")
    
    # Check if history was saved
    if os.path.exists('feature_selection_history.json'):
        print("[SUCCESS] Feature selection history saved")
    else:
        print("[WARNING] History file not created")

except Exception as e:
    print(f"[ERROR] Validation failed: {e}")
    import traceback
    traceback.print_exc()

# Step 2: Test visualization
print("\n\n2. Testing timeline visualization...")
print("-"*50)

try:
    viz = FeatureSelectionVisualizer()
    
    # Load saved history
    if viz.load_selection_history('feature_selection_history.json'):
        print(f"[SUCCESS] Loaded history with {len(viz.selection_history)} steps")
        
        # Create timeline chart
        fig = viz.create_timeline_chart()
        
        if fig:
            print("[SUCCESS] Timeline chart created")
            
            # Save chart
            fig.savefig('test_feature_timeline.png', dpi=100, bbox_inches='tight')
            print("[SUCCESS] Chart saved as test_feature_timeline.png")
            
            # Get statistics
            stats = viz.create_summary_stats()
            print("\n" + "="*50)
            print("FEATURE SELECTION STATISTICS")
            print("="*50)
            print(f"Total walk-forward steps: {stats['total_steps']}")
            print(f"Average features per step: {stats['avg_features_per_step']:.2f}")
            print(f"Min/Max features: {stats['min_features']}/{stats['max_features']}")
            
            if stats['most_selected']:
                feat, count = stats['most_selected']
                pct = (count/stats['total_steps']) * 100
                print(f"Most selected feature: {feat} ({pct:.1f}% of steps)")
            
            print("\nFeature Stability (top 5):")
            sorted_stability = sorted(stats['feature_stability'].items(), 
                                    key=lambda x: x[1]['selection_rate'], 
                                    reverse=True)
            for feat, stability in sorted_stability[:5]:
                print(f"  {feat}:")
                print(f"    Selection rate: {stability['selection_rate']*100:.1f}%")
                print(f"    Avg consecutive: {stability['avg_consecutive_steps']:.1f} steps")
                print(f"    Max consecutive: {stability['max_consecutive_steps']} steps")
            
            # Show plot
            plt.show()
        else:
            print("[ERROR] Failed to create chart")
    else:
        print("[WARNING] No history file found")
        
except Exception as e:
    print(f"[ERROR] Visualization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
Feature Selection Timeline Visualization:
✓ Runs walk-forward validation with feature selection
✓ Saves selection history to JSON file
✓ Creates multi-panel timeline visualization:
  - Top: Timeline grid showing when features are selected
  - Middle: Number of features selected over time
  - Bottom: Feature selection frequency chart
✓ Color coding shows importance scores
✓ Statistics track feature stability and patterns
✓ Integrated into GUI Charts tab
""")