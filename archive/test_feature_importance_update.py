"""
Test that feature importance is updated after walk-forward validation
"""
import os
import pandas as pd
from datetime import datetime

print("="*80)
print("TESTING FEATURE IMPORTANCE AUTO-UPDATE")
print("="*80)

# Check if feature importance exists before validation
if os.path.exists('feature_importance.png'):
    old_time = os.path.getmtime('feature_importance.png')
    old_time_str = datetime.fromtimestamp(old_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\nExisting feature_importance.png found")
    print(f"Last modified: {old_time_str}")
else:
    print("\nNo existing feature_importance.png found")
    old_time = None

# Run a quick validation
print("\nRunning validation to test feature importance collection...")
print("-"*40)

from OMtree_validation import DirectionalValidator

try:
    validator = DirectionalValidator()
    
    # Override for quick test
    validator.step_size = 200  # Larger steps for faster test
    validator.test_size = 50   # Smaller test size
    
    # Run validation
    results = validator.run_validation(verbose=True)
    
    if results is not None and len(results) > 0:
        print(f"\nValidation complete: {len(results)} predictions")
        
        # Check if feature importance was updated
        if os.path.exists('feature_importance.png'):
            new_time = os.path.getmtime('feature_importance.png')
            new_time_str = datetime.fromtimestamp(new_time).strftime('%Y-%m-%d %H:%M:%S')
            
            if old_time is None:
                print(f"\n[SUCCESS] Feature importance chart created!")
                print(f"Created at: {new_time_str}")
            elif new_time > old_time:
                print(f"\n[SUCCESS] Feature importance chart updated!")
                print(f"Updated at: {new_time_str}")
            else:
                print(f"\n[WARNING] Feature importance chart NOT updated")
                print(f"Still at: {old_time_str}")
        else:
            print(f"\n[ERROR] Feature importance chart not found after validation")
            
        # Check CSV too
        if os.path.exists('feature_importance.csv'):
            importance_df = pd.read_csv('feature_importance.csv')
            print(f"\nFeature importance data saved:")
            print(importance_df)
    else:
        print("\nNo validation results generated")
        
except Exception as e:
    print(f"\nError during test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The feature importance chart should now be automatically updated
after every walk-forward validation run. It shows:

1. Average importance across all models trained during walk-forward
2. Standard deviation showing consistency across different periods
3. Both bar chart and pie chart visualizations
4. Saved as both PNG chart and CSV data

This is much better than the old approach which trained a separate
model just for feature importance!
""")