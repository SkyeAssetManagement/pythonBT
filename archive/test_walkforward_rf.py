"""
Test walk-forward validation with RF feature selection
"""
from OMtree_validation import DirectionalValidator

print("Testing walk-forward with RF feature selection...")
print("-"*50)

try:
    # Initialize validator
    validator = DirectionalValidator('OMtree_config.ini')
    
    # Set small windows for quick test
    validator.train_size = 500
    validator.test_size = 50
    validator.step_size = 100
    
    print("Configuration:")
    print(f"  Feature selection: {'ENABLED' if validator.feature_selection_enabled else 'DISABLED'}")
    if validator.feature_selection_enabled:
        print(f"  RF trees: {validator.feature_selector.rf_n_estimators}")
        print(f"  RF max depth: {validator.feature_selector.rf_max_depth}")
        print(f"  RF min samples split: {validator.feature_selector.rf_min_samples_split}")
        print(f"  RF bootstrap fraction: {validator.feature_selector.rf_bootstrap_fraction}")
        print(f"  Importance threshold: {validator.feature_selector.importance_threshold}")
    
    print("\nRunning quick walk-forward test...")
    
    # Run validation for just a few steps
    results = validator.run_validation(verbose=True)
    
    if len(results) > 0:
        print(f"\n[SUCCESS] Walk-forward completed with {len(results)} predictions")
        
        # Check if feature selection history was recorded
        if validator.selection_history:
            print(f"[SUCCESS] Feature selection recorded {len(validator.selection_history)} steps")
            print("\nFirst few feature selections:")
            for i, step in enumerate(validator.selection_history[:3]):
                print(f"  Step {i+1}: {step['selected_features']}")
        else:
            print("[INFO] No feature selection history (may be disabled)")
    else:
        print("[WARNING] No results generated")
        
except Exception as e:
    print(f"\n[ERROR] Walk-forward failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")