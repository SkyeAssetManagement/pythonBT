"""
Test MDI feature selection using model training settings
"""
from OMtree_validation import DirectionalValidator
import configparser

print("="*80)
print("TEST: MDI Feature Selection with Model Settings")
print("="*80)

# Read config to show settings
config = configparser.ConfigParser()
config.read('OMtree_config.ini')

print("\nModel Configuration:")
print(f"  n_trees_method: {config['model']['n_trees_method']}")
print(f"  n_trees: {config['model']['n_trees']}")
print(f"  max_depth: {config['model']['max_depth']}")
print(f"  bootstrap_fraction: {config['model']['bootstrap_fraction']}")
print(f"  min_leaf_fraction: {config['model']['min_leaf_fraction']}")

print("\nFeature Selection Configuration:")
print(f"  enabled: {config['feature_selection']['enabled']}")
print(f"  min_features: {config['feature_selection']['min_features']}")
print(f"  max_features: {config['feature_selection']['max_features']}")
print(f"  selection_lookback: {config['feature_selection']['selection_lookback']}")
print(f"  importance_threshold: {config['feature_selection']['importance_threshold']}")

print("\n" + "-"*50)

try:
    # Initialize validator
    validator = DirectionalValidator('OMtree_config.ini')
    
    # Set small windows for quick test
    validator.train_size = 500
    validator.test_size = 50
    validator.step_size = 200
    
    print("\nRunning walk-forward validation...")
    print("Expected MDI settings:")
    
    # Calculate expected settings
    n_features = 8  # We have 8 features
    if validator.n_trees_method == 'per_feature':
        expected_trees = validator.n_trees_base * n_features
        print(f"  Trees for MDI: {expected_trees} ({validator.n_trees_base} per feature × {n_features} features)")
    else:
        expected_trees = validator.n_trees_base
        print(f"  Trees for MDI: {expected_trees} (absolute mode)")
    
    print(f"  Max depth: {validator.model_max_depth} (from model)")
    print(f"  Bootstrap: {validator.model_bootstrap_fraction} (from model)")
    print(f"  Min leaf fraction: {validator.model_min_leaf_fraction} (from model)")
    
    # Run validation
    results = validator.run_validation(verbose=True)
    
    if len(results) > 0:
        print(f"\n[SUCCESS] Walk-forward completed with {len(results)} predictions")
        
        # Check feature selection
        if validator.selection_history:
            print(f"[SUCCESS] Feature selection recorded {len(validator.selection_history)} steps")
            
            # Verify RF settings were used correctly
            if validator.feature_selector:
                print("\nActual MDI RF settings used:")
                print(f"  Trees: {validator.feature_selector.rf_n_estimators}")
                print(f"  Max depth: {validator.feature_selector.rf_max_depth}")
                print(f"  Min samples split: {validator.feature_selector.rf_min_samples_split}")
                print(f"  Bootstrap fraction: {validator.feature_selector.rf_bootstrap_fraction}")
                
                # Check if trees match expectation
                if validator.feature_selector.rf_n_estimators == expected_trees:
                    print("\n[OK] MDI trees match expected value!")
                else:
                    print(f"\n[WARNING] MDI trees ({validator.feature_selector.rf_n_estimators}) != expected ({expected_trees})")
        
        # Show feature selection results
        print("\nFeature selection patterns:")
        all_selections = {}
        for step in validator.selection_history:
            for feat in step['selected_features']:
                all_selections[feat] = all_selections.get(feat, 0) + 1
        
        print("Most frequently selected features:")
        for feat, count in sorted(all_selections.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = (count / len(validator.selection_history)) * 100
            print(f"  {feat}: {pct:.1f}%")
            
    else:
        print("[WARNING] No results generated")
        
except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
MDI feature selection now automatically uses model training settings:
- Trees: Matches model (per_feature mode multiplies by feature count)
- Max depth: Uses model's max_depth setting
- Bootstrap: Uses model's bootstrap_fraction
- Min samples: Derived from model's min_leaf_fraction

Only MDI-specific settings remain in [feature_selection]:
- min_features: Minimum features to select
- max_features: Maximum features to select  
- importance_threshold: Filter features below this importance
- selection_lookback: Recent samples for feature selection
""")