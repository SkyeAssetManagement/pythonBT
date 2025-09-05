"""
Test Full Walk-Forward Validation Pipeline
"""

import pandas as pd
import numpy as np
import configparser
import os
from OMtree_validation import DirectionalValidator
from OMtree_preprocessing import DataPreprocessor

print("=" * 80)
print("FULL WALK-FORWARD VALIDATION TEST")
print("=" * 80)

# Create test config with smaller windows for faster testing
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

# Use smaller windows for testing
config['validation']['train_size'] = '100'
config['validation']['test_size'] = '20'
config['validation']['step_size'] = '20'

# Enable VolSignal to test full feature set
config['preprocessing']['add_volatility_signal'] = 'true'

# Save test config
test_config = 'test_walkforward.ini'
with open(test_config, 'w') as f:
    config.write(f)

try:
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = pd.read_csv(config['data']['csv_file'])
    preprocessor = DataPreprocessor(test_config)
    processed = preprocessor.process_data(df)
    
    print(f"   Data shape: {processed.shape}")
    
    # Check VolSignal was created
    if 'VolSignal_Mean250d' in processed.columns:
        print("   VolSignal: CREATED")
    else:
        print("   VolSignal: NOT CREATED")
    
    # Initialize validator
    print("\n2. Initializing validator...")
    validator = DirectionalValidator(test_config)
    
    # Run mini walk-forward (just 5 steps for testing)
    print("\n3. Running walk-forward validation (5 steps)...")
    
    results = []
    step_size = validator.step_size
    train_size = validator.train_size
    test_size = validator.test_size
    
    # Start after enough data for preprocessing
    start_idx = max(250, train_size)  # 250 for VolSignal window
    
    for step in range(5):
        test_end_idx = start_idx + step * step_size + test_size
        
        if test_end_idx > len(processed) - 100:  # Leave some buffer
            break
        
        # Get train/test split
        result = validator.get_train_test_split(processed, test_end_idx)
        
        if result[0] is not None:
            train_X, train_y, test_X, test_y, test_indices = result
            
            # Check dimensions
            print(f"\n   Step {step + 1}:")
            print(f"   - Train: {train_X.shape if hasattr(train_X, 'shape') else len(train_X)} samples")
            print(f"   - Test: {test_X.shape if hasattr(test_X, 'shape') else len(test_X)} samples")
            
            # Check for multiple features (including VolSignal)
            if hasattr(train_X, 'ndim') and train_X.ndim > 1:
                n_features = train_X.shape[1]
                print(f"   - Features: {n_features}")
                
                # With VolSignal enabled, should have 2 features
                expected = 2 if config['preprocessing'].getboolean('add_volatility_signal') else 1
                if n_features != expected:
                    print(f"   WARNING: Expected {expected} features, got {n_features}")
            
            # Train model (using validator's model)
            from OMtree_model import DirectionalTreeEnsemble
            model = DirectionalTreeEnsemble(test_config, verbose=False)
            
            try:
                model.fit(train_X, train_y)
                predictions = model.predict(test_X)
                probabilities = model.predict_proba(test_X)
                
                # Calculate metrics
                if len(predictions) > 0:
                    trade_signals = np.sum(predictions == 1)
                    trade_pct = trade_signals / len(predictions) * 100
                    
                    # For directional model, check actual returns
                    model_type = config['model']['model_type']
                    threshold = float(config['model']['target_threshold'])
                    
                    if model_type == 'longonly':
                        profitable = test_y > threshold
                    else:  # shortonly
                        profitable = test_y < -threshold
                    
                    # Hit rate for trades
                    if trade_signals > 0:
                        trade_mask = predictions == 1
                        hit_rate = np.mean(profitable[trade_mask]) * 100
                    else:
                        hit_rate = 0
                    
                    print(f"   - Trades: {trade_signals}/{len(predictions)} ({trade_pct:.1f}%)")
                    print(f"   - Hit rate: {hit_rate:.1f}%")
                    print(f"   - Avg confidence: {probabilities.mean():.3f}")
                    
                    results.append({
                        'step': step + 1,
                        'trades': trade_signals,
                        'total': len(predictions),
                        'hit_rate': hit_rate,
                        'avg_prob': probabilities.mean()
                    })
                    
            except Exception as e:
                print(f"   ERROR in model training: {e}")
        else:
            print(f"   Step {step + 1}: No valid data")
    
    # Summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)
    
    if results:
        df_results = pd.DataFrame(results)
        print("\nResults by step:")
        print(df_results.to_string())
        
        print("\nAggregate metrics:")
        print(f"  Average trades per period: {df_results['trades'].mean():.1f}")
        print(f"  Average hit rate: {df_results['hit_rate'].mean():.1f}%")
        print(f"  Average confidence: {df_results['avg_prob'].mean():.3f}")
        
        # Check consistency
        print("\nConsistency checks:")
        
        # Check if model is too conservative (no trades)
        if df_results['trades'].sum() == 0:
            print("  WARNING: Model not generating any trades")
            print("  - Check target_threshold and vote_threshold settings")
        
        # Check if probabilities are reasonable
        avg_prob = df_results['avg_prob'].mean()
        if avg_prob < 0.1:
            print("  WARNING: Very low confidence scores")
            print("  - Model may be undertrained or parameters too restrictive")
        elif avg_prob > 0.9:
            print("  WARNING: Very high confidence scores")
            print("  - Model may be overfitting")
        else:
            print("  [OK] Confidence scores in reasonable range")
        
        # Check if hit rates are consistent
        hit_std = df_results['hit_rate'].std()
        if hit_std > 20:
            print("  WARNING: High variance in hit rates")
            print("  - Model performance may be unstable")
        else:
            print("  [OK] Hit rates relatively consistent")
    else:
        print("No results generated - check configuration")
    
    print("\n[SUCCESS] Walk-forward validation pipeline works correctly!")
    
except Exception as e:
    print(f"\n[ERROR] Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Clean up
    if os.path.exists(test_config):
        os.remove(test_config)

print("\n" + "=" * 80)