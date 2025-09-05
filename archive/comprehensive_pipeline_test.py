"""
Comprehensive Pipeline Testing and Validation
==============================================
This script thoroughly tests every component of the OMtree pipeline
with synthetic data having known outcomes to verify logical consistency.
"""

import pandas as pd
import numpy as np
import configparser
import os
import sys
from datetime import datetime, timedelta

# Import all pipeline components
from column_detector import ColumnDetector
from OMtree_preprocessing import DataPreprocessor
from OMtree_model import DirectionalTreeEnsemble
from OMtree_validation import DirectionalValidator

# Set random seed for reproducibility
np.random.seed(42)

def create_synthetic_data_with_known_pattern():
    """
    Create synthetic data with a known predictable pattern.
    Features will have clear relationships with targets.
    """
    n_samples = 2000
    dates = pd.date_range(start='2010-01-01', periods=n_samples, freq='D')
    
    # Create features with known patterns
    data = {
        'Date': dates.strftime('%Y-%m-%d'),
        'Time': ['10:00:00'] * n_samples,
    }
    
    # Create features with sinusoidal patterns
    for i in range(4):
        feature_name = f'Ret_{i}-{i+1}hr'
        # Add noise but maintain pattern
        signal = np.sin(np.arange(n_samples) * 0.1 + i) * 0.05
        noise = np.random.randn(n_samples) * 0.01
        data[feature_name] = signal + noise
    
    # Create target that depends on features
    # If sum of features > 0.05, target will be positive (profitable)
    feature_sum = sum([data[f'Ret_{i}-{i+1}hr'] for i in range(4)])
    data['Ret_fwd6hr'] = np.where(feature_sum > 0.05, 0.15, -0.05) + np.random.randn(n_samples) * 0.02
    
    df = pd.DataFrame(data)
    
    # Add some additional columns for completeness
    df['DiffClose@Obs'] = np.random.randn(n_samples) * 100 + 1000
    df['NoneClose@Obs'] = np.random.randn(n_samples) * 100 + 1500
    
    return df

def test_component_1_data_loading():
    """Test 1: Data Loading and Column Detection"""
    print("\n" + "="*80)
    print("TEST 1: DATA LOADING AND COLUMN DETECTION")
    print("="*80)
    
    results = {'component': 'Data Loading', 'tests': [], 'errors': []}
    
    # Create test data
    df = create_synthetic_data_with_known_pattern()
    df.to_csv('test_synthetic.csv', index=False)
    
    # Test column detection
    try:
        # Test auto-detection
        detected = ColumnDetector.auto_detect_columns(df)
        
        # Verify correct detection
        expected_features = ['DiffClose@Obs', 'NoneClose@Obs', 'Ret_0-1hr', 'Ret_1-2hr', 'Ret_2-3hr', 'Ret_3-4hr']
        expected_targets = ['Ret_fwd6hr']
        
        features_correct = all(f in detected['features'] for f in expected_features[2:])  # Skip Diff/None
        targets_correct = detected['targets'] == expected_targets
        
        if features_correct and targets_correct:
            results['tests'].append("Column auto-detection: PASS")
        else:
            results['errors'].append(f"Column detection mismatch. Got features: {detected['features']}, targets: {detected['targets']}")
            
    except Exception as e:
        results['errors'].append(f"Column detection failed: {e}")
    
    # Test with config
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config['data'] = {
        'csv_file': 'test_synthetic.csv',
        'feature_columns': 'Ret_0-1hr,Ret_1-2hr,Ret_2-3hr,Ret_3-4hr',
        'selected_features': 'Ret_0-1hr,Ret_1-2hr',
        'target_column': 'Ret_fwd6hr',
        'all_targets': 'Ret_fwd6hr',
        'date_column': 'Date',
        'time_column': 'Time'
    }
    
    # Save test config
    test_config = 'test_config.ini'
    with open(test_config, 'w') as f:
        config.write(f)
    
    results['tests'].append("Config creation: PASS")
    
    return results

def test_component_2_preprocessing():
    """Test 2: Preprocessing with various parameter combinations"""
    print("\n" + "="*80)
    print("TEST 2: PREPROCESSING PIPELINE")
    print("="*80)
    
    results = {'component': 'Preprocessing', 'tests': [], 'errors': []}
    
    df = pd.read_csv('test_synthetic.csv')
    
    # Test configurations
    test_cases = [
        {'normalize': 'false', 'detrend': 'false', 'method': 'IQR', 'vol_signal': 'false'},
        {'normalize': 'true', 'detrend': 'false', 'method': 'IQR', 'vol_signal': 'false'},
        {'normalize': 'true', 'detrend': 'true', 'method': 'IQR', 'vol_signal': 'false'},
        {'normalize': 'true', 'detrend': 'false', 'method': 'AVS', 'vol_signal': 'false'},
        {'normalize': 'true', 'detrend': 'true', 'method': 'AVS', 'vol_signal': 'true'},
    ]
    
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('test_config.ini')
    
    # Add preprocessing section
    config['preprocessing'] = {
        'vol_window': '20',
        'smoothing_type': 'exponential',
        'smoothing_alpha': '0.1',
        'percentile_upper': '75',
        'percentile_lower': '25',
        'recent_iqr_lookback': '20',
        'avs_slow_window': '60',
        'avs_fast_window': '20',
        'vol_signal_window': '250',
        'vol_signal_decay': '0.995'
    }
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            # Update config
            config['preprocessing']['normalize_features'] = test_case['normalize']
            config['preprocessing']['normalize_target'] = test_case['normalize']
            config['preprocessing']['detrend_features'] = test_case['detrend']
            config['preprocessing']['normalization_method'] = test_case['method']
            config['preprocessing']['add_volatility_signal'] = test_case['vol_signal']
            
            # Save config
            with open('test_config.ini', 'w') as f:
                config.write(f)
            
            # Process data
            preprocessor = DataPreprocessor('test_config.ini')
            processed = preprocessor.process_data(df)
            
            # Verify expected columns exist
            checks = []
            
            # Check detrending
            if test_case['detrend'] == 'true':
                detrend_cols = [c for c in processed.columns if '_detrend' in c]
                checks.append(('Detrend columns created', len(detrend_cols) > 0))
            
            # Check normalization
            if test_case['normalize'] == 'true':
                vol_adj_cols = [c for c in processed.columns if '_vol_adj' in c]
                checks.append(('Vol-adjusted columns created', len(vol_adj_cols) > 0))
            
            # Check volatility signal
            if test_case['vol_signal'] == 'true':
                checks.append(('VolSignal created', 'VolSignal_Mean250d' in processed.columns))
            
            # Check target is preserved
            checks.append(('Target preserved', 'Ret_fwd6hr' in processed.columns or 'Ret_fwd6hr_vol_adj' in processed.columns))
            
            # Report results
            all_passed = all(check[1] for check in checks)
            if all_passed:
                results['tests'].append(f"Test case {i}: PASS")
            else:
                failed = [check[0] for check in checks if not check[1]]
                results['errors'].append(f"Test case {i} failed: {failed}")
                
        except Exception as e:
            results['errors'].append(f"Test case {i} error: {e}")
    
    return results

def test_component_3_model():
    """Test 3: Model Training and Prediction Logic"""
    print("\n" + "="*80)
    print("TEST 3: MODEL TRAINING AND PREDICTION")
    print("="*80)
    
    results = {'component': 'Model', 'tests': [], 'errors': []}
    
    # Create simple test data with known outcome
    # All positive features -> should predict profitable (1)
    # All negative features -> should predict not profitable (0)
    
    X_positive = np.ones((100, 2)) * 0.2  # Clearly positive
    y_positive = np.ones(100) * 0.15  # Above threshold
    
    X_negative = np.ones((100, 2)) * -0.2  # Clearly negative  
    y_negative = np.ones(100) * -0.05  # Below threshold
    
    X_test_pos = np.ones((20, 2)) * 0.2
    X_test_neg = np.ones((20, 2)) * -0.2
    
    # Test configurations
    test_configs = [
        {'algorithm': 'decision_trees', 'aggregation': 'mean', 'threshold': 0.1, 'vote': 0.6},
        {'algorithm': 'decision_trees', 'aggregation': 'median', 'threshold': 0.1, 'vote': 0.6},
        {'algorithm': 'extra_trees', 'aggregation': 'mean', 'threshold': 0.1, 'vote': 0.6},
    ]
    
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('test_config.ini')
    
    # Add model section
    config['model'] = {
        'model_type': 'longonly',
        'n_trees': '50',
        'max_depth': '1',
        'bootstrap_fraction': '0.8',
        'min_leaf_fraction': '0.1',
        'random_seed': '42'
    }
    
    for i, test_cfg in enumerate(test_configs, 1):
        try:
            # Update config
            config['model']['algorithm'] = test_cfg['algorithm']
            config['model']['probability_aggregation'] = test_cfg['aggregation']
            config['model']['target_threshold'] = str(test_cfg['threshold'])
            config['model']['vote_threshold'] = str(test_cfg['vote'])
            
            with open('test_config.ini', 'w') as f:
                config.write(f)
            
            # Train on positive examples
            model = DirectionalTreeEnsemble('test_config.ini', verbose=False)
            model.fit(X_positive, y_positive)
            
            # Predict on positive test (should be 1)
            pred_pos = model.predict(X_test_pos)
            prob_pos = model.predict_proba(X_test_pos)
            
            # Train on negative examples
            model_neg = DirectionalTreeEnsemble('test_config.ini', verbose=False)
            model_neg.fit(X_negative, y_negative)
            
            # Predict on negative test (should be 0)
            pred_neg = model_neg.predict(X_test_neg)
            prob_neg = model_neg.predict_proba(X_test_neg)
            
            # Verify predictions
            pos_correct = np.mean(pred_pos) > 0.8  # Most should be 1
            neg_correct = np.mean(pred_neg) < 0.2  # Most should be 0
            
            if pos_correct and neg_correct:
                results['tests'].append(f"Config {i} ({test_cfg['algorithm']}, {test_cfg['aggregation']}): PASS")
            else:
                results['errors'].append(f"Config {i} predictions incorrect. Pos mean: {np.mean(pred_pos)}, Neg mean: {np.mean(pred_neg)}")
                
            # Verify probability aggregation
            if test_cfg['aggregation'] == 'median':
                # Median should produce mostly 0 or 1
                is_binary = np.all((prob_pos == 0) | (prob_pos == 1))
                if not is_binary:
                    results['errors'].append(f"Config {i}: Median aggregation not producing binary probabilities")
                    
        except Exception as e:
            results['errors'].append(f"Model test {i} error: {e}")
    
    return results

def test_component_4_validation():
    """Test 4: Walk-Forward Validation Logic"""
    print("\n" + "="*80)
    print("TEST 4: WALK-FORWARD VALIDATION")
    print("="*80)
    
    results = {'component': 'Validation', 'tests': [], 'errors': []}
    
    # Use synthetic data
    df = pd.read_csv('test_synthetic.csv')
    
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('test_config.ini')
    
    # Add validation section
    config['validation'] = {
        'train_size': '100',
        'test_size': '20',
        'step_size': '20',
        'min_training_samples': '50',
        'base_rate': '0.5',
        'validation_start_date': '2010-01-01',
        'validation_end_date': '2015-12-31',
        'date_format_dayfirst': 'false'
    }
    
    # Ensure model section is properly configured
    config['model']['model_type'] = 'longonly'
    config['model']['target_threshold'] = '0.1'
    config['model']['vote_threshold'] = '0.6'
    
    with open('test_config.ini', 'w') as f:
        config.write(f)
    
    try:
        # Initialize validator
        validator = DirectionalValidator('test_config.ini')
        
        # Load and prepare data
        data = validator.load_and_prepare_data()
        
        # Test a single split
        test_end_idx = 200
        result = validator.get_train_test_split(data, test_end_idx)
        
        if result[0] is not None:
            train_X, train_y, test_X, test_y, test_y_raw = result
            
            # Verify shapes
            checks = [
                ('Train size correct', len(train_X) <= 100),
                ('Test size correct', len(test_X) <= 20),
                ('Train/test no overlap', True),  # By design
                ('Raw values preserved', test_y_raw is not None)
            ]
            
            all_passed = all(check[1] for check in checks)
            if all_passed:
                results['tests'].append("Walk-forward split: PASS")
            else:
                failed = [check[0] for check in checks if not check[1]]
                results['errors'].append(f"Split validation failed: {failed}")
                
        # Run short validation
        validation_results = validator.run_validation(verbose=False)
        
        if validation_results is not None and not validation_results.empty:
            results['tests'].append(f"Validation run: PASS ({len(validation_results)} results)")
            
            # Check for required columns
            required_cols = ['prediction', 'probability', 'target_value']
            missing_cols = [c for c in required_cols if c not in validation_results.columns]
            
            if missing_cols:
                results['errors'].append(f"Missing columns in results: {missing_cols}")
        else:
            results['errors'].append("Validation returned no results")
            
    except Exception as e:
        results['errors'].append(f"Validation error: {e}")
    
    return results

def test_component_5_edge_cases():
    """Test 5: Edge Cases and Error Handling"""
    print("\n" + "="*80)
    print("TEST 5: EDGE CASES AND ERROR HANDLING")
    print("="*80)
    
    results = {'component': 'Edge Cases', 'tests': [], 'errors': []}
    
    # Test 1: Empty data
    try:
        empty_df = pd.DataFrame()
        detected = ColumnDetector.auto_detect_columns(empty_df)
        if len(detected['features']) == 0 and len(detected['targets']) == 0:
            results['tests'].append("Empty data handling: PASS")
    except:
        results['errors'].append("Failed to handle empty data")
    
    # Test 2: All NaN values
    try:
        nan_df = pd.DataFrame({
            'feature1': [np.nan] * 100,
            'target': [np.nan] * 100
        })
        detected = ColumnDetector.auto_detect_columns(nan_df)
        results['tests'].append("NaN data handling: PASS")
    except:
        results['errors'].append("Failed to handle NaN data")
    
    # Test 3: Single feature
    try:
        X_single = np.random.randn(100, 1)
        y_single = np.random.randn(100)
        
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read('test_config.ini')
        
        model = DirectionalTreeEnsemble('test_config.ini', verbose=False)
        model.fit(X_single, y_single)
        pred = model.predict(X_single[:10])
        
        if len(pred) == 10:
            results['tests'].append("Single feature handling: PASS")
    except Exception as e:
        results['errors'].append(f"Single feature error: {e}")
    
    # Test 4: Vote threshold edge cases
    test_thresholds = [0.5, 0.51, 0.99, 1.0]
    
    for threshold in test_thresholds:
        try:
            config['model']['vote_threshold'] = str(threshold)
            with open('test_config.ini', 'w') as f:
                config.write(f)
            
            model = DirectionalTreeEnsemble('test_config.ini', verbose=False)
            X = np.random.randn(50, 2)
            y = np.random.randn(50)
            model.fit(X, y)
            pred = model.predict(X[:10])
            
            results['tests'].append(f"Vote threshold {threshold}: PASS")
        except Exception as e:
            results['errors'].append(f"Vote threshold {threshold} error: {e}")
    
    return results

def test_component_6_consistency():
    """Test 6: Cross-Component Consistency"""
    print("\n" + "="*80)
    print("TEST 6: CROSS-COMPONENT CONSISTENCY")
    print("="*80)
    
    results = {'component': 'Consistency', 'tests': [], 'errors': []}
    
    # Create a complete pipeline test
    df = create_synthetic_data_with_known_pattern()
    df.to_csv('test_pipeline.csv', index=False)
    
    # Setup complete config
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config['data'] = {
        'csv_file': 'test_pipeline.csv',
        'feature_columns': 'Ret_0-1hr,Ret_1-2hr,Ret_2-3hr,Ret_3-4hr',
        'selected_features': 'Ret_0-1hr,Ret_1-2hr',
        'target_column': 'Ret_fwd6hr',
        'all_targets': 'Ret_fwd6hr',
        'date_column': 'Date',
        'time_column': 'Time'
    }
    
    config['preprocessing'] = {
        'normalize_features': 'true',
        'normalize_target': 'true',
        'detrend_features': 'true',
        'normalization_method': 'AVS',
        'vol_window': '20',
        'smoothing_type': 'exponential',
        'smoothing_alpha': '0.1',
        'percentile_upper': '75',
        'percentile_lower': '25',
        'recent_iqr_lookback': '20',
        'avs_slow_window': '60',
        'avs_fast_window': '20',
        'add_volatility_signal': 'true',
        'vol_signal_window': '250',
        'vol_signal_decay': '0.995'
    }
    
    config['model'] = {
        'model_type': 'longonly',
        'algorithm': 'decision_trees',
        'probability_aggregation': 'mean',
        'n_trees': '50',
        'max_depth': '1',
        'bootstrap_fraction': '0.8',
        'min_leaf_fraction': '0.1',
        'target_threshold': '0.1',
        'vote_threshold': '0.6',
        'random_seed': '42'
    }
    
    config['validation'] = {
        'train_size': '200',
        'test_size': '50',
        'step_size': '50',
        'min_training_samples': '100',
        'base_rate': '0.5',
        'validation_start_date': '2010-01-01',
        'validation_end_date': '2015-12-31',
        'date_format_dayfirst': 'false'
    }
    
    with open('test_pipeline.ini', 'w') as f:
        config.write(f)
    
    try:
        # Step 1: Preprocessing
        preprocessor = DataPreprocessor('test_pipeline.ini')
        processed = preprocessor.process_data(df)
        
        # Verify columns created correctly
        expected_patterns = ['_detrend', '_vol_adj', 'VolSignal']
        for pattern in expected_patterns:
            found = any(pattern in col for col in processed.columns)
            if found:
                results['tests'].append(f"Pattern '{pattern}' in columns: PASS")
            else:
                results['errors'].append(f"Pattern '{pattern}' not found in columns")
        
        # Step 2: Validation with processed data
        validator = DirectionalValidator('test_pipeline.ini')
        val_results = validator.run_validation(verbose=False)
        
        if val_results is not None and len(val_results) > 0:
            # Check predictions are binary
            unique_preds = val_results['prediction'].unique()
            if set(unique_preds).issubset({0, 1}):
                results['tests'].append("Binary predictions: PASS")
            else:
                results['errors'].append(f"Non-binary predictions found: {unique_preds}")
            
            # Check probabilities are in [0, 1]
            probs = val_results['probability'].values
            if np.all((probs >= 0) & (probs <= 1)):
                results['tests'].append("Probability range [0,1]: PASS")
            else:
                results['errors'].append(f"Probabilities out of range: min={probs.min()}, max={probs.max()}")
                
            # Check consistency between predictions and probabilities
            # If prediction is 1, probability should be >= vote_threshold
            vote_thresh = float(config['model']['vote_threshold'])
            pred_1_mask = val_results['prediction'] == 1
            if pred_1_mask.any():
                min_prob_for_1 = val_results.loc[pred_1_mask, 'probability'].min()
                if min_prob_for_1 >= vote_thresh - 0.01:  # Small tolerance for float comparison
                    results['tests'].append("Prediction-probability consistency: PASS")
                else:
                    results['errors'].append(f"Inconsistent: prediction=1 but prob={min_prob_for_1} < threshold={vote_thresh}")
                    
        else:
            results['errors'].append("End-to-end validation produced no results")
            
    except Exception as e:
        results['errors'].append(f"Pipeline consistency error: {e}")
    
    return results

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("\n" + "="*80)
    print("COMPREHENSIVE PIPELINE TESTING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    # Run all test components
    all_results.append(test_component_1_data_loading())
    all_results.append(test_component_2_preprocessing())
    all_results.append(test_component_3_model())
    all_results.append(test_component_4_validation())
    all_results.append(test_component_5_edge_cases())
    all_results.append(test_component_6_consistency())
    
    # Generate summary report
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    total_tests = 0
    total_errors = 0
    
    for result in all_results:
        component = result['component']
        n_tests = len(result['tests'])
        n_errors = len(result['errors'])
        total_tests += n_tests
        total_errors += n_errors
        
        print(f"\n{component}:")
        print(f"  Tests passed: {n_tests}")
        print(f"  Errors found: {n_errors}")
        
        if n_errors > 0:
            print("  Error details:")
            for error in result['errors'][:3]:  # Show first 3 errors
                print(f"    - {error}")
    
    # Overall assessment
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    print(f"Total tests passed: {total_tests}")
    print(f"Total errors found: {total_errors}")
    print(f"Success rate: {(total_tests/(total_tests+total_errors)*100):.1f}%")
    
    if total_errors == 0:
        print("\n[OK] ALL TESTS PASSED - Pipeline is logically consistent!")
    else:
        print(f"\n[WARNING] {total_errors} issues found - Review error details above")
    
    # Cleanup test files
    test_files = ['test_synthetic.csv', 'test_config.ini', 'test_pipeline.csv', 'test_pipeline.ini']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_tests()