"""
Test that PermuteAlpha output matches manual walk-forward validation
"""

import configparser
import tempfile
import os
import pandas as pd
import numpy as np
from datetime import datetime
from OMtree_validation import DirectionalValidator

print("="*60)
print("PERMUTE VS MANUAL VALIDATION TEST")
print("="*60)

# Test parameters
ticker = "ES"
target = "Ret_fwd3hr"
hour = "10"
direction = "longonly"
features = ['Ret_0-1hr', 'Ret_1-2hr', 'Ret_2-4hr', 'Ret_4-8hr']  # Subset of features

print(f"\nTest Configuration:")
print(f"  Ticker: {ticker}")
print(f"  Target: {target}")
print(f"  Hour: {hour}")
print(f"  Direction: {direction}")
print(f"  Features: {features}")

# Create config for manual validation
config = configparser.ConfigParser()
config.read('OMtree_config.ini')

# Create temp config
temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
temp_path = temp_config.name

# Update config
config['data']['ticker_filter'] = ticker
config['data']['target_column'] = target
config['data']['hour_filter'] = hour
config['data']['selected_features'] = ','.join(features)
config['model']['model_type'] = direction

# Write config
config.write(temp_config)
temp_config.close()

print(f"\nCreated temp config: {temp_path}")

try:
    # 1. Run manual walk-forward validation
    print("\n1. Running manual walk-forward validation...")
    validator = DirectionalValidator(temp_path)
    
    # Run validation
    results, sharpe, edge, pred_count = validator.walk_forward_validation(
        verbose=False,
        return_predictions=True
    )
    
    if results is not None and len(results) > 0:
        print(f"   Manual validation completed: {len(results)} predictions")
        print(f"   Sharpe: {sharpe:.3f}, Edge: {edge:.4f}")
        
        # Aggregate manual results to daily
        results['Date'] = pd.to_datetime(results['Date'], errors='coerce')
        manual_daily = results.groupby('Date').agg({
            'Actual_Return': 'sum',
            'Prediction': lambda x: 1 if x.any() else 0,
            'Target_Value': 'mean'
        }).reset_index()
        
        manual_daily.columns = ['Date', 'Return', 'TradeFlag', 'FeatureReturn']
        
        print(f"   Aggregated to {len(manual_daily)} days")
        
        # Save manual results
        manual_file = f"test_manual_{ticker}_{direction}_{target}_H{hour}.csv"
        manual_daily.to_csv(manual_file, index=False)
        print(f"   Saved manual results to: {manual_file}")
    else:
        print("   ERROR: Manual validation returned no results")
        manual_daily = None
    
    # 2. Now test the PermuteAlpha approach using the same logic as GUI
    if manual_daily is not None:
        print("\n2. Running PermuteAlpha-style validation...")
        
        # Create new validator for permute style
        validator2 = DirectionalValidator(temp_path)
        
        # Load and prepare data
        data = validator2.load_and_prepare_data()
        
        if data is None or len(data) == 0:
            raise Exception("No data loaded")
        
        # Get the processed columns
        feature_cols = validator2.processed_feature_columns
        target_col = validator2.processed_target_column
        
        # Initialize results lists
        all_dates = []
        all_returns = []
        all_trades = []
        all_feature_returns = []
        
        # Start walk-forward
        test_end_idx = validator2.train_size + validator2.test_size
        steps = 0
        
        while test_end_idx <= len(data):
            try:
                # Calculate indices
                test_start_idx = test_end_idx - validator2.test_size
                train_end_idx = test_start_idx
                train_start_idx = max(0, train_end_idx - validator2.train_size)
                
                # Skip if not enough training data
                if train_start_idx < 0 or (train_end_idx - train_start_idx) < validator2.min_training_samples:
                    test_end_idx += validator2.step_size
                    continue
                
                # Get data windows
                train_data = data.iloc[train_start_idx:train_end_idx]
                test_data = data.iloc[test_start_idx:test_end_idx]
                
                # Extract features and targets
                train_X = train_data[feature_cols].values
                train_y = train_data[target_col].values
                test_X = test_data[feature_cols].values
                test_y = test_data[target_col].values
                
                # Get raw target values
                raw_target_col = validator2.config['data']['target_column']
                if raw_target_col in test_data.columns:
                    test_y_raw = test_data[raw_target_col].values
                else:
                    test_y_raw = test_y
                
                # Skip if no valid data
                if len(train_X) == 0 or len(test_X) == 0:
                    test_end_idx += validator2.step_size
                    continue
                
                # Remove NaN values from training
                valid_train = ~(np.isnan(train_X).any(axis=1) | np.isnan(train_y))
                if valid_train.sum() < validator2.min_training_samples:
                    test_end_idx += validator2.step_size
                    continue
                
                train_X = train_X[valid_train]
                train_y = train_y[valid_train]
                
                # Create and train model
                from OMtree_model import DirectionalTreeEnsemble
                model = DirectionalTreeEnsemble(config_path=temp_path, verbose=False)
                model.fit(train_X, train_y)
                
                # Get predictions
                predictions = model.predict(test_X)
                
                # Get test dates
                test_dates = test_data['Date'].values
                
                # Calculate returns
                if validator2.regression_mode:
                    trade_signals = (predictions > validator2.target_threshold).astype(int)
                else:
                    trade_signals = predictions.astype(int)
                
                # Calculate actual returns
                actual_returns = trade_signals * test_y_raw
                
                # Store results
                all_dates.extend(test_dates)
                all_returns.extend(actual_returns)
                all_trades.extend(trade_signals)
                all_feature_returns.extend(test_y_raw)
                
                steps += 1
                
            except Exception as e:
                print(f"   Warning: Error in window: {str(e)}")
                pass
            
            test_end_idx += validator2.step_size
        
        if not all_dates:
            raise Exception("No valid windows found")
        
        print(f"   Permute validation completed: {steps} windows, {len(all_dates)} predictions")
        
        # Create results DataFrame
        permute_results = pd.DataFrame({
            'Date': all_dates,
            'Return': all_returns,
            'TradeFlag': all_trades,
            'FeatureReturn': all_feature_returns
        })
        
        # Aggregate to daily
        permute_results['Date'] = pd.to_datetime(permute_results['Date'], errors='coerce')
        permute_daily = permute_results.groupby('Date').agg({
            'Return': 'sum',
            'TradeFlag': lambda x: 1 if x.any() else 0,
            'FeatureReturn': 'mean'
        }).reset_index()
        
        print(f"   Aggregated to {len(permute_daily)} days")
        
        # Save permute results
        permute_file = f"test_permute_{ticker}_{direction}_{target}_H{hour}.csv"
        permute_daily.to_csv(permute_file, index=False)
        print(f"   Saved permute results to: {permute_file}")
        
        # 3. Compare results
        print("\n3. Comparing results...")
        
        # Merge on date
        comparison = pd.merge(
            manual_daily, permute_daily,
            on='Date', suffixes=('_manual', '_permute')
        )
        
        if len(comparison) > 0:
            print(f"   Matched {len(comparison)} days")
            
            # Compare key metrics
            manual_return = comparison['Return_manual'].sum()
            permute_return = comparison['Return_permute'].sum()
            
            manual_trades = comparison['TradeFlag_manual'].sum()
            permute_trades = comparison['TradeFlag_permute'].sum()
            
            print(f"\n   Total Return:")
            print(f"     Manual:  {manual_return:.4f}")
            print(f"     Permute: {permute_return:.4f}")
            print(f"     Diff:    {abs(manual_return - permute_return):.4f}")
            
            print(f"\n   Total Trading Days:")
            print(f"     Manual:  {manual_trades}")
            print(f"     Permute: {permute_trades}")
            print(f"     Diff:    {abs(manual_trades - permute_trades)}")
            
            # Check correlation
            return_corr = comparison['Return_manual'].corr(comparison['Return_permute'])
            trade_corr = comparison['TradeFlag_manual'].corr(comparison['TradeFlag_permute'])
            
            print(f"\n   Correlation:")
            print(f"     Returns:    {return_corr:.4f}")
            print(f"     TradeFlags: {trade_corr:.4f}")
            
            # Success criteria
            if return_corr > 0.95 and trade_corr > 0.95:
                print("\n[SUCCESS] Results are highly correlated!")
            else:
                print("\n[WARNING] Results differ more than expected")
                
                # Show first few differences
                print("\n   First 10 differences in returns:")
                diff_df = comparison[['Date', 'Return_manual', 'Return_permute']].head(10)
                diff_df['Diff'] = diff_df['Return_manual'] - diff_df['Return_permute']
                print(diff_df.to_string())
        else:
            print("   ERROR: No matching dates found")
    
except Exception as e:
    print(f"\n[ERROR] Test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # Clean up temp file
    if os.path.exists(temp_path):
        os.unlink(temp_path)
        print(f"\nCleaned up temp config")

print("="*60)