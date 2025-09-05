"""
Test complete PermuteAlpha workflow
"""

import configparser
import tempfile
import os
import pandas as pd
import numpy as np
from datetime import datetime
from OMtree_validation import DirectionalValidator

print("="*60)
print("PERMUTE ALPHA WORKFLOW TEST")
print("="*60)

# Test parameters
ticker = "ES"
target = "Ret_fwd3hr"
hour = "10"
direction = "longonly"

# Create a test config
config = configparser.ConfigParser()
config.read('OMtree_config.ini')

# Create temp config for testing
temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
temp_path = temp_config.name

# Update config for this permutation
config['data']['ticker_filter'] = ticker
config['data']['target_column'] = target
config['data']['hour_filter'] = hour
config['model']['model_type'] = direction

# Write config
config.write(temp_config)
temp_config.close()

print(f"\nTest config created: {temp_path}")
print(f"Testing permutation: {ticker}_{direction}_{target}_H{hour}")

try:
    # Create validator
    print("\n1. Creating validator...")
    validator = DirectionalValidator(temp_path)
    
    # Load and prepare data
    print("2. Loading and preparing data...")
    data = validator.load_and_prepare_data()
    
    if data is None or len(data) == 0:
        raise Exception("No data loaded")
    
    print(f"   Data shape: {data.shape}")
    
    # Get the processed feature columns and target
    feature_cols = validator.processed_feature_columns
    target_col = validator.processed_target_column
    
    if not feature_cols:
        raise Exception("No feature columns found")
    
    print(f"   Features: {len(feature_cols)} columns")
    print(f"   Target: {target_col}")
    
    # Get date column
    date_col = 'Date'
    if date_col not in data.columns:
        # Try to find date column
        for col in data.columns:
            if 'date' in col.lower():
                date_col = col
                break
    
    # Initialize results lists
    all_dates = []
    all_returns = []
    all_trades = []
    all_feature_returns = []
    
    # Run walk-forward for first 3 windows only (for quick test)
    test_end_idx = validator.train_size + validator.test_size
    windows_processed = 0
    max_windows = 3
    
    print(f"\n3. Running walk-forward (first {max_windows} windows)...")
    
    while test_end_idx <= len(data) and windows_processed < max_windows:
        try:
            # Calculate indices
            test_start_idx = test_end_idx - validator.test_size
            train_end_idx = test_start_idx
            train_start_idx = max(0, train_end_idx - validator.train_size)
            
            # Skip if not enough training data
            if train_start_idx < 0 or (train_end_idx - train_start_idx) < validator.min_training_samples:
                test_end_idx += validator.step_size
                continue
            
            # Get training data
            train_data = data.iloc[train_start_idx:train_end_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]
            
            # Extract features and targets
            train_X = train_data[feature_cols].values
            train_y = train_data[target_col].values
            test_X = test_data[feature_cols].values
            test_y = test_data[target_col].values
            
            # Get raw target values
            raw_target_col = validator.config['data']['target_column']
            if raw_target_col in test_data.columns:
                test_y_raw = test_data[raw_target_col].values
            else:
                test_y_raw = test_y
            
            # Skip if no valid data
            if len(train_X) == 0 or len(test_X) == 0:
                test_end_idx += validator.step_size
                continue
            
            # Remove NaN values from training
            valid_train = ~(np.isnan(train_X).any(axis=1) | np.isnan(train_y))
            if valid_train.sum() < validator.min_training_samples:
                test_end_idx += validator.step_size
                continue
            
            train_X = train_X[valid_train]
            train_y = train_y[valid_train]
            
            # Create and train model
            from OMtree_model import DirectionalTreeEnsemble
            
            # Use the config_path
            model = DirectionalTreeEnsemble(
                config_path=temp_path,
                verbose=False
            )
            
            # Fit model
            model.fit(train_X, train_y)
            
            # Get predictions
            predictions = model.predict(test_X)
            
            # Get test dates
            test_dates = test_data[date_col].values
            
            # Calculate returns
            if validator.regression_mode:
                # In regression mode, threshold predictions
                trade_signals = (predictions > validator.target_threshold).astype(int)
            else:
                # In classification mode, predictions are already binary
                trade_signals = predictions.astype(int)
            
            # Calculate actual returns
            actual_returns = trade_signals * test_y_raw
            
            # Store results
            all_dates.extend(test_dates)
            all_returns.extend(actual_returns)
            all_trades.extend(trade_signals)
            all_feature_returns.extend(test_y_raw)
            
            windows_processed += 1
            print(f"   Window {windows_processed}: {len(test_dates)} predictions, {trade_signals.sum()} trades")
            
        except Exception as e:
            print(f"   Warning: Error in window: {str(e)}")
            pass
        
        test_end_idx += validator.step_size
    
    if not all_dates:
        raise Exception("No valid windows found")
    
    print(f"\n4. Creating results DataFrame...")
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': all_dates,
        'Return': all_returns,
        'TradeFlag': all_trades,
        'FeatureReturn': all_feature_returns
    })
    
    print(f"   Total rows: {len(results_df)}")
    print(f"   Total trades: {results_df['TradeFlag'].sum()}")
    print(f"   Average return when trading: {results_df[results_df['TradeFlag'] == 1]['Return'].mean():.4f}")
    
    # Convert dates and aggregate to daily
    results_df['Date'] = pd.to_datetime(results_df['Date'], errors='coerce')
    
    # Remove any NaT dates
    results_df = results_df.dropna(subset=['Date'])
    
    if len(results_df) == 0:
        raise Exception("No valid dates in results")
    
    # Aggregate to daily level
    daily_df = results_df.groupby('Date').agg({
        'Return': 'sum',
        'TradeFlag': lambda x: 1 if x.any() else 0,
        'FeatureReturn': 'mean'
    }).reset_index()
    
    print(f"\n5. Daily aggregated results:")
    print(f"   Days: {len(daily_df)}")
    print(f"   Trading days: {daily_df['TradeFlag'].sum()}")
    print(f"   Total return: {daily_df['Return'].sum():.4f}")
    
    # Save sample output
    output_path = f"test_permute_{ticker}_{direction}_{target}_H{hour}.csv"
    daily_df.to_csv(output_path, index=False)
    print(f"\n6. Sample output saved to: {output_path}")
    
    # Show first few rows
    print("\nFirst 5 rows of output:")
    print(daily_df.head())
    
    # Verify the TradeFlag × FeatureReturn = Return relationship
    print("\n7. Verifying TradeFlag × FeatureReturn = Return:")
    sample_rows = daily_df.head(10)
    for idx, row in sample_rows.iterrows():
        expected = row['TradeFlag'] * row['FeatureReturn']
        actual = row['Return']
        match = "✓" if abs(expected - actual) < 0.0001 else "✗"
        print(f"   Row {idx}: {row['TradeFlag']:.0f} × {row['FeatureReturn']:.4f} = {expected:.4f} vs {actual:.4f} {match}")
    
    print("\n[SUCCESS] PermuteAlpha workflow test completed!")
    
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