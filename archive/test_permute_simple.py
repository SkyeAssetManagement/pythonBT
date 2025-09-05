"""
Simple test of PermuteAlpha using walk-forward validation
"""

import configparser
import tempfile
import os
import pandas as pd
from OMtree_validation import DirectionalValidator

print("="*60)
print("SIMPLE PERMUTE TEST")
print("="*60)

# Test parameters
ticker = "ES"
target = "Ret_fwd3hr"
hour = "10"
direction = "longonly"
features = ['Ret_0-1hr', 'Ret_1-2hr', 'Ret_2-4hr', 'Ret_4-8hr']

print(f"\nTest Configuration:")
print(f"  Ticker: {ticker}")
print(f"  Target: {target}")
print(f"  Hour: {hour}")
print(f"  Direction: {direction}")
print(f"  Features: {features}")

# Create config
config = configparser.ConfigParser()
config.read('OMtree_config.ini')

# Create temp config
temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
temp_path = temp_config.name

# Update config for this permutation
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
    # Run validation (same as PermuteAlpha would)
    print("\nRunning walk-forward validation...")
    validator = DirectionalValidator(temp_path)
    
    # Run validation
    results_df = validator.run_validation(verbose=False)
    
    if results_df is None or len(results_df) == 0:
        raise Exception("Walk-forward validation returned no results")
    
    print(f"Validation completed with {len(results_df)} results")
    
    # Load the detailed predictions
    results_file = f'walkforward_results_{validator.model_type}.csv'
    if os.path.exists(results_file):
        detailed_results = pd.read_csv(results_file)
        
        print(f"\nLoaded detailed results from {results_file}")
        print(f"  Rows: {len(detailed_results)}")
        print(f"  Columns: {list(detailed_results.columns)}")
        
        # Convert to PermuteAlpha format
        permute_df = pd.DataFrame({
            'Date': detailed_results['date'],
            'Return': detailed_results['pnl'],
            'TradeFlag': detailed_results['signal'].astype(int),
            'FeatureReturn': detailed_results['actual']
        })
        
        # Aggregate to daily
        permute_df['Date'] = pd.to_datetime(permute_df['Date'], errors='coerce')
        permute_df = permute_df.dropna(subset=['Date'])
        
        daily_df = permute_df.groupby('Date').agg({
            'Return': 'sum',
            'TradeFlag': lambda x: 1 if x.any() else 0,
            'FeatureReturn': 'mean'
        }).reset_index()
        
        print(f"\nAggregated to {len(daily_df)} daily results")
        
        # Show sample
        print("\nFirst 5 rows of output:")
        print(daily_df.head())
        
        # Calculate stats
        total_return = daily_df['Return'].sum()
        trading_days = daily_df['TradeFlag'].sum()
        avg_return = daily_df[daily_df['TradeFlag'] == 1]['Return'].mean() if trading_days > 0 else 0
        
        print(f"\nStatistics:")
        print(f"  Total Return: {total_return:.4f}")
        print(f"  Trading Days: {trading_days}")
        print(f"  Avg Return per Trade Day: {avg_return:.4f}")
        
        # Verify relationship
        print(f"\nVerifying TradeFlag × FeatureReturn = Return:")
        sample = daily_df.head(10)
        errors = 0
        for _, row in sample.iterrows():
            expected = row['TradeFlag'] * row['FeatureReturn']
            actual = row['Return']
            if abs(expected - actual) > 0.0001:
                errors += 1
                print(f"  Error: {row['Date']}: {row['TradeFlag']} × {row['FeatureReturn']:.4f} = {expected:.4f} vs {actual:.4f}")
        
        if errors == 0:
            print(f"  All {len(sample)} sample rows match correctly!")
        
        # Save output
        output_file = f"test_{ticker}_{direction}_{target}_H{hour}.csv"
        daily_df.to_csv(output_file, index=False)
        print(f"\nSaved output to: {output_file}")
        
        print("\n[SUCCESS] Test completed successfully!")
        
    else:
        print(f"\n[ERROR] Detailed results file not found: {results_file}")
        
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