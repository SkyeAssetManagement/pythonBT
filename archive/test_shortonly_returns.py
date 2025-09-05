"""
Test that shortonly returns are correctly inverted
"""

import configparser
import tempfile
import os
import pandas as pd
import numpy as np

print("="*60)
print("SHORT vs LONG RETURNS TEST")
print("="*60)

# Test parameters
ticker = "ES"
target = "Ret_fwd3hr"
hour = "10"
features = ['Ret_0-1hr', 'Ret_1-2hr', 'Ret_2-4hr', 'Ret_4-8hr']

# We'll run both longonly and shortonly and compare
for direction in ['longonly', 'shortonly']:
    print(f"\n{'='*30}")
    print(f"Testing {direction.upper()}")
    print(f"{'='*30}")
    
    # Create config
    config = configparser.ConfigParser()
    config.read('OMtree_config.ini')
    
    # Create temp config
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
    temp_path = temp_config.name
    
    # Update config for this test
    config['data']['ticker_filter'] = ticker
    config['data']['target_column'] = target
    config['data']['hour_filter'] = hour
    config['data']['selected_features'] = ','.join(features)
    config['model']['model_type'] = direction
    
    # Write config
    config.write(temp_config)
    temp_config.close()
    
    try:
        from OMtree_validation import DirectionalValidator
        
        # Create validator
        validator = DirectionalValidator(temp_path)
        
        # Run validation
        results_df = validator.run_validation(verbose=False)
        
        # Load the detailed results
        results_file = f'walkforward_results_{direction}.csv'
        if os.path.exists(results_file):
            detailed_results = pd.read_csv(results_file)
            
            # Show first 10 rows where trades were made
            trades_df = detailed_results[detailed_results['signal'] == 1].head(10)
            
            if len(trades_df) > 0:
                print(f"\nFirst 10 trades for {direction}:")
                print("Actual (raw target) | PnL")
                print("-" * 30)
                for _, row in trades_df.iterrows():
                    print(f"{row['actual']:>10.4f} | {row['pnl']:>10.4f}")
                
                # Calculate totals
                total_actual = trades_df['actual'].sum()
                total_pnl = trades_df['pnl'].sum()
                
                print("-" * 30)
                print(f"Total actual: {total_actual:.4f}")
                print(f"Total PnL:    {total_pnl:.4f}")
                
                # For shortonly, PnL should be opposite sign of actual
                if direction == 'shortonly':
                    if np.sign(total_actual) != np.sign(total_pnl) or total_actual == 0:
                        print("✓ CORRECT: Short PnL is inverted from actual returns")
                    else:
                        print("✗ ERROR: Short PnL should be inverted!")
                else:
                    if np.sign(total_actual) == np.sign(total_pnl) or total_actual == 0:
                        print("✓ CORRECT: Long PnL matches actual returns")
                    else:
                        print("✗ ERROR: Long PnL should match actual returns!")
                
                # Save a sample output
                output_file = f"test_{direction}_{ticker}_{target}_H{hour}.csv"
                
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
                
                daily_df.to_csv(output_file, index=False)
                print(f"\nSaved output to: {output_file}")
            else:
                print(f"\nNo trades found for {direction}")
        else:
            print(f"\nResults file not found: {results_file}")
            
    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# Now compare the two output files
print(f"\n{'='*60}")
print("COMPARING LONG vs SHORT OUTPUTS")
print(f"{'='*60}")

try:
    long_file = f"test_longonly_{ticker}_{target}_H{hour}.csv"
    short_file = f"test_shortonly_{ticker}_{target}_H{hour}.csv"
    
    if os.path.exists(long_file) and os.path.exists(short_file):
        long_df = pd.read_csv(long_file)
        short_df = pd.read_csv(short_file)
        
        # Merge on date to compare
        comparison = pd.merge(long_df, short_df, on='Date', suffixes=('_long', '_short'))
        
        # Show first 10 rows where both had trades
        both_trades = comparison[(comparison['TradeFlag_long'] == 1) & (comparison['TradeFlag_short'] == 1)].head(10)
        
        if len(both_trades) > 0:
            print("\nDays where both long and short traded:")
            print("Date | Return_Long | Return_Short | FeatureReturn")
            print("-" * 60)
            
            for _, row in both_trades.iterrows():
                # For the same day, if long makes money, short should lose (and vice versa)
                expected_short = -row['FeatureReturn'] if row['TradeFlag_short'] == 1 else 0
                match = "✓" if abs(row['Return_short'] - expected_short) < 0.0001 else "✗"
                print(f"{row['Date'][:10]} | {row['Return_long']:>8.4f} | {row['Return_short']:>8.4f} | {row['FeatureReturn_long']:>8.4f} {match}")
        
        # Check overall statistics
        print(f"\nOverall Statistics:")
        print(f"  Long total return:  {long_df['Return'].sum():.4f}")
        print(f"  Short total return: {short_df['Return'].sum():.4f}")
        print(f"  Long trade days:    {long_df['TradeFlag'].sum()}")
        print(f"  Short trade days:   {short_df['TradeFlag'].sum()}")
        
        if abs(long_df['Return'].sum() + short_df['Return'].sum()) < 0.1:
            print("\n✗ ERROR: Long and short returns are too similar - they should be different!")
        else:
            print("\n✓ SUCCESS: Long and short returns are different as expected")
            
    else:
        print("Output files not found for comparison")
        
except Exception as e:
    print(f"Comparison failed: {str(e)}")

print("="*60)