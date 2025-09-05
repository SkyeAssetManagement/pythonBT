import pandas as pd
import numpy as np
from OMtree_preprocessing import DataPreprocessor
import configparser

# Load config
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

df = pd.read_csv(config['data']['csv_file'])

print("="*80)
print("DEBUGGING NORMALIZATION IMPLEMENTATION")
print("="*80)

# Test with normalization OFF
config['preprocessing']['normalize_target'] = 'false'
with open('OMtree_config.ini', 'w') as f:
    config.write(f)
    
preprocessor_no_norm = DataPreprocessor('OMtree_config.ini')
data_no_norm = preprocessor_no_norm.process_data(df)

# Test with normalization ON
config['preprocessing']['normalize_target'] = 'true'
with open('OMtree_config.ini', 'w') as f:
    config.write(f)
    
preprocessor_with_norm = DataPreprocessor('OMtree_config.ini')
data_with_norm = preprocessor_with_norm.process_data(df)

target_col = config['data']['target_column']

# Check what columns were created
print(f"\nColumns in no_norm data containing target name:")
target_cols_no_norm = [col for col in data_no_norm.columns if target_col in col]
print(target_cols_no_norm)

print(f"\nColumns in with_norm data containing target name:")
target_cols_with_norm = [col for col in data_with_norm.columns if target_col in col]
print(target_cols_with_norm)

# Check values from both
print("\nChecking values at index 300 (should be past vol_window):")
idx = 300
print(f"\nWithout normalization:")
for col in target_cols_no_norm:
    val = data_no_norm[col].iloc[idx]
    print(f"  {col}: {val:.6f}")

print(f"\nWith normalization:")
for col in target_cols_with_norm:
    val = data_with_norm[col].iloc[idx]
    print(f"  {col}: {val:.6f}")

# Check if IQR is being calculated
if f'{target_col}_iqr' in data_with_norm.columns:
    iqr_values = data_with_norm[f'{target_col}_iqr'].iloc[250:350]
    print(f"\nIQR values (indices 250-350):")
    print(f"  Min: {iqr_values.min():.4f}")
    print(f"  Max: {iqr_values.max():.4f}")
    print(f"  Mean: {iqr_values.mean():.4f}")
    print(f"  First 5 values: {iqr_values.iloc[:5].values}")

# Check raw vs adjusted
print("\n" + "="*50)
print("CHECKING RAW VS ADJUSTED VALUES")
print("="*50)

start_idx = 250
for i in range(start_idx, start_idx + 10):
    raw_val = data_with_norm[target_col].iloc[i]
    adj_val = data_with_norm[f'{target_col}_vol_adj'].iloc[i]
    if f'{target_col}_iqr' in data_with_norm.columns:
        iqr_val = data_with_norm[f'{target_col}_iqr'].iloc[i]
        calc_norm = raw_val / iqr_val if iqr_val != 0 and not np.isnan(iqr_val) else np.nan
        print(f"Index {i}: Raw={raw_val:.4f}, IQR={iqr_val:.4f}, Adj={adj_val:.4f}, Calc={calc_norm:.4f}")
    else:
        print(f"Index {i}: Raw={raw_val:.4f}, Adj={adj_val:.4f}")