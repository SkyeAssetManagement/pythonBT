import pandas as pd
import numpy as np
from OMtree_preprocessing import DataPreprocessor
import configparser

# Load config
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

# Test with normalization OFF
config['preprocessing']['normalize_target'] = 'false'
preprocessor_no_norm = DataPreprocessor('OMtree_config.ini')
df = pd.read_csv(config['data']['csv_file'])

print("="*80)
print("TESTING NORMALIZATION EFFECT ON TARGET")
print("="*80)

# Process without normalization
data_no_norm = preprocessor_no_norm.process_data(df)
target_col = config['data']['target_column']
target_col_adj = f"{target_col}_vol_adj"

# Get the actual column used for training
if target_col_adj in data_no_norm.columns:
    target_no_norm = data_no_norm[target_col_adj].values
else:
    target_no_norm = data_no_norm[target_col].values

# Process with normalization
config['preprocessing']['normalize_target'] = 'true'
preprocessor_with_norm = DataPreprocessor('OMtree_config.ini')
data_with_norm = preprocessor_with_norm.process_data(df)

# Get the actual column used for training
if target_col_adj in data_with_norm.columns:
    target_with_norm = data_with_norm[target_col_adj].values
else:
    target_with_norm = data_with_norm[target_col].values

print(f"No norm using column: {target_col_adj if target_col_adj in data_no_norm.columns else target_col}")
print(f"With norm using column: {target_col_adj if target_col_adj in data_with_norm.columns else target_col}")

# Check starting from index 250 (our offset)
start_idx = 250
end_idx = start_idx + 500  # Check 500 samples

print(f"\nChecking indices {start_idx} to {end_idx}:")

# Remove NaN values for comparison
valid_mask = ~(np.isnan(target_no_norm[start_idx:end_idx]) | np.isnan(target_with_norm[start_idx:end_idx]))
target_no_norm_valid = target_no_norm[start_idx:end_idx][valid_mask]
target_with_norm_valid = target_with_norm[start_idx:end_idx][valid_mask]

print(f"Valid samples: {len(target_no_norm_valid)}")

# Check if signs match
signs_no_norm = np.sign(target_no_norm_valid)
signs_with_norm = np.sign(target_with_norm_valid)
sign_match = (signs_no_norm == signs_with_norm).all()

print(f"\nSign match (positive/negative): {sign_match}")
if not sign_match:
    mismatches = np.where(signs_no_norm != signs_with_norm)[0]
    print(f"Number of sign mismatches: {len(mismatches)}")
    print(f"First 5 mismatches:")
    for i in mismatches[:5]:
        print(f"  Index {start_idx + i}: No norm={target_no_norm_valid[i]:.6f} (sign={signs_no_norm[i]}), "
              f"With norm={target_with_norm_valid[i]:.6f} (sign={signs_with_norm[i]})")

# Check classification at threshold=0
threshold = 0.0
class_no_norm = (target_no_norm_valid > threshold).astype(int)
class_with_norm = (target_with_norm_valid > threshold).astype(int)
class_match = (class_no_norm == class_with_norm).all()

print(f"\nClassification match (threshold={threshold}): {class_match}")
if not class_match:
    class_mismatches = np.where(class_no_norm != class_with_norm)[0]
    print(f"Number of classification mismatches: {len(class_mismatches)}")
    print(f"Classification match rate: {(class_no_norm == class_with_norm).mean()*100:.1f}%")

# Check if there are any exact zeros
zeros_no_norm = np.sum(target_no_norm_valid == 0)
zeros_with_norm = np.sum(target_with_norm_valid == 0)
print(f"\nExact zeros in non-normalized: {zeros_no_norm}")
print(f"Exact zeros in normalized: {zeros_with_norm}")

# Check very small values near zero
near_zero_threshold = 1e-10
near_zeros_no_norm = np.sum(np.abs(target_no_norm_valid) < near_zero_threshold)
near_zeros_with_norm = np.sum(np.abs(target_with_norm_valid) < near_zero_threshold)
print(f"\nValues near zero (<{near_zero_threshold}):")
print(f"  Non-normalized: {near_zeros_no_norm}")
print(f"  Normalized: {near_zeros_with_norm}")

# Sample some values to see the transformation
print("\nSample values (first 10):")
for i in range(min(10, len(target_no_norm_valid))):
    ratio = target_with_norm_valid[i] / target_no_norm_valid[i] if target_no_norm_valid[i] != 0 else np.inf
    print(f"  Index {start_idx + i}: No norm={target_no_norm_valid[i]:.6f}, "
          f"With norm={target_with_norm_valid[i]:.6f}, Ratio={ratio:.4f}")