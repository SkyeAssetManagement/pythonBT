"""
Fix feature columns in config to match actual data
"""

import pandas as pd
import configparser

# Load data to see actual columns
df = pd.read_csv('DTSmlDATA7x7.csv')

# Find all feature columns (Ret_ but not Ret_fwd)
feature_cols = [col for col in df.columns if col.startswith('Ret_') and not col.startswith('Ret_fwd')]
target_cols = [col for col in df.columns if col.startswith('Ret_fwd')]

print("=== ACTUAL DATA COLUMNS ===")
print(f"Features in data: {feature_cols}")
print(f"Targets in data: {target_cols}")

# Load config
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

print("\n=== CURRENT CONFIG ===")
print(f"feature_columns: {config['data']['feature_columns']}")
print(f"all_targets: {config['data']['all_targets']}")

print("\n=== UPDATING CONFIG ===")
# Update config with correct feature columns
config['data']['feature_columns'] = ','.join(feature_cols)

# Save updated config
with open('OMtree_config.ini', 'w') as f:
    config.write(f)

print(f"Updated feature_columns to: {','.join(feature_cols)}")
print("Config file updated successfully!")