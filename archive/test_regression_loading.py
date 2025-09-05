"""
Test regression module data loading
"""

import configparser
import pandas as pd

# Load config
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

print("=== CONFIG DATA SECTION ===")
for key, value in config['data'].items():
    print(f"{key}: {value}")

print("\n=== PARSING FEATURES ===")

# Parse features exactly as regression module does
features = []
if 'feature_columns' in config['data']:
    features = [f.strip() for f in config['data']['feature_columns'].split(',')]
    print(f"Found feature_columns: {features}")
elif 'selected_features' in config['data']:
    features = [f.strip() for f in config['data']['selected_features'].split(',')]
    print(f"Found selected_features: {features}")

print(f"\nTotal features parsed: {len(features)}")
print(f"Features list: {features}")

# Parse targets
targets = []
if 'all_targets' in config['data']:
    targets = [t.strip() for t in config['data']['all_targets'].split(',')]
    print(f"\nFound all_targets: {targets}")
elif 'target_column' in config['data']:
    targets = [config['data']['target_column']]
    print(f"\nFound target_column: {targets}")

print(f"\nTotal targets parsed: {len(targets)}")

# Load actual data
print("\n=== CHECKING DATA FILE ===")
df = pd.read_csv(config['data']['csv_file'])
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)[:20]}...")  # Show first 20 columns

# Check which features/targets exist
features_exist = [f for f in features if f in df.columns]
targets_exist = [t for t in targets if t in df.columns]

print(f"\nFeatures in data: {len(features_exist)}/{len(features)}")
print(f"Missing features: {[f for f in features if f not in df.columns]}")

print(f"\nTargets in data: {len(targets_exist)}/{len(targets)}")
print(f"Missing targets: {[t for t in targets if t not in df.columns]}")