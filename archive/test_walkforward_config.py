"""
Test script to verify walk-forward uses selected features from config
"""
import configparser
import pandas as pd

# Read config
config = configparser.ConfigParser()
config.read('OMtree_config.ini')

print("Configuration Test Report")
print("=" * 60)

# Check selected features
if 'selected_features' in config['data']:
    selected_features = [f.strip() for f in config['data']['selected_features'].split(',')]
    print(f"\nSelected Features in Config: {len(selected_features)}")
    for feature in selected_features:
        print(f"  - {feature}")
else:
    print("\nWARNING: No selected_features found in config!")

# Check target
target = config['data']['target_column']
print(f"\nTarget Column: {target}")

# Check validation dates
val_start = config['validation'].get('validation_start_date', 'Not set')
val_end = config['validation'].get('validation_end_date', 'Not set')
print(f"\nValidation Period:")
print(f"  Start: {val_start}")
print(f"  End: {val_end}")

# Check model parameters
print(f"\nModel Configuration:")
print(f"  Type: {config['model']['model_type']}")
print(f"  Trees: {config['model']['n_trees']}")
print(f"  Threshold: {config['model']['target_threshold']}")

# Load data to verify columns exist
try:
    df = pd.read_csv(config['data']['csv_file'])
    print(f"\nData File: {config['data']['csv_file']}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    
    # Verify selected features exist
    missing_features = []
    for feature in selected_features:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        print(f"\nWARNING: Missing features in data:")
        for f in missing_features:
            print(f"  - {f}")
    else:
        print(f"\n[OK] All selected features found in data")
    
    # Verify target exists
    if target in df.columns:
        print(f"[OK] Target column '{target}' found in data")
    else:
        print(f"[ERROR] Target column '{target}' NOT found in data!")
        
except Exception as e:
    print(f"\nError loading data: {e}")

print("\n" + "=" * 60)
print("Configuration test complete")