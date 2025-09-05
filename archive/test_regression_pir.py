"""
Test regression tab column detection for PIR file
"""

import pandas as pd
import configparser

# Load PIR file
df = pd.read_csv('DTSmlDATA_PIR.csv')

print("=== TESTING REGRESSION TAB LOGIC FOR PIR FILE ===\n")

# Simulate regression tab's column detection logic
feature_columns = []
target_columns = []
feature_columns_vol = []
target_columns_vol = []

# First try to load from config
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

config_loaded = False

if 'data' in config:
    features = []
    targets = []
    
    # Get all target columns
    if 'all_targets' in config['data']:
        targets = [t.strip() for t in config['data']['all_targets'].split(',')]
    elif 'target_column' in config['data']:
        targets = [config['data']['target_column']]
    
    # Get feature columns - prioritize feature_columns over selected_features
    if 'feature_columns' in config['data']:
        features = [f.strip() for f in config['data']['feature_columns'].split(',')]
    elif 'selected_features' in config['data']:
        features = [f.strip() for f in config['data']['selected_features'].split(',')]
    
    print(f"Config features: {features}")
    print(f"Config targets: {targets}")
    
    if features or targets:
        config_loaded = True
        # Store for later use
        feature_columns = [col for col in features if col in df.columns]
        target_columns = [col for col in targets if col in df.columns]
        
        print(f"\nFeatures found in PIR data: {feature_columns}")
        print(f"Targets found in PIR data: {target_columns}")
        
        # Also check for volatility-adjusted versions
        for col in feature_columns:
            vol_col = f'{col}_vol_adj'
            if vol_col in df.columns:
                feature_columns_vol.append(vol_col)
        for col in target_columns:
            vol_col = f'{col}_vol_adj'
            if vol_col in df.columns:
                target_columns_vol.append(vol_col)

if not config_loaded or not feature_columns:
    print("\nConfig features not found in PIR data, using auto-detection...")
    # Auto-detect features and targets
    feature_columns = []
    target_columns = []
    
    for col in df.columns:
        if col.startswith('Ret_fwd'):
            target_columns.append(col)
        elif col.startswith('Ret_') and not col.startswith('Ret_fwd'):
            feature_columns.append(col)
        elif col in ['DiffClose@Obs', 'NoneClose@Obs']:
            feature_columns.append(col)
    
    print(f"Auto-detected features: {feature_columns}")
    print(f"Auto-detected targets: {target_columns}")

print(f"\n=== FINAL RESULT ===")
print(f"Features for regression tab: {len(feature_columns)} columns")
print(f"  {feature_columns}")
print(f"Targets for regression tab: {len(target_columns)} columns")
print(f"  {target_columns}")