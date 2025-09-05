"""
Debug column issue with PIR file
"""

import pandas as pd
import configparser
from column_detector import ColumnDetector

# Load PIR data 
df = pd.read_csv('DTSmlDATA_PIR.csv')
print("PIR file columns:")
print(df.columns.tolist())

# Check if Ret_16hr exists
print(f"\n'Ret_16hr' in columns: {'Ret_16hr' in df.columns}")

# Load config
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

print(f"\nConfig selected_features: {config['data']['selected_features']}")

# Test preprocessing to see what columns it creates
from OMtree_preprocessing import DataPreprocessor

config['data']['csv_file'] = 'DTSmlDATA_PIR.csv'
test_config = 'test_pir_debug.ini'
with open(test_config, 'w') as f:
    config.write(f)

preprocessor = DataPreprocessor(test_config)
processed = preprocessor.process_data(df)

print(f"\nProcessed columns:")
print(processed.columns.tolist())

# Check if Ret_16hr exists after preprocessing
print(f"\n'Ret_16hr' in processed: {'Ret_16hr' in processed.columns}")
print(f"'Ret_16hr_vol_adj' in processed: {'Ret_16hr_vol_adj' in processed.columns}")

# Now check validation
from OMtree_validation import DirectionalValidator

validator = DirectionalValidator(test_config)
print(f"\nValidator selected_features: {validator.selected_features}")

# Load and prepare data
data = validator.load_and_prepare_data()
print(f"\nPrepared data columns:")
print([c for c in data.columns if 'Ret_16hr' in c or 'Ret_8-16hr' in c])

# Test the column selection logic
print("\nTesting column selection logic:")
feature = 'Ret_16hr'
vol_adj_col = f'{feature}_vol_adj'

print(f"Looking for: {feature}")
print(f"Vol adj version: {vol_adj_col}")
print(f"'{vol_adj_col}' in data.columns: {vol_adj_col in data.columns}")
print(f"'{feature}' in data.columns: {feature in data.columns}")

# Find similar columns
similar = ColumnDetector.find_similar_columns(data, feature, threshold=0.5)
print(f"\nSimilar columns found: {similar[:3] if similar else 'None'}")

# Clean up
import os
if os.path.exists(test_config):
    os.remove(test_config)