"""
Test walkforward with PIR file to reproduce the error
"""

import pandas as pd
import configparser
from OMtree_validation import DirectionalValidator

# Update config to use PIR file
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('OMtree_config.ini')

print(f"Current config file: {config['data']['csv_file']}")
print(f"Current selected_features: {config['data']['selected_features']}")

# Check what file is actually being loaded
csv_file = config['data']['csv_file']
df = pd.read_csv(csv_file)
print(f"\nActually loaded: {csv_file}")
print(f"Columns in loaded file: {[c for c in df.columns if 'Ret_16hr' in c or 'Ret_8-16hr' in c]}")

# The issue: config says DTSmlDATA7x7.csv but user wants to run on PIR
# When user runs walkforward on PIR, they need to update the csv_file in config!

print("\n" + "="*60)
print("ISSUE IDENTIFIED:")
print("="*60)
print("The config file points to DTSmlDATA7x7.csv")
print("But you want to run walkforward on DTSmlDATA_PIR.csv")
print("")
print("DTSmlDATA7x7.csv has: Ret_8-16hr")
print("DTSmlDATA_PIR.csv has: Ret_16hr")
print("")
print("When Ret_16hr is not found in DTSmlDATA7x7.csv,")
print("the system finds the similar column Ret_8-16hr")
print("")
print("SOLUTION: Update config file to set:")
print("  csv_file = DTSmlDATA_PIR.csv")
print("before running walkforward on PIR data")
print("="*60)