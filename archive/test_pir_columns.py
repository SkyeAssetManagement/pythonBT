"""
Test column detection for PIR file
"""

import pandas as pd
from column_detector import ColumnDetector

# Load PIR file
df = pd.read_csv('DTSmlDATA_PIR.csv')

print("=== PIR FILE ANALYSIS ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Test auto-detection
detected = ColumnDetector.auto_detect_columns(df)
print(f"\nAuto-detected features: {detected['features']}")
print(f"Auto-detected targets: {detected['targets']}")

# Check numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print(f"\nNumeric columns: {numeric_cols}")

# Categorize columns manually for verification
backward_returns = []
forward_returns = []
other = []

for col in df.columns:
    if 'fwd' in col.lower():
        forward_returns.append(col)
    elif col.startswith('Ret_') and 'fwd' not in col.lower():
        backward_returns.append(col)
    else:
        other.append(col)

print(f"\nManual categorization:")
print(f"Backward returns: {backward_returns}")
print(f"Forward returns: {forward_returns}")
print(f"Other: {other}")