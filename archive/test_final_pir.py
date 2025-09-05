"""
Final test of PIR file handling in regression tab
"""

import pandas as pd
from column_detector import ColumnDetector

# Load PIR file
df = pd.read_csv('DTSmlDATA_PIR.csv')

print("=== FINAL PIR FILE HANDLING TEST ===\n")

# Test ColumnDetector (as used in updated regression tab)
detected = ColumnDetector.auto_detect_columns(df)

print(f"ColumnDetector Results:")
print(f"Features detected: {len(detected['features'])} columns")
for f in detected['features']:
    print(f"  - {f}")

print(f"\nTargets detected: {len(detected['targets'])} columns")
for t in detected['targets']:
    print(f"  - {t}")

# Verify backward returns are included
backward_returns_found = [f for f in detected['features'] if f.startswith('Ret_') and 'fwd' not in f]
print(f"\nBackward returns found: {len(backward_returns_found)}")
print(f"  {backward_returns_found}")

if len(backward_returns_found) == 8:
    print("\n[SUCCESS] All 8 backward returns (Ret_1hr through Ret_128hr) recognized as features")
else:
    print(f"\n[ERROR] Expected 8 backward returns, found {len(backward_returns_found)}")

# Check if PIR columns would be detected if they existed
test_pir_columns = ['PIR_1hr', 'PIR_16hr', 'Impact_Ratio', 'Price_Impact']
for col in test_pir_columns:
    col_lower = col.lower()
    would_detect = any(pattern in col_lower for pattern in ['pir', 'ratio', 'impact'])
    print(f"Would detect '{col}': {would_detect}")