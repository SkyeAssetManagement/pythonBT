"""
Debug the hanging issue
"""

import sys
import time
from OMtree_validation import DirectionalValidator

print("Starting debug test...")
print("="*60)

# Add verbose output
start_time = time.time()

try:
    print("1. Creating validator...")
    sys.stdout.flush()
    
    validator = DirectionalValidator('OMtree_config.ini')
    
    print("2. Validator created successfully")
    print(f"   Feature selection enabled: {validator.feature_selection_enabled}")
    sys.stdout.flush()
    
    print("3. Loading data...")
    sys.stdout.flush()
    
    data = validator.load_and_prepare_data()
    
    print("4. Data loaded successfully")
    print(f"   Data shape: {data.shape}")
    sys.stdout.flush()
    
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Completed in {elapsed:.2f} seconds")
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n[ERROR] Failed after {elapsed:.2f} seconds")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()

print("="*60)