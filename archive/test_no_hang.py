"""
Quick test to verify the system doesn't hang with current config
"""

import pandas as pd
import numpy as np
from OMtree_preprocessing import DataPreprocessor
import time

print("Testing preprocessing with current config...")
print("="*60)

# Create simple test data
np.random.seed(42)
df = pd.DataFrame({
    'Date': pd.date_range(start='2020-01-01', periods=200, freq='h'),
    'Time': pd.date_range(start='2020-01-01', periods=200, freq='h').strftime('%H:%M:%S'),
    'Ret_0-1hr': np.random.randn(200) * 0.01,
    'Ret_1-2hr': np.random.randn(200) * 0.01,
    'Ret_2-4hr': np.random.randn(200) * 0.01,
    'Ret_128-256hr': np.random.randn(200) * 0.01
})

start_time = time.time()

try:
    # Load current config
    preprocessor = DataPreprocessor('OMtree_config.ini')
    
    print(f"Config loaded successfully")
    print(f"  Normalization method: {preprocessor.normalization_method}")
    print(f"  Vol window: {preprocessor.vol_window}")
    print(f"  Winsorize enabled: {preprocessor.winsorize_enabled}")
    print(f"  IQR weighting: {preprocessor.iqr_weighting_enabled}")
    
    # Process data
    print("\nProcessing data...")
    processed, features, target = preprocessor.process_data(df)
    
    elapsed = time.time() - start_time
    
    print(f"\n[SUCCESS] Preprocessing completed in {elapsed:.2f} seconds")
    print(f"  Processed shape: {processed.shape}")
    print(f"  Feature columns: {len(features)}")
    print(f"  Target column: {target}")
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n[ERROR] Failed after {elapsed:.2f} seconds")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test completed")