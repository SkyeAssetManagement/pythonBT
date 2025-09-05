"""
Test that preprocessing shows correct smoothing information
"""

import numpy as np
import pandas as pd
from OMtree_preprocessing import DataPreprocessor

# Create simple test data
df = pd.DataFrame({
    'Date': pd.date_range(start='2020-01-01', periods=100, freq='h'),
    'test_feature': np.random.randn(100),
    'test_target': np.random.randn(100)
})

# Create preprocessor with default config
preprocessor = DataPreprocessor('OMtree_config.ini')

print("Preprocessing Configuration")
print("="*60)
print(f"Normalization method: {preprocessor.normalization_method}")
print(f"Vol window: {preprocessor.vol_window}")
print(f"Smoothing type: {preprocessor.smoothing_type}")
print(f"IQR lookback: {preprocessor.recent_iqr_lookback}")
print(f"Calculated alpha: {preprocessor.smoothing_alpha:.4f}")
print(f"  Formula: 2/(n+1) = 2/({preprocessor.recent_iqr_lookback}+1) = {preprocessor.smoothing_alpha:.4f}")
print("="*60)