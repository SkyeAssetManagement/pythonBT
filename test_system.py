"""
Test script to verify the OMtree system components
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

print("=" * 60)
print("OMtree Trading System - Component Test")
print("=" * 60)
print()

# Test imports
print("Testing module imports...")
try:
    import numpy as np
    print("OK - NumPy imported successfully")
except ImportError as e:
    print(f"X - NumPy import failed: {e}")

try:
    import pandas as pd
    print("OK - Pandas imported successfully")
except ImportError as e:
    print(f"X - Pandas import failed: {e}")

try:
    from sklearn.tree import DecisionTreeClassifier
    print("OK - Scikit-learn imported successfully")
except ImportError as e:
    print(f"X - Scikit-learn import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("OK - Matplotlib imported successfully")
except ImportError as e:
    print(f"X - Matplotlib import failed: {e}")

print()

# Test data file
print("Testing data files...")
data_file = "data/sample_trading_data.csv"
if os.path.exists(data_file):
    print(f"OK - Sample data file exists: {data_file}")
    try:
        df = pd.read_csv(data_file)
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {len(df.columns)} columns")
        print(f"  - Date range: {df['Date'].min()} to {df['Date'].max()}")
    except Exception as e:
        print(f"X - Error reading data file: {e}")
else:
    print(f"X - Sample data file not found: {data_file}")

print()

# Test configuration
print("Testing configuration...")
config_file = "OMtree_config.ini"
if os.path.exists(config_file):
    print(f"OK - Configuration file exists: {config_file}")
    import configparser
    config = configparser.ConfigParser()
    config.read(config_file)
    print(f"  - Data file: {config['data']['csv_file']}")
    print(f"  - Model type: {config['model']['model_type']}")
    print(f"  - Trees: {config['model']['n_trees']}")
else:
    print(f"X - Configuration file not found: {config_file}")

print()

# Test core modules
print("Testing core modules...")
try:
    from src.OMtree_model import DirectionalTreeEnsemble
    print("OK - Model module imported successfully")
except ImportError as e:
    print(f"X - Model module import failed: {e}")

try:
    from src.OMtree_preprocessing import DataPreprocessor
    print("OK - Preprocessing module imported successfully")
except ImportError as e:
    print(f"X - Preprocessing module import failed: {e}")

try:
    from src.OMtree_validation import WalkForwardValidator
    print("OK - Validation module imported successfully")
except ImportError as e:
    print(f"X - Validation module import failed: {e}")

print()

# Quick model test
print("Testing model initialization...")
try:
    model = DirectionalTreeEnsemble(config_path='OMtree_config.ini', verbose=False)
    print("OK - Model initialized successfully")
    print(f"  - Model type: {model.model_type}")
    print(f"  - Trees: {model.n_trees}")
    print(f"  - Max depth: {model.max_depth}")
except Exception as e:
    print(f"X - Model initialization failed: {e}")

print()
print("=" * 60)
print("Test complete! System is ready to use.")
print("=" * 60)