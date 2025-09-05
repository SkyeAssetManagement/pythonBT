#!/usr/bin/env python3
"""
Debug the data structure to see what fields are available
"""

import sys
sys.path.append('src')

from src.data.parquet_converter import ParquetConverter
import pandas as pd

def debug_data_structure():
    """Debug the data structure to understand the field names"""
    
    print("=== DEBUGGING DATA STRUCTURE ===")
    
    try:
        # Load data using parquet converter
        parquet_converter = ParquetConverter()
        data = parquet_converter.load_or_convert("GC", "1m", "diffAdjusted")
        
        if data:
            print("SUCCESS: Data loaded")
            print(f"Data type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"Dictionary keys: {list(data.keys())}")
                for key, value in data.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: type={type(value)}")
                        
                # Show first few values of each field
                print("\nFirst 3 values of each field:")
                for key, value in data.items():
                    if hasattr(value, '__len__') and len(value) > 0:
                        try:
                            if hasattr(value, 'iloc'):  # pandas Series
                                print(f"  {key}: {value.iloc[:3].tolist()}")
                            else:  # numpy array
                                print(f"  {key}: {value[:3].tolist()}")
                        except:
                            print(f"  {key}: {str(value)[:100]}")
                            
            elif isinstance(data, pd.DataFrame):
                print(f"DataFrame columns: {list(data.columns)}")
                print(f"DataFrame shape: {data.shape}")
                print("\nFirst 3 rows:")
                print(data.head(3))
                
        else:
            print("ERROR: No data loaded")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_structure()