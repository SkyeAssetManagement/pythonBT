#!/usr/bin/env python3
"""
Detailed Precision Detection Debug Script
Find out exactly why precision is being detected as 2 instead of 5
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def debug_precision_detection():
    """Debug precision detection step by step"""
    
    print("=== DETAILED PRECISION DEBUG ===")
    
    # Test with exact AD data that should be 5 decimals
    ad_data = np.array([0.65432, 0.65445, 0.65401, 0.65467, 0.65423])
    print(f"Test data: {ad_data}")
    print(f"Data types: {ad_data.dtype}")
    
    # Step through the detection logic manually
    print("\n--- Manual Detection Process ---")
    
    # 1. Check non-zero mask
    non_zero_mask = ad_data != 0
    print(f"Non-zero mask: {non_zero_mask}")
    print(f"Any non-zero values: {np.any(non_zero_mask)}")
    
    if np.any(non_zero_mask):
        sample_values = ad_data[non_zero_mask][:500]
        print(f"Sample values: {sample_values}")
        
        max_precision = 0
        
        for i, value in enumerate(sample_values):
            print(f"\n--- Processing value {i+1}: {value} ---")
            
            if np.isnan(value) or np.isinf(value):
                print("  Skipping: NaN or Inf")
                continue
            
            # Check string representation
            value_str = str(value)
            print(f"  String representation: '{value_str}'")
            
            if '.' in value_str and not value_str.endswith('.'):
                decimal_part = value_str.split('.')[1]
                print(f"  Decimal part: '{decimal_part}'")
                
                if 'e' not in decimal_part.lower():
                    meaningful_decimals = len(decimal_part.rstrip('0'))
                    print(f"  Raw meaningful decimals: {meaningful_decimals}")
                    
                    # Check capping logic
                    original_meaningful = meaningful_decimals
                    if meaningful_decimals > 6:
                        print(f"  Applying cap (>{6} decimals)")
                        if value < 10:
                            meaningful_decimals = min(meaningful_decimals, 5)
                            print(f"  Forex cap (value < 10): {meaningful_decimals}")
                        else:
                            meaningful_decimals = min(meaningful_decimals, 4)
                            print(f"  Large value cap: {meaningful_decimals}")
                    
                    print(f"  Final meaningful decimals: {meaningful_decimals}")
                    max_precision = max(max_precision, meaningful_decimals)
                    print(f"  Current max precision: {max_precision}")
                else:
                    print("  Skipping: Scientific notation detected")
            else:
                print("  Skipping: No decimal point or ends with decimal")
        
        print(f"\n--- Final Detection Logic ---")
        print(f"Max precision before checks: {max_precision}")
        
        # Check fallback logic
        if max_precision == 0 and np.any(ad_data % 1 != 0):
            print("Applying fallback logic (max_precision == 0 but has decimals)")
            sample_fractional = ad_data[ad_data < 1.0]
            print(f"Fractional values (< 1.0): {sample_fractional}")
            if len(sample_fractional) > 0:
                print("Forex fallback: setting to 5 decimals")
                max_precision = 5
        
        # Minimum precision check
        if max_precision < 2:
            print(f"Applying minimum precision (was {max_precision}, setting to 2)")
            max_precision = 2
        
        print(f"FINAL DETECTED PRECISION: {max_precision}")
    
    # Now test with the actual DashboardManager
    print("\n--- Testing with DashboardManager ---")
    from src.dashboard.dashboard_manager import DashboardManager
    
    dm = DashboardManager()
    detected = dm._detect_precision(ad_data)
    print(f"DashboardManager detected: {detected}")
    
    # Test with different data types
    print("\n--- Testing Data Type Variations ---")
    
    # Float64 (default)
    test1 = np.array([0.65432, 0.65445], dtype=np.float64)
    print(f"Float64: {test1} -> {dm._detect_precision(test1)}")
    
    # Float32
    test2 = np.array([0.65432, 0.65445], dtype=np.float32)
    print(f"Float32: {test2} -> {dm._detect_precision(test2)}")
    
    # From string conversion
    test3 = np.array([float('0.65432'), float('0.65445')])
    print(f"From strings: {test3} -> {dm._detect_precision(test3)}")
    
    # Check if string representations are being corrupted
    print("\n--- String Representation Analysis ---")
    for val in ad_data:
        print(f"Value: {val}")
        print(f"  str(): '{str(val)}'")
        print(f"  repr(): {repr(val)}")
        print(f"  format .5f: '{val:.5f}'")
        print(f"  format .10f: '{val:.10f}'")

if __name__ == "__main__":
    debug_precision_detection()