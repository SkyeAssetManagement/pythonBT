#!/usr/bin/env python3
"""
Precision Verification Script
Quick test to verify 5 decimal places are working for AD data
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def verify_precision():
    """Verify precision system is working correctly"""
    
    print("PRECISION VERIFICATION TEST")
    print("="*40)
    
    from src.dashboard.dashboard_manager import DashboardManager
    
    # Test AD data
    ad_data = np.array([0.65432, 0.65445, 0.65401, 0.65467, 0.65423])
    
    dm = DashboardManager()
    detected = dm._detect_precision(ad_data)
    
    print(f"AD data: {ad_data[:3]}")
    print(f"Detected precision: {detected}")
    print(f"Expected: 5")
    print(f"Status: {'PASS' if detected == 5 else 'FAIL'}")
    
    # Test formatting
    test_price = 0.65432
    formatted = f"{test_price:.{detected}f}"
    expected = "0.65432"
    
    print(f"\nFormat test: {test_price} -> {formatted}")
    print(f"Expected: {expected}")
    print(f"Status: {'PASS' if formatted == expected else 'FAIL'}")
    
    print(f"\n{'='*40}")
    print(f"RESULT: AD data will show {detected} decimal places")
    print(f"Y-axis will show: 0.65432, 0.65445, etc.")
    print(f"{'='*40}")

if __name__ == "__main__":
    verify_precision()