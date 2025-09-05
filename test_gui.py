#!/usr/bin/env python
"""
Test script to verify GUI is working correctly
"""

import sys
import os

print("=" * 60)
print("Testing OMtree GUI")
print("=" * 60)

# Add src to path
sys.path.insert(0, 'src')

# Test imports
try:
    print("1. Testing core imports...")
    from OMtree_gui import OMtreeGUI
    import tkinter as tk
    print("   [OK] GUI module loaded")
    
    print("\n2. Testing src imports...")
    from src.OMtree_model import DirectionalTreeEnsemble
    from src.OMtree_validation import DirectionalValidator
    from src.OMtree_preprocessing import DataPreprocessor
    print("   [OK] Core modules loaded")
    
    print("\n3. Testing GUI initialization...")
    root = tk.Tk()
    root.withdraw()  # Hide window for test
    app = OMtreeGUI(root)
    print("   [OK] GUI initialized successfully")
    
    print("\n4. Checking data folder...")
    if os.path.exists('data'):
        csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        print(f"   [OK] Found {len(csv_files)} CSV files in data/")
    else:
        print("   [WARNING] Data folder not found")
    
    print("\n5. Checking configuration...")
    if os.path.exists('OMtree_config.ini'):
        print("   [OK] Configuration file exists")
    else:
        print("   [WARNING] Configuration file not found")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED - GUI is ready to use!")
    print("=" * 60)
    print("\nTo launch the GUI:")
    print("  • Double-click: run_gui.bat")
    print("  • Command line: python OMtree_gui.py")
    print("  • Quick launch: python launch.py")
    
    root.destroy()
    
except Exception as e:
    print(f"\n[ERROR]: {e}")
    print("\nPlease check:")
    print("1. You're in the OM3 directory")
    print("2. All dependencies are installed")
    print("3. The src/ folder exists with all modules")
    sys.exit(1)