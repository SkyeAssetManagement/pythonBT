"""
Test script to launch the GUI and verify Data View tab
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Launching OMtree GUI with Data View tab...")
print("=" * 60)
print("\nTo test the Data View tab:")
print("1. Navigate to the 'Data View' tab")
print("2. Load DTSmlDATA7x7.csv")
print("3. Select a column (e.g., Ret_fwd6hr)")
print("4. Try different preprocessing options:")
print("   - Volatility normalization")
print("   - Smoothing (exponential or simple)")
print("5. Click 'Analyze Selected Column' to see:")
print("   - Statistics tab: Comprehensive descriptive stats")
print("   - Distribution tab: Histogram, box plot, Q-Q plot, CDF")
print("   - Time Series tab: Time series with moving average")
print("   - Rolling Stats tab: 90-bar rolling mean and stdev")
print("\n" + "=" * 60)

# Import and run the GUI
try:
    from OMtree_gui_v3 import OMtreeGUI
    import tkinter as tk
    
    root = tk.Tk()
    app = OMtreeGUI(root)
    
    print("\nGUI launched successfully!")
    print("Close the window to exit.")
    
    root.mainloop()
    
except Exception as e:
    print(f"\nError launching GUI: {e}")
    import traceback
    traceback.print_exc()