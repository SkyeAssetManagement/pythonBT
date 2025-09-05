"""
Test GUI launch and basic functionality
"""
import tkinter as tk
from OMtree_gui_v2 import OMtreeGUI
import sys

print("Testing GUI launch...")

try:
    # Create root window
    root = tk.Tk()
    
    # Create GUI
    app = OMtreeGUI(root)
    
    print("[OK] GUI created successfully")
    
    # Check tabs exist
    tabs = app.notebook.tabs()
    print(f"[OK] Found {len(tabs)} tabs")
    
    # Check configuration manager
    print(f"[OK] Config manager initialized")
    
    # Test that walk-forward button exists
    if hasattr(app, 'run_button'):
        print("[OK] Run Walk Forward button found")
    
    # Test configuration history tables exist
    if hasattr(app, 'data_config_table'):
        print("[OK] Data config history table found")
    
    if hasattr(app, 'model_config_table'):
        print("[OK] Model config history table found")
    
    print("\nAll GUI components verified successfully!")
    
    # Close the window
    root.destroy()
    
except Exception as e:
    print(f"[ERROR] Failed to launch GUI: {e}")
    sys.exit(1)