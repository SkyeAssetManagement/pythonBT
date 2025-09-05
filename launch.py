#!/usr/bin/env python
"""
OMtree GUI Launcher
Simple script to launch the OMtree GUI
"""

import sys
import os

print("=" * 60)
print("OMtree Trading Model")
print("=" * 60)

# Check if GUI exists
if os.path.exists('OMtree_gui.py'):
    print("Launching GUI...")
    print("-" * 60)
    
    # Run the GUI
    from OMtree_gui import OMtreeGUI
    import tkinter as tk
    
    root = tk.Tk()
    app = OMtreeGUI(root)
    root.mainloop()
else:
    print("ERROR: OMtree_gui.py not found!")
    print("Please ensure you're in the OM3 directory")
    sys.exit(1)