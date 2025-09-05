#!/usr/bin/env python
"""
OMtree GUI Launcher
Checks dependencies and launches the trading model GUI
"""

import sys
import subprocess
import importlib

def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'tkinter': 'tkinter',
        'PIL': 'pillow',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    for module, package in required.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nTo install missing packages, run:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("="*60)
    print("OMtree Trading Model GUI")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("\nAll dependencies satisfied.")
    print("Launching GUI...")
    
    try:
        # Import and run the GUI
        from OMtree_gui_v3 import OMtreeGUI
        import tkinter as tk
        
        root = tk.Tk()
        app = OMtreeGUI(root)
        
        # Set close protocol
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        root.mainloop()
        
    except Exception as e:
        print(f"\nError launching GUI: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()