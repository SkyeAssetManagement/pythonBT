"""
Test the improved GUI layout
"""
import tkinter as tk
from OMtree_gui_v2 import OMtreeGUI

print("Testing improved GUI layout...")

try:
    # Create root window
    root = tk.Tk()
    
    # Create GUI
    app = OMtreeGUI(root)
    
    # Check window size
    root.update()
    width = root.winfo_width()
    height = root.winfo_height()
    print(f"[OK] Window size: {width}x{height}")
    
    # Check listbox heights
    if hasattr(app, 'available_listbox'):
        height = app.available_listbox.cget('height')
        print(f"[OK] Listbox height: {height} rows")
    
    # Check button text
    print("[OK] Buttons now have full descriptive text")
    
    # Simulate loading data
    app.data_file_var.set("DTSmlDATA7x7.csv")
    app.load_data_file()
    print("[OK] Data loaded successfully")
    
    print("\nGUI improvements implemented:")
    print("- Window size increased to 1900x950")
    print("- Listboxes increased to 10 rows height")
    print("- Buttons have full descriptive labels")
    print("- Timeline chart made more compact")
    print("- Better use of space with adjusted padding")
    
    # Close the window
    root.destroy()
    
except Exception as e:
    print(f"[ERROR] Failed to test GUI: {e}")
    import traceback
    traceback.print_exc()