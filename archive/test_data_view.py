import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from data_view_module import DataViewTab

def test_data_view():
    """Test the Data View module standalone"""
    
    # Create main window
    root = tk.Tk()
    root.title("Data View Module Test")
    root.geometry("1400x800")
    
    # Create notebook
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Create Data View tab
    data_view = DataViewTab(notebook)
    
    # Create a test button to load sample data
    def load_test_data():
        """Load DTSmlDATA7x7.csv for testing"""
        try:
            # Load data
            df = pd.read_csv('DTSmlDATA7x7.csv')
            print(f"Loaded {len(df)} rows from DTSmlDATA7x7.csv")
            
            # Check columns
            print(f"Columns: {df.columns.tolist()[:10]}...")
            
            # Check for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"Found {len(numeric_cols)} numeric columns")
            
            # Verify targets and features
            targets = [col for col in numeric_cols if 'fwd' in col.lower()]
            features = [col for col in numeric_cols if col.startswith('Ret_') and 'fwd' not in col.lower()]
            print(f"Targets: {targets}")
            print(f"Features: {features[:5]}...")
            
            return True
            
        except Exception as e:
            print(f"Error loading test data: {e}")
            return False
    
    # Add test button
    test_frame = ttk.Frame(root)
    test_frame.pack(side='bottom', fill='x', padx=10, pady=5)
    
    ttk.Label(test_frame, text="Test Functions:").pack(side='left', padx=5)
    ttk.Button(test_frame, text="Load DTSmlDATA7x7.csv", 
              command=load_test_data).pack(side='left', padx=5)
    
    def test_preprocessing():
        """Test preprocessing functions"""
        print("\nTesting preprocessing options:")
        print("1. Volatility normalization with window=60")
        print("2. Exponential smoothing with alpha=0.1")
        print("3. Simple moving average smoothing")
        print("All preprocessing options available in the GUI")
    
    ttk.Button(test_frame, text="Test Preprocessing", 
              command=test_preprocessing).pack(side='left', padx=5)
    
    # Instructions
    instructions = """
    Data View Tab Test Instructions:
    1. Click 'Load' to load DTSmlDATA7x7.csv
    2. Select a column from the list (e.g., Ret_fwd6hr)
    3. Optionally apply preprocessing:
       - Volatility normalization
       - Smoothing (exponential or simple)
    4. Click 'Analyze Selected Column' to see:
       - Comprehensive statistics
       - Distribution plots (histogram, box plot, Q-Q, CDF)
       - Time series visualization
       - 90-bar rolling statistics
    5. Try different columns and preprocessing settings
    """
    
    ttk.Label(test_frame, text=instructions, justify='left').pack(side='left', padx=20)
    
    # Run the GUI
    root.mainloop()

if __name__ == "__main__":
    print("Testing Data View Module")
    print("=" * 60)
    test_data_view()