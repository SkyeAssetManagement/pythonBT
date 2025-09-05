"""
Test script to verify VolSignal feature appears in GUI
"""
import pandas as pd
from OMtree_preprocessing import DataPreprocessor

# Load and preprocess data to create VolSignal feature
print("Loading data...")
df = pd.read_csv('DTSmlDATA7x7.csv')
print(f"Original columns: {df.columns.tolist()[:10]}...")

# Preprocess to add VolSignal
print("\nPreprocessing data to add VolSignal feature...")
preprocessor = DataPreprocessor()
processed_df = preprocessor.process_data(df)

# Check what columns we have now
vol_signal_cols = [col for col in processed_df.columns if 'VolSignal' in col]
print(f"\nVolSignal columns found: {vol_signal_cols}")

# Save preprocessed data for GUI to load
output_file = 'DTSmlDATA7x7_with_volsignal.csv'
processed_df.to_csv(output_file, index=False)
print(f"\nSaved preprocessed data to: {output_file}")

print("\nInstructions:")
print("1. Launch the GUI: python launch_gui.py")
print(f"2. Load the file: {output_file}")
print("3. You should now see 'VolSignal_Max250d' in the features list")
print("\nAlternatively, the VolSignal feature is automatically added")
print("when you run walk-forward validation with the original data file.")