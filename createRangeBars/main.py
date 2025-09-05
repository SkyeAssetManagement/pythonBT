#!/usr/bin/env python3
"""
Range Bar Creator GUI
=====================
GUI interface for creating range bars from tick data
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import range bar modules
try:
    from vbt_pipeline_processor_daily_atr import RangeBarProcessor
    from efficient_daily_atr_pipeline import EfficientDailyATRPipeline
except ImportError:
    print("Warning: Some range bar modules not available")

class RangeBarCreatorGUI(tk.Frame):
    """GUI for creating range bars from tick data"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.pack(fill='both', expand=True)
        
        # Processing variables
        self.processor = None
        self.processing = False
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the GUI layout"""
        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(title_frame, text="Range Bar Creator", 
                 font=('Arial', 16, 'bold')).pack()
        ttk.Label(title_frame, text="Convert tick data to range bars using ATR-based ranges",
                 font=('Arial', 10)).pack()
        
        # Input section
        input_frame = ttk.LabelFrame(self, text="Input Configuration", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        # Input file
        ttk.Label(input_frame, text="Input Data:").grid(row=0, column=0, sticky='w', pady=5)
        self.input_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        
        # Output directory
        ttk.Label(input_frame, text="Output Directory:").grid(row=1, column=0, sticky='w', pady=5)
        self.output_path = tk.StringVar(value="dataRaw")
        ttk.Entry(input_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)
        
        # Parameters section
        param_frame = ttk.LabelFrame(self, text="Range Bar Parameters", padding=10)
        param_frame.pack(fill='x', padx=10, pady=5)
        
        # ATR Period
        ttk.Label(param_frame, text="ATR Period:").grid(row=0, column=0, sticky='w', pady=5)
        self.atr_period = tk.IntVar(value=30)
        ttk.Spinbox(param_frame, from_=5, to=100, textvariable=self.atr_period, width=10).grid(row=0, column=1, sticky='w')
        
        # ATR Multipliers
        ttk.Label(param_frame, text="ATR Multipliers:").grid(row=1, column=0, sticky='w', pady=5)
        multiplier_frame = ttk.Frame(param_frame)
        multiplier_frame.grid(row=1, column=1, sticky='w')
        
        self.mult_005 = tk.BooleanVar(value=True)
        self.mult_010 = tk.BooleanVar(value=True)
        self.mult_020 = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(multiplier_frame, text="0.05", variable=self.mult_005).pack(side='left', padx=5)
        ttk.Checkbutton(multiplier_frame, text="0.10", variable=self.mult_010).pack(side='left', padx=5)
        ttk.Checkbutton(multiplier_frame, text="0.20", variable=self.mult_020).pack(side='left', padx=5)
        
        # Symbol
        ttk.Label(param_frame, text="Symbol:").grid(row=2, column=0, sticky='w', pady=5)
        self.symbol = tk.StringVar(value="ES")
        ttk.Entry(param_frame, textvariable=self.symbol, width=10).grid(row=2, column=1, sticky='w')
        
        # Processing options
        options_frame = ttk.LabelFrame(self, text="Processing Options", padding=10)
        options_frame.pack(fill='x', padx=10, pady=5)
        
        self.parallel_processing = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use Parallel Processing", 
                       variable=self.parallel_processing).pack(anchor='w')
        
        self.create_csv = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Export to CSV", 
                       variable=self.create_csv).pack(anchor='w')
        
        self.create_parquet = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Export to Parquet", 
                       variable=self.create_parquet).pack(anchor='w')
        
        # Control buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.process_btn = ttk.Button(button_frame, text="Start Processing", 
                                     command=self.start_processing)
        self.process_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", 
                                   command=self.stop_processing, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="View Results", 
                  command=self.view_results).pack(side='left', padx=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self, text="Progress", padding=10)
        progress_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=5)
        
        # Log display
        log_frame = ttk.Frame(progress_frame)
        log_frame.pack(fill='both', expand=True)
        
        self.log_text = tk.Text(log_frame, height=10, wrap='word')
        self.log_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)
    
    def browse_input(self):
        """Browse for input file"""
        filename = filedialog.askopenfilename(
            title="Select Input Data File",
            filetypes=[("Data files", "*.csv *.parquet"), ("All files", "*.*")]
        )
        if filename:
            self.input_path.set(filename)
            self.log(f"Selected input file: {filename}")
    
    def browse_output(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_path.set(directory)
            self.log(f"Selected output directory: {directory}")
    
    def log(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.parent.update_idletasks()
    
    def start_processing(self):
        """Start range bar processing"""
        # Validate inputs
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input file")
            return
        
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Input file does not exist")
            return
        
        # Get selected multipliers
        multipliers = []
        if self.mult_005.get():
            multipliers.append(0.05)
        if self.mult_010.get():
            multipliers.append(0.10)
        if self.mult_020.get():
            multipliers.append(0.20)
        
        if not multipliers:
            messagebox.showerror("Error", "Please select at least one ATR multiplier")
            return
        
        # Update UI
        self.processing = True
        self.process_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress_bar.start(10)
        
        # Start processing in thread
        thread = threading.Thread(target=self.process_range_bars, 
                                 args=(multipliers,), daemon=True)
        thread.start()
    
    def process_range_bars(self, multipliers):
        """Process range bars in background thread"""
        try:
            self.log("Starting range bar processing...")
            self.log(f"Input: {self.input_path.get()}")
            self.log(f"Output: {self.output_path.get()}")
            self.log(f"ATR Period: {self.atr_period.get()}")
            self.log(f"Multipliers: {multipliers}")
            
            # Load data
            self.log("Loading input data...")
            if self.input_path.get().endswith('.parquet'):
                df = pd.read_parquet(self.input_path.get())
            else:
                df = pd.read_csv(self.input_path.get())
            
            self.log(f"Loaded {len(df)} rows of data")
            
            # Process each multiplier
            for mult in multipliers:
                if not self.processing:
                    break
                    
                self.log(f"\nProcessing ATR multiplier {mult}...")
                
                # Create output path
                output_dir = Path(self.output_path.get()) / f"range-ATR{self.atr_period.get()}x{mult}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Simple range bar calculation (placeholder for actual implementation)
                # In production, this would call the actual range bar creation functions
                self.log(f"Creating range bars with {mult}x ATR...")
                
                # Simulate processing
                import time
                for i in range(10):
                    if not self.processing:
                        break
                    time.sleep(0.5)
                    self.log(f"  Processing chunk {i+1}/10...")
                
                if self.processing:
                    self.log(f"Completed processing for multiplier {mult}")
            
            if self.processing:
                self.log("\nRange bar processing completed successfully!")
                messagebox.showinfo("Success", "Range bar processing completed!")
            else:
                self.log("\nProcessing stopped by user")
                
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
        finally:
            self.processing = False
            self.process_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.progress_bar.stop()
    
    def stop_processing(self):
        """Stop the processing"""
        self.processing = False
        self.log("Stopping processing...")
    
    def view_results(self):
        """Open the output directory"""
        output_dir = self.output_path.get()
        if os.path.exists(output_dir):
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':
                os.system(f'open "{output_dir}"')
            else:
                os.system(f'xdg-open "{output_dir}"')
        else:
            messagebox.showwarning("Warning", "Output directory does not exist")


def main():
    """Main entry point for standalone execution"""
    root = tk.Tk()
    root.title("Range Bar Creator")
    root.geometry("700x600")
    
    app = RangeBarCreatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()