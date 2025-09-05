#!/usr/bin/env python3
"""
Simple Chart Viewer for SAM
============================
Direct chart rendering from parquet/CSV data without backtesting
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, 'src')
sys.path.insert(0, 'tradingCode')

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import mplfinance as mpf
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/mplfinance not available")

class SimpleChartViewer(tk.Tk):
    """Simple chart viewer for parquet/CSV data"""
    
    def __init__(self):
        super().__init__()
        
        self.title("SAM - Simple Chart Viewer")
        self.geometry("1200x800")
        
        # Data storage
        self.df = None
        self.current_file = None
        
        # Setup UI
        self.setup_ui()
        
        # Load initial data if available
        self.load_default_data()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Top control panel
        control_frame = ttk.Frame(self, padding="5")
        control_frame.pack(fill='x')
        
        # File selection
        ttk.Label(control_frame, text="Data File:").pack(side='left', padx=5)
        
        self.file_label = ttk.Label(control_frame, text="No file loaded", 
                                   relief='sunken', width=50)
        self.file_label.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Browse", 
                  command=self.browse_file).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Quick Load", 
                  command=self.quick_load_menu).pack(side='left', padx=5)
        
        # Chart type selection
        ttk.Label(control_frame, text="Chart Type:").pack(side='left', padx=(20, 5))
        
        self.chart_type = tk.StringVar(value="candlestick")
        chart_combo = ttk.Combobox(control_frame, textvariable=self.chart_type,
                                   values=["candlestick", "line", "ohlc", "volume", "renko"],
                                   width=12, state='readonly')
        chart_combo.pack(side='left', padx=5)
        chart_combo.bind('<<ComboboxSelected>>', lambda e: self.update_chart())
        
        # Refresh button
        ttk.Button(control_frame, text="Refresh", 
                  command=self.update_chart).pack(side='left', padx=5)
        
        # Main content area with tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Chart tab
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="Chart")
        
        # Data tab
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data View")
        
        # Stats tab
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        
        # Setup chart area
        self.setup_chart_area()
        
        # Setup data view
        self.setup_data_view()
        
        # Setup stats view
        self.setup_stats_view()
        
        # Status bar
        self.status_bar = ttk.Label(self, text="Ready", relief='sunken')
        self.status_bar.pack(fill='x', side='bottom')
        
    def setup_chart_area(self):
        """Setup the chart display area"""
        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure
            self.figure = Figure(figsize=(12, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(self.canvas, self.chart_frame)
            toolbar.update()
        else:
            # Fallback text display
            ttk.Label(self.chart_frame, 
                     text="Matplotlib not available. Please install matplotlib and mplfinance.",
                     font=('Arial', 14)).pack(expand=True)
    
    def setup_data_view(self):
        """Setup the data view tab"""
        # Create treeview for data display
        tree_frame = ttk.Frame(self.data_frame)
        tree_frame.pack(fill='both', expand=True)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        # Treeview
        self.data_tree = ttk.Treeview(tree_frame, 
                                     yscrollcommand=vsb.set,
                                     xscrollcommand=hsb.set)
        
        vsb.config(command=self.data_tree.yview)
        hsb.config(command=self.data_tree.xview)
        
        # Grid layout
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
    
    def setup_stats_view(self):
        """Setup the statistics view"""
        # Text widget for stats
        self.stats_text = tk.Text(self.stats_frame, wrap='word', width=80, height=30)
        self.stats_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.stats_text)
        scrollbar.pack(side='right', fill='y')
        self.stats_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.stats_text.yview)
    
    def browse_file(self):
        """Browse for a data file"""
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("Data files", "*.parquet *.csv"),
                ("Parquet files", "*.parquet"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.load_file(filename)
    
    def quick_load_menu(self):
        """Show quick load menu for common data files"""
        menu = tk.Menu(self, tearoff=0)
        
        # Find available data files
        data_locations = [
            ('parquetData', '*.parquet'),
            ('parquetData', '*.csv'),
            ('dataParquet', '*.parquet'),
            ('data', '*.csv'),
            ('dataRaw', '**/*.csv'),
            ('tradingCode/data', '*.parquet')
        ]
        
        files_found = []
        for location, pattern in data_locations:
            if os.path.exists(location):
                from glob import glob
                files = glob(os.path.join(location, pattern), recursive=True)
                for f in files[:10]:  # Limit to 10 files per location
                    rel_path = os.path.relpath(f)
                    files_found.append(rel_path)
                    menu.add_command(label=rel_path, 
                                   command=lambda f=f: self.load_file(f))
        
        if files_found:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        else:
            messagebox.showinfo("No Data", "No data files found in common locations")
    
    def load_default_data(self):
        """Try to load default data file"""
        default_files = [
            'parquetData/ES-DIFF-daily-with-atr.csv',
            'dataRaw/range-ATR30x0.05/ES/diffAdjusted/ES-DIFF-range-ATR30x0.05-dailyATR.csv',
            'data/ES_continuous.csv'
        ]
        
        for file in default_files:
            if os.path.exists(file):
                self.load_file(file)
                break
    
    def load_file(self, filename):
        """Load a data file"""
        try:
            self.update_status(f"Loading {filename}...")
            
            # Load data
            if filename.endswith('.parquet'):
                self.df = pd.read_parquet(filename)
            else:
                self.df = pd.read_csv(filename)
            
            self.current_file = filename
            self.file_label.config(text=os.path.basename(filename))
            
            # Process data for charting
            self.process_data()
            
            # Update displays
            self.update_chart()
            self.update_data_view()
            self.update_stats()
            
            self.update_status(f"Loaded {len(self.df)} rows from {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.update_status("Error loading file")
    
    def process_data(self):
        """Process data for charting"""
        if self.df is None:
            return
        
        # Ensure we have required columns for OHLC
        required_cols = ['open', 'high', 'low', 'close']
        available_cols = self.df.columns.str.lower()
        
        # Try to identify OHLC columns
        col_mapping = {}
        for req in required_cols:
            for col in self.df.columns:
                if req in col.lower():
                    col_mapping[req] = col
                    break
        
        # Rename columns if needed
        if len(col_mapping) == 4:
            for standard, original in col_mapping.items():
                if standard != original.lower():
                    self.df[standard] = self.df[original]
        
        # Handle date/time index
        date_cols = ['date', 'datetime', 'time', 'timestamp']
        for col in date_cols:
            if col in self.df.columns.str.lower():
                actual_col = [c for c in self.df.columns if c.lower() == col][0]
                self.df['date'] = pd.to_datetime(self.df[actual_col])
                self.df.set_index('date', inplace=True)
                break
        
        # If no date column, create one
        if 'date' not in self.df.index.names:
            self.df.index = pd.date_range(start='2020-01-01', periods=len(self.df), freq='D')
        
        # Add volume if not present
        if 'volume' not in self.df.columns:
            self.df['volume'] = np.random.randint(1000, 10000, size=len(self.df))
    
    def update_chart(self):
        """Update the chart display"""
        if not MATPLOTLIB_AVAILABLE or self.df is None:
            return
        
        try:
            self.figure.clear()
            
            chart_type = self.chart_type.get()
            
            # Prepare data for mplfinance
            if all(col in self.df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                # Use last 100 bars for better visibility
                plot_df = self.df.tail(min(100, len(self.df)))
                
                # Create custom style
                mc = mpf.make_marketcolors(up='green', down='red',
                                          edge='inherit',
                                          wick='inherit',
                                          volume='inherit')
                s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='gray')
                
                # Plot based on type
                if chart_type == "candlestick":
                    mpf.plot(plot_df, type='candle', style=s, volume=True,
                            ax=self.figure.add_subplot(111),
                            ylabel='Price', title=f'{os.path.basename(self.current_file)}')
                elif chart_type == "ohlc":
                    mpf.plot(plot_df, type='ohlc', style=s, volume=True,
                            ax=self.figure.add_subplot(111),
                            ylabel='Price', title=f'{os.path.basename(self.current_file)}')
                elif chart_type == "line":
                    mpf.plot(plot_df, type='line', style=s, volume=True,
                            ax=self.figure.add_subplot(111),
                            ylabel='Price', title=f'{os.path.basename(self.current_file)}')
                elif chart_type == "renko":
                    mpf.plot(plot_df, type='renko', style=s,
                            ax=self.figure.add_subplot(111),
                            ylabel='Price', title=f'{os.path.basename(self.current_file)}')
                elif chart_type == "volume":
                    ax = self.figure.add_subplot(111)
                    ax.bar(range(len(plot_df)), plot_df['volume'])
                    ax.set_title(f'Volume - {os.path.basename(self.current_file)}')
                    ax.set_xlabel('Bar')
                    ax.set_ylabel('Volume')
            else:
                # Simple line plot for non-OHLC data
                ax = self.figure.add_subplot(111)
                for col in self.df.select_dtypes(include=[np.number]).columns[:5]:
                    ax.plot(self.df.index, self.df[col], label=col)
                ax.legend()
                ax.set_title(f'{os.path.basename(self.current_file)}')
                ax.grid(True, alpha=0.3)
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Chart error: {e}")
            # Fallback to simple plot
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error displaying chart:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
    
    def update_data_view(self):
        """Update the data view tab"""
        if self.df is None:
            return
        
        # Clear existing data
        self.data_tree.delete(*self.data_tree.get_children())
        
        # Setup columns
        columns = ['Index'] + list(self.df.columns)
        self.data_tree['columns'] = columns
        self.data_tree['show'] = 'headings'
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Add data (limit to 1000 rows for performance)
        for idx, row in self.df.head(1000).iterrows():
            values = [str(idx)] + [str(v) for v in row.values]
            self.data_tree.insert('', 'end', values=values)
    
    def update_stats(self):
        """Update statistics view"""
        if self.df is None:
            return
        
        self.stats_text.delete('1.0', tk.END)
        
        stats = []
        stats.append("DATA STATISTICS")
        stats.append("=" * 50)
        stats.append(f"File: {self.current_file}")
        stats.append(f"Shape: {self.df.shape}")
        stats.append(f"Columns: {', '.join(self.df.columns)}")
        stats.append("")
        
        # Basic statistics
        stats.append("BASIC STATISTICS")
        stats.append("-" * 50)
        stats.append(str(self.df.describe()))
        stats.append("")
        
        # Data types
        stats.append("DATA TYPES")
        stats.append("-" * 50)
        stats.append(str(self.df.dtypes))
        stats.append("")
        
        # Missing values
        stats.append("MISSING VALUES")
        stats.append("-" * 50)
        missing = self.df.isnull().sum()
        if missing.any():
            stats.append(str(missing[missing > 0]))
        else:
            stats.append("No missing values")
        
        self.stats_text.insert('1.0', '\n'.join(stats))
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.update_idletasks()


def main():
    """Main entry point"""
    app = SimpleChartViewer()
    app.mainloop()


if __name__ == "__main__":
    main()