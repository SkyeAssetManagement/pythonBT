import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from scipy import stats
from date_parser import FlexibleDateParser
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.OMtree_preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class DataViewTab:
    def __init__(self, parent_notebook):
        self.parent = parent_notebook
        self.df = None
        self.df_processed = None
        self.selected_column = None
        self.preprocessing_applied = False
        
        # Create Data View tab
        self.data_frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.data_frame, text='Data View')
        
        self.setup_data_view_tab()
    
    def setup_data_view_tab(self):
        """Setup the Data View tab with controls and visualization areas"""
        # Main container
        main_container = ttk.Frame(self.data_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_container, text="Data Selection & Settings", padding=10)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        # File selection
        file_frame = ttk.LabelFrame(left_panel, text="Data File", padding=5)
        file_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(file_frame, text="File:").pack(side='left')
        self.file_entry = ttk.Entry(file_frame, width=25)
        self.file_entry.pack(side='left', padx=5)
        self.file_entry.insert(0, "DTSmlDATA7x7.csv")
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Load", command=self.load_data).pack(side='left')
        
        # Column selection
        column_frame = ttk.LabelFrame(left_panel, text="Select Column to Analyze", padding=5)
        column_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Search box
        search_frame = ttk.Frame(column_frame)
        search_frame.pack(fill='x', pady=(0, 5))
        ttk.Label(search_frame, text="Search:").pack(side='left')
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        self.search_entry.pack(side='left', padx=5)
        self.search_var.trace('w', self.filter_columns)
        
        # Column listbox
        list_frame = ttk.Frame(column_frame)
        list_frame.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.column_listbox = tk.Listbox(list_frame, height=10, yscrollcommand=scrollbar.set, exportselection=False)
        self.column_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.column_listbox.yview)
        self.column_listbox.bind('<<ListboxSelect>>', self.on_column_select)
        
        # Column type indicator
        self.column_type_label = ttk.Label(column_frame, text="", foreground='blue')
        self.column_type_label.pack(pady=5)
        
        # Preprocessing settings (independent from main config)
        preprocess_frame = ttk.LabelFrame(left_panel, text="Preprocessing Settings", padding=5)
        preprocess_frame.pack(fill='x', pady=(0, 10))
        
        # Normalization method
        ttk.Label(preprocess_frame, text="Normalization Method:").grid(row=0, column=0, sticky='w', pady=2)
        self.norm_method_var = tk.StringVar(value="IQR")
        norm_combo = ttk.Combobox(preprocess_frame, textvariable=self.norm_method_var, 
                                 values=["None", "IQR", "AVS"], width=10, state='readonly')
        norm_combo.grid(row=0, column=1, sticky='w', pady=2)
        
        # Method-specific parameters frame
        self.method_params_frame = ttk.Frame(preprocess_frame)
        self.method_params_frame.grid(row=1, column=0, columnspan=2, sticky='w', pady=(5, 0))
        
        # IQR parameters
        self.iqr_frame = ttk.Frame(self.method_params_frame)
        ttk.Label(self.iqr_frame, text="IQR Window:").grid(row=0, column=0, sticky='w', pady=2)
        self.vol_window_var = tk.IntVar(value=60)
        vol_spinbox = ttk.Spinbox(self.iqr_frame, from_=10, to=500, textvariable=self.vol_window_var, width=10)
        vol_spinbox.grid(row=0, column=1, sticky='w', pady=2)
        
        # AVS parameters
        self.avs_frame = ttk.Frame(self.method_params_frame)
        ttk.Label(self.avs_frame, text="Slow Window:").grid(row=0, column=0, sticky='w', pady=2)
        self.avs_slow_var = tk.IntVar(value=60)
        avs_slow_spinbox = ttk.Spinbox(self.avs_frame, from_=20, to=200, textvariable=self.avs_slow_var, width=10)
        avs_slow_spinbox.grid(row=0, column=1, sticky='w', pady=2)
        
        ttk.Label(self.avs_frame, text="Fast Window:").grid(row=1, column=0, sticky='w', pady=2)
        self.avs_fast_var = tk.IntVar(value=20)
        avs_fast_spinbox = ttk.Spinbox(self.avs_frame, from_=5, to=100, textvariable=self.avs_fast_var, width=10)
        avs_fast_spinbox.grid(row=1, column=1, sticky='w', pady=2)
        
        # Function to switch parameter frames
        def update_method_params(*args):
            method = self.norm_method_var.get()
            if method == "AVS":
                self.iqr_frame.grid_remove()
                self.avs_frame.grid(row=0, column=0, sticky='w')
            elif method == "IQR":
                self.avs_frame.grid_remove()
                self.iqr_frame.grid(row=0, column=0, sticky='w')
            else:  # None
                self.iqr_frame.grid_remove()
                self.avs_frame.grid_remove()
        
        norm_combo.bind('<<ComboboxSelected>>', update_method_params)
        # Initialize with IQR visible
        self.iqr_frame.grid(row=0, column=0, sticky='w')
        self.avs_frame.grid_remove()
        
        # Smoothing
        ttk.Label(preprocess_frame, text="Smoothing:").grid(row=2, column=0, sticky='w', pady=2)
        self.smoothing_var = tk.StringVar(value="none")
        smoothing_combo = ttk.Combobox(preprocess_frame, textvariable=self.smoothing_var, values=["none", "exponential", "simple"], width=12)
        smoothing_combo.grid(row=2, column=1, sticky='w', pady=2)
        
        ttk.Label(preprocess_frame, text="Smoothing Alpha:").grid(row=3, column=0, sticky='w', pady=2)
        self.alpha_var = tk.DoubleVar(value=0.1)
        alpha_spinbox = ttk.Spinbox(preprocess_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.alpha_var, width=10)
        alpha_spinbox.grid(row=3, column=1, sticky='w', pady=2)
        
        # Apply preprocessing button
        ttk.Button(preprocess_frame, text="Apply Preprocessing", command=self.apply_preprocessing).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Reset button
        ttk.Button(preprocess_frame, text="Reset to Raw Data", command=self.reset_to_raw).grid(row=5, column=0, columnspan=2)
        
        # Status indicator
        self.preprocess_status = ttk.Label(preprocess_frame, text="Using: Raw Data", foreground='green')
        self.preprocess_status.grid(row=6, column=0, columnspan=2, pady=5)
        
        # Analysis button
        ttk.Button(left_panel, text="Analyze Selected Column", command=self.analyze_column, style='Accent.TButton').pack(pady=10)
        
        # Right panel - Results
        right_panel = ttk.Frame(main_container)
        right_panel.grid(row=0, column=1, sticky='nsew')
        
        # Create notebook for different views
        self.results_notebook = ttk.Notebook(right_panel)
        self.results_notebook.pack(fill='both', expand=True)
        
        # Statistics tab
        stats_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(stats_frame, text='Statistics')
        
        # Create text widget for stats with scrollbar
        stats_scroll = ttk.Scrollbar(stats_frame)
        stats_scroll.pack(side='right', fill='y')
        self.stats_text = tk.Text(stats_frame, wrap='word', yscrollcommand=stats_scroll.set, height=30, width=80)
        self.stats_text.pack(fill='both', expand=True)
        stats_scroll.config(command=self.stats_text.yview)
        
        # Distribution tab
        self.dist_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.dist_frame, text='Distribution')
        
        # Time Series tab
        self.ts_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.ts_frame, text='Time Series')
        
        # Rolling Stats tab
        self.rolling_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.rolling_frame, text='Rolling Stats')
        
        # Configure grid weights
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=3)
        main_container.rowconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to load data")
        status_bar = ttk.Label(self.data_frame, textvariable=self.status_var, relief='sunken', anchor='w')
        status_bar.pack(side='bottom', fill='x', padx=10, pady=(0, 5))
    
    def load_data(self):
        """Load data from CSV file"""
        try:
            filename = self.file_entry.get()
            self.df = pd.read_csv(filename)
            self.df_processed = self.df.copy()
            
            # Parse dates for time series analysis
            date_columns = FlexibleDateParser.get_date_columns(self.df)
            try:
                parsed_dates = FlexibleDateParser.parse_dates(self.df)
                self.df['parsed_datetime'] = parsed_dates
                self.df_processed['parsed_datetime'] = parsed_dates
            except:
                # If date parsing fails, create a simple index
                self.df['parsed_datetime'] = pd.date_range(start='2000-01-01', periods=len(self.df), freq='D')
                self.df_processed['parsed_datetime'] = self.df['parsed_datetime']
            
            # Populate column list
            self.populate_column_list()
            
            self.status_var.set(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
            
            # Reset preprocessing status
            self.preprocessing_applied = False
            self.preprocess_status.config(text="Using: Raw Data", foreground='green')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Failed to load data")
    
    def populate_column_list(self):
        """Populate the column listbox"""
        self.column_listbox.delete(0, tk.END)
        
        if self.df is not None:
            # Get numeric columns only (excluding datetime)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Sort columns: targets first, then features, then others
            targets = [col for col in numeric_cols if 'fwd' in col.lower()]
            features = [col for col in numeric_cols if col.startswith('Ret_') and 'fwd' not in col.lower()]
            others = [col for col in numeric_cols if col not in targets and col not in features]
            
            # Add to listbox with categories
            if targets:
                self.column_listbox.insert(tk.END, "=== TARGETS ===")
                for col in targets:
                    self.column_listbox.insert(tk.END, f"  {col}")
            
            if features:
                self.column_listbox.insert(tk.END, "=== FEATURES ===")
                for col in features:
                    self.column_listbox.insert(tk.END, f"  {col}")
            
            if others:
                self.column_listbox.insert(tk.END, "=== OTHER ===")
                for col in others:
                    self.column_listbox.insert(tk.END, f"  {col}")
    
    def browse_file(self):
        """Browse for a data file"""
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="data" if os.path.exists("data") else "."
        )
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
    
    def filter_columns(self, *args):
        """Filter columns based on search text"""
        if self.df is None:
            return
        
        search_text = self.search_var.get().lower()
        self.column_listbox.delete(0, tk.END)
        
        if not search_text:
            self.populate_column_list()
            return
        
        # Filter numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        filtered_cols = [col for col in numeric_cols if search_text in col.lower()]
        
        for col in filtered_cols:
            self.column_listbox.insert(tk.END, col)
    
    def on_column_select(self, event):
        """Handle column selection"""
        selection = self.column_listbox.curselection()
        if selection:
            item = self.column_listbox.get(selection[0]).strip()
            if not item.startswith('==='):  # Skip category headers
                self.selected_column = item
                
                # Update column type indicator
                if 'fwd' in item.lower():
                    self.column_type_label.config(text=f"Type: Target (forward-looking)", foreground='red')
                elif item.startswith('Ret_'):
                    self.column_type_label.config(text=f"Type: Feature (historical)", foreground='blue')
                else:
                    self.column_type_label.config(text=f"Type: Other", foreground='green')
    
    def apply_preprocessing(self):
        """Apply preprocessing to the data"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        try:
            self.df_processed = self.df.copy()
            
            # Get numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Apply normalization based on selected method
            norm_method = self.norm_method_var.get()
            
            if norm_method == "IQR":
                vol_window = self.vol_window_var.get()
                
                for col in numeric_cols:
                    # Calculate rolling IQR
                    q75 = self.df_processed[col].rolling(window=vol_window, min_periods=1).quantile(0.75)
                    q25 = self.df_processed[col].rolling(window=vol_window, min_periods=1).quantile(0.25)
                    iqr = q75 - q25
                    iqr = iqr.replace(0, np.nan).ffill().fillna(1)
                    
                    # Normalize by IQR (no centering)
                    self.df_processed[f'{col}_norm'] = self.df_processed[col] / iqr
                    
            elif norm_method == "AVS":
                slow_window = self.avs_slow_var.get()
                fast_window = self.avs_fast_var.get()
                
                for col in numeric_cols:
                    # Calculate AVS
                    abs_series = np.abs(self.df_processed[col])
                    vol_slow = abs_series.rolling(window=slow_window, min_periods=20).mean()
                    vol_fast = abs_series.rolling(window=fast_window, min_periods=10).mean()
                    
                    # Adaptive weighting
                    vol_ratio = vol_fast / vol_slow
                    vol_ratio = vol_ratio.clip(0.5, 2.0)
                    weight = 0.3 + 0.3 * np.abs(vol_ratio - 1)
                    weight = np.minimum(weight, 0.6)
                    
                    # Adaptive volatility
                    adaptive_vol = weight * vol_fast + (1 - weight) * vol_slow
                    adaptive_vol = adaptive_vol.replace(0, np.nan).ffill().fillna(1)
                    
                    # Normalize by AVS
                    self.df_processed[f'{col}_norm'] = self.df_processed[col] / adaptive_vol
            
            # Apply smoothing if selected and normalization was applied
            if norm_method != "None":
                if self.smoothing_var.get() == 'exponential':
                    alpha = self.alpha_var.get()
                    for col in numeric_cols:
                        if f'{col}_norm' in self.df_processed.columns:
                            self.df_processed[f'{col}_norm'] = self.df_processed[f'{col}_norm'].ewm(alpha=alpha, adjust=False).mean()
                elif self.smoothing_var.get() == 'simple':
                    window = int(1 / self.alpha_var.get()) if self.alpha_var.get() > 0 else 10
                    for col in numeric_cols:
                        if f'{col}_norm' in self.df_processed.columns:
                            self.df_processed[f'{col}_norm'] = self.df_processed[f'{col}_norm'].rolling(window=window, min_periods=1).mean()
            else:
                # Apply smoothing to raw data if selected
                if self.smoothing_var.get() == 'exponential':
                    alpha = self.alpha_var.get()
                    for col in numeric_cols:
                        self.df_processed[col] = self.df_processed[col].ewm(alpha=alpha, adjust=False).mean()
                elif self.smoothing_var.get() == 'simple':
                    window = int(1 / self.alpha_var.get()) if self.alpha_var.get() > 0 else 10
                    for col in numeric_cols:
                        self.df_processed[col] = self.df_processed[col].rolling(window=window, min_periods=1).mean()
            
            self.preprocessing_applied = True
            
            # Update status
            status_text = "Using: "
            if norm_method == "IQR":
                status_text += f"IQR Normalized (window={self.vol_window_var.get()})"
            elif norm_method == "AVS":
                status_text += f"AVS Normalized (slow={self.avs_slow_var.get()}, fast={self.avs_fast_var.get()})"
            else:
                status_text += "Raw Data"
            
            if self.smoothing_var.get() != 'none':
                status_text += f" + {self.smoothing_var.get()} smoothing"
            
            self.preprocess_status.config(text=status_text, foreground='blue')
            self.status_var.set("Preprocessing applied")
            
            # Re-populate column list if normalization was applied
            if norm_method != "None":
                self.populate_column_list_processed()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply preprocessing: {str(e)}")
    
    def populate_column_list_processed(self):
        """Populate column list with processed columns"""
        self.column_listbox.delete(0, tk.END)
        
        if self.df_processed is not None:
            # Get all numeric columns
            numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns.tolist()
            
            # Separate normalized columns
            norm_cols = [col for col in numeric_cols if '_norm' in col]
            raw_cols = [col for col in numeric_cols if '_norm' not in col and '_avs_vol' not in col and '_iqr' not in col]
            
            if norm_cols:
                self.column_listbox.insert(tk.END, "=== NORMALIZED ===")
                for col in norm_cols:
                    self.column_listbox.insert(tk.END, f"  {col}")
            
            if raw_cols:
                self.column_listbox.insert(tk.END, "=== RAW COLUMNS ===")
                for col in raw_cols:
                    self.column_listbox.insert(tk.END, f"  {col}")
    
    def reset_to_raw(self):
        """Reset to raw data"""
        if self.df is not None:
            self.df_processed = self.df.copy()
            self.preprocessing_applied = False
            self.preprocess_status.config(text="Using: Raw Data", foreground='green')
            self.populate_column_list()
            self.status_var.set("Reset to raw data")
    
    def analyze_column(self):
        """Analyze the selected column"""
        if self.selected_column is None:
            messagebox.showwarning("Warning", "Please select a column to analyze")
            return
        
        if self.df_processed is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        # Get the column data
        if self.selected_column in self.df_processed.columns:
            column_data = self.df_processed[self.selected_column].dropna()
        else:
            messagebox.showerror("Error", f"Column '{self.selected_column}' not found")
            return
        
        # Calculate statistics
        self.calculate_statistics(column_data)
        
        # Create visualizations
        self.create_distribution_plots(column_data)
        self.create_time_series_plot(column_data)
        self.create_rolling_stats_plot(column_data)
        
        self.status_var.set(f"Analyzed column: {self.selected_column}")
    
    def calculate_statistics(self, data):
        """Calculate and display comprehensive statistics"""
        stats_text = []
        stats_text.append("=" * 60)
        stats_text.append(f"STATISTICS FOR: {self.selected_column}")
        stats_text.append("=" * 60)
        
        # Basic statistics
        stats_text.append("\nBASIC STATISTICS:")
        stats_text.append("-" * 40)
        stats_text.append(f"Count:          {len(data):,}")
        stats_text.append(f"Mean:           {data.mean():.6f}")
        stats_text.append(f"Median:         {data.median():.6f}")
        stats_text.append(f"Std Dev:        {data.std():.6f}")
        stats_text.append(f"Variance:       {data.var():.6f}")
        stats_text.append(f"Min:            {data.min():.6f}")
        stats_text.append(f"Max:            {data.max():.6f}")
        stats_text.append(f"Range:          {data.max() - data.min():.6f}")
        
        # Percentiles
        stats_text.append("\nPERCENTILES:")
        stats_text.append("-" * 40)
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats_text.append(f"{p:3d}th percentile: {data.quantile(p/100):.6f}")
        
        # Distribution characteristics
        stats_text.append("\nDISTRIBUTION CHARACTERISTICS:")
        stats_text.append("-" * 40)
        stats_text.append(f"Skewness:       {stats.skew(data):.6f}")
        stats_text.append(f"Kurtosis:       {stats.kurtosis(data):.6f}")
        stats_text.append(f"IQR:            {data.quantile(0.75) - data.quantile(0.25):.6f}")
        
        # Normality test
        if len(data) > 8:
            _, p_value = stats.normaltest(data)
            stats_text.append(f"Normality test p-value: {p_value:.6f}")
            if p_value < 0.05:
                stats_text.append("  → Data is NOT normally distributed (p < 0.05)")
            else:
                stats_text.append("  → Data appears normally distributed (p >= 0.05)")
        
        # Zero crossings and sign statistics
        stats_text.append("\nSIGN STATISTICS:")
        stats_text.append("-" * 40)
        positive_count = (data > 0).sum()
        negative_count = (data < 0).sum()
        zero_count = (data == 0).sum()
        stats_text.append(f"Positive values: {positive_count:,} ({positive_count/len(data)*100:.1f}%)")
        stats_text.append(f"Negative values: {negative_count:,} ({negative_count/len(data)*100:.1f}%)")
        stats_text.append(f"Zero values:     {zero_count:,} ({zero_count/len(data)*100:.1f}%)")
        
        # Autocorrelation
        if len(data) > 10:
            stats_text.append("\nAUTOCORRELATION:")
            stats_text.append("-" * 40)
            for lag in [1, 5, 10, 20]:
                if len(data) > lag:
                    autocorr = data.autocorr(lag=lag)
                    stats_text.append(f"Lag {lag:2d}: {autocorr:.4f}")
        
        # Update text widget
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, '\n'.join(stats_text))
    
    def create_distribution_plots(self, data):
        """Create distribution visualizations"""
        # Clear previous plots
        for widget in self.dist_frame.winfo_children():
            widget.destroy()
        
        # Create figure with subplots
        fig = Figure(figsize=(12, 8), tight_layout=True)
        
        # 1. Histogram with KDE
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(data, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        
        # Add KDE
        kde_data = np.linspace(data.min(), data.max(), 100)
        kde = stats.gaussian_kde(data)
        ax1.plot(kde_data, kde(kde_data), 'r-', linewidth=2, label='KDE')
        
        # Add normal distribution overlay
        mu, sigma = data.mean(), data.std()
        normal_dist = stats.norm.pdf(kde_data, mu, sigma)
        ax1.plot(kde_data, normal_dist, 'g--', linewidth=2, label='Normal')
        
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution of {self.selected_column}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax2 = fig.add_subplot(2, 2, 2)
        box_data = ax2.boxplot(data, vert=True, patch_artist=True)
        box_data['boxes'][0].set_facecolor('lightblue')
        ax2.set_ylabel('Value')
        ax2.set_title('Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        ax3 = fig.add_subplot(2, 2, 3)
        stats.probplot(data, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal)')
        ax3.grid(True, alpha=0.3)
        
        # 4. CDF
        ax4 = fig.add_subplot(2, 2, 4)
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, cdf, 'b-', linewidth=2)
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Function')
        ax4.grid(True, alpha=0.3)
        
        # Add mean and median lines
        ax4.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.4f}')
        ax4.axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.4f}')
        ax4.legend()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.dist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_time_series_plot(self, data):
        """Create time series visualization"""
        # Clear previous plots
        for widget in self.ts_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig = Figure(figsize=(12, 8), tight_layout=True)
        
        # Get datetime index
        if 'parsed_datetime' in self.df_processed.columns:
            time_index = self.df_processed['parsed_datetime'][:len(data)]
        else:
            time_index = range(len(data))
        
        # 1. Time series plot
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(time_index, data.values, 'b-', linewidth=0.5, label='Value')
        
        # Add rolling mean
        if len(data) > 20:
            rolling_mean = pd.Series(data.values).rolling(window=20, min_periods=1).mean()
            ax1.plot(time_index, rolling_mean, 'r-', linewidth=2, label='20-period MA')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.set_title(f'Time Series: {self.selected_column}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns/Changes plot
        ax2 = fig.add_subplot(2, 1, 2)
        changes = pd.Series(data.values).diff()
        ax2.plot(time_index, changes, 'g-', linewidth=0.5, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add standard deviation bands
        std = changes.std()
        ax2.axhline(y=std, color='red', linestyle='--', alpha=0.5, label=f'±1 STD')
        ax2.axhline(y=-std, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=2*std, color='orange', linestyle='--', alpha=0.3, label=f'±2 STD')
        ax2.axhline(y=-2*std, color='orange', linestyle='--', alpha=0.3)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Change')
        ax2.set_title('First Differences (Returns)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.ts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_rolling_stats_plot(self, data):
        """Create 90-bar rolling statistics plot"""
        # Clear previous plots
        for widget in self.rolling_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig = Figure(figsize=(12, 8), tight_layout=True)
        
        # Calculate rolling statistics
        window = 90
        data_series = pd.Series(data.values)
        rolling_mean = data_series.rolling(window=window, min_periods=1).mean()
        rolling_std = data_series.rolling(window=window, min_periods=1).std()
        rolling_min = data_series.rolling(window=window, min_periods=1).min()
        rolling_max = data_series.rolling(window=window, min_periods=1).max()
        
        # Get time index
        if 'parsed_datetime' in self.df_processed.columns:
            time_index = self.df_processed['parsed_datetime'][:len(data)]
        else:
            time_index = range(len(data))
        
        # 1. Rolling mean and standard deviation bands
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(time_index, data_series.values, 'gray', linewidth=0.5, alpha=0.5, label='Raw Data')
        ax1.plot(time_index, rolling_mean, 'b-', linewidth=2, label=f'{window}-bar Mean')
        
        # Add standard deviation bands
        ax1.fill_between(time_index, 
                         rolling_mean - rolling_std, 
                         rolling_mean + rolling_std, 
                         alpha=0.3, color='blue', label=f'±1 STD')
        ax1.fill_between(time_index, 
                         rolling_mean - 2*rolling_std, 
                         rolling_mean + 2*rolling_std, 
                         alpha=0.1, color='blue', label=f'±2 STD')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.set_title(f'{window}-bar Rolling Mean with Standard Deviation Bands')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling standard deviation
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(time_index, rolling_std, 'r-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title(f'{window}-bar Rolling Standard Deviation')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling min/max envelope
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(time_index, data_series.values, 'gray', linewidth=0.5, alpha=0.5, label='Raw Data')
        ax3.plot(time_index, rolling_min, 'g-', linewidth=1.5, label=f'{window}-bar Min')
        ax3.plot(time_index, rolling_max, 'r-', linewidth=1.5, label=f'{window}-bar Max')
        ax3.fill_between(time_index, rolling_min, rolling_max, alpha=0.2, color='yellow', label='Min-Max Range')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Value')
        ax3.set_title(f'{window}-bar Rolling Min/Max Envelope')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.rolling_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def apply_preprocessing(self):
        """Apply preprocessing to loaded data (for Data View tab)"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        
        try:
            # Collect preprocessing settings from the new preprocessing widgets
            settings = {}
            settings['normalization_method'] = self.preproc_widgets['normalization_method'].get()
            settings['normalize_features'] = self.preproc_widgets['normalize_features'].get() == 'true'
            settings['normalize_target'] = self.preproc_widgets['normalize_target'].get() == 'true'
            settings['detrend_features'] = self.preproc_widgets['detrend_features'].get() == 'true'
            
            if settings['normalization_method'] != 'None':
                if settings['normalization_method'] in ['IQR', 'LOGIT_RANK']:
                    settings['vol_window'] = int(self.preproc_widgets['vol_window'].get())
                    settings['winsorize_enabled'] = self.preproc_widgets['winsorize_enabled'].get() == 'true'
                    settings['winsorize_percentile'] = int(self.preproc_widgets['winsorize_percentile'].get())
                elif settings['normalization_method'] == 'AVS':
                    settings['avs_slow_window'] = int(self.preproc_widgets['avs_slow_window'].get())
                    settings['avs_fast_window'] = int(self.preproc_widgets['avs_fast_window'].get())
            
            # Apply preprocessing using the DataPreprocessor class
            preprocessor = DataPreprocessor()
            # Create a temporary config-like object for the preprocessor
            import configparser
            temp_config = configparser.ConfigParser()
            temp_config.add_section('preprocessing')
            for key, value in settings.items():
                temp_config.set('preprocessing', key, str(value))
            
            preprocessor.config = temp_config
            
            # Apply preprocessing
            self.df_processed = preprocessor.process_data(self.df.copy())
            
            # Update status
            self.preproc_status_label.config(text=f"Preprocessing applied ({settings['normalization_method']})")
            
            # Update column analysis if a column is selected
            if self.selected_column:
                self.update_analysis()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error applying preprocessing: {str(e)}")
            self.preproc_status_label.config(text="Error in preprocessing", foreground='red')