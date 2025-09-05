import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
import configparser
import os
import warnings
from date_parser import FlexibleDateParser
from column_detector import ColumnDetector
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.OMtree_preprocessing import DataPreprocessor
warnings.filterwarnings('ignore')

class RegressionAnalysisTab:
    def __init__(self, parent_notebook):
        self.parent = parent_notebook
        self.df = None
        self.df_filtered = None  # For filtered data
        self.df_processed = None  # For processed data
        self.selected_x_vars = []
        self.selected_y_vars = []
        self.target_threshold = 0.1  # Default threshold
        self.preprocessing_settings = {}  # Independent preprocessing settings
        self.use_normalized_data = False  # Initialize normalized data flag
        
        # Initialize column lists
        self.feature_columns = []
        self.target_columns = []
        self.feature_columns_vol = []
        self.target_columns_vol = []
        
        # Create regression tab
        self.regression_frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.regression_frame, text='Regression Analysis')
        
        self.setup_regression_tab()
    
    def setup_regression_tab(self):
        """Setup the regression analysis tab with filtering options"""
        # Main container with three panels
        main_container = ttk.Frame(self.regression_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Variable selection
        left_panel = ttk.LabelFrame(main_container, text="Variable Selection", padding=10)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        # File selection
        file_frame = ttk.Frame(left_panel)
        file_frame.pack(fill='x', pady=(0, 10))
        ttk.Label(file_frame, text="Data File:").pack(side='left')
        self.file_entry = ttk.Entry(file_frame, width=30)
        self.file_entry.pack(side='left', padx=5)
        self.file_entry.insert(0, "data/DTSmlDATA_AllTickers_Hourly.csv")
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Load", command=self.load_data).pack(side='left')
        
        # === DATA FILTERING OPTIONS ===
        data_filter_frame = ttk.LabelFrame(left_panel, text="Data Filters", padding=5)
        data_filter_frame.pack(fill='x', pady=(0, 10))
        
        # Ticker filter
        ticker_frame = ttk.Frame(data_filter_frame)
        ticker_frame.pack(fill='x', pady=2)
        self.ticker_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ticker_frame, text="Filter by Ticker:", variable=self.ticker_filter_var,
                       command=self.toggle_ticker_filter).pack(side='left')
        self.ticker_combo = ttk.Combobox(ticker_frame, values=[], state='disabled', width=15)
        self.ticker_combo.pack(side='left', padx=(5, 0))
        
        # Hour filter
        hour_frame = ttk.Frame(data_filter_frame)
        hour_frame.pack(fill='x', pady=2)
        self.hour_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(hour_frame, text="Filter by Hour:", variable=self.hour_filter_var,
                       command=self.toggle_hour_filter).pack(side='left')
        self.hour_spin_var = tk.IntVar(value=0)
        self.hour_spin = ttk.Spinbox(hour_frame, from_=0, to=23, textvariable=self.hour_spin_var,
                                     state='disabled', width=15)
        self.hour_spin.pack(side='left', padx=(5, 0))
        
        # Apply filters button
        ttk.Button(data_filter_frame, text="Apply Filters", command=self.apply_data_filters).pack(pady=(5, 0))
        
        # Analysis type selection
        type_frame = ttk.LabelFrame(left_panel, text="Analysis Type", padding=5)
        type_frame.pack(fill='x', pady=(0, 10))
        self.analysis_type = tk.StringVar(value="features_to_targets")
        ttk.Radiobutton(type_frame, text="Features → Targets", 
                       variable=self.analysis_type, value="features_to_targets",
                       command=self.update_variable_lists).pack(anchor='w')
        ttk.Radiobutton(type_frame, text="Features → Features", 
                       variable=self.analysis_type, value="features_to_features",
                       command=self.update_variable_lists).pack(anchor='w')
        ttk.Radiobutton(type_frame, text="Targets → Targets", 
                       variable=self.analysis_type, value="targets_to_targets",
                       command=self.update_variable_lists).pack(anchor='w')
        ttk.Radiobutton(type_frame, text="Custom Selection", 
                       variable=self.analysis_type, value="custom",
                       command=self.update_variable_lists).pack(anchor='w')
        
        # === PREPROCESSING OPTIONS ===
        preproc_frame = ttk.LabelFrame(left_panel, text="Preprocessing Options (Independent)", padding=5)
        preproc_frame.pack(fill='x', pady=(0, 10))
        
        # Store preprocessing widgets
        self.preproc_widgets = {}
        self.iqr_widgets = []
        self.avs_widgets = []
        
        # Normalization method
        ttk.Label(preproc_frame, text='Normalization Method:').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        norm_method = ttk.Combobox(preproc_frame, values=['None', 'IQR', 'AVS', 'LOGIT_RANK'], width=15, state='readonly')
        norm_method.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        norm_method.set('IQR')  # Default
        self.preproc_widgets['normalization_method'] = norm_method
        
        # Method description
        self.norm_method_desc = ttk.Label(preproc_frame, text='IQR: Interquartile Range normalization', 
                                         font=('Arial', 8), foreground='gray')
        self.norm_method_desc.grid(row=0, column=2, padx=5, pady=2, sticky='w')
        
        def update_norm_description(*args):
            method = norm_method.get()
            if method == 'AVS':
                self.norm_method_desc.config(text='AVS: Adaptive Volatility Scaling')
                for widget in self.avs_widgets:
                    widget.grid()
                for widget in self.iqr_widgets:
                    widget.grid_remove()
            elif method == 'LOGIT_RANK':
                self.norm_method_desc.config(text='Logit-Rank: Percentile rank with logit transform')
                for widget in self.avs_widgets:
                    widget.grid_remove()
                for widget in self.iqr_widgets:
                    widget.grid_remove()
            elif method == 'None':
                self.norm_method_desc.config(text='No normalization applied')
                for widget in self.avs_widgets:
                    widget.grid_remove()
                for widget in self.iqr_widgets:
                    widget.grid_remove()
            else:  # IQR
                self.norm_method_desc.config(text='IQR: Interquartile Range normalization')
                for widget in self.iqr_widgets:
                    widget.grid()
                for widget in self.avs_widgets:
                    widget.grid_remove()
        
        norm_method.bind('<<ComboboxSelected>>', update_norm_description)
        
        # Preprocessing parameters
        preproc_params = [
            ('normalize_features', 'Normalize Features', 'combo', ['true', 'false'], 'both', 1),
            ('normalize_target', 'Normalize Target', 'combo', ['true', 'false'], 'both', 2),
            ('detrend_features', 'Detrend Features', 'combo', ['false', 'true'], 'both', 3),
            ('vol_window', 'Lookback Window', 'spinbox', (10, 500, 10), 'iqr', 4),
            ('winsorize_enabled', 'Enable Winsorization', 'combo', ['false', 'true'], 'iqr', 5),
            ('winsorize_percentile', 'Winsorize %', 'spinbox', (1, 25, 1), 'iqr', 6),
            ('avs_slow_window', 'AVS Slow Window', 'spinbox', (20, 200, 10), 'avs', 7),
            ('avs_fast_window', 'AVS Fast Window', 'spinbox', (5, 100, 5), 'avs', 8),
        ]
        
        for key, label, widget_type, options, param_group, row in preproc_params:
            label_widget = ttk.Label(preproc_frame, text=label + ':')
            label_widget.grid(row=row, column=0, sticky='w', padx=5, pady=2)
            
            if widget_type == 'combo':
                widget = ttk.Combobox(preproc_frame, values=options, width=15, state='readonly')
                if key == 'normalize_features':
                    widget.set('true')
                else:
                    widget.set(options[0])
            elif widget_type == 'spinbox':
                min_val, max_val, increment = options
                widget = ttk.Spinbox(preproc_frame, from_=min_val, to=max_val, 
                                    increment=increment, width=15)
                if key == 'vol_window':
                    widget.set(60)
                elif key == 'winsorize_percentile':
                    widget.set(5)
                elif key == 'avs_slow_window':
                    widget.set(100)
                elif key == 'avs_fast_window':
                    widget.set(20)
            
            widget.grid(row=row, column=1, padx=5, pady=2, sticky='w')
            self.preproc_widgets[key] = widget
            
            # Add to appropriate widget list for showing/hiding
            if param_group == 'iqr':
                self.iqr_widgets.extend([label_widget, widget])
            elif param_group == 'avs':
                self.avs_widgets.extend([label_widget, widget])
                # Initially hide AVS widgets
                label_widget.grid_remove()
                widget.grid_remove()
        
        # Add processing button
        process_frame = ttk.Frame(preproc_frame)
        process_frame.grid(row=9, column=0, columnspan=3, pady=10)
        ttk.Button(process_frame, text="Apply Preprocessing", command=self.apply_preprocessing).pack(side='left', padx=5)
        self.preproc_status_label = ttk.Label(process_frame, text="", foreground='green')
        self.preproc_status_label.pack(side='left', padx=10)
        
        # === TARGET FILTERING SECTION ===
        filter_frame = ttk.LabelFrame(left_panel, text="Target Filtering", padding=5)
        filter_frame.pack(fill='x', pady=(0, 10))
        
        # Filter type selection
        filter_type_frame = ttk.Frame(filter_frame)
        filter_type_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(filter_type_frame, text="Filter Type:").pack(side='left', padx=(0, 10))
        self.filter_type = tk.StringVar(value="ALL")
        ttk.Radiobutton(filter_type_frame, text="ALL", 
                       variable=self.filter_type, value="ALL",
                       command=self.update_filter_info).pack(side='left', padx=5)
        ttk.Radiobutton(filter_type_frame, text="UP (> threshold)", 
                       variable=self.filter_type, value="UP",
                       command=self.update_filter_info).pack(side='left', padx=5)
        ttk.Radiobutton(filter_type_frame, text="DOWN (< -threshold)", 
                       variable=self.filter_type, value="DOWN",
                       command=self.update_filter_info).pack(side='left', padx=5)
        
        # Threshold setting
        threshold_frame = ttk.Frame(filter_frame)
        threshold_frame.pack(fill='x', pady=5)
        
        ttk.Label(threshold_frame, text="Threshold:").pack(side='left', padx=(0, 10))
        self.threshold_var = tk.StringVar(value="0.1")
        self.threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.pack(side='left', padx=5)
        
        ttk.Button(threshold_frame, text="Load from Config", 
                  command=self.load_threshold_from_config).pack(side='left', padx=10)
        
        # Filter info label
        self.filter_info = ttk.Label(filter_frame, text="Filter: ALL data points", 
                                    foreground='blue', font=('Arial', 9))
        self.filter_info.pack(anchor='w', pady=(5, 0))
        
        # Variable lists
        lists_frame = ttk.Frame(left_panel)
        lists_frame.pack(fill='both', expand=True)
        
        # X Variables (Independent)
        x_frame = ttk.LabelFrame(lists_frame, text="X Variables (Independent)", padding=5)
        x_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        # Create listbox first
        x_list_frame = ttk.Frame(x_frame)
        x_list_frame.pack(fill='both', expand=True)
        
        x_scroll = ttk.Scrollbar(x_list_frame)
        x_scroll.pack(side='right', fill='y')
        self.x_listbox = tk.Listbox(x_list_frame, selectmode='multiple', 
                                    yscrollcommand=x_scroll.set, height=8, exportselection=False)
        self.x_listbox.pack(side='left', fill='both', expand=True)
        x_scroll.config(command=self.x_listbox.yview)
        
        # X variable buttons
        x_button_frame = ttk.Frame(x_frame)
        x_button_frame.pack(fill='x', pady=(5, 0))
        ttk.Button(x_button_frame, text="Select All", 
                  command=lambda: self.select_all_x()).pack(side='left', padx=2)
        ttk.Button(x_button_frame, text="Deselect All", 
                  command=lambda: self.deselect_all_x()).pack(side='left', padx=2)
        
        # Y Variables (Dependent)
        y_frame = ttk.LabelFrame(lists_frame, text="Y Variables (Dependent)", padding=5)
        y_frame.grid(row=0, column=1, sticky='nsew')
        
        # Create listbox first
        y_list_frame = ttk.Frame(y_frame)
        y_list_frame.pack(fill='both', expand=True)
        
        y_scroll = ttk.Scrollbar(y_list_frame)
        y_scroll.pack(side='right', fill='y')
        self.y_listbox = tk.Listbox(y_list_frame, selectmode='multiple',
                                    yscrollcommand=y_scroll.set, height=8, exportselection=False)
        self.y_listbox.pack(side='left', fill='both', expand=True)
        y_scroll.config(command=self.y_listbox.yview)
        
        # Y variable buttons
        y_button_frame = ttk.Frame(y_frame)
        y_button_frame.pack(fill='x', pady=(5, 0))
        ttk.Button(y_button_frame, text="Select All", 
                  command=lambda: self.select_all_y()).pack(side='left', padx=2)
        ttk.Button(y_button_frame, text="Deselect All", 
                  command=lambda: self.deselect_all_y()).pack(side='left', padx=2)
        
        lists_frame.columnconfigure(0, weight=1)
        lists_frame.columnconfigure(1, weight=1)
        lists_frame.rowconfigure(0, weight=1)
        
        # Selection info
        self.selection_info = ttk.Label(left_panel, text="", foreground='blue')
        self.selection_info.pack(pady=5)
        
        # Run button
        ttk.Button(left_panel, text="Run Regression Analysis", 
                  command=self.run_regression, style='Accent.TButton').pack(pady=10)
        
        # Update selection info button for manual update
        ttk.Button(left_panel, text="Update Selection Count", 
                  command=self.update_selection_info).pack(pady=5)
        
        # Right panel - Results display
        right_panel = ttk.LabelFrame(main_container, text="Analysis Results", padding=10)
        right_panel.grid(row=0, column=1, sticky='nsew')
        
        # Create notebook for different result views
        self.results_notebook = ttk.Notebook(right_panel)
        self.results_notebook.pack(fill='both', expand=True)
        
        # Matrix view tab
        self.matrix_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.matrix_frame, text='Matrix View')
        
        # Summary stats tab
        self.summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_frame, text='Summary Statistics')
        
        summary_scroll = ttk.Scrollbar(self.summary_frame)
        summary_scroll.pack(side='right', fill='y')
        self.summary_text = tk.Text(self.summary_frame, wrap='none', 
                                   yscrollcommand=summary_scroll.set, height=25)
        self.summary_text.pack(fill='both', expand=True)
        summary_scroll.config(command=self.summary_text.yview)
        
        # Configure grid weights
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=3)
        main_container.rowconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to load data")
        status_bar = ttk.Label(self.regression_frame, textvariable=self.status_var, 
                              relief='sunken', anchor='w')
        status_bar.pack(side='bottom', fill='x', padx=10, pady=(0, 5))
    
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
    
    def load_threshold_from_config(self):
        """Load threshold from config file"""
        try:
            config = configparser.ConfigParser()
            if config.read('OMtree_config.ini'):
                if 'model' in config and 'target_threshold' in config['model']:
                    threshold = float(config['model']['target_threshold'])
                    self.threshold_var.set(str(threshold))
                    self.target_threshold = threshold
                    messagebox.showinfo("Success", f"Loaded threshold: {threshold}")
                    self.update_filter_info()
                else:
                    messagebox.showwarning("Not Found", "No target_threshold found in config")
            else:
                messagebox.showwarning("Error", "Could not read OMtree_config.ini")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load threshold: {str(e)}")
    
    def apply_preprocessing(self):
        """Apply preprocessing to loaded data"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        
        try:
            # Collect preprocessing settings
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
            
            # Apply preprocessing using the same preprocessor as Model Tester
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
            
            # Update variable lists if data was preprocessed
            self.update_variable_lists()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error applying preprocessing: {str(e)}")
            self.preproc_status_label.config(text="Error in preprocessing", foreground='red')
    
    def update_filter_info(self):
        """Update the filter information label"""
        filter_type = self.filter_type.get()
        try:
            threshold = float(self.threshold_var.get())
            self.target_threshold = threshold
        except:
            threshold = self.target_threshold
        
        if filter_type == "ALL":
            self.filter_info.config(text="Filter: ALL data points")
        elif filter_type == "UP":
            self.filter_info.config(text=f"Filter: UP movements > {threshold:.3f}")
        elif filter_type == "DOWN":
            self.filter_info.config(text=f"Filter: DOWN movements < -{threshold:.3f}")
    
    def apply_target_filter(self, df, target_columns):
        """Apply filtering to target columns based on filter settings"""
        filter_type = self.filter_type.get()
        
        if filter_type == "ALL":
            self.status_var.set(f"Using ALL data: {len(df)} rows")
            return df
        
        try:
            threshold = float(self.threshold_var.get())
        except:
            threshold = self.target_threshold
        
        # Create a mask for filtering based on ANY target column meeting criteria
        # This ensures we get meaningful data for regression
        mask = pd.Series([False] * len(df))
        
        # Determine which columns to use based on data type selection
        columns_to_check = []
        for col in target_columns:
            if self.use_normalized_data:
                # Check for normalized version first
                vol_adj_col = f'{col}_vol_adj'
                if vol_adj_col in df.columns:
                    columns_to_check.append(vol_adj_col)
                else:
                    columns_to_check.append(col)
            else:
                columns_to_check.append(col)
        
        # Apply filter - use OR logic across all target columns
        for col in columns_to_check:
            if col in df.columns:
                col_data = pd.to_numeric(df[col], errors='coerce')
                if filter_type == "UP":
                    mask = mask | (col_data > threshold)
                elif filter_type == "DOWN":
                    mask = mask | (col_data < -threshold)
        
        # If no target columns or all NaN, return original
        if not mask.any():
            messagebox.showwarning("Filter Warning", 
                                  f"No data points meet the {filter_type} filter criteria (threshold={threshold})")
            return df
        
        filtered_df = df[mask].copy()
        
        # Report filtering statistics
        original_count = len(df)
        filtered_count = len(filtered_df)
        percent_retained = (filtered_count / original_count * 100) if original_count > 0 else 0
        
        # Detailed status message
        if filter_type == "UP":
            filter_desc = f"target > {threshold}"
        else:
            filter_desc = f"target < -{threshold}"
        
        self.status_var.set(f"Filter {filter_type} ({filter_desc}): {filtered_count}/{original_count} rows ({percent_retained:.1f}%)")
        
        # Check if we have enough data for meaningful regression
        if filtered_count < 30:
            messagebox.showwarning("Low Data Warning", 
                                  f"Only {filtered_count} data points after filtering. Results may be unreliable.")
        
        return filtered_df
    
    def load_data(self):
        """Load data from CSV file with flexible date parsing"""
        try:
            filename = self.file_entry.get()
            self.df = pd.read_csv(filename)
            
            # Try to load from config first
            config = configparser.ConfigParser()
            config_loaded = False
            
            if config.read('OMtree_config.ini'):
                # Load threshold
                if 'model' in config and 'target_threshold' in config['model']:
                    self.target_threshold = float(config['model']['target_threshold'])
                    self.threshold_var.set(str(self.target_threshold))
                
                # Parse dates using flexible parser
                date_columns = FlexibleDateParser.get_date_columns(self.df)
                
                # Get date/time column info from config or auto-detect
                date_col = config['data'].get('date_column', date_columns.get('date_column'))
                time_col = config['data'].get('time_column', date_columns.get('time_column'))
                datetime_col = date_columns.get('datetime_column')
                
                # Check dayfirst setting
                dayfirst = None
                if 'validation' in config and 'date_format_dayfirst' in config['validation']:
                    dayfirst_str = config['validation']['date_format_dayfirst'].lower()
                    if dayfirst_str == 'true':
                        dayfirst = True
                    elif dayfirst_str == 'false':
                        dayfirst = False
                
                # Try to parse dates
                try:
                    parsed_dates = FlexibleDateParser.parse_dates(
                        self.df,
                        date_column=date_col,
                        time_column=time_col,
                        datetime_column=datetime_col,
                        dayfirst=dayfirst
                    )
                    
                    # Add parsed dates to dataframe
                    self.df['parsed_datetime'] = parsed_dates
                    
                    # Check for validation_end_date and filter data
                    if 'validation' in config and 'validation_end_date' in config['validation']:
                        validation_end_date = pd.to_datetime(config['validation']['validation_end_date'])
                        
                        original_len = len(self.df)
                        self.df = self.df[self.df['parsed_datetime'] <= validation_end_date].copy()
                        filtered_len = len(self.df)
                        if original_len > filtered_len:
                            messagebox.showinfo("Data Filtered", 
                                              f"Data filtered to validation_end_date ({validation_end_date.strftime('%Y-%m-%d')})\n"
                                              f"Using {filtered_len} of {original_len} rows\n"
                                              f"({original_len - filtered_len} rows reserved for out-of-sample)")
                
                except Exception as e:
                    print(f"Could not parse dates: {e}")
                    # Continue without date filtering
                
                # Try to categorize columns from config
                if 'data' in config:
                    features = []
                    targets = []
                    
                    # Get all target columns
                    if 'all_targets' in config['data']:
                        targets = [t.strip() for t in config['data']['all_targets'].split(',')]
                    elif 'target_column' in config['data']:
                        targets = [config['data']['target_column']]
                    
                    # Get feature columns - prioritize feature_columns over selected_features
                    if 'feature_columns' in config['data']:
                        features = [f.strip() for f in config['data']['feature_columns'].split(',')]
                    elif 'selected_features' in config['data']:
                        # Note: selected_features is typically a subset used for model training
                        # For regression analysis, we want all features if possible
                        features = [f.strip() for f in config['data']['selected_features'].split(',')]
                    
                    if features or targets:
                        config_loaded = True
                        # Store for later use
                        self.feature_columns = [col for col in features if col in self.df.columns]
                        self.target_columns = [col for col in targets if col in self.df.columns]
                        
                        # Also check for volatility-adjusted versions
                        self.feature_columns_vol = []
                        self.target_columns_vol = []
                        for col in self.feature_columns:
                            vol_col = f'{col}_vol_adj'
                            if vol_col in self.df.columns:
                                self.feature_columns_vol.append(vol_col)
                        for col in self.target_columns:
                            vol_col = f'{col}_vol_adj'
                            if vol_col in self.df.columns:
                                self.target_columns_vol.append(vol_col)
            
            if not config_loaded or not self.feature_columns:
                # Use ColumnDetector for better auto-detection
                print("Using ColumnDetector for auto-detection...")
                detected = ColumnDetector.auto_detect_columns(self.df)
                self.feature_columns = detected['features']
                self.target_columns = detected['targets']
                
                # Check for volatility-adjusted versions
                self.feature_columns_vol = []
                self.target_columns_vol = []
                for col in self.feature_columns:
                    vol_col = f'{col}_vol_adj'
                    if vol_col in self.df.columns:
                        self.feature_columns_vol.append(vol_col)
                for col in self.target_columns:
                    vol_col = f'{col}_vol_adj'
                    if vol_col in self.df.columns:
                        self.target_columns_vol.append(vol_col)
            
            # Update ticker dropdown if Ticker column exists
            if 'Ticker' in self.df.columns:
                unique_tickers = sorted(self.df['Ticker'].unique())
                self.ticker_combo['values'] = unique_tickers
                if unique_tickers:
                    self.ticker_combo.set(unique_tickers[0])
            
            # Keep a copy of the original unfiltered data
            self.df_original = self.df.copy()
            
            # Update variable lists
            self.update_variable_lists()
            
            # Report what we found
            n_features = len(self.feature_columns)
            n_features_vol = len(self.feature_columns_vol)
            n_targets = len(self.target_columns)
            n_targets_vol = len(self.target_columns_vol)
            
            status_msg = f"Loaded {len(self.df)} rows with {n_features} features"
            if n_features_vol > 0:
                status_msg += f" ({n_features_vol} vol-adjusted)"
            status_msg += f" and {n_targets} targets"
            if n_targets_vol > 0:
                status_msg += f" ({n_targets_vol} vol-adjusted)"
            
            self.status_var.set(status_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Failed to load data")
    
    def toggle_ticker_filter(self):
        """Enable/disable ticker filter controls"""
        if self.ticker_filter_var.get():
            self.ticker_combo.config(state='readonly')
        else:
            self.ticker_combo.config(state='disabled')
    
    def toggle_hour_filter(self):
        """Enable/disable hour filter controls"""
        if self.hour_filter_var.get():
            self.hour_spin.config(state='normal')
        else:
            self.hour_spin.config(state='disabled')
    
    def apply_data_filters(self):
        """Apply ticker and hour filters to the data"""
        if not hasattr(self, 'df_original'):
            messagebox.showwarning("No Data", "Please load data first")
            return
        
        # Start with original data
        self.df = self.df_original.copy()
        
        # Apply ticker filter
        if self.ticker_filter_var.get() and 'Ticker' in self.df.columns:
            selected_ticker = self.ticker_combo.get()
            if selected_ticker:
                self.df = self.df[self.df['Ticker'] == selected_ticker].copy()
        
        # Apply hour filter
        if self.hour_filter_var.get() and 'Hour' in self.df.columns:
            selected_hour = self.hour_spin_var.get()
            self.df = self.df[self.df['Hour'] == selected_hour].copy()
        
        # Update status
        filter_info = []
        if self.ticker_filter_var.get() and 'Ticker' in self.df.columns:
            filter_info.append(f"Ticker={self.ticker_combo.get()}")
        if self.hour_filter_var.get() and 'Hour' in self.df.columns:
            filter_info.append(f"Hour={self.hour_spin_var.get()}")
        
        if filter_info:
            filter_text = f" (Filtered: {', '.join(filter_info)})"
        else:
            filter_text = ""
        
        # Update variable lists and status
        self.update_variable_lists()
        
        n_features = len(self.feature_columns) if hasattr(self, 'feature_columns') else 0
        n_targets = len(self.target_columns) if hasattr(self, 'target_columns') else 0
        
        status_msg = f"Data: {len(self.df)} rows{filter_text} with {n_features} features and {n_targets} targets"
        self.status_var.set(status_msg)
        
        # Clear any existing analysis
        if hasattr(self, 'results_text'):
            self.results_text.delete(1.0, tk.END)
    
    def update_variable_lists(self):
        """Update the X and Y variable lists based on analysis type and data type"""
        if not hasattr(self, 'feature_columns') or not hasattr(self, 'target_columns'):
            return
        
        analysis_type = self.analysis_type.get()
        
        # Clear existing lists
        self.x_listbox.delete(0, tk.END)
        self.y_listbox.delete(0, tk.END)
        
        # Use processed data columns if available, otherwise use available columns
        if self.df_processed is not None:
            # Get columns from processed data
            available_columns = list(self.df_processed.columns)
            features_to_show = [col for col in available_columns if col in self.feature_columns or any(col.startswith(base + '_') for base in self.feature_columns)]
            targets_to_show = [col for col in available_columns if col in self.target_columns or any(col.startswith(base + '_') for base in self.target_columns)]
        elif self.use_normalized_data and len(self.feature_columns_vol) > 0:
            # Use volatility-adjusted columns if available
            features_to_show = self.feature_columns_vol if self.feature_columns_vol else self.feature_columns
            targets_to_show = self.target_columns_vol if self.target_columns_vol else self.target_columns
        else:
            # Use raw columns
            features_to_show = self.feature_columns
            targets_to_show = self.target_columns
        
        if analysis_type == "features_to_targets":
            # X = features, Y = targets
            for col in features_to_show:
                self.x_listbox.insert(tk.END, col)
            for col in targets_to_show:
                self.y_listbox.insert(tk.END, col)
            
            # Auto-select all
            self.select_all_x()
            self.select_all_y()
            
        elif analysis_type == "features_to_features":
            # Both X and Y = features
            for col in features_to_show:
                self.x_listbox.insert(tk.END, col)
                self.y_listbox.insert(tk.END, col)
            
            # Auto-select all
            self.select_all_x()
            self.select_all_y()
            
        elif analysis_type == "targets_to_targets":
            # Both X and Y = targets
            for col in targets_to_show:
                self.x_listbox.insert(tk.END, col)
                self.y_listbox.insert(tk.END, col)
            
            # Auto-select all
            self.select_all_x()
            self.select_all_y()
            
        elif analysis_type == "custom":
            # All columns available for both
            all_cols = features_to_show + targets_to_show
            for col in all_cols:
                self.x_listbox.insert(tk.END, col)
                self.y_listbox.insert(tk.END, col)
        
        self.update_selection_info()
    
    def select_all_x(self):
        """Select all X variables"""
        self.x_listbox.selection_set(0, tk.END)
        self.update_selection_info()
    
    def deselect_all_x(self):
        """Deselect all X variables"""
        self.x_listbox.selection_clear(0, tk.END)
        self.update_selection_info()
    
    def select_all_y(self):
        """Select all Y variables"""
        self.y_listbox.selection_set(0, tk.END)
        self.update_selection_info()
    
    def deselect_all_y(self):
        """Deselect all Y variables"""
        self.y_listbox.selection_clear(0, tk.END)
        self.update_selection_info()
    
    def update_selection_info(self):
        """Update the selection information label"""
        x_selected = len(self.x_listbox.curselection())
        y_selected = len(self.y_listbox.curselection())
        self.selection_info.config(text=f"Selected: {x_selected} X variables, {y_selected} Y variables")
    
    def run_regression(self):
        """Run regression analysis with filtering"""
        if self.df is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        # Use processed data if available, otherwise use raw data
        working_df = self.df_processed if self.df_processed is not None else self.df
        
        # Get selected variables
        x_indices = self.x_listbox.curselection()
        y_indices = self.y_listbox.curselection()
        
        if not x_indices or not y_indices:
            messagebox.showerror("Error", "Please select both X and Y variables")
            return
        
        self.selected_x_vars = [self.x_listbox.get(i) for i in x_indices]
        self.selected_y_vars = [self.y_listbox.get(i) for i in y_indices]
        
        # Apply target filtering
        self.df_filtered = self.apply_target_filter(working_df, self.selected_y_vars)
        
        if len(self.df_filtered) < 10:
            messagebox.showerror("Error", f"Too few data points after filtering ({len(self.df_filtered)}). Try adjusting filter settings.")
            return
        
        # Create results matrix
        results_matrix = pd.DataFrame(index=self.selected_y_vars, columns=self.selected_x_vars)
        p_value_matrix = pd.DataFrame(index=self.selected_y_vars, columns=self.selected_x_vars)
        
        # Summary statistics
        summary_stats = []
        summary_stats.append("="*60)
        summary_stats.append("REGRESSION ANALYSIS RESULTS")
        summary_stats.append("="*60)
        summary_stats.append(f"\nData Type: {'VOLATILITY NORMALIZED' if self.use_normalized_data else 'RAW DATA'}")
        summary_stats.append(f"Filter Type: {self.filter_type.get()}")
        if self.filter_type.get() != "ALL":
            summary_stats.append(f"Threshold: {self.threshold_var.get()}")
        summary_stats.append(f"Data Points: {len(self.df_filtered)} / {len(self.df)} ({len(self.df_filtered)/len(self.df)*100:.1f}%)")
        summary_stats.append(f"\nX Variables: {', '.join(self.selected_x_vars)}")
        summary_stats.append(f"Y Variables: {', '.join(self.selected_y_vars)}")
        summary_stats.append("\n" + "-"*60)
        
        # Track actual observations used for each regression
        obs_count_matrix = pd.DataFrame(index=self.selected_y_vars, columns=self.selected_x_vars)
        
        # Run regressions
        for y_var in self.selected_y_vars:
            for x_var in self.selected_x_vars:
                try:
                    # Get data from filtered dataframe
                    x_data = pd.to_numeric(self.df_filtered[x_var], errors='coerce')
                    y_data = pd.to_numeric(self.df_filtered[y_var], errors='coerce')
                    
                    # Remove NaN values
                    valid_mask = ~(x_data.isna() | y_data.isna())
                    x_clean = x_data[valid_mask].values.reshape(-1, 1)
                    y_clean = y_data[valid_mask].values
                    
                    # Store observation count
                    obs_count_matrix.loc[y_var, x_var] = len(x_clean)
                    
                    # Need at least 3 points for meaningful regression
                    if len(x_clean) > 2:
                        # Fit regression
                        model = LinearRegression()
                        model.fit(x_clean, y_clean)
                        
                        # Calculate R²
                        r2 = r2_score(y_clean, model.predict(x_clean))
                        results_matrix.loc[y_var, x_var] = r2
                        
                        # Calculate p-value using scipy.stats for accuracy
                        _, _, _, p_value, _ = stats.linregress(x_clean.flatten(), y_clean)
                        p_value_matrix.loc[y_var, x_var] = p_value
                    else:
                        results_matrix.loc[y_var, x_var] = np.nan
                        p_value_matrix.loc[y_var, x_var] = np.nan
                        
                except Exception as e:
                    print(f"Error in regression {x_var} -> {y_var}: {str(e)}")
                    results_matrix.loc[y_var, x_var] = np.nan
                    p_value_matrix.loc[y_var, x_var] = np.nan
                    obs_count_matrix.loc[y_var, x_var] = 0
        
        # Add summary for each Y variable
        for y_var in self.selected_y_vars:
            summary_stats.append(f"\n{y_var}:")
            for x_var in self.selected_x_vars:
                r2_val = results_matrix.loc[y_var, x_var]
                p_val = p_value_matrix.loc[y_var, x_var]
                obs_count = obs_count_matrix.loc[y_var, x_var]
                if not pd.isna(r2_val):
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    summary_stats.append(f"  vs {x_var}: R²={r2_val:.4f}, p={p_val:.4f} {sig} (n={obs_count})")
                else:
                    summary_stats.append(f"  vs {x_var}: No valid data (n={obs_count})")
        
        # Add observation count summary
        summary_stats.append("\n" + "-"*60)
        summary_stats.append("OBSERVATION COUNTS:")
        min_obs = obs_count_matrix.min().min()
        max_obs = obs_count_matrix.max().max()
        mean_obs = obs_count_matrix.mean().mean()
        summary_stats.append(f"  Min: {min_obs:.0f}, Max: {max_obs:.0f}, Mean: {mean_obs:.1f}")
        
        # Display results
        self.display_results(results_matrix, p_value_matrix, summary_stats)
    
    def display_results(self, results_matrix, p_value_matrix, summary_stats):
        """Display regression results with 4-chart layout"""
        # Clear previous results
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        
        # Create figure with 4 subplots
        fig = plt.Figure(figsize=(14, 10), tight_layout=True)
        
        # Get filter information for titles
        filter_info = f" [{self.filter_type.get()} filter"
        if self.filter_type.get() != "ALL":
            filter_info += f", threshold={self.threshold_var.get()}"
        filter_info += "]"
        
        # Convert to numeric for plotting
        r2_matrix = results_matrix.astype(float)
        pval_matrix = p_value_matrix.astype(float)
        
        # Calculate coefficient matrix and correlation matrix
        coef_matrix = pd.DataFrame(index=results_matrix.index, columns=results_matrix.columns)
        corr_matrix = pd.DataFrame(index=results_matrix.index, columns=results_matrix.columns)
        
        for y_var in self.selected_y_vars:
            for x_var in self.selected_x_vars:
                try:
                    x_data = pd.to_numeric(self.df_filtered[x_var], errors='coerce')
                    y_data = pd.to_numeric(self.df_filtered[y_var], errors='coerce')
                    valid_mask = ~(x_data.isna() | y_data.isna())
                    
                    if valid_mask.sum() > 1:
                        # Linear regression for coefficient
                        x_clean = x_data[valid_mask].values.reshape(-1, 1)
                        y_clean = y_data[valid_mask].values
                        model = LinearRegression()
                        model.fit(x_clean, y_clean)
                        coef_matrix.loc[y_var, x_var] = model.coef_[0]
                        
                        # Correlation
                        corr_matrix.loc[y_var, x_var] = np.corrcoef(x_clean.flatten(), y_clean)[0, 1]
                    else:
                        coef_matrix.loc[y_var, x_var] = np.nan
                        corr_matrix.loc[y_var, x_var] = np.nan
                except:
                    coef_matrix.loc[y_var, x_var] = np.nan
                    corr_matrix.loc[y_var, x_var] = np.nan
        
        # Chart 1: R² Values with significance stars
        ax1 = fig.add_subplot(2, 2, 1)
        
        # Create annotation matrix with significance stars
        annot_r2 = pd.DataFrame(index=r2_matrix.index, columns=r2_matrix.columns)
        for i in r2_matrix.index:
            for j in r2_matrix.columns:
                r2_val = r2_matrix.loc[i, j]
                p_val = pval_matrix.loc[i, j]
                if pd.isna(r2_val):
                    annot_r2.loc[i, j] = 'NaN'
                else:
                    sig = ''
                    if not pd.isna(p_val):
                        if p_val < 0.001:
                            sig = '***'
                        elif p_val < 0.01:
                            sig = '**'
                        elif p_val < 0.05:
                            sig = '*'
                    annot_r2.loc[i, j] = f'{r2_val:.3f}{sig}'
        
        sns.heatmap(r2_matrix, annot=annot_r2, fmt='', cmap='RdYlGn',
                   xticklabels=results_matrix.columns,
                   yticklabels=results_matrix.index,
                   cbar_kws={'label': 'R² Score'},
                   ax=ax1, vmin=0, vmax=1,
                   annot_kws={'size': 6})
        ax1.set_title(f'R² Values{filter_info}', fontweight='bold', fontsize=10)
        ax1.set_xlabel('X Variables (Independent)', fontsize=9)
        ax1.set_ylabel('Y Variables (Dependent)', fontsize=9)
        
        # Chart 2: Regression Coefficients
        ax2 = fig.add_subplot(2, 2, 2)
        coef_numeric = coef_matrix.astype(float)
        # Use diverging colormap for coefficients
        vmax = np.nanmax(np.abs(coef_numeric.values)) if not np.isnan(coef_numeric.values).all() else 1
        sns.heatmap(coef_numeric, annot=True, fmt='.3f', cmap='RdBu_r',
                   xticklabels=coef_matrix.columns,
                   yticklabels=coef_matrix.index,
                   cbar_kws={'label': 'Coefficient'},
                   ax=ax2, center=0, vmin=-vmax, vmax=vmax,
                   annot_kws={'size': 7})
        ax2.set_title(f'Regression Coefficients{filter_info}', fontweight='bold', fontsize=10)
        ax2.set_xlabel('X Variables (Independent)', fontsize=9)
        ax2.set_ylabel('Y Variables (Dependent)', fontsize=9)
        
        # Chart 3: P-Values with confidence level classification (1, 2, 3)
        ax3 = fig.add_subplot(2, 2, 3)
        
        # Create confidence level matrix (1, 2, 3) instead of actual p-values
        confidence_matrix = pd.DataFrame(index=pval_matrix.index, columns=pval_matrix.columns)
        annot_conf = pd.DataFrame(index=pval_matrix.index, columns=pval_matrix.columns)
        
        for i in pval_matrix.index:
            for j in pval_matrix.columns:
                p = pval_matrix.loc[i, j]
                if pd.isna(p):
                    confidence_matrix.loc[i, j] = np.nan
                    annot_conf.loc[i, j] = ''
                else:
                    if p < 0.001:
                        confidence_matrix.loc[i, j] = 3  # Highest confidence
                        annot_conf.loc[i, j] = '3'
                    elif p < 0.01:
                        confidence_matrix.loc[i, j] = 2  # Medium confidence
                        annot_conf.loc[i, j] = '2'
                    elif p < 0.05:
                        confidence_matrix.loc[i, j] = 1  # Low confidence
                        annot_conf.loc[i, j] = '1'
                    else:
                        confidence_matrix.loc[i, j] = 0  # Not significant
                        annot_conf.loc[i, j] = ''
        
        # Convert to numeric for proper coloring
        conf_numeric = confidence_matrix.astype(float)
        
        # Use discrete color levels for confidence
        sns.heatmap(conf_numeric, annot=annot_conf, fmt='', cmap='RdYlGn',
                   xticklabels=pval_matrix.columns,
                   yticklabels=pval_matrix.index,
                   cbar_kws={'label': 'Confidence Level'},
                   ax=ax3, vmin=0, vmax=3,
                   annot_kws={'size': 14, 'weight': 'bold'},
                   cbar=True)
        
        ax3.set_title(f'Confidence Levels (3: p<0.001, 2: p<0.01, 1: p<0.05){filter_info}', fontweight='bold', fontsize=10)
        ax3.set_xlabel('X Variables (Independent)', fontsize=9)
        ax3.set_ylabel('Y Variables (Dependent)', fontsize=9)
        
        # Chart 4: Correlation Matrix
        ax4 = fig.add_subplot(2, 2, 4)
        corr_numeric = corr_matrix.astype(float)
        sns.heatmap(corr_numeric, annot=True, fmt='.3f', cmap='coolwarm',
                   xticklabels=corr_matrix.columns,
                   yticklabels=corr_matrix.index,
                   cbar_kws={'label': 'Correlation'},
                   ax=ax4, center=0, vmin=-1, vmax=1,
                   annot_kws={'size': 7})
        ax4.set_title(f'Correlation Coefficients{filter_info}', fontweight='bold', fontsize=10)
        ax4.set_xlabel('X Variables (Independent)', fontsize=9)
        ax4.set_ylabel('Y Variables (Dependent)', fontsize=9)
        
        # Adjust layout
        fig.suptitle(f'Regression Analysis Results - {len(self.df_filtered)}/{len(self.df)} data points ({len(self.df_filtered)/len(self.df)*100:.1f}%)', 
                    fontsize=12, fontweight='bold', y=1.02)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.matrix_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Update summary text
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, '\n'.join(summary_stats))
        
        # Add legend at the bottom
        legend_text = "\n\n" + "="*60
        legend_text += "\nConfidence Levels: 3 = p<0.001 (highest), 2 = p<0.01 (medium), 1 = p<0.05 (low)"
        legend_text += "\nR² interpretation: 0=no correlation, 1=perfect correlation"
        legend_text += "\nData Type: " + ("Volatility Normalized" if self.use_normalized_data else "Raw Data")
        legend_text += "\n" + "="*60
        self.summary_text.insert(tk.END, legend_text)
        
        self.status_var.set("Regression analysis complete")