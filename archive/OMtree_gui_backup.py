import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import configparser
import subprocess
import threading
import pandas as pd
import numpy as np
import os
from datetime import datetime
from PIL import Image, ImageTk
import json

# Import the regression analysis module
from regression_gui_module import RegressionAnalysisTab
# Import the timeline visualization component
from timeline_component import TimelineVisualization

class OMtreeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OMtree Trading Model - Enhanced Configuration & Analysis")
        self.root.geometry("1500x900")
        
        # Store config file path
        self.config_file = 'OMtree_config.ini'
        self.df = None
        self.available_columns = []
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.config_tab = ttk.Frame(self.notebook)
        self.run_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.charts_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text='Data & Fields')
        self.notebook.add(self.config_tab, text='Model Configuration')
        self.notebook.add(self.run_tab, text='Run Validation')
        self.notebook.add(self.results_tab, text='Performance Stats')
        self.notebook.add(self.charts_tab, text='Charts')
        
        # Add regression analysis tab
        self.regression_analysis = RegressionAnalysisTab(self.notebook)
        
        # Initialize tabs
        self.setup_data_tab()
        self.setup_config_tab()
        self.setup_run_tab()
        self.setup_results_tab()
        self.setup_charts_tab()
        
        # Load initial config
        self.load_config()
        
        # Process tracking
        self.validation_process = None
        
    def setup_data_tab(self):
        """Setup the data loading and field selection tab"""
        main_frame = ttk.Frame(self.data_tab, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Data File Selection", padding=10)
        file_frame.pack(fill='x', pady=(0, 10))
        
        file_input_frame = ttk.Frame(file_frame)
        file_input_frame.pack(fill='x')
        
        ttk.Label(file_input_frame, text="CSV File:").pack(side='left')
        self.data_file_var = tk.StringVar(value="DTSmlDATA7x7.csv")
        self.data_file_entry = ttk.Entry(file_input_frame, textvariable=self.data_file_var, width=40)
        self.data_file_entry.pack(side='left', padx=5)
        
        ttk.Button(file_input_frame, text="Browse", command=self.browse_data_file).pack(side='left', padx=2)
        ttk.Button(file_input_frame, text="Load Data", command=self.load_data_file).pack(side='left', padx=5)
        
        # Data preview
        preview_frame = ttk.LabelFrame(file_frame, text="Data Preview", padding=5)
        preview_frame.pack(fill='x', pady=(10, 0))
        
        self.data_info_label = ttk.Label(preview_frame, text="No data loaded")
        self.data_info_label.pack(anchor='w')
        
        # CRITICAL VALIDATION PERIOD SETTINGS - PROMINENT DISPLAY
        validation_frame = ttk.LabelFrame(main_frame, text="⚠️ VALIDATION PERIOD - CONTROLS OUT-OF-SAMPLE DATA ⚠️", 
                                        padding=15)
        validation_frame.pack(fill='x', pady=(10, 10))
        validation_frame.configure(relief='raised', borderwidth=3)
        
        # Warning message
        warning_label = ttk.Label(validation_frame, 
                                text="⚠️ IMPORTANT: These dates control how much data is used for training vs out-of-sample testing",
                                font=('Arial', 11, 'bold'), foreground='red')
        warning_label.pack(pady=(0, 10))
        
        # Date controls frame
        dates_frame = ttk.Frame(validation_frame)
        dates_frame.pack(fill='x', pady=(0, 10))
        
        # Validation Start Date
        start_frame = ttk.Frame(dates_frame)
        start_frame.pack(side='left', padx=20)
        ttk.Label(start_frame, text="Validation Start Date:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.validation_start_var = tk.StringVar(value="2010-01-01")
        self.validation_start_entry = ttk.Entry(start_frame, textvariable=self.validation_start_var, 
                                               width=15, font=('Arial', 11, 'bold'))
        self.validation_start_entry.pack()
        ttk.Label(start_frame, text="(YYYY-MM-DD format)", font=('Arial', 8)).pack()
        
        # Validation End Date
        end_frame = ttk.Frame(dates_frame)
        end_frame.pack(side='left', padx=20)
        ttk.Label(end_frame, text="Validation End Date:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.validation_end_var = tk.StringVar(value="2023-12-31")
        self.validation_end_entry = ttk.Entry(end_frame, textvariable=self.validation_end_var, 
                                             width=15, font=('Arial', 11, 'bold'))
        self.validation_end_entry.pack()
        ttk.Label(end_frame, text="(YYYY-MM-DD format)", font=('Arial', 8)).pack()
        
        # Update button
        ttk.Button(dates_frame, text="Update Timeline", 
                  command=self.update_timeline, style='Accent.TButton').pack(side='left', padx=20)
        
        # Timeline visualization
        timeline_frame = ttk.Frame(validation_frame)
        timeline_frame.pack(fill='both', expand=True, pady=(10, 0))
        self.timeline = TimelineVisualization(timeline_frame)
        
        # Initialize with default dates
        self.data_start_date = None
        self.data_end_date = None
        
        # Field selection section
        field_frame = ttk.LabelFrame(main_frame, text="Field Selection", padding=10)
        field_frame.pack(fill='both', expand=True)
        
        # Create three columns: Available, Features, Targets
        columns_frame = ttk.Frame(field_frame)
        columns_frame.pack(fill='both', expand=True)
        
        # Available columns
        avail_frame = ttk.LabelFrame(columns_frame, text="Available Columns", padding=5)
        avail_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        avail_scroll = ttk.Scrollbar(avail_frame)
        avail_scroll.pack(side='right', fill='y')
        self.available_listbox = tk.Listbox(avail_frame, selectmode='multiple',
                                           yscrollcommand=avail_scroll.set, exportselection=False)
        self.available_listbox.pack(fill='both', expand=True)
        avail_scroll.config(command=self.available_listbox.yview)
        
        # Buttons column
        button_frame = ttk.Frame(columns_frame)
        button_frame.grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="→ Features →", 
                  command=self.move_to_features).pack(pady=5)
        ttk.Button(button_frame, text="← Remove ←", 
                  command=self.remove_from_features).pack(pady=5)
        ttk.Label(button_frame, text="").pack(pady=10)  # Spacer
        ttk.Button(button_frame, text="→ Targets →", 
                  command=self.move_to_targets).pack(pady=5)
        ttk.Button(button_frame, text="← Remove ←", 
                  command=self.remove_from_targets).pack(pady=5)
        
        # Feature columns
        feature_frame = ttk.LabelFrame(columns_frame, text="Feature Columns", padding=5)
        feature_frame.grid(row=0, column=2, sticky='nsew', padx=5)
        
        feature_scroll = ttk.Scrollbar(feature_frame)
        feature_scroll.pack(side='right', fill='y')
        self.feature_listbox = tk.Listbox(feature_frame, selectmode='multiple',
                                         yscrollcommand=feature_scroll.set, exportselection=False)
        self.feature_listbox.pack(fill='both', expand=True)
        feature_scroll.config(command=self.feature_listbox.yview)
        
        # Target columns
        target_frame = ttk.LabelFrame(columns_frame, text="Target Columns", padding=5)
        target_frame.grid(row=0, column=3, sticky='nsew', padx=(5, 0))
        
        target_scroll = ttk.Scrollbar(target_frame)
        target_scroll.pack(side='right', fill='y')
        self.target_listbox = tk.Listbox(target_frame, selectmode='multiple',
                                        yscrollcommand=target_scroll.set, exportselection=False)
        self.target_listbox.pack(fill='both', expand=True)
        target_scroll.config(command=self.target_listbox.yview)
        
        # Configure grid weights
        columns_frame.columnconfigure(0, weight=1)
        columns_frame.columnconfigure(2, weight=1)
        columns_frame.columnconfigure(3, weight=1)
        columns_frame.rowconfigure(0, weight=1)
        
        # Additional options
        options_frame = ttk.LabelFrame(main_frame, text="Column Settings", padding=10)
        options_frame.pack(fill='x', pady=(10, 0))
        
        date_frame = ttk.Frame(options_frame)
        date_frame.pack(fill='x')
        
        ttk.Label(date_frame, text="Date Column:").pack(side='left')
        self.date_column_var = tk.StringVar(value="Date")
        self.date_column_combo = ttk.Combobox(date_frame, textvariable=self.date_column_var, width=20)
        self.date_column_combo.pack(side='left', padx=5)
        
        ttk.Label(date_frame, text="Time Column (optional):").pack(side='left', padx=(20, 0))
        self.time_column_var = tk.StringVar(value="Time")
        self.time_column_combo = ttk.Combobox(date_frame, textvariable=self.time_column_var, width=20)
        self.time_column_combo.pack(side='left', padx=5)
        
        # Save configuration button
        ttk.Button(options_frame, text="Save Field Configuration", 
                  command=self.save_field_config, style='Accent.TButton').pack(pady=10)
        
        # Auto-detect button
        ttk.Button(options_frame, text="Auto-Detect Fields", 
                  command=self.auto_detect_fields).pack(pady=5)
    
    def browse_data_file(self):
        """Browse for data file"""
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.data_file_var.set(filename)
    
    def load_data_file(self):
        """Load the selected data file"""
        try:
            filename = self.data_file_var.get()
            self.df = pd.read_csv(filename)
            
            # Update available columns
            self.available_columns = list(self.df.columns)
            
            # Update listboxes
            self.available_listbox.delete(0, tk.END)
            for col in self.available_columns:
                self.available_listbox.insert(tk.END, col)
            
            # Update date/time column combos
            self.date_column_combo['values'] = self.available_columns
            self.time_column_combo['values'] = [''] + self.available_columns
            
            # Get date range from data
            date_col = self.date_column_var.get()
            if date_col in self.df.columns:
                self.df[date_col] = pd.to_datetime(self.df[date_col])
                self.data_start_date = self.df[date_col].min()
                self.data_end_date = self.df[date_col].max()
            else:
                # Try to find a date column
                for col in ['Date', 'date', 'DateTime', 'datetime']:
                    if col in self.df.columns:
                        self.df[col] = pd.to_datetime(self.df[col])
                        self.data_start_date = self.df[col].min()
                        self.data_end_date = self.df[col].max()
                        break
            
            # Update info label
            info_text = f"Loaded: {len(self.df)} rows × {len(self.df.columns)} columns"
            if self.data_start_date and self.data_end_date:
                info_text += f"\nDate range: {self.data_start_date.strftime('%Y-%m-%d')} to {self.data_end_date.strftime('%Y-%m-%d')}"
            self.data_info_label.config(text=info_text)
            
            # Update timeline visualization
            if self.data_start_date and self.data_end_date:
                self.update_timeline()
            
            # Try auto-detection
            self.auto_detect_fields()
            
            messagebox.showinfo("Success", f"Data loaded successfully!\n{info_text}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def auto_detect_fields(self):
        """Auto-detect features and targets based on column naming"""
        if not self.available_columns:
            return
        
        # Clear current selections
        self.feature_listbox.delete(0, tk.END)
        self.target_listbox.delete(0, tk.END)
        
        # Detect based on common patterns
        for col in self.available_columns:
            col_lower = col.lower()
            
            # Skip date/time columns
            if any(x in col_lower for x in ['date', 'time', 'ticker', 'symbol']):
                continue
            
            # Check for target patterns
            if col.startswith('Ret_fwd') or 'forward' in col_lower or 'target' in col_lower:
                self.target_listbox.insert(tk.END, col)
            # Check for feature patterns
            elif (col.startswith('Ret_') and 'fwd' not in col) or any(x in col_lower for x in ['feature', 'input', 'predictor']):
                self.feature_listbox.insert(tk.END, col)
            # If numeric and no clear pattern, consider as potential feature
            elif self.df[col].dtype in [np.float64, np.int64]:
                self.feature_listbox.insert(tk.END, col)
    
    def move_to_features(self):
        """Move selected columns to features"""
        selected = self.available_listbox.curselection()
        for idx in reversed(selected):
            col = self.available_listbox.get(idx)
            # Check if not already in features
            if col not in self.feature_listbox.get(0, tk.END):
                self.feature_listbox.insert(tk.END, col)
    
    def remove_from_features(self):
        """Remove selected columns from features"""
        selected = self.feature_listbox.curselection()
        for idx in reversed(selected):
            self.feature_listbox.delete(idx)
    
    def move_to_targets(self):
        """Move selected columns to targets"""
        selected = self.available_listbox.curselection()
        for idx in reversed(selected):
            col = self.available_listbox.get(idx)
            # Check if not already in targets
            if col not in self.target_listbox.get(0, tk.END):
                self.target_listbox.insert(tk.END, col)
    
    def remove_from_targets(self):
        """Remove selected columns from targets"""
        selected = self.target_listbox.curselection()
        for idx in reversed(selected):
            self.target_listbox.delete(idx)
    
    def update_timeline(self):
        """Update the timeline visualization with current dates"""
        if not self.data_start_date or not self.data_end_date:
            messagebox.showwarning("Warning", "Please load data first to see the timeline")
            return
        
        try:
            validation_start = pd.to_datetime(self.validation_start_var.get())
            validation_end = pd.to_datetime(self.validation_end_var.get())
            
            # Validate dates
            if validation_start >= validation_end:
                messagebox.showerror("Error", "Validation end date must be after start date")
                return
            
            if validation_end > self.data_end_date:
                messagebox.showwarning("Warning", 
                                      f"Validation end date ({validation_end.strftime('%Y-%m-%d')}) is after data end date ({self.data_end_date.strftime('%Y-%m-%d')})")
            
            # Update timeline visualization
            train_days, test_days, val_days = self.timeline.update_timeline(
                self.data_start_date, 
                self.data_end_date,
                validation_start,
                validation_end
            )
            
            # Update info label with statistics
            info_text = f"Training: {train_days:,} days | Out-of-Sample: {test_days:,} days"
            if test_days > 0:
                oos_pct = (test_days / (train_days + test_days)) * 100
                info_text += f" ({oos_pct:.1f}% OOS)"
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update timeline: {str(e)}")
    
    def save_field_config(self):
        """Save the field configuration to config file"""
        try:
            # Get selected features and targets
            features = list(self.feature_listbox.get(0, tk.END))
            targets = list(self.target_listbox.get(0, tk.END))
            
            if not features:
                messagebox.showwarning("Warning", "Please select at least one feature column")
                return
            
            if not targets:
                messagebox.showwarning("Warning", "Please select at least one target column")
                return
            
            # Update config file
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            # Update data section
            if 'data' not in config:
                config['data'] = {}
            
            config['data']['csv_file'] = self.data_file_var.get()
            config['data']['feature_columns'] = ','.join(features)
            config['data']['selected_features'] = ','.join(features)
            config['data']['target_column'] = targets[0]  # Primary target
            config['data']['all_targets'] = ','.join(targets)  # All targets
            config['data']['date_column'] = self.date_column_var.get()
            
            if self.time_column_var.get():
                config['data']['time_column'] = self.time_column_var.get()
            
            # Update validation section
            if 'validation' not in config:
                config['validation'] = {}
            
            config['validation']['validation_start_date'] = self.validation_start_var.get()
            config['validation']['validation_end_date'] = self.validation_end_var.get()
            
            # Save config
            with open(self.config_file, 'w') as f:
                config.write(f)
            
            messagebox.showinfo("Success", 
                              f"Configuration saved!\n"
                              f"Features: {len(features)} columns\n"
                              f"Targets: {len(targets)} columns")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def setup_config_tab(self):
        """Setup the configuration editor tab"""
        # Create main frame with scrollbar
        canvas = tk.Canvas(self.config_tab)
        scrollbar = ttk.Scrollbar(self.config_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Store config widgets
        self.config_widgets = {}
        
        # Define config structure
        self.config_structure = {
            'Preprocessing': {
                'section': 'preprocessing',
                'fields': [
                    ('normalize_features', 'Normalize Features', 'Apply volatility normalization', 'combo', ['true', 'false']),
                    ('normalize_target', 'Normalize Target', 'Apply target normalization', 'combo', ['true', 'false']),
                    ('vol_window', 'Volatility Window', 'Rolling window for volatility (days)', 'spinbox', (10, 500, 10)),
                    ('smoothing_type', 'Smoothing Type', 'Volatility smoothing method', 'combo', ['exponential', 'linear', 'none']),
                    ('smoothing_alpha', 'Smoothing Alpha', 'Exponential smoothing factor', 'spinbox', (0.01, 1.0, 0.01)),
                ]
            },
            'Model Parameters': {
                'section': 'model',
                'fields': [
                    ('model_type', 'Model Type', 'Trading direction', 'combo', ['longonly', 'shortonly', 'both']),
                    ('n_trees', 'Number of Trees', 'Trees in ensemble', 'spinbox', (10, 1000, 10)),
                    ('max_depth', 'Max Depth', 'Maximum tree depth', 'spinbox', (1, 10, 1)),
                    ('bootstrap_fraction', 'Bootstrap Fraction', 'Data fraction per tree', 'spinbox', (0.1, 1.0, 0.05)),
                    ('min_leaf_fraction', 'Min Leaf Fraction', 'Min samples in leaf', 'spinbox', (0.01, 0.5, 0.01)),
                    ('target_threshold', 'Target Threshold', 'Threshold for profitable trade', 'spinbox', (0.0, 0.5, 0.01)),
                    ('vote_threshold', 'Vote Threshold', 'Fraction of trees to vote', 'spinbox', (0.5, 1.0, 0.05)),
                ]
            },
            'Validation': {
                'section': 'validation',
                'fields': [
                    ('train_size', 'Training Size', 'Training window size', 'spinbox', (100, 5000, 100)),
                    ('test_size', 'Test Size', 'Test window size', 'spinbox', (10, 500, 10)),
                    ('step_size', 'Step Size', 'Walk-forward step', 'spinbox', (1, 100, 5)),
                    ('base_rate', 'Base Rate', 'Baseline success rate', 'spinbox', (0.0, 1.0, 0.01)),
                ]
            }
        }
        
        # Create config sections
        row = 0
        for section_name, section_info in self.config_structure.items():
            frame = ttk.LabelFrame(scrollable_frame, text=section_name, padding=10)
            frame.grid(row=row, column=0, sticky='ew', padx=10, pady=5)
            
            for i, (key, label, tooltip, widget_type, options) in enumerate(section_info['fields']):
                ttk.Label(frame, text=label + ':').grid(row=i, column=0, sticky='w', padx=5, pady=2)
                
                if widget_type == 'combo':
                    widget = ttk.Combobox(frame, values=options, width=20)
                elif widget_type == 'spinbox':
                    min_val, max_val, increment = options
                    widget = ttk.Spinbox(frame, from_=min_val, to=max_val, 
                                        increment=increment, width=20)
                else:  # entry
                    widget = ttk.Entry(frame, width=30)
                
                widget.grid(row=i, column=1, padx=5, pady=2)
                self.config_widgets[f"{section_info['section']}.{key}"] = widget
            
            row += 1
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.grid(row=row, column=0, pady=10)
        
        ttk.Button(button_frame, text="Load Config", command=self.load_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_config).pack(side='left', padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_run_tab(self):
        """Setup the run validation tab"""
        # Create main frame
        main_frame = ttk.Frame(self.run_tab, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Validation Control", padding=10)
        control_frame.pack(fill='x')
        
        # Run button
        self.run_button = ttk.Button(control_frame, text="Run Validation", 
                                    command=self.run_validation, style='Accent.TButton')
        self.run_button.pack(side='left', padx=5)
        
        # Stop button
        self.stop_button = ttk.Button(control_frame, text="Stop", 
                                     command=self.stop_validation, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           mode='indeterminate', length=200)
        self.progress_bar.pack(side='left', padx=20)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side='left', padx=10)
        
        # Output console
        console_frame = ttk.LabelFrame(main_frame, text="Output Console", padding=10)
        console_frame.pack(fill='both', expand=True, pady=10)
        
        self.console_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, 
                                                      height=20, bg='black', fg='green')
        self.console_text.pack(fill='both', expand=True)
    
    def setup_results_tab(self):
        """Setup the results display tab"""
        # Create main frame
        main_frame = ttk.Frame(self.results_tab, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # Summary stats frame
        stats_frame = ttk.LabelFrame(main_frame, text="Performance Summary", padding=10)
        stats_frame.pack(fill='x')
        
        self.stats_text = tk.Text(stats_frame, height=10, wrap='word')
        self.stats_text.pack(fill='both', expand=True)
        
        # Detailed results frame
        results_frame = ttk.LabelFrame(main_frame, text="Detailed Results", padding=10)
        results_frame.pack(fill='both', expand=True, pady=10)
        
        # Create treeview for results
        columns = ('Metric', 'Value', 'Benchmark', 'Improvement')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings')
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150)
        
        self.results_tree.pack(fill='both', expand=True)
        
        # Refresh button
        ttk.Button(main_frame, text="Refresh Results", 
                  command=self.load_results).pack(pady=5)
    
    def setup_charts_tab(self):
        """Setup the charts display tab"""
        # Create main frame
        main_frame = ttk.Frame(self.charts_tab, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # Chart selection
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(fill='x')
        
        ttk.Label(chart_frame, text="Select Chart:").pack(side='left', padx=5)
        
        self.chart_var = tk.StringVar(value="cumulative_pnl")
        charts = [
            ("Cumulative P&L", "cumulative_pnl"),
            ("Accuracy Over Time", "accuracy_time"),
            ("Feature Importance", "feature_importance"),
            ("Prediction Distribution", "prediction_dist")
        ]
        
        for text, value in charts:
            ttk.Radiobutton(chart_frame, text=text, variable=self.chart_var, 
                          value=value).pack(side='left', padx=5)
        
        ttk.Button(chart_frame, text="Load Chart", 
                  command=self.load_chart).pack(side='left', padx=20)
        
        # Chart display area
        self.chart_label = ttk.Label(main_frame, text="No chart loaded")
        self.chart_label.pack(fill='both', expand=True, pady=10)
    
    def load_config(self):
        """Load configuration from file"""
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            for section_name, section_info in self.config_structure.items():
                section = section_info['section']
                if section in config:
                    for key, _, _, _, _ in section_info['fields']:
                        widget_key = f"{section}.{key}"
                        if widget_key in self.config_widgets and key in config[section]:
                            widget = self.config_widgets[widget_key]
                            value = config[section][key]
                            
                            if isinstance(widget, ttk.Combobox):
                                widget.set(value)
                            elif isinstance(widget, (ttk.Entry, ttk.Spinbox)):
                                widget.delete(0, tk.END)
                                widget.insert(0, value)
            
            # Load validation dates if present
            if 'validation' in config:
                if 'validation_start_date' in config['validation']:
                    self.validation_start_var.set(config['validation']['validation_start_date'])
                if 'validation_end_date' in config['validation']:
                    self.validation_end_var.set(config['validation']['validation_end_date'])
            
            self.console_text.insert(tk.END, "Configuration loaded successfully\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {str(e)}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            for section_name, section_info in self.config_structure.items():
                section = section_info['section']
                if section not in config:
                    config[section] = {}
                
                for key, _, _, _, _ in section_info['fields']:
                    widget_key = f"{section}.{key}"
                    if widget_key in self.config_widgets:
                        widget = self.config_widgets[widget_key]
                        config[section][key] = widget.get()
            
            with open(self.config_file, 'w') as f:
                config.write(f)
            
            messagebox.showinfo("Success", "Configuration saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {str(e)}")
    
    def reset_config(self):
        """Reset configuration to defaults"""
        if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
            # Set default values
            defaults = {
                'preprocessing.normalize_features': 'true',
                'preprocessing.normalize_target': 'true',
                'preprocessing.vol_window': '60',
                'preprocessing.smoothing_type': 'exponential',
                'preprocessing.smoothing_alpha': '0.1',
                'model.model_type': 'longonly',
                'model.n_trees': '200',
                'model.max_depth': '1',
                'model.bootstrap_fraction': '0.8',
                'model.min_leaf_fraction': '0.2',
                'model.target_threshold': '0.1',
                'model.vote_threshold': '0.7',
                'validation.train_size': '1000',
                'validation.test_size': '100',
                'validation.step_size': '50',
                'validation.base_rate': '0.42'
            }
            
            for key, value in defaults.items():
                if key in self.config_widgets:
                    widget = self.config_widgets[key]
                    if isinstance(widget, ttk.Combobox):
                        widget.set(value)
                    else:
                        widget.delete(0, tk.END)
                        widget.insert(0, value)
            
            messagebox.showinfo("Success", "Settings reset to defaults")
    
    def run_validation(self):
        """Run the validation process"""
        self.console_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.console_text.insert(tk.END, "Starting validation process...\n")
        self.console_text.see(tk.END)
        
        # Update UI
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar.start()
        self.status_label.config(text="Running...")
        
        # Start validation in separate thread
        thread = threading.Thread(target=self._run_validation_thread)
        thread.daemon = True
        thread.start()
    
    def _run_validation_thread(self):
        """Run validation in separate thread"""
        try:
            # Run the validation script
            process = subprocess.Popen(
                ['python', 'OMtree_walkforward.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.validation_process = process
            
            # Read output line by line
            for line in process.stdout:
                self.root.after(0, self._update_console, line)
            
            process.wait()
            
            if process.returncode == 0:
                self.root.after(0, self._validation_complete, True)
            else:
                self.root.after(0, self._validation_complete, False)
                
        except Exception as e:
            self.root.after(0, self._validation_error, str(e))
    
    def _update_console(self, text):
        """Update console with new text"""
        self.console_text.insert(tk.END, text)
        self.console_text.see(tk.END)
    
    def _validation_complete(self, success):
        """Handle validation completion"""
        self.progress_bar.stop()
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        if success:
            self.status_label.config(text="Completed successfully")
            self.console_text.insert(tk.END, "\nValidation completed successfully!\n")
            messagebox.showinfo("Success", "Validation completed successfully!")
            self.load_results()
        else:
            self.status_label.config(text="Failed")
            self.console_text.insert(tk.END, "\nValidation failed!\n")
    
    def _validation_error(self, error):
        """Handle validation error"""
        self.progress_bar.stop()
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Error")
        self.console_text.insert(tk.END, f"\nError: {error}\n")
        messagebox.showerror("Error", f"Validation failed: {error}")
    
    def stop_validation(self):
        """Stop the validation process"""
        if self.validation_process:
            self.validation_process.terminate()
            self.console_text.insert(tk.END, "\nValidation stopped by user\n")
            self.progress_bar.stop()
            self.run_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_label.config(text="Stopped")
    
    def load_results(self):
        """Load and display results"""
        try:
            if not os.path.exists('OMtree_results.csv'):
                self.stats_text.delete('1.0', tk.END)
                self.stats_text.insert('1.0', "No results file found")
                return
            
            df = pd.read_csv('OMtree_results.csv')
            
            # Calculate metrics from actual columns
            trades = df[df['prediction'] == 1]
            hit_rate = trades['actual_profitable'].mean() if len(trades) > 0 else 0
            total_trades = len(trades)
            total_observations = len(df)
            
            # Get model type from config
            config = configparser.ConfigParser()
            config.read('OMtree_config.ini')
            model_type = config['model']['model_type']
            base_rate = float(config['validation']['base_rate'])
            
            # Calculate P&L based on model type
            if len(trades) > 0:
                if model_type == 'longonly':
                    cumulative_pnl = trades['target_value'].sum()
                else:  # shortonly
                    cumulative_pnl = -trades['target_value'].sum()
            else:
                cumulative_pnl = 0
            
            edge = hit_rate - base_rate
            
            # Display summary stats
            self.stats_text.delete('1.0', tk.END)
            summary = f"Total Observations: {total_observations:,}\n"
            summary += f"Total Trades: {total_trades:,}\n"
            summary += f"Trading Frequency: {total_trades/total_observations:.1%}\n"
            summary += f"Hit Rate: {hit_rate:.2%}\n"
            summary += f"Base Rate: {base_rate:.2%}\n"
            summary += f"Edge: {edge:+.2%}\n"
            summary += f"Cumulative P&L: {cumulative_pnl:.2f}\n"
            summary += f"Avg Confidence: {trades['probability'].mean():.3f}\n" if len(trades) > 0 else ""
            self.stats_text.insert('1.0', summary)
            
            # Load performance metrics if available
            if os.path.exists('OMtree_performance.csv'):
                perf_df = pd.read_csv('OMtree_performance.csv')
                latest = perf_df.iloc[-1]
                
                # Clear tree
                self.results_tree.delete(*self.results_tree.get_children())
                
                # Add metrics from actual columns
                metrics = [
                    ('Hit Rate', f"{float(latest.get('hit_rate', 0)):.1%}", f"{base_rate:.1%}", 
                     f"{(float(latest.get('hit_rate', 0)) - base_rate) / base_rate * 100:.1f}%"),
                    ('Edge', f"{float(latest.get('edge', 0)):.1%}", '0%', 
                     f"+{float(latest.get('edge', 0)) * 100:.1f}%"),
                    ('Sharpe Ratio', f"{float(latest.get('sharpe_ratio', 0)):.2f}", '1.0', 
                     f"{(float(latest.get('sharpe_ratio', 0)) - 1.0) / 1.0 * 100:.1f}%"),
                    ('Positive Months', f"{float(latest.get('positive_months_pct', 0)):.1%}", '50%', 
                     f"{(float(latest.get('positive_months_pct', 0)) - 0.5) / 0.5 * 100:.1f}%"),
                    ('Total P&L', f"{float(latest.get('total_pnl', 0)):.2f}", '0', 
                     f"+{float(latest.get('total_pnl', 0)):.2f}"),
                ]
                
                for metric in metrics:
                    self.results_tree.insert('', 'end', values=metric)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results: {str(e)}")
    
    def load_chart(self):
        """Load and display selected chart"""
        chart_type = self.chart_var.get()
        
        # Get model type from config
        config = configparser.ConfigParser()
        config.read('OMtree_config.ini')
        model_type = config['model']['model_type']
        
        # Map chart types to actual generated files
        chart_files = {
            'cumulative_pnl': f'OMtree_comprehensive_{model_type}.png',
            'accuracy_time': f'OMtree_comprehensive_{model_type}.png',
            'feature_importance': f'OMtree_progression_{model_type}.png',
            'prediction_dist': f'OMtree_progression_{model_type}.png'
        }
        
        chart_file = chart_files.get(chart_type, f'OMtree_comprehensive_{model_type}.png')
        
        if os.path.exists(chart_file):
            try:
                img = Image.open(chart_file)
                # Scale image to fit while maintaining aspect ratio
                img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.chart_label.config(image=photo, text="")
                self.chart_label.image = photo  # Keep reference
            except Exception as e:
                self.chart_label.config(text=f"Error loading chart: {str(e)}")
        else:
            self.chart_label.config(text=f"Chart file not found: {chart_file}\nRun validation first to generate charts.")

def main():
    root = tk.Tk()
    app = OMtreeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()