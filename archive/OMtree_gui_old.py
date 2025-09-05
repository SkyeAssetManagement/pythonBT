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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# Import the regression analysis module
from regression_gui_module import RegressionAnalysisTab
# Import the timeline visualization component
from timeline_component import TimelineVisualization

class OMtreeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OMtree Trading Model - Enhanced Configuration & Analysis")
        self.root.geometry("1600x900")
        
        # Store config file path
        self.config_file = 'OMtree_config.ini'
        self.df = None
        self.available_columns = []
        
        # Create menu bar
        self.create_menu()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
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
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Configuration", command=self.save_all_config)
        file_menu.add_command(label="Save Configuration As...", command=self.save_config_as)
        file_menu.add_command(label="Load Configuration", command=self.load_config_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Regression Analysis", command=lambda: self.notebook.select(5))
        tools_menu.add_command(label="View Results", command=self.load_results)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
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
        self.validation_end_var = tk.StringVar(value="2015-12-31")
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
    
    def setup_config_tab(self):
        """Setup the configuration editor tab with feature/target selection"""
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
        
        # Feature/Target Selection Frame at the top
        selection_frame = ttk.LabelFrame(scrollable_frame, text="Model Features & Target Selection", padding=15)
        selection_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)
        
        # Feature selection
        feature_select_frame = ttk.Frame(selection_frame)
        feature_select_frame.pack(fill='x', pady=5)
        
        ttk.Label(feature_select_frame, text="Select Features for Model:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        # Feature checkboxes frame
        self.feature_check_frame = ttk.Frame(feature_select_frame)
        self.feature_check_frame.pack(fill='x', padx=20, pady=5)
        self.feature_check_vars = {}
        
        # Target selection
        target_select_frame = ttk.Frame(selection_frame)
        target_select_frame.pack(fill='x', pady=(10, 5))
        
        ttk.Label(target_select_frame, text="Select Target for Model:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        # Target radio buttons frame
        self.target_radio_frame = ttk.Frame(target_select_frame)
        self.target_radio_frame.pack(fill='x', padx=20, pady=5)
        self.selected_target_var = tk.StringVar()
        
        # Update button
        ttk.Button(selection_frame, text="Load Available Features/Targets", 
                  command=self.update_feature_target_selection).pack(pady=10)
        
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
        row = 1
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
    
    def update_feature_target_selection(self):
        """Update feature and target selection based on Data & Fields tab"""
        # Clear existing widgets
        for widget in self.feature_check_frame.winfo_children():
            widget.destroy()
        for widget in self.target_radio_frame.winfo_children():
            widget.destroy()
        
        # Get features from feature listbox
        features = list(self.feature_listbox.get(0, tk.END))
        targets = list(self.target_listbox.get(0, tk.END))
        
        if not features:
            ttk.Label(self.feature_check_frame, text="No features defined in Data & Fields tab").pack()
        else:
            # Create checkboxes for features
            for i, feature in enumerate(features):
                var = tk.BooleanVar(value=True)
                self.feature_check_vars[feature] = var
                cb = ttk.Checkbutton(self.feature_check_frame, text=feature, variable=var)
                cb.grid(row=i//3, column=i%3, sticky='w', padx=5, pady=2)
        
        if not targets:
            ttk.Label(self.target_radio_frame, text="No targets defined in Data & Fields tab").pack()
        else:
            # Create radio buttons for targets
            for i, target in enumerate(targets):
                rb = ttk.Radiobutton(self.target_radio_frame, text=target, 
                                   variable=self.selected_target_var, value=target)
                rb.grid(row=i//3, column=i%3, sticky='w', padx=5, pady=2)
            # Select first target by default
            if targets:
                self.selected_target_var.set(targets[0])
    
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
        
        # Console output
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding=10)
        console_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        self.console_text = scrolledtext.ScrolledText(console_frame, wrap='word', height=20)
        self.console_text.pack(fill='both', expand=True)
    
    def setup_results_tab(self):
        """Setup performance statistics tab with improved formatting"""
        # Create main container
        main_frame = ttk.Frame(self.results_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create paned window for stats and chart
        paned = ttk.PanedWindow(main_frame, orient='horizontal')
        paned.pack(fill='both', expand=True)
        
        # Left side - Statistics
        stats_frame = ttk.LabelFrame(paned, text="Performance Metrics", padding=10)
        paned.add(stats_frame, weight=1)
        
        # Summary statistics in a grid layout
        summary_frame = ttk.Frame(stats_frame)
        summary_frame.pack(fill='x', pady=(0, 10))
        
        self.stats_labels = {}
        stats_layout = [
            ('total_obs', 'Total Observations:', 0, 0),
            ('total_trades', 'Total Trades:', 1, 0),
            ('trade_freq', 'Trading Frequency:', 2, 0),
            ('hit_rate', 'Hit Rate:', 3, 0),
            ('base_rate', 'Base Rate:', 0, 2),
            ('edge', 'Edge:', 1, 2),
            ('cumulative_pnl', 'Cumulative P&L:', 2, 2),
            ('avg_confidence', 'Avg Confidence:', 3, 2),
        ]
        
        for key, label, row, col in stats_layout:
            ttk.Label(summary_frame, text=label, font=('Arial', 10, 'bold')).grid(
                row=row, column=col, sticky='w', padx=10, pady=5)
            value_label = ttk.Label(summary_frame, text="-", font=('Arial', 10))
            value_label.grid(row=row, column=col+1, sticky='w', padx=10, pady=5)
            self.stats_labels[key] = value_label
        
        # Performance table
        table_frame = ttk.LabelFrame(stats_frame, text="Key Performance Indicators", padding=10)
        table_frame.pack(fill='both', expand=True, pady=10)
        
        # Create treeview for metrics
        columns = ('Metric', 'Value', 'Benchmark', 'Difference')
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120)
        
        self.results_tree.pack(fill='both', expand=True, side='left')
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.results_tree.yview)
        tree_scroll.pack(side='right', fill='y')
        self.results_tree.configure(yscrollcommand=tree_scroll.set)
        
        # Right side - Equity curve chart
        chart_frame = ttk.LabelFrame(paned, text="Cumulative Equity Curve", padding=10)
        paned.add(chart_frame, weight=2)
        
        # Create matplotlib figure for equity curve
        self.equity_figure = plt.Figure(figsize=(8, 6), facecolor='white')
        self.equity_canvas = FigureCanvasTkAgg(self.equity_figure, master=chart_frame)
        self.equity_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Refresh button
        refresh_frame = ttk.Frame(stats_frame)
        refresh_frame.pack(fill='x', pady=10)
        ttk.Button(refresh_frame, text="Refresh Results", command=self.load_results,
                  style='Accent.TButton').pack()
    
    def setup_charts_tab(self):
        """Setup charts display tab to take full screen"""
        # Create main frame
        main_frame = ttk.Frame(self.charts_tab)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Chart selection frame (minimal height)
        select_frame = ttk.Frame(main_frame)
        select_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(select_frame, text="Select Chart:").pack(side='left', padx=5)
        
        self.chart_var = tk.StringVar()
        chart_options = [
            ('Comprehensive Analysis', 'comprehensive'),
            ('Yearly Progression', 'progression'),
            ('Feature Importance', 'feature_importance'),
            ('Regression Analysis', 'regression')
        ]
        
        for i, (label, value) in enumerate(chart_options):
            ttk.Radiobutton(select_frame, text=label, variable=self.chart_var, 
                           value=value, command=self.load_chart).pack(side='left', padx=10)
        
        self.chart_var.set('comprehensive')
        
        # Chart display frame (takes most of the space)
        chart_display_frame = ttk.LabelFrame(main_frame, text="Chart Display", padding=5)
        chart_display_frame.pack(fill='both', expand=True)
        
        # Create scrollable canvas for large charts
        chart_canvas = tk.Canvas(chart_display_frame, bg='white')
        chart_scroll_y = ttk.Scrollbar(chart_display_frame, orient='vertical', command=chart_canvas.yview)
        chart_scroll_x = ttk.Scrollbar(chart_display_frame, orient='horizontal', command=chart_canvas.xview)
        
        chart_canvas.configure(yscrollcommand=chart_scroll_y.set, xscrollcommand=chart_scroll_x.set)
        
        chart_scroll_y.pack(side='right', fill='y')
        chart_scroll_x.pack(side='bottom', fill='x')
        chart_canvas.pack(side='left', fill='both', expand=True)
        
        # Create frame inside canvas for chart
        self.chart_frame = ttk.Frame(chart_canvas)
        chart_canvas.create_window((0, 0), window=self.chart_frame, anchor='nw')
        
        # Chart label for displaying images
        self.chart_label = ttk.Label(self.chart_frame, text="No chart loaded\nRun validation to generate charts")
        self.chart_label.pack(fill='both', expand=True)
        
        # Update scroll region when chart changes
        self.chart_frame.bind('<Configure>', 
                             lambda e: chart_canvas.configure(scrollregion=chart_canvas.bbox("all")))
    
    # --- Helper Methods ---
    
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
            return
        
        try:
            validation_start = pd.to_datetime(self.validation_start_var.get())
            validation_end = pd.to_datetime(self.validation_end_var.get())
            
            # Validate dates
            if validation_start >= validation_end:
                messagebox.showerror("Error", "Validation end date must be after start date")
                return
            
            # Update timeline visualization
            train_days, test_days, val_days = self.timeline.update_timeline(
                self.data_start_date, 
                self.data_end_date,
                validation_start,
                validation_end
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update timeline: {str(e)}")
    
    def save_field_config(self):
        """Save the field configuration to config file"""
        try:
            # Get selected features and targets
            features = list(self.feature_listbox.get(0, tk.END))
            targets = list(self.target_listbox.get(0, tk.END))
            
            if not features or not targets:
                return
            
            # Get selected features from Model Config if available
            selected_features = []
            if hasattr(self, 'feature_check_vars'):
                for feature, var in self.feature_check_vars.items():
                    if var.get() and feature in features:
                        selected_features.append(feature)
            else:
                selected_features = features
            
            # Get selected target from Model Config if available
            selected_target = self.selected_target_var.get() if self.selected_target_var.get() else targets[0]
            
            # Update config file
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            # Update data section
            if 'data' not in config:
                config['data'] = {}
            
            config['data']['csv_file'] = self.data_file_var.get()
            config['data']['feature_columns'] = ','.join(features)
            config['data']['selected_features'] = ','.join(selected_features)
            config['data']['target_column'] = selected_target
            config['data']['all_targets'] = ','.join(targets)
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
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def load_config(self):
        """Load configuration from file"""
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            # Load validation dates if present
            if 'validation' in config:
                if 'validation_start_date' in config['validation']:
                    self.validation_start_var.set(config['validation']['validation_start_date'])
                if 'validation_end_date' in config['validation']:
                    self.validation_end_var.set(config['validation']['validation_end_date'])
            
            # Load config widgets if they exist
            if hasattr(self, 'config_widgets'):
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
            
            # Update feature/target selection
            self.update_feature_target_selection()
            
        except Exception as e:
            pass  # Silent fail on initial load
    
    def save_config(self):
        """Save configuration to file"""
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            # Save all widget values
            for section_name, section_info in self.config_structure.items():
                section = section_info['section']
                if section not in config:
                    config[section] = {}
                
                for key, _, _, _, _ in section_info['fields']:
                    widget_key = f"{section}.{key}"
                    if widget_key in self.config_widgets:
                        widget = self.config_widgets[widget_key]
                        if isinstance(widget, ttk.Combobox):
                            config[section][key] = widget.get()
                        elif isinstance(widget, (ttk.Entry, ttk.Spinbox)):
                            config[section][key] = widget.get()
            
            # Save selected features and target
            if hasattr(self, 'feature_check_vars'):
                selected_features = []
                for feature, var in self.feature_check_vars.items():
                    if var.get():
                        selected_features.append(feature)
                if selected_features and 'data' in config:
                    config['data']['selected_features'] = ','.join(selected_features)
            
            if self.selected_target_var.get() and 'data' in config:
                config['data']['target_column'] = self.selected_target_var.get()
            
            with open(self.config_file, 'w') as f:
                config.write(f)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {str(e)}")
    
    def reset_config(self):
        """Reset configuration to defaults"""
        # Reset all widgets to default values
        pass
    
    def run_validation(self):
        """Run the validation process"""
        # Save config first
        self.save_field_config()
        self.save_config()
        
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
    
    def stop_validation(self):
        """Stop the validation process"""
        if self.validation_process:
            self.validation_process.terminate()
            self.validation_process = None
            self.progress_bar.stop()
            self.run_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_label.config(text="Stopped")
    
    def load_results(self):
        """Load and display results with equity curve"""
        try:
            if not os.path.exists('OMtree_results.csv'):
                return
            
            df = pd.read_csv('OMtree_results.csv')
            df['date'] = pd.to_datetime(df['date'])
            
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
                    trades['pnl'] = trades['target_value']
                else:  # shortonly
                    trades['pnl'] = -trades['target_value']
                cumulative_pnl = trades['pnl'].sum()
            else:
                cumulative_pnl = 0
            
            edge = hit_rate - base_rate
            
            # Update summary labels
            self.stats_labels['total_obs'].config(text=f"{total_observations:,}")
            self.stats_labels['total_trades'].config(text=f"{total_trades:,}")
            self.stats_labels['trade_freq'].config(text=f"{total_trades/total_observations:.1%}")
            self.stats_labels['hit_rate'].config(text=f"{hit_rate:.2%}")
            self.stats_labels['base_rate'].config(text=f"{base_rate:.2%}")
            self.stats_labels['edge'].config(text=f"{edge:+.2%}")
            self.stats_labels['cumulative_pnl'].config(text=f"{cumulative_pnl:.2f}")
            if len(trades) > 0:
                self.stats_labels['avg_confidence'].config(text=f"{trades['probability'].mean():.3f}")
            
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
            
            # Plot equity curve
            if len(trades) > 0:
                self.plot_equity_curve(trades)
                
        except Exception as e:
            pass  # Silent fail
    
    def plot_equity_curve(self, trades):
        """Plot cumulative equity curve"""
        try:
            # Clear previous plot
            self.equity_figure.clear()
            ax = self.equity_figure.add_subplot(111)
            
            # Sort trades by date
            trades_sorted = trades.sort_values('date')
            
            # Calculate cumulative P&L
            cumulative_pnl = trades_sorted['pnl'].cumsum()
            
            # Plot
            ax.plot(trades_sorted['date'], cumulative_pnl, 'b-', linewidth=2, label='Cumulative P&L')
            ax.fill_between(trades_sorted['date'], 0, cumulative_pnl, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Format
            ax.set_title('Cumulative Equity Curve', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative P&L')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format dates on x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            self.equity_figure.tight_layout()
            self.equity_canvas.draw()
            
        except Exception as e:
            pass
    
    def load_chart(self):
        """Load and display selected chart"""
        chart_type = self.chart_var.get()
        
        # Get model type from config
        config = configparser.ConfigParser()
        config.read('OMtree_config.ini')
        model_type = config['model']['model_type']
        
        # Map chart types to actual generated files
        chart_files = {
            'comprehensive': f'OMtree_comprehensive_{model_type}.png',
            'progression': f'OMtree_progression_{model_type}.png',
            'feature_importance': 'feature_importance.png',  # Will need to generate this
            'regression': 'regression_analysis_matrix.png'
        }
        
        chart_file = chart_files.get(chart_type, f'OMtree_comprehensive_{model_type}.png')
        
        if os.path.exists(chart_file):
            try:
                img = Image.open(chart_file)
                # Display at full size
                photo = ImageTk.PhotoImage(img)
                self.chart_label.config(image=photo, text="")
                self.chart_label.image = photo  # Keep reference
            except Exception as e:
                self.chart_label.config(text=f"Error loading chart: {str(e)}")
        else:
            self.chart_label.config(text=f"Chart file not found: {chart_file}\nRun validation first to generate charts.")
    
    def save_all_config(self):
        """Save all configuration"""
        self.save_field_config()
        self.save_config()
    
    def save_config_as(self):
        """Save configuration with a new name"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".ini",
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
        )
        if filename:
            self.config_file = filename
            self.save_all_config()
    
    def load_config_file(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
        )
        if filename:
            self.config_file = filename
            self.load_config()
    
    def show_about(self):
        """Show about dialog"""
        about_text = """OMtree Trading Model
Version 2.0

Enhanced walk-forward validation system
for directional trading strategies.

© 2024 - All rights reserved"""
        messagebox.showinfo("About", about_text)

def main():
    root = tk.Tk()
    app = OMtreeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()