import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import configparser
import subprocess
import threading
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for core modules
sys.path.insert(0, 'src')
from datetime import datetime
from PIL import Image, ImageTk
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# Import modules from src
from src.regression_gui_module_v3 import RegressionAnalysisTab
from src.timeline_component_v2 import TimelineVisualization
from src.config_manager import ConfigurationManager
from src.performance_stats import calculate_performance_stats, format_stats_for_display
from src.date_parser import FlexibleDateParser
from src.data_view_module import DataViewTab
from src.tree_visualizer import TreeVisualizer

class OMtreeGUI:
    def export_treeview_to_csv(self, tree, filepath):
        """Export treeview data to CSV file"""
        try:
            # Get column headers
            columns = [tree.heading(col)['text'] for col in tree['columns']]
            
            # Get all rows
            rows = []
            for item in tree.get_children():
                rows.append(tree.item(item)['values'])
            
            # Create DataFrame and save
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(filepath, index=False)
            print(f"Exported treeview to {filepath}")
        except Exception as e:
            print(f"Error exporting treeview: {e}")
    
    def __init__(self, root):
        self.root = root
        self.root.title("OMtree Trading Model - Enhanced Configuration & Analysis")
        # Larger window size to fit all content
        self.root.geometry("2100x1200")
        
        # Set minimum window size
        self.root.minsize(1920, 1000)
        
        # Try to maximize window on startup
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except:
                pass  # Fall back to specified geometry
        
        # Initialize configuration manager
        self.config_manager = ConfigurationManager()
        
        # Store config file path
        self.config_file = 'OMtree_config.ini'
        self.last_data_file = None  # Track last loaded file
        self.df = None
        self.available_columns = []
        
        # Process tracking
        self.validation_process = None
        
        # Create menu bar
        self.create_menu()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs (removed Run Walk Forward tab)
        self.data_tab = ttk.Frame(self.notebook)
        self.config_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.tree_tab = ttk.Frame(self.notebook)
        self.permute_alpha_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text='Data & Fields')
        self.notebook.add(self.config_tab, text='Model Tester')  # Renamed from Model Configuration
        self.notebook.add(self.results_tab, text='Performance Stats & Charts')
        self.notebook.add(self.tree_tab, text='Tree Visualizer')
        self.notebook.add(self.permute_alpha_tab, text='PermuteAlpha')
        
        # Add regression analysis tab
        self.regression_analysis = RegressionAnalysisTab(self.notebook)
        
        # Add data view tab
        self.data_view = DataViewTab(self.notebook)
        
        # Initialize tabs
        self.setup_data_tab()
        self.setup_model_tester_tab()  # Renamed and integrated walk forward
        self.setup_results_tab()
        self.setup_tree_tab()
        self.setup_permute_alpha_tab()
        
        # Load initial config
        self.load_config()
        
        # Auto-load last data file
        self.auto_load_last_file()
        
        # Auto-load most recent settings files
        self.load_recent_settings()
        
    def auto_load_last_file(self):
        """Auto-load the last used data file on startup"""
        try:
            # Check if there's a saved last file in config
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            if 'data' in config and 'last_file' in config['data']:
                last_file = config['data']['last_file']
                if os.path.exists(last_file):
                    self.data_file_var.set(last_file)
                    self.load_data_file()
                    return
            
            # Otherwise try to find a data file in data folder
            if os.path.exists('data'):
                data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
                if data_files:
                    # Load the first CSV file found
                    self.data_file_var.set(f"data/{data_files[0]}")
                    self.load_data_file()
        except Exception as e:
            print(f"Could not auto-load data: {e}")
    
    def clear_settings(self):
        """Clear all current settings"""
        if messagebox.askyesno("Clear Settings", "Clear all current settings?"):
            self.data_file_var.set("")
            self.validation_start_var.set("")
            self.validation_end_var.set("")
            if hasattr(self, 'feature_listbox'):
                self.feature_listbox.delete(0, tk.END)
            if hasattr(self, 'target_listbox'):
                self.target_listbox.delete(0, tk.END)
    
    def load_last_file(self):
        """Manually load the last used file"""
        self.auto_load_last_file()
        if not self.data_file_var.get():
            messagebox.showinfo("No Data", "No previous data file found")
    
    def save_config(self):
        """Save current configuration"""
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            # Save the current file as last file
            if 'data' not in config:
                config['data'] = {}
            config['data']['last_file'] = self.data_file_var.get()
            
            with open(self.config_file, 'w') as f:
                config.write(f)
            
            messagebox.showinfo("Success", "Configuration saved")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Create toolbar frame
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side='top', fill='x', padx=2, pady=2)
        
        # Add toolbar buttons with text icons
        ttk.Button(toolbar, text="📂 Load Data Settings", 
                  command=self.load_data_settings, width=18).pack(side='left', padx=2)
        ttk.Button(toolbar, text="💾 Save Data Settings", 
                  command=self.save_data_settings, width=18).pack(side='left', padx=2)
        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=5)
        ttk.Button(toolbar, text="📁 Load Model Settings", 
                  command=self.load_model_settings, width=18).pack(side='left', padx=2)
        ttk.Button(toolbar, text="💾 Save Model Settings", 
                  command=self.save_model_settings, width=18).pack(side='left', padx=2)
        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=5)
        ttk.Button(toolbar, text="📁 Load Permute Settings", 
                  command=self.load_permute_settings, width=20).pack(side='left', padx=2)
        ttk.Button(toolbar, text="💾 Save Permute Settings", 
                  command=self.save_permute_settings, width=20).pack(side='left', padx=2)
        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=5)
        ttk.Button(toolbar, text="▶ Run Test", 
                  command=self.run_walkforward, width=12).pack(side='left', padx=2)
        ttk.Button(toolbar, text="📊 Results", 
                  command=lambda: self.notebook.select(self.results_tab), width=10).pack(side='left', padx=2)
        
        # File menu - Project operations
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_command(label="Save Project As...", command=self.save_project_as)
        file_menu.add_command(label="Load Project", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Save Data Settings...", command=self.save_data_settings)
        file_menu.add_command(label="Load Data Settings...", command=self.load_data_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Save Model Settings...", command=self.save_model_settings)
        file_menu.add_command(label="Load Model Settings...", command=self.load_model_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Save Permute Settings...", command=self.save_permute_settings)
        file_menu.add_command(label="Load Permute Settings...", command=self.load_permute_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Regression Analysis", command=lambda: self.notebook.select(4))
        tools_menu.add_command(label="Refresh Results", command=self.refresh_current_view)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        
    def setup_data_tab(self):
        """Setup the data loading and field selection tab with improved layout"""
        # Add current settings file label at the top
        settings_frame = ttk.Frame(self.data_tab)
        settings_frame.pack(fill='x', padx=10, pady=(5, 0))
        ttk.Label(settings_frame, text="Current Data Settings File:", font=('Arial', 11, 'bold')).pack(side='left', padx=(0, 5))
        self.current_data_settings_label = ttk.Label(settings_frame, text="None", font=('Arial', 11), foreground='blue')
        self.current_data_settings_label.pack(side='left')
        
        # Main container with better proportions
        main_paned = ttk.PanedWindow(self.data_tab, orient='horizontal')
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left side - Data loading and field selection
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)
        
        # Right side - Configuration history
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # === LEFT SIDE ===
        # File selection section
        file_frame = ttk.LabelFrame(left_frame, text="Data File Selection", padding=10)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        file_input_frame = ttk.Frame(file_frame)
        file_input_frame.pack(fill='x')
        
        ttk.Label(file_input_frame, text="CSV File:").pack(side='left', padx=5)
        self.data_file_var = tk.StringVar(value="data/DTSmlDATA7x7.csv")
        self.data_file_entry = ttk.Entry(file_input_frame, textvariable=self.data_file_var, width=50)
        self.data_file_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        ttk.Button(file_input_frame, text="Browse", command=self.browse_data_file).pack(side='left', padx=5)
        ttk.Button(file_input_frame, text="Load", command=self.load_data_file, 
                  style='Accent.TButton').pack(side='left', padx=5)
        
        # Data info
        self.data_info_label = ttk.Label(file_frame, text="No data loaded", foreground='gray')
        self.data_info_label.pack(anchor='w', pady=(5, 0))
        
        # Filter frame container for ticker and hour filters
        self.filters_frame = ttk.Frame(file_frame)
        self.filters_frame.pack(fill='x', pady=(5, 0))
        
        # Ticker filter (for multi-ticker data)
        self.ticker_filter_frame = ttk.Frame(self.filters_frame)
        self.ticker_filter_frame.pack(side='left', padx=(0, 20))
        
        ttk.Label(self.ticker_filter_frame, text="Ticker:").pack(side='left', padx=5)
        self.ticker_filter_var = tk.StringVar(value="All")
        self.ticker_filter_combo = ttk.Combobox(self.ticker_filter_frame, textvariable=self.ticker_filter_var,
                                                width=12, state='readonly')
        self.ticker_filter_combo['values'] = ['All']
        self.ticker_filter_combo.pack(side='left', padx=5)
        self.ticker_filter_combo.bind('<<ComboboxSelected>>', self.apply_filters)
        
        # Initially hide ticker filter
        self.ticker_filter_frame.pack_forget()
        
        # Time filter frame (for hourly data)
        self.time_filter_frame = ttk.Frame(self.filters_frame)
        self.time_filter_frame.pack(side='left')
        
        ttk.Label(self.time_filter_frame, text="Hour:").pack(side='left', padx=5)
        self.hour_filter_var = tk.StringVar(value="All")
        self.hour_filter_combo = ttk.Combobox(self.time_filter_frame, textvariable=self.hour_filter_var, 
                                               width=12, state='readonly')
        self.hour_filter_combo['values'] = ['All']
        self.hour_filter_combo.pack(side='left', padx=5)
        self.hour_filter_combo.bind('<<ComboboxSelected>>', self.apply_filters)
        
        # Initially hide the time filter
        self.time_filter_frame.pack_forget()
        
        # Initially hide the entire filters frame
        self.filters_frame.pack_forget()
        
        # Validation period with improved layout
        validation_frame = ttk.LabelFrame(left_frame, text="Validation Period Settings", padding=10)
        validation_frame.pack(fill='x', padx=5, pady=5)
        
        # Dates in a single row
        dates_frame = ttk.Frame(validation_frame)
        dates_frame.pack(fill='x')
        
        # Start date
        ttk.Label(dates_frame, text="Start Date:").pack(side='left', padx=5)
        self.validation_start_var = tk.StringVar(value="2010-01-01")
        self.validation_start_entry = ttk.Entry(dates_frame, textvariable=self.validation_start_var, width=12)
        self.validation_start_entry.pack(side='left', padx=5)
        self.validation_start_entry.bind('<FocusOut>', lambda e: self.update_timeline())
        
        ttk.Label(dates_frame, text="End Date:").pack(side='left', padx=(20, 5))
        self.validation_end_var = tk.StringVar(value="2015-12-31")
        self.validation_end_entry = ttk.Entry(dates_frame, textvariable=self.validation_end_var, width=12)
        self.validation_end_entry.pack(side='left', padx=5)
        self.validation_end_entry.bind('<FocusOut>', lambda e: self.update_timeline())
        
        # Info label
        info_label = ttk.Label(dates_frame, text="(Data after End Date is out-of-sample)", 
                              foreground='blue')
        info_label.pack(side='left', padx=20)
        
        # Timeline visualization - compact and non-overlapping
        timeline_frame = ttk.Frame(validation_frame, height=80)
        timeline_frame.pack(fill='x', pady=(5, 0))
        timeline_frame.pack_propagate(False)
        self.timeline = TimelineVisualization(timeline_frame)
        
        # Field selection with better layout
        field_frame = ttk.LabelFrame(left_frame, text="Field Selection", padding=10)
        field_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create three columns with better proportions
        columns_frame = ttk.Frame(field_frame)
        columns_frame.pack(fill='both', expand=True)
        
        # Available columns
        avail_frame = ttk.LabelFrame(columns_frame, text="Available Columns", padding=5)
        avail_frame.grid(row=0, column=0, sticky='nsew', padx=3)
        
        # Scrollbar for available listbox
        avail_scroll = ttk.Scrollbar(avail_frame)
        avail_scroll.pack(side='right', fill='y')
        
        self.available_listbox = tk.Listbox(avail_frame, selectmode='multiple', height=12, 
                                           exportselection=False, yscrollcommand=avail_scroll.set)
        self.available_listbox.pack(fill='both', expand=True)
        avail_scroll.config(command=self.available_listbox.yview)
        
        # Buttons column with better layout
        button_frame = ttk.Frame(columns_frame)
        button_frame.grid(row=0, column=1, padx=10, sticky='ns')
        
        # Center the buttons vertically
        button_container = ttk.Frame(button_frame)
        button_container.pack(expand=True)
        
        ttk.Button(button_container, text="Add to Features →", 
                  command=self.move_to_features, width=18).pack(pady=3)
        ttk.Button(button_container, text="← Remove Features", 
                  command=self.remove_from_features, width=18).pack(pady=3)
        ttk.Label(button_container, text="").pack(pady=10)  # Spacer
        ttk.Button(button_container, text="Add to Targets →", 
                  command=self.move_to_targets, width=18).pack(pady=3)
        ttk.Button(button_container, text="← Remove Targets", 
                  command=self.remove_from_targets, width=18).pack(pady=3)
        
        # Feature columns
        feature_frame = ttk.LabelFrame(columns_frame, text="Selected Features", padding=5)
        feature_frame.grid(row=0, column=2, sticky='nsew', padx=3)
        
        feature_scroll = ttk.Scrollbar(feature_frame)
        feature_scroll.pack(side='right', fill='y')
        
        self.feature_listbox = tk.Listbox(feature_frame, selectmode='multiple', height=12, 
                                         exportselection=False, yscrollcommand=feature_scroll.set)
        self.feature_listbox.pack(fill='both', expand=True)
        feature_scroll.config(command=self.feature_listbox.yview)
        
        # Target columns
        target_frame = ttk.LabelFrame(columns_frame, text="Selected Targets", padding=5)
        target_frame.grid(row=0, column=3, sticky='nsew', padx=3)
        
        target_scroll = ttk.Scrollbar(target_frame)
        target_scroll.pack(side='right', fill='y')
        
        self.target_listbox = tk.Listbox(target_frame, selectmode='multiple', height=12, 
                                        exportselection=False, yscrollcommand=target_scroll.set)
        self.target_listbox.pack(fill='both', expand=True)
        target_scroll.config(command=self.target_listbox.yview)
        
        # Configure grid weights for proper resizing
        columns_frame.columnconfigure(0, weight=2)
        columns_frame.columnconfigure(2, weight=2)
        columns_frame.columnconfigure(3, weight=2)
        columns_frame.rowconfigure(0, weight=1)
        
        # Model Feature and Target Selection Section
        model_selection_frame = ttk.LabelFrame(left_frame, text="Model Features and Target Selection", padding=10)
        model_selection_frame.pack(fill='x', padx=5, pady=5)
        
        # Feature selection with Select All/Clear All buttons
        feature_label_frame = ttk.Frame(model_selection_frame)
        feature_label_frame.pack(anchor='w', fill='x')
        ttk.Label(feature_label_frame, text="Select Features for Model:").pack(side='left')
        ttk.Label(feature_label_frame, text=" (features will be normalized based on settings)", 
                 font=('Arial', 10), foreground='gray').pack(side='left')
        
        # Select All / Clear All buttons
        feature_button_frame = ttk.Frame(model_selection_frame)
        feature_button_frame.pack(anchor='w', pady=(5, 0))
        ttk.Button(feature_button_frame, text="Select All", 
                  command=lambda: self.select_all_features(True)).pack(side='left', padx=5)
        ttk.Button(feature_button_frame, text="Clear All", 
                  command=lambda: self.select_all_features(False)).pack(side='left', padx=5)
        
        self.feature_check_frame = ttk.Frame(model_selection_frame)
        self.feature_check_frame.pack(fill='x', padx=20, pady=5)
        
        # Random noise features section
        noise_frame = ttk.LabelFrame(model_selection_frame, text="Noise Testing Features", padding=5)
        noise_frame.pack(fill='x', pady=(10, 5))
        
        noise_control_frame = ttk.Frame(noise_frame)
        noise_control_frame.pack(fill='x')
        
        ttk.Label(noise_control_frame, text="Add random noise features:").pack(side='left', padx=5)
        self.num_noise_features = tk.IntVar(value=0)
        ttk.Spinbox(noise_control_frame, from_=0, to=10, increment=1, 
                   textvariable=self.num_noise_features, width=10).pack(side='left', padx=5)
        ttk.Label(noise_control_frame, text="(Gaussian random data for testing feature selection)", 
                 font=('Arial', 10), foreground='gray').pack(side='left', padx=5)
        
        self.noise_features_frame = ttk.Frame(noise_frame)
        self.noise_features_frame.pack(fill='x', padx=20, pady=(5, 0))
        self.feature_check_vars = {}
        
        # Target selection
        ttk.Label(model_selection_frame, text="Select Target for Model:").pack(anchor='w', pady=(10, 0))
        
        self.target_radio_frame = ttk.Frame(model_selection_frame)
        self.target_radio_frame.pack(fill='x', padx=20, pady=5)
        self.selected_target_var = tk.StringVar()
        
        # Update button
        ttk.Button(model_selection_frame, text="Load Available Features/Targets", 
                  command=self.update_feature_target_selection).pack(pady=5)
        
        # === RIGHT SIDE - Quick Actions ===
        actions_frame = ttk.LabelFrame(right_frame, text="Quick Actions", padding=10)
        actions_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(actions_frame, text="Load Last File", 
                  command=self.load_last_file).pack(fill='x', pady=2)
        ttk.Button(actions_frame, text="Clear Settings", 
                  command=self.clear_settings).pack(fill='x', pady=2)
        ttk.Button(actions_frame, text="Save Configuration", 
                  command=self.save_config).pack(fill='x', pady=2)
        
        # Add spacer to push content up
        ttk.Frame(right_frame).pack(fill='both', expand=True)
        
        # History features removed - not needed
    
    def setup_model_tester_tab(self):
        """Setup the model configuration and testing tab"""
        # Add current settings file label at the top
        settings_frame = ttk.Frame(self.config_tab)
        settings_frame.pack(fill='x', padx=10, pady=(5, 0))
        ttk.Label(settings_frame, text="Current Model Settings File:", font=('Arial', 11, 'bold')).pack(side='left', padx=(0, 5))
        self.current_model_settings_label = ttk.Label(settings_frame, text="None", font=('Arial', 11), foreground='blue')
        self.current_model_settings_label.pack(side='left')
        
        # Create three-column layout
        main_frame = ttk.Frame(self.config_tab)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left column - Model & Preprocessing (30% width)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Middle column - Feature Selection & Walk-Forward (30% width)
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right column - Walk Forward Control and Console (40% width)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='left', fill='both', expand=True, padx=(5, 0))
        
        # === LEFT COLUMN - Model Configuration ===
        config_canvas = tk.Canvas(left_frame)
        config_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=config_canvas.yview)
        config_scrollable = ttk.Frame(config_canvas)
        
        config_scrollable.bind(
            "<Configure>",
            lambda e: config_canvas.configure(scrollregion=config_canvas.bbox("all"))
        )
        
        config_canvas.create_window((0, 0), window=config_scrollable, anchor="nw")
        config_canvas.configure(yscrollcommand=config_scrollbar.set)
        
        # Store config widgets
        self.config_widgets = {}
        
        # Note: Feature/Target selection has been moved to Data tab
        
        # Model Parameters
        model_frame = ttk.LabelFrame(config_scrollable, text="Model Parameters", padding=10)
        model_frame.pack(fill='x', padx=5, pady=5)
        
        model_params = [
            ('model.model_type', 'Model Type', 'combo', ['longonly', 'shortonly']),
            ('model.algorithm', 'Algorithm', 'combo', ['decision_trees', 'extra_trees']),
            ('model.tree_criterion', 'Tree Criterion', 'combo', ['mse', 'mae', 'friedman_mse', 'gini', 'entropy']),
            ('model.regression_mode', 'Regression Mode', 'combo', ['false', 'true']),
            ('model.convert_to_binary', 'Convert to Binary', 'combo', ['true', 'false']),
            ('model.probability_aggregation', 'Aggregation Method', 'combo', ['mean', 'median']),
            ('model.balanced_bootstrap', 'Balanced Bootstrap', 'combo', ['false', 'true']),
            ('model.n_trees_method', 'Trees Method', 'combo', ['absolute', 'per_feature']),
            ('model.n_trees', 'Number of Trees', 'spinbox', (10, 1000, 10)),
            ('model.max_depth', 'Max Depth', 'spinbox', (1, 10, 1)),
            ('model.bootstrap_fraction', 'Bootstrap Fraction', 'spinbox', (0.1, 1.0, 0.1)),
            ('model.min_leaf_fraction', 'Min Leaf Fraction', 'spinbox', (0.01, 0.5, 0.01)),
            ('model.target_threshold', 'Target Threshold', 'spinbox', (0.0, 0.5, 0.01)),
            ('model.vote_threshold', 'Vote Threshold (Binary)', 'spinbox', (0.5, 1.0, 0.05)),
            ('model.trade_prediction_threshold', 'Trade Threshold (Raw)', 'spinbox', (0.0, 0.1, 0.001)),
            ('model.auto_calibrate_threshold', 'Auto-Calibrate Threshold', 'combo', ['false', 'true']),
            ('model.target_prediction_rate', 'Target Prediction Rate', 'spinbox', (0.05, 0.50, 0.05)),
            ('model.calibration_lookback', 'Calibration Lookback', 'spinbox', (30, 300, 10)),
            ('model.random_seed', 'Random Seed', 'spinbox', (1, 99999, 1)),
            ('model.n_jobs', 'CPU Cores (n_jobs)', 'spinbox', (-1, 64, 1)),
        ]
        
        for i, (key, label, widget_type, options) in enumerate(model_params):
            ttk.Label(model_frame, text=label + ':').grid(row=i, column=0, sticky='w', padx=5, pady=2)
            
            if widget_type == 'combo':
                widget = ttk.Combobox(model_frame, values=options, width=20, state='readonly')
            elif widget_type == 'spinbox':
                min_val, max_val, increment = options
                widget = ttk.Spinbox(model_frame, from_=min_val, to=max_val, 
                                    increment=increment, width=20)
            
            widget.grid(row=i, column=1, padx=5, pady=2, sticky='w')
            self.config_widgets[key] = widget
            
            # Add descriptions for all parameters
            if key == 'model.model_type':
                desc = ttk.Label(model_frame, text='Long-only or short-only positions', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            # Add description for algorithm type
            if key == 'model.algorithm':
                self.algorithm_desc = ttk.Label(model_frame, text='', font=('Arial', 10), foreground='gray')
                self.algorithm_desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
                
                def update_algorithm_desc(*args):
                    algo = widget.get()
                    if algo == 'extra_trees':
                        self.algorithm_desc.config(text='More randomized, reduces overfitting')
                    else:
                        self.algorithm_desc.config(text='Standard random forest approach')
                
                widget.bind('<<ComboboxSelected>>', update_algorithm_desc)
                update_algorithm_desc()  # Set initial description
            
            # Add description for tree criterion
            if key == 'model.tree_criterion':
                desc = ttk.Label(model_frame, text='Splitting metric: mse/mae/friedman for reg, gini/entropy for class', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            # Add description for regression mode
            if key == 'model.regression_mode':
                self.regression_mode_desc = ttk.Label(model_frame, text='', font=('Arial', 10), foreground='gray')
                self.regression_mode_desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
                
                def update_regression_mode_desc(*args):
                    mode = widget.get()
                    if mode == 'true':
                        self.regression_mode_desc.config(text='Use regressors (continuous targets)')
                    else:
                        self.regression_mode_desc.config(text='Use classifiers (binary labels)')
                
                widget.bind('<<ComboboxSelected>>', update_regression_mode_desc)
                update_regression_mode_desc()  # Set initial description
            
            # Add description for convert to binary
            if key == 'model.convert_to_binary':
                self.convert_desc = ttk.Label(model_frame, text='', font=('Arial', 10), foreground='gray')
                self.convert_desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
                
                def update_convert_desc(*args):
                    mode = widget.get()
                    regression_mode = self.config_widgets.get('model.regression_mode')
                    if regression_mode and regression_mode.get() == 'true':
                        if mode == 'true':
                            self.convert_desc.config(text='Binary voting: use Vote Threshold')
                        else:
                            self.convert_desc.config(text='Raw aggregation: use Trade Threshold')
                    else:
                        self.convert_desc.config(text='Only applies in regression mode')
                
                widget.bind('<<ComboboxSelected>>', update_convert_desc)
                update_convert_desc()
            
            # Add description for aggregation method
            if key == 'model.probability_aggregation':
                self.prob_agg_desc = ttk.Label(model_frame, text='How to combine tree predictions', font=('Arial', 10), foreground='gray')
                self.prob_agg_desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            # Add description for trade prediction threshold
            if key == 'model.trade_prediction_threshold':
                desc = ttk.Label(model_frame, text='Min return to trigger trade (raw mode)', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            # Add description for balanced bootstrap
            if key == 'model.balanced_bootstrap':
                desc = ttk.Label(model_frame, text='Equal samples from each class', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            # Add descriptions for auto-calibration
            if key == 'model.auto_calibrate_threshold':
                self.auto_calib_desc = ttk.Label(model_frame, text='', font=('Arial', 10), foreground='gray')
                self.auto_calib_desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
                
                def update_auto_calib_desc(*args):
                    enabled = widget.get() == 'true'
                    if enabled:
                        self.auto_calib_desc.config(text='Dynamic threshold adjustment')
                    else:
                        self.auto_calib_desc.config(text='Use fixed thresholds')
                
                widget.bind('<<ComboboxSelected>>', update_auto_calib_desc)
                update_auto_calib_desc()
            
            if key == 'model.target_prediction_rate':
                desc = ttk.Label(model_frame, text='% of predictions to trigger (0.2 = 20%)', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            if key == 'model.calibration_lookback':
                desc = ttk.Label(model_frame, text='Training samples for calibration', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            if key == 'model.random_seed':
                desc = ttk.Label(model_frame, text='For reproducible results', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            # Add description for n_jobs
            if key == 'model.n_jobs':
                desc = ttk.Label(model_frame, text='-1=all cores, 1=single, -2=all but one', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            # Add description for trees method
            if key == 'model.n_trees_method':
                self.trees_method_desc = ttk.Label(model_frame, text='', font=('Arial', 10), foreground='gray')
                self.trees_method_desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
                
                def update_trees_method_desc(*args):
                    method = widget.get()
                    if method == 'per_feature':
                        self.trees_method_desc.config(text='Trees = N × features')
                    else:
                        self.trees_method_desc.config(text='Fixed tree count')
                    self.update_trees_label()
                
                widget.bind('<<ComboboxSelected>>', update_trees_method_desc)
                update_trees_method_desc()
            
            # Update trees label when n_trees changes
            if key == 'model.n_trees':
                self.trees_label = ttk.Label(model_frame, text='', font=('Arial', 10), foreground='gray')
                self.trees_label.grid(row=i, column=2, padx=5, pady=2, sticky='w')
                
                def on_trees_change(*args):
                    self.update_trees_label()
                
                widget.bind('<KeyRelease>', on_trees_change)
                widget.bind('<ButtonRelease>', on_trees_change)
            
            # Add descriptions for remaining parameters
            if key == 'model.max_depth':
                desc = ttk.Label(model_frame, text='Tree depth (1=stumps, higher=complex)', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            if key == 'model.bootstrap_fraction':
                desc = ttk.Label(model_frame, text='Data fraction per tree (0.5-0.8 typical)', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            if key == 'model.min_leaf_fraction':
                desc = ttk.Label(model_frame, text='Min % samples for leaf node', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            if key == 'model.target_threshold':
                desc = ttk.Label(model_frame, text='Binary classification threshold', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
            
            if key == 'model.vote_threshold':
                desc = ttk.Label(model_frame, text='Min % trees agreeing (binary mode)', font=('Arial', 10), foreground='gray')
                desc.grid(row=i, column=2, padx=5, pady=2, sticky='w')
        
        # Preprocessing Parameters
        preproc_frame = ttk.LabelFrame(config_scrollable, text="Preprocessing", padding=10)
        preproc_frame.pack(fill='x', padx=5, pady=5)
        
        # Add normalization method selector
        ttk.Label(preproc_frame, text='Normalization Method:').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        norm_method = ttk.Combobox(preproc_frame, values=['IQR', 'AVS', 'LOGIT_RANK'], width=20, state='readonly')
        norm_method.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['preprocessing.normalization_method'] = norm_method
        
        # Method description
        self.norm_method_desc = ttk.Label(preproc_frame, text='IQR: Interquartile Range normalization', 
                                         font=('Arial', 10), foreground='gray')
        self.norm_method_desc.grid(row=0, column=2, padx=5, pady=2, sticky='w')
        
        def update_norm_description(*args):
            method = norm_method.get()
            if method == 'AVS':
                self.norm_method_desc.config(text='AVS: Adaptive Volatility Scaling')
                # Show AVS parameters
                for widget in self.avs_widgets:
                    widget.grid()
                # Hide IQR parameters
                for widget in self.iqr_widgets:
                    widget.grid_remove()
            elif method == 'LOGIT_RANK':
                self.norm_method_desc.config(text='Logit-Rank: Percentile rank with logit transform')
                # Hide AVS parameters
                for widget in self.avs_widgets:
                    widget.grid_remove()
                # Show only window size from IQR parameters  
                for widget in self.iqr_widgets:
                    widget.grid_remove()
                # Show vol_window which is needed for logit-rank
                if 'preprocessing.vol_window' in self.config_widgets:
                    self.config_widgets['preprocessing.vol_window'].grid()
            else:
                self.norm_method_desc.config(text='IQR: Interquartile Range normalization')
                # Show IQR parameters
                for widget in self.iqr_widgets:
                    widget.grid()
                # Hide AVS parameters
                for widget in self.avs_widgets:
                    widget.grid_remove()
        
        norm_method.bind('<<ComboboxSelected>>', update_norm_description)
        
        # Store widget lists for showing/hiding
        self.iqr_widgets = []
        self.avs_widgets = []
        
        preproc_params = [
            ('preprocessing.normalize_features', 'Normalize Features', 'combo', ['true', 'false'], 'both'),
            ('preprocessing.normalize_target', 'Normalize Target', 'combo', ['true', 'false'], 'both'),
            ('preprocessing.detrend_features', 'Detrend Features (subtract median)', 'combo', ['true', 'false'], 'both'),
            ('preprocessing.vol_window', 'Lookback Window', 'spinbox', (10, 500, 10), 'iqr'),
            ('preprocessing.winsorize_enabled', 'Enable Winsorization', 'combo', ['false', 'true'], 'iqr'),
            ('preprocessing.winsorize_percentile', 'Winsorize %', 'spinbox', (1, 25, 1), 'iqr'),
            ('preprocessing.avs_slow_window', 'AVS Slow Window', 'spinbox', (20, 200, 10), 'avs'),
            ('preprocessing.avs_fast_window', 'AVS Fast Window', 'spinbox', (5, 100, 5), 'avs'),
            ('preprocessing.add_volatility_signal', 'Add Volatility Signal', 'combo', ['false', 'true'], 'both'),
        ]
        
        for i, param_tuple in enumerate(preproc_params):
            key, label, widget_type, options, param_group = param_tuple
            
            label_widget = ttk.Label(preproc_frame, text=label + ':')
            label_widget.grid(row=i+1, column=0, sticky='w', padx=5, pady=2)
            
            if widget_type == 'combo':
                widget = ttk.Combobox(preproc_frame, values=options, width=20, state='readonly')
            elif widget_type == 'spinbox':
                min_val, max_val, increment = options
                widget = ttk.Spinbox(preproc_frame, from_=min_val, to=max_val, 
                                    increment=increment, width=20)
            elif widget_type == 'entry':
                widget = ttk.Entry(preproc_frame, width=20)
            
            widget.grid(row=i+1, column=1, padx=5, pady=2, sticky='w')
            self.config_widgets[key] = widget
            
            # Add descriptions for preprocessing parameters
            desc_text = ''
            if key == 'preprocessing.normalize_features':
                desc_text = 'Scale features to standard range'
            elif key == 'preprocessing.normalize_target':
                desc_text = 'Scale target values'
            elif key == 'preprocessing.detrend_features':
                desc_text = 'Remove feature median'
            elif key == 'preprocessing.vol_window':
                desc_text = 'Lookback for IQR/rank calc'
            elif key == 'preprocessing.winsorize_enabled':
                desc_text = 'Clip extreme values before IQR'
            elif key == 'preprocessing.winsorize_percentile':
                desc_text = 'Clip at this percentile (e.g., 5%)'
            elif key == 'preprocessing.avs_slow_window':
                desc_text = 'Long-term volatility window'
            elif key == 'preprocessing.avs_fast_window':
                desc_text = 'Short-term volatility window'
            
            if desc_text:
                desc_label = ttk.Label(preproc_frame, text=desc_text, font=('Arial', 10), foreground='gray')
                desc_label.grid(row=i+1, column=2, padx=5, pady=2, sticky='w')
            
            # Add to appropriate widget list
            if param_group == 'iqr':
                self.iqr_widgets.extend([label_widget, widget])
            elif param_group == 'avs':
                self.avs_widgets.extend([label_widget, widget])
                # Initially hide AVS widgets
                label_widget.grid_remove()
                widget.grid_remove()
        
        # Feature Selection and Walk-Forward Validation have been moved to the middle column
        
        # Add a label with instructions at the bottom
        info_label = ttk.Label(config_scrollable, 
                              text="Configuration is automatically saved when running tests",
                              font=('Arial', 9, 'italic'), foreground='gray')
        info_label.pack(pady=10)
        
        config_canvas.pack(side="left", fill="both", expand=True)
        config_scrollbar.pack(side="right", fill="y")
        
        # === MIDDLE COLUMN - Feature Selection and Walk-Forward Settings ===
        middle_canvas = tk.Canvas(middle_frame)
        middle_scrollbar = ttk.Scrollbar(middle_frame, orient="vertical", command=middle_canvas.yview)
        middle_scrollable = ttk.Frame(middle_canvas)
        
        middle_scrollable.bind(
            "<Configure>",
            lambda e: middle_canvas.configure(scrollregion=middle_canvas.bbox("all"))
        )
        
        middle_canvas.create_window((0, 0), window=middle_scrollable, anchor="nw")
        middle_canvas.configure(yscrollcommand=middle_scrollbar.set)
        
        # Move Feature Selection here
        feature_selection_frame = ttk.LabelFrame(middle_scrollable, text="Random Forest MDI Feature Selection", padding=10)
        feature_selection_frame.pack(fill='x', padx=5, pady=5)
        
        # Enable/disable feature selection
        ttk.Label(feature_selection_frame, text="Enable Feature Selection:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.feature_selection_enabled_var = tk.StringVar(value='false')
        feature_selection_enabled_combo = ttk.Combobox(feature_selection_frame, 
                                                       textvariable=self.feature_selection_enabled_var,
                                                       values=['false', 'true'], width=20, state='readonly')
        feature_selection_enabled_combo.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['feature_selection.enabled'] = feature_selection_enabled_combo
        
        # Add callback to enable/disable settings
        feature_selection_enabled_combo.bind('<<ComboboxSelected>>', lambda e: self.toggle_feature_selection_settings())
        
        # Store references to feature selection controls for enable/disable
        self.fs_controls = []
        
        # Selection parameters
        ttk.Label(feature_selection_frame, text="Min Features:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        min_features_spin = ttk.Spinbox(feature_selection_frame, from_=1, to=10, increment=1, width=20)
        min_features_spin.grid(row=1, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['feature_selection.min_features'] = min_features_spin
        self.fs_controls.append(min_features_spin)
        
        ttk.Label(feature_selection_frame, text="Max Features:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        max_features_spin = ttk.Spinbox(feature_selection_frame, from_=1, to=20, increment=1, width=20)
        max_features_spin.grid(row=2, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['feature_selection.max_features'] = max_features_spin
        self.fs_controls.append(max_features_spin)
        
        ttk.Label(feature_selection_frame, text="Selection Lookback:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        lookback_spin = ttk.Spinbox(feature_selection_frame, from_=100, to=2000, increment=100, width=20)
        lookback_spin.grid(row=3, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['feature_selection.selection_lookback'] = lookback_spin
        self.fs_controls.append(lookback_spin)
        
        # Threshold Mode Selection
        ttk.Label(feature_selection_frame, text="Threshold Mode:").grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.threshold_mode_var = tk.StringVar(value='minimum')
        threshold_mode_combo = ttk.Combobox(feature_selection_frame, 
                                           textvariable=self.threshold_mode_var,
                                           values=['minimum', 'cumulative'], width=20, state='readonly')
        threshold_mode_combo.grid(row=4, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['feature_selection.threshold_mode'] = threshold_mode_combo
        self.fs_controls.append(threshold_mode_combo)
        
        # Minimum Importance Threshold (for minimum mode)
        self.min_importance_label = ttk.Label(feature_selection_frame, text="Min Importance:")
        self.min_importance_label.grid(row=5, column=0, sticky='w', padx=5, pady=2)
        threshold_spin = ttk.Spinbox(feature_selection_frame, from_=0.0, to=0.5, increment=0.01, width=20)
        threshold_spin.grid(row=5, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['feature_selection.importance_threshold'] = threshold_spin
        self.fs_controls.append(threshold_spin)
        
        # Cumulative Importance Threshold (for cumulative mode)
        self.cumulative_label = ttk.Label(feature_selection_frame, text="Cumulative %:")
        self.cumulative_label.grid(row=6, column=0, sticky='w', padx=5, pady=2)
        cumulative_spin = ttk.Spinbox(feature_selection_frame, from_=0.5, to=1.0, increment=0.05, width=20)
        cumulative_spin.set(0.95)  # Default to 95%
        cumulative_spin.grid(row=6, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['feature_selection.cumulative_threshold'] = cumulative_spin
        self.fs_controls.append(cumulative_spin)
        
        # Set initial state of feature selection controls
        self.toggle_feature_selection_settings()
        
        # Walk-Forward Validation Settings
        validation_frame = ttk.LabelFrame(middle_scrollable, text="Walk-Forward Validation Settings", padding=10)
        validation_frame.pack(fill='x', padx=5, pady=5)
        
        # Training Window Size
        ttk.Label(validation_frame, text='Training Window Size:').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        train_size_widget = ttk.Spinbox(validation_frame, from_=100, to=5000, increment=100, width=20)
        train_size_widget.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['validation.train_size'] = train_size_widget
        ttk.Label(validation_frame, text='Number of days to use for training the model', 
                 font=('Arial', 10), foreground='gray').grid(row=0, column=2, padx=5, pady=2, sticky='w')
        
        # Testing Window Size
        ttk.Label(validation_frame, text='Testing Window Size:').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        test_size_widget = ttk.Spinbox(validation_frame, from_=10, to=500, increment=10, width=20)
        test_size_widget.grid(row=1, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['validation.test_size'] = test_size_widget
        ttk.Label(validation_frame, text='Number of days to test predictions', 
                 font=('Arial', 10), foreground='gray').grid(row=1, column=2, padx=5, pady=2, sticky='w')
        
        # Step Size
        ttk.Label(validation_frame, text='Step Size:').grid(row=2, column=0, sticky='w', padx=5, pady=2)
        step_size_widget = ttk.Spinbox(validation_frame, from_=1, to=200, increment=10, width=20)
        step_size_widget.grid(row=2, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['validation.step_size'] = step_size_widget
        ttk.Label(validation_frame, text='Days to advance window in each iteration', 
                 font=('Arial', 10), foreground='gray').grid(row=2, column=2, padx=5, pady=2, sticky='w')
        
        # Min Training Samples
        ttk.Label(validation_frame, text='Min Training Samples:').grid(row=3, column=0, sticky='w', padx=5, pady=2)
        min_training_widget = ttk.Spinbox(validation_frame, from_=50, to=1000, increment=50, width=20)
        min_training_widget.set(100)  # Default value
        min_training_widget.grid(row=3, column=1, padx=5, pady=2, sticky='w')
        self.config_widgets['validation.min_training_samples'] = min_training_widget
        ttk.Label(validation_frame, text='Minimum samples required for training', 
                 font=('Arial', 10), foreground='gray').grid(row=3, column=2, padx=5, pady=2, sticky='w')
        
        # Add informational label
        info_label = ttk.Label(validation_frame, 
                              text="Walk-forward validation tests the model on unseen future data by sliding the training/test windows forward in time",
                              font=('Arial', 8, 'italic'), foreground='gray', wraplength=500)
        info_label.grid(row=4, column=0, columnspan=3, pady=(10, 0), padx=5)
        
        middle_canvas.pack(side="left", fill="both", expand=True)
        middle_scrollbar.pack(side="right", fill="y")
        
        # === RIGHT COLUMN - Walk Forward Control and Console ===
        # Control panel at top
        control_frame = ttk.LabelFrame(right_frame, text="Walk Forward Testing", padding=10)
        control_frame.pack(fill='x', padx=5, pady=(5, 10))
        
        # Create a horizontal layout for status and buttons
        top_row = ttk.Frame(control_frame)
        top_row.pack(fill='x', pady=(0, 10))
        
        # Status on the left
        self.validation_status = ttk.Label(top_row, text="Status: Ready", foreground='green', font=('Arial', 10, 'bold'))
        self.validation_status.pack(side='left', padx=10)
        
        # Run controls on the right
        button_frame = ttk.Frame(top_row)
        button_frame.pack(side='right', padx=10)
        
        self.run_button = ttk.Button(button_frame, text="Run Walk Forward Test", 
                                    command=self.run_walkforward, style='Accent.TButton')
        self.run_button.pack(side='left', padx=2)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self.stop_validation, state='disabled')
        self.stop_button.pack(side='left', padx=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=(0, 10))
        
        
        # Quick stats in a compact format
        stats_frame = ttk.Frame(control_frame)
        stats_frame.pack(fill='x')
        
        self.last_run_stats = ttk.Label(stats_frame, text="No runs yet", font=('Consolas', 9))
        self.last_run_stats.pack(anchor='w')
        
        # Console output - now much bigger
        console_frame = ttk.LabelFrame(right_frame, text="Console Output", padding=5)
        console_frame.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        
        # Make console text area bigger with better font
        self.console_text = scrolledtext.ScrolledText(console_frame, wrap='word', 
                                                      height=35, width=80,
                                                      font=('Consolas', 9))
        self.console_text.pack(fill='both', expand=True)
    
    def setup_results_tab(self):
        """Setup performance statistics tab with charts"""
        # Create main container
        main_frame = ttk.Frame(self.results_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Chart selection at top
        select_frame = ttk.Frame(main_frame)
        select_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(select_frame, text="View:").pack(side='left', padx=5)
        
        self.chart_var = tk.StringVar()
        chart_options = [
            ('Trade Stats', 'tradestats'),
            ('Feature Selection Timeline', 'feature_timeline'),
        ]
        
        for label, value in chart_options:
            ttk.Radiobutton(select_frame, text=label, variable=self.chart_var, 
                           value=value, command=self.switch_view).pack(side='left', padx=10)
        
        self.chart_var.set('tradestats')
        
        ttk.Button(select_frame, text="Refresh", 
                  command=self.refresh_current_view, style='Accent.TButton').pack(side='right', padx=5)
        
        # Content frame that switches between stats and charts
        self.content_frame = ttk.Frame(main_frame)
        self.content_frame.pack(fill='both', expand=True, pady=5)
        
        # Initialize with tradestats view
        self.setup_chart_view()
        self.load_tradestats_charts()
        
    
    def switch_view(self):
        """Switch between different chart views"""
        view_type = self.chart_var.get()
        
        self.setup_chart_view()
        
        if view_type == 'tradestats':
            self.load_tradestats_charts()
        elif view_type == 'feature_timeline':
            self.load_feature_timeline()
    
    def setup_chart_view(self):
        """Setup chart view in content frame"""
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Chart display frame
        chart_display_frame = ttk.LabelFrame(self.content_frame, text="Chart Display", padding=5)
        chart_display_frame.pack(fill='both', expand=True)
        
        # Create label for chart
        self.chart_label = ttk.Label(chart_display_frame, text="No chart loaded")
        self.chart_label.pack(fill='both', expand=True)
    
    def refresh_current_view(self):
        """Refresh the current view"""
        view_type = self.chart_var.get()
        
        if view_type == 'tradestats':
            self.load_tradestats_charts()
        elif view_type == 'feature_timeline':
            self.load_feature_timeline()
    
    def setup_tree_tab(self):
        """Setup the Tree Visualizer tab with educational information"""
        main_frame = ttk.Frame(self.tree_tab)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize tree visualizer
        self.tree_visualizer = TreeVisualizer()
        
        # Educational panel at top
        edu_frame = ttk.LabelFrame(main_frame, text="Understanding Decision Trees", padding=10)
        edu_frame.pack(fill='x', pady=(0, 10))
        
        edu_text = tk.Text(edu_frame, height=8, wrap='word', font=('Arial', 10))
        edu_text.pack(fill='x')
        edu_text.insert('1.0', """Decision trees are machine learning models that make predictions by learning simple decision rules from data features.

How to read the tree:
• Each NODE shows a decision rule (e.g., "feature < threshold")
• GREEN paths indicate the condition is TRUE
• RED paths indicate the condition is FALSE
• LEAF nodes (endpoints) show the final prediction value
• Darker colors indicate stronger predictions (positive or negative)

The ensemble combines multiple trees, each voting on the final prediction. This reduces overfitting and improves accuracy.""")
        edu_text.config(state='disabled', bg='#f0f0f0')
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 5))
        
        # Load model button
        ttk.Button(control_frame, text="Load Final Model", 
                  command=self.load_tree_model).pack(side='left', padx=5)
        
        # Tree selection
        ttk.Label(control_frame, text="Tree #:").pack(side='left', padx=(20, 5))
        self.tree_index_var = tk.IntVar(value=1)
        self.tree_spinbox = ttk.Spinbox(control_frame, from_=1, to=1, width=10, 
                                        textvariable=self.tree_index_var,
                                        command=self.update_tree_display)
        self.tree_spinbox.pack(side='left', padx=5)
        
        # Display mode
        ttk.Label(control_frame, text="View:").pack(side='left', padx=(20, 5))
        self.tree_view_var = tk.StringVar(value='single')
        ttk.Radiobutton(control_frame, text="Single Tree", variable=self.tree_view_var,
                       value='single', command=self.update_tree_display).pack(side='left', padx=5)
        ttk.Radiobutton(control_frame, text="Aggregate Summary", variable=self.tree_view_var,
                       value='aggregate', command=self.update_tree_display).pack(side='left', padx=5)
        
        # Model info display
        info_frame = ttk.LabelFrame(main_frame, text="Model Statistics", padding=5)
        info_frame.pack(fill='x', pady=(0, 5))
        
        self.tree_info_text = tk.Text(info_frame, height=4, wrap='word')
        self.tree_info_text.pack(fill='x')
        self.tree_info_text.insert('1.0', "No model loaded. Click 'Load Final Model' to begin.")
        self.tree_info_text.config(state='disabled')
        
        # Tree visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Tree Visualization", padding=5)
        viz_frame.pack(fill='both', expand=True)
        
        # Create matplotlib canvas for tree display
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt
        
        self.tree_figure = plt.Figure(figsize=(12, 8), facecolor='white')
        self.tree_ax = self.tree_figure.add_subplot(111)
        self.tree_ax.text(0.5, 0.5, "No tree loaded", ha='center', va='center', fontsize=14)
        self.tree_ax.axis('off')
        
        self.tree_canvas = FigureCanvasTkAgg(self.tree_figure, master=viz_frame)
        self.tree_canvas.draw()
        self.tree_canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def load_tree_model(self):
        """Load the saved model for visualization"""
        try:
            # Try to load the model
            success = self.tree_visualizer.load_model()
            
            if success:
                # Update info display
                self.tree_info_text.config(state='normal')
                self.tree_info_text.delete('1.0', tk.END)
                self.tree_info_text.insert('1.0', self.tree_visualizer.get_model_info())
                self.tree_info_text.config(state='disabled')
                
                # Update spinbox range
                n_trees = len(self.tree_visualizer.trees)
                self.tree_spinbox.config(to=n_trees)
                
                # Display first tree or aggregate
                self.update_tree_display()
                
                messagebox.showinfo("Success", f"Model loaded successfully!\n{n_trees} trees available for visualization.")
            else:
                messagebox.showerror("Error", "No trees found in the model file.")
                
        except FileNotFoundError:
            messagebox.showerror("Error", "Model file 'final_model.pkl' not found.\nPlease run a walk-forward test first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def update_tree_display(self):
        """Update the tree visualization based on current selection"""
        if not self.tree_visualizer.trees:
            return
            
        try:
            if self.tree_view_var.get() == 'single':
                # Display single tree
                tree_idx = self.tree_index_var.get() - 1  # Convert to 0-based index
                self.tree_visualizer.plot_single_tree(tree_idx, self.tree_figure, self.tree_ax)
            else:
                # Display aggregate summary
                self.tree_visualizer.plot_aggregate_summary(self.tree_figure, self.tree_ax)
            
            # Refresh canvas
            self.tree_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display tree:\n{str(e)}")
    
    
    def setup_permute_alpha_tab(self):
        """Setup the PermuteAlpha tab for systematic parameter testing"""
        # Main container
        main_frame = ttk.Frame(self.permute_alpha_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left side - Selection lists
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Right side - Results and summary
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True)
        
        # === LEFT SIDE - Selection Lists ===
        
        # Tickers selection
        self.ticker_frame = ttk.LabelFrame(left_frame, text="Tickers", padding=10)
        self.ticker_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        ticker_buttons = ttk.Frame(self.ticker_frame)
        ticker_buttons.pack(fill='x')
        ttk.Button(ticker_buttons, text="Select All", 
                  command=lambda: self.select_all_permute_items('tickers', True)).pack(side='left', padx=2)
        ttk.Button(ticker_buttons, text="Clear All", 
                  command=lambda: self.select_all_permute_items('tickers', False)).pack(side='left', padx=2)
        
        ticker_scroll = ttk.Scrollbar(self.ticker_frame)
        ticker_scroll.pack(side='right', fill='y')
        self.permute_ticker_listbox = tk.Listbox(self.ticker_frame, selectmode='multiple', 
                                                 exportselection=False,
                                                 yscrollcommand=ticker_scroll.set, height=6)
        self.permute_ticker_listbox.pack(fill='both', expand=True)
        ticker_scroll.config(command=self.permute_ticker_listbox.yview)
        
        # Targets selection
        self.target_frame = ttk.LabelFrame(left_frame, text="Targets *", padding=10)
        self.target_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        target_buttons = ttk.Frame(self.target_frame)
        target_buttons.pack(fill='x')
        ttk.Button(target_buttons, text="Select All",
                  command=lambda: self.select_all_permute_items('targets', True)).pack(side='left', padx=2)
        ttk.Button(target_buttons, text="Clear All",
                  command=lambda: self.select_all_permute_items('targets', False)).pack(side='left', padx=2)
        
        target_scroll = ttk.Scrollbar(self.target_frame)
        target_scroll.pack(side='right', fill='y')
        self.permute_target_listbox = tk.Listbox(self.target_frame, selectmode='multiple',
                                                 exportselection=False,
                                                 yscrollcommand=target_scroll.set, height=6)
        self.permute_target_listbox.pack(fill='both', expand=True)
        target_scroll.config(command=self.permute_target_listbox.yview)
        
        # Hours selection
        self.hour_frame = ttk.LabelFrame(left_frame, text="Hours", padding=10)
        self.hour_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        hour_buttons = ttk.Frame(self.hour_frame)
        hour_buttons.pack(fill='x')
        ttk.Button(hour_buttons, text="Select All",
                  command=lambda: self.select_all_permute_items('hours', True)).pack(side='left', padx=2)
        ttk.Button(hour_buttons, text="Clear All",
                  command=lambda: self.select_all_permute_items('hours', False)).pack(side='left', padx=2)
        
        hour_scroll = ttk.Scrollbar(self.hour_frame)
        hour_scroll.pack(side='right', fill='y')
        self.permute_hour_listbox = tk.Listbox(self.hour_frame, selectmode='multiple',
                                               exportselection=False,
                                               yscrollcommand=hour_scroll.set, height=6)
        self.permute_hour_listbox.pack(fill='both', expand=True)
        hour_scroll.config(command=self.permute_hour_listbox.yview)
        
        # Features selection (common for all permutations)
        features_frame = ttk.LabelFrame(left_frame, text="Features (Used for ALL permutations) *", padding=10)
        features_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        feature_buttons = ttk.Frame(features_frame)
        feature_buttons.pack(fill='x')
        ttk.Button(feature_buttons, text="Select All",
                  command=lambda: self.select_all_permute_items('features', True)).pack(side='left', padx=2)
        ttk.Button(feature_buttons, text="Clear All",
                  command=lambda: self.select_all_permute_items('features', False)).pack(side='left', padx=2)
        ttk.Button(feature_buttons, text="Use Model Tab Selection",
                  command=self.use_model_tab_features).pack(side='left', padx=2)
        
        feature_scroll = ttk.Scrollbar(features_frame)
        feature_scroll.pack(side='right', fill='y')
        self.permute_feature_listbox = tk.Listbox(features_frame, selectmode='multiple',
                                                  exportselection=False,
                                                  yscrollcommand=feature_scroll.set, height=6)
        self.permute_feature_listbox.pack(fill='both', expand=True)
        feature_scroll.config(command=self.permute_feature_listbox.yview)
        
        # Direction selection (radio buttons since it's mutually exclusive)
        direction_frame = ttk.LabelFrame(left_frame, text="Direction", padding=10)
        direction_frame.pack(fill='x', pady=(0, 10))
        
        self.permute_direction_var = tk.StringVar(value="longonly")
        
        ttk.Radiobutton(direction_frame, text="Long Only", 
                       variable=self.permute_direction_var, value="longonly").pack(anchor='w', pady=2)
        ttk.Radiobutton(direction_frame, text="Short Only",
                       variable=self.permute_direction_var, value="shortonly").pack(anchor='w', pady=2)
        ttk.Radiobutton(direction_frame, text="Both (Long and Short separately)",
                       variable=self.permute_direction_var, value="both").pack(anchor='w', pady=2)
        
        # Permutation count and estimate
        estimate_frame = ttk.LabelFrame(left_frame, text="Permutation Estimate", padding=10)
        estimate_frame.pack(fill='x', pady=(0, 10))
        
        self.permute_count_label = ttk.Label(estimate_frame, text="Total Permutations: 0", font=('Arial', 11, 'bold'))
        self.permute_count_label.pack(anchor='w', pady=2)
        
        self.permute_time_label = ttk.Label(estimate_frame, text="Estimated Time: 0 minutes")
        self.permute_time_label.pack(anchor='w', pady=2)
        
        self.permute_per_second_label = ttk.Label(estimate_frame, text="(~5 seconds per permutation)", foreground='gray')
        self.permute_per_second_label.pack(anchor='w', pady=2)
        
        # Add timing button
        timing_button_frame = ttk.Frame(estimate_frame)
        timing_button_frame.pack(fill='x', pady=(5, 0))
        
        self.time_single_button = ttk.Button(timing_button_frame, text="Time Single Iteration", 
                                            command=self.time_single_iteration)
        self.time_single_button.pack(side='left', padx=(0, 5))
        
        self.timing_status_label = ttk.Label(timing_button_frame, text="", foreground='blue')
        self.timing_status_label.pack(side='left')
        
        # Parallel processing options
        parallel_frame = ttk.LabelFrame(left_frame, text="Parallel Processing", padding=10)
        parallel_frame.pack(fill='x', pady=(0, 10))
        
        # Enable parallel processing checkbox for PermuteAlpha
        self.permute_use_parallel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parallel_frame, text="Use Parallel Processing", 
                       variable=self.permute_use_parallel_var,
                       command=self.update_parallel_settings).pack(anchor='w')
        
        # Number of workers
        workers_frame = ttk.Frame(parallel_frame)
        workers_frame.pack(fill='x', pady=5)
        
        ttk.Label(workers_frame, text="Workers:").pack(side='left')
        
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        default_workers = max(1, max_workers - 1)
        
        self.num_workers_var = tk.IntVar(value=default_workers)
        self.workers_spin = ttk.Spinbox(workers_frame, from_=1, to=max_workers, 
                                        textvariable=self.num_workers_var,
                                        width=5, state='normal')
        self.workers_spin.pack(side='left', padx=5)
        
        ttk.Label(workers_frame, text=f"(Max: {max_workers} cores)", 
                 font=('Arial', 10), foreground='gray').pack(side='left', padx=5)
        
        # Speedup estimate
        self.speedup_label = ttk.Label(parallel_frame, 
                                       text=f"Estimated speedup: {default_workers:.1f}x faster",
                                       font=('Arial', 10), foreground='green')
        self.speedup_label.pack(anchor='w', pady=2)
        
        # Control buttons
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill='x', pady=10)
        
        self.permute_go_button = ttk.Button(control_frame, text="Run Permutations", 
                                           command=self.run_permute_alpha, style='Accent.TButton')
        self.permute_go_button.pack(side='left', padx=5)
        
        self.permute_stop_button = ttk.Button(control_frame, text="Stop", 
                                             command=self.stop_permute_alpha, state='disabled')
        self.permute_stop_button.pack(side='left', padx=5)
        
        # Save/Load settings buttons
        ttk.Separator(control_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        
        ttk.Button(control_frame, text="Save Settings", 
                  command=self.save_permute_settings).pack(side='left', padx=2)
        
        ttk.Button(control_frame, text="Load Settings", 
                  command=self.load_permute_settings).pack(side='left', padx=2)
        
        # Progress
        self.permute_progress_label = ttk.Label(control_frame, text="Ready")
        self.permute_progress_label.pack(side='left', padx=20)
        
        # Overall permutation progress
        ttk.Label(left_frame, text="Overall Progress:", font=('Arial', 10)).pack(anchor='w', pady=(5, 2))
        self.permute_progress = ttk.Progressbar(left_frame, mode='determinate')
        self.permute_progress.pack(fill='x', pady=(0, 5))
        
        # Individual permutation progress  
        ttk.Label(left_frame, text="Current Permutation:", font=('Arial', 10)).pack(anchor='w', pady=(5, 2))
        self.permute_individual_progress = ttk.Progressbar(left_frame, mode='determinate')
        self.permute_individual_progress.pack(fill='x', pady=(0, 5))
        
        # Current step label
        self.permute_step_label = ttk.Label(left_frame, text="", font=('Arial', 11), foreground='gray')
        self.permute_step_label.pack(anchor='w')
        
        # === RIGHT SIDE - Results ===
        
        # Overall Statistics at the top
        overall_stats_frame = ttk.LabelFrame(right_frame, text="Overall Statistics", padding=10)
        overall_stats_frame.pack(fill='x', pady=(0, 10))
        
        self.overall_stats_text = tk.Text(overall_stats_frame, height=8, wrap='word', 
                                          font=('Consolas', 11))
        self.overall_stats_text.pack(fill='x')
        self.overall_stats_text.insert('1.0', "No permutations run yet")
        self.overall_stats_text.config(state='disabled')
        
        # Summary report as a sortable table
        summary_frame = ttk.LabelFrame(right_frame, text="Summary Report (Click column headers to sort)", padding=10)
        summary_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Create treeview for results table with tradestats.md metrics
        columns = ('Ticker', 'Direction', 'Target', 'Hour', 'NumObs', 'Years', 'NumTrades', 
                  'TradeFreq', 'Win%', 'AvgLoss', 'AvgProfit', 'AvgPnL', 'Expectancy',
                  'BestDay', 'WorstDay', 'Annual%', 'MaxDD%', 'Sharpe', 'ProfitDD', 'UPI')
        
        self.permute_summary_tree = ttk.Treeview(summary_frame, columns=columns, 
                                                 show='headings', height=12)
        
        # Configure columns with appropriate widths
        col_widths = {
            'Ticker': 55,
            'Direction': 65,
            'Target': 90,
            'Hour': 40,
            'NumObs': 60,
            'Years': 45,
            'NumTrades': 65,
            'TradeFreq': 65,
            'Win%': 50,
            'AvgLoss': 60,
            'AvgProfit': 65,
            'AvgPnL': 55,
            'Expectancy': 70,
            'BestDay': 60,
            'WorstDay': 65,
            'Annual%': 60,
            'MaxDD%': 60,
            'Sharpe': 55,
            'ProfitDD': 60,
            'UPI': 45
        }
        
        for col in columns:
            self.permute_summary_tree.heading(col, text=col,
                                             command=lambda c=col: self.sort_permute_results(c))
            self.permute_summary_tree.column(col, width=col_widths.get(col, 80), anchor='center')
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(summary_frame, orient='vertical', 
                                 command=self.permute_summary_tree.yview)
        y_scroll.pack(side='right', fill='y')
        
        x_scroll = ttk.Scrollbar(summary_frame, orient='horizontal',
                                 command=self.permute_summary_tree.xview)
        x_scroll.pack(side='bottom', fill='x')
        
        self.permute_summary_tree.configure(yscrollcommand=y_scroll.set,
                                           xscrollcommand=x_scroll.set)
        self.permute_summary_tree.pack(fill='both', expand=True)
        
        # Store data for sorting
        self.permute_results_data = []
        self.permute_sort_reverse = {}
        
        # Failed combinations log
        failed_frame = ttk.LabelFrame(right_frame, text="Failed Combinations", padding=10)
        failed_frame.pack(fill='both', expand=True)
        
        failed_scroll = ttk.Scrollbar(failed_frame)
        failed_scroll.pack(side='right', fill='y')
        
        self.permute_failed_text = scrolledtext.ScrolledText(failed_frame, wrap=tk.WORD, height=8,
                                                            yscrollcommand=failed_scroll.set)
        self.permute_failed_text.pack(fill='both', expand=True)
        failed_scroll.config(command=self.permute_failed_text.yview)
        
        # Output directory selection
        output_frame = ttk.LabelFrame(right_frame, text="Output Settings", padding=10)
        output_frame.pack(fill='x', pady=10)
        
        path_frame = ttk.Frame(output_frame)
        path_frame.pack(fill='x')
        
        ttk.Label(path_frame, text="Output Path:").pack(side='left')
        self.permute_output_path = tk.StringVar(value="PermuteAlpha_Results")
        self.permute_output_entry = ttk.Entry(path_frame, textvariable=self.permute_output_path, width=30)
        self.permute_output_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        ttk.Button(path_frame, text="Browse", 
                  command=self.browse_permute_output_path).pack(side='left', padx=2)
        ttk.Button(path_frame, text="Open", 
                  command=self.open_permute_results_folder).pack(side='left', padx=2)
        
        # Initialize listboxes with data if available
        self.update_permute_selections()
        
        # Bind selection events to update count
        self.permute_ticker_listbox.bind('<<ListboxSelect>>', lambda e: self.update_permute_count())
        self.permute_target_listbox.bind('<<ListboxSelect>>', lambda e: self.update_permute_count())
        self.permute_hour_listbox.bind('<<ListboxSelect>>', lambda e: self.update_permute_count())
        self.permute_feature_listbox.bind('<<ListboxSelect>>', lambda e: self.update_permute_count())
        self.permute_direction_var.trace('w', lambda *args: self.update_permute_count())
        
        # Initialize state
        self.permute_running = False
        self.permute_thread = None
        self.seconds_per_permutation = 5  # Default estimate
        
        # Initial count update
        self.update_permute_count()
        
    # === Helper Methods ===
    
    def validate_configuration(self):
        """Validate that all required configuration is present"""
        errors = []
        
        # Check data file
        if not self.data_file_var.get():
            errors.append("No data file specified")
        elif not os.path.exists(self.data_file_var.get()):
            errors.append(f"Data file not found: {self.data_file_var.get()}")
        
        # Check features and targets
        features = list(self.feature_listbox.get(0, tk.END))
        targets = list(self.target_listbox.get(0, tk.END))
        
        if not features:
            errors.append("No features selected")
        if not targets:
            errors.append("No targets selected")
        
        # Check model configuration
        selected_features = [f for f, var in self.feature_check_vars.items() if var.get()]
        if not selected_features:
            errors.append("No features selected for model")
        
        if not self.selected_target_var.get():
            errors.append("No target selected for model")
        
        # Check dates
        try:
            pd.to_datetime(self.validation_start_var.get())
            pd.to_datetime(self.validation_end_var.get())
        except:
            errors.append("Invalid date format")
        
        return errors
    
    
    def run_walkforward(self):
        """Run walk forward validation with pre-validation checks"""
        # Validate configuration
        errors = self.validate_configuration()
        if errors:
            messagebox.showerror("Configuration Error", 
                               "Please fix the following errors:\n\n" + "\n".join(errors))
            return
        
        # Update status
        self.validation_status.config(text="Status: Running...", foreground='orange')
        
        # Configuration is auto-saved via apply_selections_to_config
        
        # Save current config to INI file for walkforward script
        self.apply_selections_to_config()
        
        self.console_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.console_text.insert(tk.END, "Starting walk forward validation...\n")
        self.console_text.insert(tk.END, "Configuration saved to history.\n")
        self.console_text.see(tk.END)
        
        # Update UI
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar.start()
        
        # Start validation in separate thread
        thread = threading.Thread(target=self._run_validation_thread)
        thread.daemon = True
        thread.start()
    
    def _run_validation_thread(self):
        """Run validation in background thread"""
        try:
            # Always use sequential processing
            self.root.after(0, self._update_console, "Starting walk-forward validation...\n")
            self._run_sequential_validation()
                
        except Exception as e:
            self.root.after(0, self._validation_error, str(e))
    
    def _run_sequential_validation(self):
        """Run validation using the original sequential method"""
        # Run the walk-forward script
        process = subprocess.Popen(
            ['python', 'OMtree_walkforward.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        self.validation_process = process
        
        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                self.root.after(0, self._update_console, line)
        
        process.wait()
        
        # Update UI on completion
        self.root.after(0, self._validation_complete, process.returncode)
    
    def _update_console(self, text):
        """Update console output (called from main thread)"""
        self.console_text.insert(tk.END, text)
        self.console_text.see(tk.END)
        
        # Check for completion indicators
        if "hit rate" in text.lower():
            self.last_run_stats.config(text=text.strip())
    
    def _validation_complete(self, return_code):
        """Handle validation completion"""
        self.progress_bar.stop()
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        if return_code == 0:
            self.validation_status.config(text="Status: Completed Successfully", foreground='green')
            self.console_text.insert(tk.END, "\nValidation completed successfully!\n")
            
            # Switch to Performance Stats tab
            self.root.after(500, lambda: self.notebook.select(self.results_tab))
            
            # Set view to tradestats and load the charts
            self.root.after(700, lambda: self.chart_var.set('tradestats'))
            self.root.after(900, self.load_tradestats_charts)
        else:
            self.validation_status.config(text="Status: Completed with Errors", foreground='red')
            self.console_text.insert(tk.END, f"\nValidation completed with errors (code {return_code})\n")
    
    def _validation_error(self, error_msg):
        """Handle validation error"""
        self.progress_bar.stop()
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.validation_status.config(text="Status: Error", foreground='red')
        
        self.console_text.insert(tk.END, f"\nError: {error_msg}\n")
        messagebox.showerror("Validation Error", f"An error occurred:\n{error_msg}")
    
    def stop_validation(self):
        """Stop running validation"""
        if self.validation_process:
            self.validation_process.terminate()
            self.console_text.insert(tk.END, "\nValidation stopped by user.\n")
            self.validation_status.config(text="Status: Stopped", foreground='orange')
    
    # === Existing methods (simplified versions) ===
    
    def browse_data_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.data_file_var.set(filename)
    
    def apply_filters(self, event=None):
        """Apply ticker and/or hour filters to the loaded data"""
        if not hasattr(self, 'df_unfiltered'):
            return
            
        # Start with unfiltered data
        self.df = self.df_unfiltered.copy()
        
        # Apply ticker filter if applicable
        selected_ticker = self.ticker_filter_var.get()
        if 'Ticker' in self.df.columns and selected_ticker != 'All':
            self.df = self.df[self.df['Ticker'] == selected_ticker].copy()
        
        # Apply hour filter if applicable
        selected_hour = self.hour_filter_var.get()
        if 'Hour' in self.df.columns and selected_hour != 'All':
            try:
                hour_val = int(selected_hour)
                self.df = self.df[self.df['Hour'] == hour_val].copy()
            except:
                pass
        
        # Update info label with filter status
        rows_text = f"{len(self.df):,} rows"
        filters_applied = []
        
        if 'Ticker' in self.df_unfiltered.columns and selected_ticker != 'All':
            filters_applied.append(f"ticker={selected_ticker}")
        
        if 'Hour' in self.df_unfiltered.columns and selected_hour != 'All':
            filters_applied.append(f"hour={selected_hour}")
        
        if filters_applied:
            rows_text += f" (filtered: {', '.join(filters_applied)})"
        
        self.data_info_label.config(
            text=f"Loaded: {rows_text} × {len(self.df.columns)} columns",
            foreground='green'
        )
        
        # Update the timeline if dates are present
        if 'Date' in self.df.columns:
            self.update_timeline()
            
        # Update PermuteAlpha selections
        self.update_permute_selections()
    
    def load_data_file(self):
        try:
            filename = self.data_file_var.get()
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File not found: {filename}")
            
            self.df = pd.read_csv(filename)
            self.df_unfiltered = self.df.copy()  # Keep unfiltered copy
            self.available_columns = list(self.df.columns)
            
            # Check what filters are needed
            has_ticker = 'Ticker' in self.df.columns
            has_hour = 'Hour' in self.df.columns
            
            # Show/hide filters based on data
            if has_ticker or has_hour:
                self.filters_frame.pack(fill='x', pady=(5, 0))
                
                # Handle ticker filter
                if has_ticker:
                    self.ticker_filter_frame.pack(side='left', padx=(0, 20))
                    unique_tickers = sorted(self.df['Ticker'].unique())
                    ticker_values = ['All'] + list(unique_tickers)
                    self.ticker_filter_combo['values'] = ticker_values
                    self.ticker_filter_combo.set('All')
                else:
                    self.ticker_filter_frame.pack_forget()
                    self.ticker_filter_combo.set('All')
                
                # Handle hour filter
                if has_hour:
                    self.time_filter_frame.pack(side='left')
                    unique_hours = sorted(self.df['Hour'].unique())
                    hour_values = ['All'] + [str(h) for h in unique_hours]
                    self.hour_filter_combo['values'] = hour_values
                    self.hour_filter_combo.set('All')
                else:
                    self.time_filter_frame.pack_forget()
                    self.hour_filter_combo.set('All')
            else:
                # Hide all filters
                self.filters_frame.pack_forget()
                self.ticker_filter_combo.set('All')
                self.hour_filter_combo.set('All')
            
            # Clear existing selections
            self.available_listbox.delete(0, tk.END)
            self.feature_listbox.delete(0, tk.END)
            self.target_listbox.delete(0, tk.END)
            
            # Add all columns to available
            for col in self.available_columns:
                if col not in ['Date', 'Time', 'Hour', 'Ticker']:
                    self.available_listbox.insert(tk.END, col)
            
            # Auto-detect and populate features/targets
            for col in self.available_columns:
                if col.startswith('Ret_fwd'):
                    self.target_listbox.insert(tk.END, col)
                elif col.startswith('Ret_') and not col.startswith('Ret_fwd'):
                    self.feature_listbox.insert(tk.END, col)
                elif col in ['NoneTradePrice']:
                    self.feature_listbox.insert(tk.END, col)
            
            # Update info label
            rows_text = f"{len(self.df):,} rows"
            info_parts = []
            if has_ticker:
                n_tickers = len(self.df['Ticker'].unique())
                info_parts.append(f"{n_tickers} ticker{'s' if n_tickers > 1 else ''}")
            if has_hour:
                info_parts.append("hourly")
            if info_parts:
                rows_text += f" ({', '.join(info_parts)})"
            self.data_info_label.config(
                text=f"Loaded: {rows_text} × {len(self.df.columns)} columns",
                foreground='green'
            )
            
            # Set date columns
            self.date_column_var = tk.StringVar(value="Date" if "Date" in self.df.columns else "")
            self.time_column_var = tk.StringVar(value="Time" if "Time" in self.df.columns else "")
            
            # Update timeline using flexible date parser
            date_columns = FlexibleDateParser.get_date_columns(self.df)
            
            # Get date/time column info from config or auto-detect
            config = configparser.ConfigParser()
            date_col = None
            time_col = None
            datetime_col = None
            dayfirst = None
            
            if config.read('OMtree_config.ini'):
                date_col = config['data'].get('date_column', date_columns.get('date_column'))
                time_col = config['data'].get('time_column', date_columns.get('time_column'))
                datetime_col = date_columns.get('datetime_column')
                
                # Check dayfirst setting
                if 'validation' in config and 'date_format_dayfirst' in config['validation']:
                    dayfirst_str = config['validation']['date_format_dayfirst'].lower()
                    if dayfirst_str == 'true':
                        dayfirst = True
                    elif dayfirst_str == 'false':
                        dayfirst = False
            else:
                date_col = date_columns.get('date_column')
                time_col = date_columns.get('time_column')
                datetime_col = date_columns.get('datetime_column')
            
            # Try to parse dates
            try:
                parsed_dates = FlexibleDateParser.parse_dates(
                    self.df,
                    date_column=date_col,
                    time_column=time_col,
                    datetime_column=datetime_col,
                    dayfirst=dayfirst
                )
                
                self.df['parsed_datetime'] = parsed_dates
                self.data_start_date = parsed_dates.min()
                self.data_end_date = parsed_dates.max()
                self.update_timeline()
                
            except Exception as e:
                print(f"Could not parse dates for timeline: {e}")
                # Try to find any existing date column
                for col in ['Date', 'date', 'DateTime', 'parsed_datetime']:
                    if col in self.df.columns:
                        try:
                            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                            self.data_start_date = self.df[col].min()
                            self.data_end_date = self.df[col].max()
                            self.update_timeline()
                            break
                        except:
                            continue
            
            # Update feature/target selection in model tab
            self.update_feature_target_selection()
            
        except Exception as e:
            self.data_info_label.config(text=f"Error: {str(e)}", foreground='red')
            messagebox.showerror("Load Error", f"Failed to load data:\n{str(e)}")
    
    def update_timeline(self):
        """Update timeline visualization"""
        if hasattr(self, 'data_start_date') and hasattr(self, 'data_end_date'):
            try:
                validation_start = pd.to_datetime(self.validation_start_var.get())
                validation_end = pd.to_datetime(self.validation_end_var.get())
                
                self.timeline.update_timeline(
                    self.data_start_date, 
                    self.data_end_date,
                    validation_start,
                    validation_end
                )
            except Exception as e:
                print(f"Timeline update error: {e}")
    
    def move_to_features(self):
        """Move selected items to features"""
        selected = self.available_listbox.curselection()
        for idx in reversed(selected):
            item = self.available_listbox.get(idx)
            if item not in self.feature_listbox.get(0, tk.END):
                self.feature_listbox.insert(tk.END, item)
    
    def remove_from_features(self):
        """Remove selected items from features"""
        selected = self.feature_listbox.curselection()
        for idx in reversed(selected):
            self.feature_listbox.delete(idx)
    
    def move_to_targets(self):
        """Move selected items to targets"""
        selected = self.available_listbox.curselection()
        for idx in reversed(selected):
            item = self.available_listbox.get(idx)
            if item not in self.target_listbox.get(0, tk.END):
                self.target_listbox.insert(tk.END, item)
    
    def remove_from_targets(self):
        """Remove selected items from targets"""
        selected = self.target_listbox.curselection()
        for idx in reversed(selected):
            self.target_listbox.delete(idx)
    
    def update_feature_display(self):
        """Deprecated - engineered features removed"""
        pass
    
    def update_trees_label(self):
        """Update the trees count label based on method and selected features"""
        try:
            if hasattr(self, 'trees_label') and hasattr(self, 'config_widgets'):
                method = self.config_widgets.get('model.n_trees_method', None)
                n_trees_widget = self.config_widgets.get('model.n_trees', None)
                
                if method and n_trees_widget:
                    method_value = method.get()
                    n_trees_base = int(float(n_trees_widget.get()))
                    
                    if method_value == 'per_feature':
                        # Count selected features
                        if hasattr(self, 'feature_check_vars'):
                            n_features = len([var for var in self.feature_check_vars.values() if var.get()])
                        else:
                            # Fallback: count from config
                            selected = self.config.get('data', 'selected_features', fallback='')
                            n_features = len([f.strip() for f in selected.split(',') if f.strip()])
                        
                        if n_features == 0:
                            n_features = 1  # Default to 1 if none selected
                        actual_trees = n_trees_base * n_features
                        self.trees_label.config(text=f'Total: {actual_trees} trees')
                    else:
                        self.trees_label.config(text='')
        except Exception as e:
            print(f"Error updating trees label: {e}")
    
    def restore_feature_target_selection(self, config):
        """Restore feature and target selection from config"""
        # First update the UI to create checkboxes
        self.update_feature_target_selection()
        
        # Then set the selected features
        if 'data' in config:
            if 'selected_features' in config['data']:
                selected = [f.strip() for f in config['data']['selected_features'].split(',')]
                for feature, var in self.feature_check_vars.items():
                    var.set(feature in selected)
            
            if 'target_column' in config['data']:
                self.selected_target_var.set(config['data']['target_column'])
    
    def update_feature_target_selection(self):
        """Update feature and target selection in model config"""
        # Clear existing
        for widget in self.feature_check_frame.winfo_children():
            widget.destroy()
        for widget in self.target_radio_frame.winfo_children():
            widget.destroy()
        for widget in self.noise_features_frame.winfo_children():
            widget.destroy()
        
        # No longer need to clear engineered features (removed)
        
        # First remove any existing RandomNoise features from listbox
        all_items = list(self.feature_listbox.get(0, tk.END))
        for idx in range(len(all_items) - 1, -1, -1):
            if all_items[idx].startswith('RandomNoise_'):
                self.feature_listbox.delete(idx)
        
        # Add noise features to the feature_listbox if requested
        num_noise = self.num_noise_features.get()
        if num_noise > 0:
            for i in range(num_noise):
                noise_name = f"RandomNoise_{i+1}"
                self.feature_listbox.insert(tk.END, noise_name)
        
        # Add features from feature listbox (now includes noise features)
        features = list(self.feature_listbox.get(0, tk.END))
        self.feature_check_vars = {}
        
        # Separate regular and noise features
        regular_features = [f for f in features if not f.startswith('RandomNoise_')]
        noise_features = [f for f in features if f.startswith('RandomNoise_')]
        
        # Add regular features to checkboxes
        for i, feature in enumerate(regular_features):
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(self.feature_check_frame, text=feature, variable=var)
            cb.grid(row=i//3, column=i%3, sticky='w', padx=5, pady=2)
            self.feature_check_vars[feature] = var
        
        # Display noise features in separate section
        if noise_features:
            ttk.Label(self.noise_features_frame, text="Noise features added:", 
                     font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky='w', pady=(0,5))
            
            for i, noise_name in enumerate(noise_features):
                var = tk.BooleanVar(value=True)  # Default to selected
                cb = ttk.Checkbutton(self.noise_features_frame, text=noise_name, variable=var)
                cb.grid(row=1 + i//4, column=i%4, sticky='w', padx=5, pady=2)
                self.feature_check_vars[noise_name] = var
        
        # Add targets from target listbox
        targets = list(self.target_listbox.get(0, tk.END))
        
        for i, target in enumerate(targets):
            rb = ttk.Radiobutton(self.target_radio_frame, text=target, 
                               variable=self.selected_target_var, value=target)
            rb.grid(row=i//3, column=i%3, sticky='w', padx=5, pady=2)
    
    def select_all_features(self, select=True):
        """Select or deselect all feature checkboxes"""
        for var in self.feature_check_vars.values():
            var.set(select)
    
    def save_data_config_to_history(self):
        """Save current data configuration to history"""
        features = list(self.feature_listbox.get(0, tk.END))
        targets = list(self.target_listbox.get(0, tk.END))
        
        if not features or not targets:
            messagebox.showwarning("Incomplete Configuration", 
                                  "Please select features and targets before saving.")
            return
        
        config_id = self.config_manager.save_data_config(
            csv_file=self.data_file_var.get(),
            validation_start=self.validation_start_var.get(),
            validation_end=self.validation_end_var.get(),
            date_column=self.date_column_var.get() if hasattr(self, 'date_column_var') else 'Date',
            time_column=self.time_column_var.get() if hasattr(self, 'time_column_var') else '',
            features=features,
            targets=targets
        )
        
        # Data history removed
        return config_id
    
    def save_model_config_to_history(self):
        """Save current model configuration (compatibility method)"""
        selected_features = [f for f, var in self.feature_check_vars.items() if var.get()]
        
        if not selected_features or not self.selected_target_var.get():
            messagebox.showwarning("Incomplete Configuration", 
                                  "Please select model features and target before saving.")
            return None
        
        model_params = {}
        for key, widget in self.config_widgets.items():
            value = widget.get()
            section, param = key.split('.')
            
            if param in ['normalize_features', 'normalize_target', 'detrend_features']:
                value = value.lower() == 'true'
            elif param != 'model_type':
                try:
                    value = float(value)
                except:
                    pass
            
            model_params[param] = value
        
        # Save to config manager if it exists
        if hasattr(self, 'config_manager'):
            config_id = self.config_manager.save_model_config(
                selected_features=selected_features,
                selected_target=self.selected_target_var.get(),
                model_params=model_params
            )
        else:
            config_id = "model_config_" + datetime.now().strftime('%Y%m%d_%H%M%S')
        return config_id
    
    # Data history refresh removed
    
    # Model history refresh removed
    
    # Data config history loading removed
        pass  # Removed
    
    def load_model_config_from_history(self, event=None):
        """Compatibility method - no longer needed without history"""
        pass
    
    def apply_selections_to_config(self):
        """Apply current selections to config file for walkforward script"""
        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        # Update the CSV file to match what's loaded in the GUI
        if 'data' not in config:
            config['data'] = {}
        config['data']['csv_file'] = self.data_file_var.get()
        
        # Save filters if applicable
        if hasattr(self, 'df'):
            if 'Hour' in self.df.columns:
                config['data']['hour_filter'] = self.hour_filter_var.get()
            if 'Ticker' in self.df.columns:
                config['data']['ticker_filter'] = self.ticker_filter_var.get()
        
        # Update all available features and targets from the loaded file
        all_features = []
        all_targets = []
        for i in range(self.feature_listbox.size()):
            all_features.append(self.feature_listbox.get(i))
        for i in range(self.target_listbox.size()):
            all_targets.append(self.target_listbox.get(i))
        
        if all_features:
            config['data']['feature_columns'] = ','.join(all_features)
        if all_targets:
            config['data']['all_targets'] = ','.join(all_targets)
        
        # Get selected features from checkboxes
        selected_features = [f for f, var in self.feature_check_vars.items() if var.get()]
        
        # Update config
        if selected_features and 'data' in config:
            config['data']['selected_features'] = ','.join(selected_features)
        
        if self.selected_target_var.get() and 'data' in config:
            config['data']['target_column'] = self.selected_target_var.get()
        
        # Update validation dates
        if 'validation' not in config:
            config['validation'] = {}
        
        config['validation']['validation_start_date'] = self.validation_start_var.get()
        config['validation']['validation_end_date'] = self.validation_end_var.get()
        
        # Update model parameters
        for key, widget in self.config_widgets.items():
            section, param = key.split('.')
            if section not in config:
                config[section] = {}
            
            # Handle BooleanVar specially
            if isinstance(widget, tk.BooleanVar):
                config[section][param] = 'true' if widget.get() else 'false'
            else:
                config[section][param] = widget.get()
        
        # Save to file
        with open(self.config_file, 'w') as f:
            config.write(f)
    
    def load_config(self):
        """Load configuration from INI file"""
        if os.path.exists(self.config_file):
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            # Load data settings
            if 'data' in config:
                self.data_file_var.set(config['data'].get('csv_file', ''))
                # Load filters if present
                if 'hour_filter' in config['data']:
                    self.hour_filter_var.set(config['data'].get('hour_filter', 'All'))
                if 'ticker_filter' in config['data']:
                    self.ticker_filter_var.set(config['data'].get('ticker_filter', 'All'))
            
            # Load validation dates
            if 'validation' in config:
                self.validation_start_var.set(config['validation'].get('validation_start_date', '2010-01-01'))
                self.validation_end_var.set(config['validation'].get('validation_end_date', '2015-12-31'))
            
            # Load feature and target columns into listboxes
            if 'data' in config:
                # Load feature columns
                if 'feature_columns' in config['data']:
                    features = [f.strip() for f in config['data']['feature_columns'].split(',')]
                    self.feature_listbox.delete(0, tk.END)
                    for feature in features:
                        self.feature_listbox.insert(tk.END, feature)
                
                # Load target columns
                if 'all_targets' in config['data']:
                    targets = [t.strip() for t in config['data']['all_targets'].split(',')]
                    self.target_listbox.delete(0, tk.END)
                    for target in targets:
                        self.target_listbox.insert(tk.END, target)
                
                # Load selected features and target
                if 'selected_features' in config['data'] and 'target_column' in config['data']:
                    # This will populate the checkboxes after listboxes are loaded
                    self.root.after(100, self.restore_feature_target_selection, config)
            
            # Load model parameters
            for key, widget in self.config_widgets.items():
                section, param = key.split('.')
                if section in config and param in config[section]:
                    # Handle BooleanVar specially
                    if isinstance(widget, tk.BooleanVar):
                        widget.set(config[section][param].lower() == 'true')
                    else:
                        widget.set(config[section][param])
            
            # Trigger normalization method update to show/hide correct widgets
            if 'preprocessing.normalization_method' in self.config_widgets:
                norm_widget = self.config_widgets['preprocessing.normalization_method']
                method = norm_widget.get()
                if method == 'AVS':
                    self.norm_method_desc.config(text='AVS: Adaptive Volatility Scaling')
                    # Show AVS parameters
                    for widget in self.avs_widgets:
                        widget.grid()
                    # Hide IQR parameters
                    for widget in self.iqr_widgets:
                        widget.grid_remove()
                else:
                    self.norm_method_desc.config(text='IQR: Interquartile Range normalization')
                    # Show IQR parameters
                    for widget in self.iqr_widgets:
                        widget.grid()
                    # Hide AVS parameters
                    for widget in self.avs_widgets:
                        widget.grid_remove()
            
            # Update feature selection controls state
            self.toggle_feature_selection_settings()
            
            # Update engineered features display after loading
            if hasattr(self, 'update_feature_display'):
                self.update_feature_display()
    
    
    def load_regular_charts(self):
        """Load and display regular performance charts"""
        try:
            # Clear any existing chart first
            for widget in self.chart_label.winfo_children():
                widget.destroy()
            
            # Get results file from config
            import configparser
            config = configparser.ConfigParser(inline_comment_prefixes='#')
            config.read('OMtree_config.ini')
            results_file = config.get('output', 'results_file', fallback='OMtree_results.csv')
            
            # Check if we have results data
            if not os.path.exists(results_file):
                self.chart_label.config(text="No results available. Please run walk-forward validation first.")
                return
            
            # Load results - always reload from disk to get latest
            df = pd.read_csv(results_file)
            print(f"Loaded {len(df)} rows from {results_file}")
            
            # Get model type - already have config loaded
            model_type = config.get('model', 'model_type', fallback='longonly')
            print(f"Model type: {model_type}")
            
            # Create basic performance charts
            from src.performance_stats import calculate_performance_stats
            stats_dict = calculate_performance_stats(df, model_type=model_type)
            
            # Create equity curve and drawdown chart
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            fig = plt.Figure(figsize=(14, 8), facecolor='white')
            
            # Equity curve
            ax1 = fig.add_subplot(2, 1, 1)
            if 'equity_curve' in stats_dict:
                equity = stats_dict['equity_curve']
                ax1.plot(range(len(equity)), equity, 'b-', linewidth=2)
                ax1.set_title(f'Cumulative Equity Curve - {model_type.title()}', fontweight='bold')
                ax1.set_xlabel('Trade Number')
                ax1.set_ylabel('Cumulative Return')
                ax1.grid(True, alpha=0.3)
            
            # Drawdown
            ax2 = fig.add_subplot(2, 1, 2)
            if 'drawdown' in stats_dict:
                dd = stats_dict['drawdown']
                ax2.fill_between(range(len(dd)), 0, dd, color='red', alpha=0.3)
                ax2.plot(range(len(dd)), dd, 'r-', linewidth=1)
                ax2.set_title('Drawdown', fontweight='bold')
                ax2.set_xlabel('Trade Number')
                ax2.set_ylabel('Drawdown (%)')
                ax2.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.chart_label)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            self.chart_label.config(text=f"Error loading charts: {str(e)}")
            print(f"Charts error: {e}")
            import traceback
            traceback.print_exc()
    
    
    def load_feature_timeline(self):
        """Load and display feature selection timeline"""
        try:
            # Import visualizer
            from feature_selection_visualizer import FeatureSelectionVisualizer
            
            # Create visualizer
            viz = FeatureSelectionVisualizer()
            
            # Try to load saved history
            if not viz.load_selection_history('results/feature_selection_history.json'):
                # Show message if no history available
                msg_label = ttk.Label(self.chart_label, 
                                     text="No feature selection history available.\n\n"
                                          "Run walk-forward validation with feature selection enabled\n"
                                          "to generate timeline visualization.",
                                     font=('Arial', 11))
                msg_label.pack(expand=True)
                return
            
            # Create timeline chart widget
            chart_widget = viz.create_timeline_chart(self.chart_label)
            
            if chart_widget:
                # Pack the widget
                chart_widget.pack(fill='both', expand=True)
                
                # Add stats summary
                stats = viz.create_summary_stats()
                if stats:
                    stats_text = f"Total Steps: {stats['total_steps']} | "
                    stats_text += f"Avg Features/Step: {stats['avg_features_per_step']:.1f} | "
                    if stats['most_selected']:
                        feat, count = stats['most_selected']
                        stats_text += f"Most Selected: {feat} ({count}/{stats['total_steps']})"
                    
                    self.chart_label.config(text="")  # Clear any existing text
                    
                    # Create stats label at bottom
                    stats_frame = ttk.Frame(self.chart_label)
                    stats_frame.pack(side='bottom', fill='x', pady=5)
                    ttk.Label(stats_frame, text=stats_text, 
                             font=('Arial', 11), foreground='gray').pack()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.chart_label.config(text=f"Error loading feature timeline: {e}")
    
    def display_tradestats_charts(self, stats_dict):
        """Display tradestats charts in a separate window"""
        try:
            from src.tradestats_charts import create_all_charts
            
            # Create a new top-level window for charts
            chart_window = tk.Toplevel(self.root)
            chart_window.title("Tradestats Performance Charts")
            chart_window.geometry("1400x900")
            
            # Create the charts
            fig = create_all_charts(stats_dict)
            
            # Display in the window
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, master=chart_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Add a toolbar for navigation
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, chart_window)
            toolbar.update()
            
        except Exception as e:
            print(f"Error displaying tradestats charts: {e}")
            import traceback
            traceback.print_exc()
    
    def load_tradestats_charts(self):
        """Load and display tradestats charts"""
        try:
            # Get results file from config
            import configparser
            config = configparser.ConfigParser(inline_comment_prefixes='#')
            config.read('OMtree_config.ini')
            results_file = config.get('output', 'results_file', fallback='OMtree_results.csv')
            
            # Check if we have results data
            if not os.path.exists(results_file):
                self.chart_label.config(text="No results available. Please run walk-forward validation first.")
                return
            
            # Load full results
            df = pd.read_csv(results_file)
            
            # Get model type
            model_type = self.config_widgets.get('model.model_type', None)
            if model_type:
                model_type = model_type.get()
            else:
                import configparser
                config = configparser.ConfigParser(inline_comment_prefixes='#')
                config.read('OMtree_config.ini')
                model_type = config.get('model', 'model_type', fallback='longonly')
            
            # Calculate performance stats with full dataset
            from src.performance_stats import calculate_performance_stats
            stats_dict = calculate_performance_stats(df, model_type=model_type)
            
            # Create charts
            from src.tradestats_charts import create_all_charts
            fig = create_all_charts(stats_dict)
            
            # Embed in tkinter
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, master=self.chart_label)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            self.chart_label.config(text=f"Error loading tradestats charts: {str(e)}")
            print(f"Tradestats charts error: {e}")
            import traceback
            traceback.print_exc()
    
    def resize_chart(self, event=None):
        """Auto-resize chart to fit window"""
        if self.current_chart_path and os.path.exists(self.current_chart_path):
            try:
                # Get frame dimensions
                if event:
                    width = event.width
                    height = event.height
                else:
                    self.chart_label.update()
                    width = self.chart_label.winfo_width()
                    height = self.chart_label.winfo_height()
                
                # Skip if dimensions too small
                if width < 100 or height < 100:
                    return
                
                # Load and resize image
                img = Image.open(self.current_chart_path)
                img.thumbnail((width-10, height-10), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Update label
                self.chart_label.config(image=photo, text="")
                self.chart_label.image = photo  # Keep reference
                
            except Exception as e:
                print(f"Chart resize error: {e}")
    
    def generate_feature_importance(self):
        """Generate feature importance chart"""
        try:
            subprocess.run(['python', 'generate_feature_importance.py'], check=True)
            messagebox.showinfo("Success", "Feature importance chart generated successfully!")
            
            # Auto-load the chart
            self.chart_var.set('feature_importance')
            self.load_chart()
            
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "Failed to generate feature importance chart")
    
    def save_data_settings(self):
        """Save data & fields settings to a JSON file"""
        # Create data_settings directory if it doesn't exist
        settings_dir = "data_settings"
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
        
        # Get filename from user
        filename = filedialog.asksaveasfilename(
            initialdir=settings_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Data Settings"
        )
        
        if not filename:
            return
        
        # Collect all data settings
        settings = {}
        
        # Save data file path
        settings['data_file'] = self.data_file_var.get() if hasattr(self, 'data_file_var') else ""
        
        # Save ticker selection
        if hasattr(self, 'ticker_listbox'):
            selected_indices = self.ticker_listbox.curselection()
            all_tickers = [self.ticker_listbox.get(i) for i in range(self.ticker_listbox.size())]
            selected_tickers = [all_tickers[i] for i in selected_indices]
            settings['selected_tickers'] = selected_tickers
            settings['all_tickers'] = all_tickers
        
        # Save hour selection
        if hasattr(self, 'hour_listbox'):
            selected_indices = self.hour_listbox.curselection()
            all_hours = [self.hour_listbox.get(i) for i in range(self.hour_listbox.size())]
            selected_hours = [all_hours[i] for i in selected_indices]
            settings['selected_hours'] = selected_hours
            settings['all_hours'] = all_hours
        
        # Save feature selection
        if hasattr(self, 'feature_check_vars'):
            selected_features = [f for f, var in self.feature_check_vars.items() if var.get()]
            settings['selected_features'] = selected_features
            settings['all_features'] = list(self.feature_check_vars.keys())
        
        # Save target selection
        if hasattr(self, 'selected_target_var'):
            settings['selected_target'] = self.selected_target_var.get()
        
        # Save validation dates
        if hasattr(self, 'validation_start_var'):
            settings['validation_start'] = self.validation_start_var.get()
        if hasattr(self, 'validation_end_var'):
            settings['validation_end'] = self.validation_end_var.get()
        
        # Save to JSON file
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)
            
            # Update current settings label
            self.current_data_settings_label.config(text=os.path.basename(filename))
            
            # Save as most recent data settings file
            self.save_recent_files('data', filename)
            
            messagebox.showinfo("Success", f"Data settings saved to:\n{os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings:\n{str(e)}")
    
    def save_data_settings_to_file(self, filename):
        """Save data settings to a specific file (for auto-save)"""
        # Collect all data settings
        settings = {}
        
        # Basic settings
        settings['data_file'] = self.data_file_var.get()
        
        # Features and targets from listboxes
        settings['features'] = []
        for i in range(self.feature_listbox.size()):
            settings['features'].append(self.feature_listbox.get(i))
        
        settings['targets'] = []
        for i in range(self.target_listbox.size()):
            settings['targets'].append(self.target_listbox.get(i))
        
        # Selected features and target
        if hasattr(self, 'feature_check_vars'):
            settings['selected_features'] = [f for f, var in self.feature_check_vars.items() if var.get()]
        else:
            settings['selected_features'] = []
        
        if hasattr(self, 'selected_target_var'):
            settings['selected_target'] = self.selected_target_var.get()
        else:
            settings['selected_target'] = ""
        
        # Filter settings
        settings['ticker_filter'] = self.ticker_filter_var.get()
        settings['hour_filter'] = self.hour_filter_var.get()
        
        # Validation dates
        settings['validation_start'] = self.validation_start_var.get()
        settings['validation_end'] = self.validation_end_var.get()
        
        # Save to JSON file
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            raise Exception(f"Failed to save data settings: {str(e)}")
    
    def save_permute_settings(self):
        """Save PermuteAlpha settings to a JSON file"""
        # Create permute_settings directory if it doesn't exist
        settings_dir = "permute_settings"
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
        
        # Get filename from user
        filename = filedialog.asksaveasfilename(
            initialdir=settings_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Permute Settings"
        )
        
        if not filename:
            return
        
        # Collect all permute settings
        settings = {}
        
        # Save ticker selections
        if hasattr(self, 'permute_ticker_listbox'):
            selected_indices = self.permute_ticker_listbox.curselection()
            all_tickers = [self.permute_ticker_listbox.get(i) for i in range(self.permute_ticker_listbox.size())]
            selected_tickers = [all_tickers[i] for i in selected_indices]
            settings['selected_tickers'] = selected_tickers
            settings['all_tickers'] = all_tickers
        
        # Save target selections
        if hasattr(self, 'permute_target_listbox'):
            selected_indices = self.permute_target_listbox.curselection()
            all_targets = [self.permute_target_listbox.get(i) for i in range(self.permute_target_listbox.size())]
            selected_targets = [all_targets[i] for i in selected_indices]
            settings['selected_targets'] = selected_targets
            settings['all_targets'] = all_targets
        
        # Save hour selections
        if hasattr(self, 'permute_hour_listbox'):
            selected_indices = self.permute_hour_listbox.curselection()
            all_hours = [self.permute_hour_listbox.get(i) for i in range(self.permute_hour_listbox.size())]
            selected_hours = [all_hours[i] for i in selected_indices]
            settings['selected_hours'] = selected_hours
            settings['all_hours'] = all_hours
        
        # Save feature selections
        if hasattr(self, 'permute_feature_listbox'):
            selected_indices = self.permute_feature_listbox.curselection()
            all_features = [self.permute_feature_listbox.get(i) for i in range(self.permute_feature_listbox.size())]
            selected_features = [all_features[i] for i in selected_indices]
            settings['selected_features'] = selected_features
            settings['all_features'] = all_features
        
        # Save direction setting
        if hasattr(self, 'permute_direction_var'):
            settings['direction'] = self.permute_direction_var.get()
        
        # Save parallel processing settings
        if hasattr(self, 'permute_use_parallel_var'):
            settings['use_parallel'] = self.permute_use_parallel_var.get()
        if hasattr(self, 'num_workers_var'):
            settings['num_workers'] = self.num_workers_var.get()
        
        # Save output path
        if hasattr(self, 'permute_output_path'):
            settings['output_path'] = self.permute_output_path.get()
        
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=2)
            
            messagebox.showinfo("Success", f"Permute settings saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings:\n{str(e)}")
    
    def save_permute_settings_to_file(self, filename):
        """Save permute settings to a specific file (for auto-save)"""
        # Collect all permute settings
        settings = {}
        
        # Save output path
        if hasattr(self, 'permute_output_path'):
            settings['output_path'] = self.permute_output_path.get()
        
        # Save ticker selections
        if hasattr(self, 'permute_ticker_listbox'):
            selected_indices = self.permute_ticker_listbox.curselection()
            all_tickers = [self.permute_ticker_listbox.get(i) for i in range(self.permute_ticker_listbox.size())]
            selected_tickers = [all_tickers[i] for i in selected_indices]
            settings['selected_tickers'] = selected_tickers
            settings['all_tickers'] = all_tickers
        
        # Save target selections
        if hasattr(self, 'permute_target_listbox'):
            selected_indices = self.permute_target_listbox.curselection()
            all_targets = [self.permute_target_listbox.get(i) for i in range(self.permute_target_listbox.size())]
            selected_targets = [all_targets[i] for i in selected_indices]
            settings['selected_targets'] = selected_targets
            settings['all_targets'] = all_targets
        
        # Save hour selections
        if hasattr(self, 'permute_hour_listbox'):
            selected_indices = self.permute_hour_listbox.curselection()
            all_hours = [self.permute_hour_listbox.get(i) for i in range(self.permute_hour_listbox.size())]
            selected_hours = [all_hours[i] for i in selected_indices]
            settings['selected_hours'] = selected_hours
            settings['all_hours'] = all_hours
        
        # Save feature selections
        if hasattr(self, 'permute_feature_listbox'):
            selected_indices = self.permute_feature_listbox.curselection()
            all_features = [self.permute_feature_listbox.get(i) for i in range(self.permute_feature_listbox.size())]
            selected_features = [all_features[i] for i in selected_indices]
            settings['selected_features'] = selected_features
            settings['all_features'] = all_features
        
        # Save direction setting
        if hasattr(self, 'permute_direction_var'):
            settings['direction'] = self.permute_direction_var.get()
        
        # Save parallel processing settings
        if hasattr(self, 'permute_use_parallel_var'):
            settings['use_parallel'] = self.permute_use_parallel_var.get()
        if hasattr(self, 'num_workers_var'):
            settings['num_workers'] = self.num_workers_var.get()
        
        # Save to JSON file
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            raise Exception(f"Failed to save permute settings: {str(e)}")
    
    def load_permute_settings(self):
        """Load PermuteAlpha settings from a JSON file"""
        # Default to permute_settings directory
        settings_dir = "permute_settings"
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
        
        # Get filename from user
        filename = filedialog.askopenfilename(
            initialdir=settings_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Permute Settings"
        )
        
        if not filename:
            return
        
        try:
            import json
            with open(filename, 'r') as f:
                settings = json.load(f)
            
            # Load ticker selections
            if 'selected_tickers' in settings and hasattr(self, 'permute_ticker_listbox'):
                # Clear current selection
                self.permute_ticker_listbox.selection_clear(0, tk.END)
                # Select saved tickers
                all_tickers = [self.permute_ticker_listbox.get(i) for i in range(self.permute_ticker_listbox.size())]
                for ticker in settings['selected_tickers']:
                    if ticker in all_tickers:
                        idx = all_tickers.index(ticker)
                        self.permute_ticker_listbox.selection_set(idx)
            
            # Load target selections
            if 'selected_targets' in settings and hasattr(self, 'permute_target_listbox'):
                # Clear current selection
                self.permute_target_listbox.selection_clear(0, tk.END)
                # Select saved targets
                all_targets = [self.permute_target_listbox.get(i) for i in range(self.permute_target_listbox.size())]
                for target in settings['selected_targets']:
                    if target in all_targets:
                        idx = all_targets.index(target)
                        self.permute_target_listbox.selection_set(idx)
            
            # Load hour selections
            if 'selected_hours' in settings and hasattr(self, 'permute_hour_listbox'):
                # Clear current selection
                self.permute_hour_listbox.selection_clear(0, tk.END)
                # Select saved hours
                all_hours = [self.permute_hour_listbox.get(i) for i in range(self.permute_hour_listbox.size())]
                for hour in settings['selected_hours']:
                    if str(hour) in all_hours:
                        idx = all_hours.index(str(hour))
                        self.permute_hour_listbox.selection_set(idx)
            
            # Load feature selections
            if 'selected_features' in settings and hasattr(self, 'permute_feature_listbox'):
                # Clear current selection
                self.permute_feature_listbox.selection_clear(0, tk.END)
                # Select saved features
                all_features = [self.permute_feature_listbox.get(i) for i in range(self.permute_feature_listbox.size())]
                for feature in settings['selected_features']:
                    if feature in all_features:
                        idx = all_features.index(feature)
                        self.permute_feature_listbox.selection_set(idx)
            
            # Load direction setting
            if 'direction' in settings and hasattr(self, 'permute_direction_var'):
                self.permute_direction_var.set(settings['direction'])
            
            # Load parallel processing settings
            if 'use_parallel' in settings and hasattr(self, 'permute_use_parallel_var'):
                self.permute_use_parallel_var.set(settings['use_parallel'])
                self.update_parallel_settings()
            if 'num_workers' in settings and hasattr(self, 'num_workers_var'):
                self.num_workers_var.set(settings['num_workers'])
            
            # Load output path
            if 'output_path' in settings and hasattr(self, 'permute_output_path'):
                self.permute_output_path.set(settings['output_path'])
            
            # Update the permutation count after loading
            self.update_permute_count()
            
            messagebox.showinfo("Success", f"Permute settings loaded from:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings:\n{str(e)}")
    
    def load_data_settings(self):
        """Load data & fields settings from a JSON file"""
        # Default to data_settings directory
        settings_dir = "data_settings"
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
        
        # Get filename from user
        filename = filedialog.askopenfilename(
            initialdir=settings_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Data Settings"
        )
        
        if not filename:
            return
        
        try:
            import json
            with open(filename, 'r') as f:
                settings = json.load(f)
            
            # Apply data file path
            if 'data_file' in settings and hasattr(self, 'data_file_var'):
                self.data_file_var.set(settings['data_file'])
                # Load the data file if it exists
                if os.path.exists(settings['data_file']):
                    self.load_data_file()
            
            # Apply ticker selection
            if 'selected_tickers' in settings and hasattr(self, 'ticker_listbox'):
                # Clear current selection
                self.ticker_listbox.selection_clear(0, tk.END)
                # Select the saved tickers
                all_tickers = [self.ticker_listbox.get(i) for i in range(self.ticker_listbox.size())]
                for ticker in settings['selected_tickers']:
                    if ticker in all_tickers:
                        idx = all_tickers.index(ticker)
                        self.ticker_listbox.selection_set(idx)
            
            # Apply hour selection
            if 'selected_hours' in settings and hasattr(self, 'hour_listbox'):
                # Clear current selection
                self.hour_listbox.selection_clear(0, tk.END)
                # Select the saved hours
                all_hours = [self.hour_listbox.get(i) for i in range(self.hour_listbox.size())]
                for hour in settings['selected_hours']:
                    if str(hour) in all_hours:
                        idx = all_hours.index(str(hour))
                        self.hour_listbox.selection_set(idx)
            
            # Apply feature selection
            if 'selected_features' in settings and hasattr(self, 'feature_check_vars'):
                # Clear all first
                for var in self.feature_check_vars.values():
                    var.set(False)
                # Set selected ones
                for feature in settings['selected_features']:
                    if feature in self.feature_check_vars:
                        self.feature_check_vars[feature].set(True)
            
            # Apply target selection
            if 'selected_target' in settings and hasattr(self, 'selected_target_var'):
                self.selected_target_var.set(settings['selected_target'])
            
            # Apply validation dates
            if 'validation_start' in settings and hasattr(self, 'validation_start_var'):
                self.validation_start_var.set(settings['validation_start'])
            if 'validation_end' in settings and hasattr(self, 'validation_end_var'):
                self.validation_end_var.set(settings['validation_end'])
            
            # Update current settings label
            self.current_data_settings_label.config(text=os.path.basename(filename))
            
            # Save as most recent data settings file
            self.save_recent_files('data', filename)
            
            messagebox.showinfo("Success", f"Data settings loaded from:\n{os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings:\n{str(e)}")
    
    def save_recent_files(self, settings_type, filename):
        """Save the most recently used settings file"""
        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        if 'recent_files' not in config:
            config['recent_files'] = {}
        
        if settings_type == 'data':
            config['recent_files']['data_settings'] = filename
        elif settings_type == 'model':
            config['recent_files']['model_settings'] = filename
        
        with open(self.config_file, 'w') as f:
            config.write(f)
    
    def toggle_feature_selection_settings(self):
        """Enable or disable feature selection controls based on toggle state"""
        # Get the current state of feature selection
        fs_enabled = self.feature_selection_enabled_var.get().lower() == "true"
        
        # Set the state for all feature selection controls
        state = 'normal' if fs_enabled else 'disabled'
        for control in self.fs_controls:
            control.config(state=state)
    
    def load_recent_settings(self):
        """Load the most recently used settings files on startup"""
        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        if 'recent_files' in config:
            # Load recent data settings
            if 'data_settings' in config['recent_files']:
                data_file = config['recent_files']['data_settings']
                if os.path.exists(data_file):
                    try:
                        import json
                        with open(data_file, 'r') as f:
                            settings = json.load(f)
                        # Apply settings (simplified version, reuse load logic)
                        self.apply_data_settings(settings)
                        self.current_data_settings_label.config(text=os.path.basename(data_file))
                    except:
                        pass
            
            # Load recent model settings
            if 'model_settings' in config['recent_files']:
                model_file = config['recent_files']['model_settings']
                if os.path.exists(model_file):
                    try:
                        import json
                        with open(model_file, 'r') as f:
                            settings = json.load(f)
                        self.apply_model_settings(settings)
                        self.current_model_settings_label.config(text=os.path.basename(model_file))
                    except:
                        pass
    
    def apply_data_settings(self, settings):
        """Apply data settings without showing dialog"""
        # Apply data file path
        if 'data_file' in settings and hasattr(self, 'data_file_var'):
            self.data_file_var.set(settings['data_file'])
            if os.path.exists(settings['data_file']):
                self.load_data_file()
        
        # Apply feature selection
        if 'selected_features' in settings and hasattr(self, 'feature_check_vars'):
            for var in self.feature_check_vars.values():
                var.set(False)
            for feature in settings['selected_features']:
                if feature in self.feature_check_vars:
                    self.feature_check_vars[feature].set(True)
        
        # Apply target
        if 'selected_target' in settings and hasattr(self, 'selected_target_var'):
            self.selected_target_var.set(settings['selected_target'])
        
        # Apply validation dates
        if 'validation_start' in settings and hasattr(self, 'validation_start_var'):
            self.validation_start_var.set(settings['validation_start'])
        if 'validation_end' in settings and hasattr(self, 'validation_end_var'):
            self.validation_end_var.set(settings['validation_end'])
    
    def apply_model_settings(self, settings):
        """Apply model settings without showing dialog"""
        for key, value in settings.items():
            if key == 'selected_features':
                if hasattr(self, 'feature_check_vars'):
                    for var in self.feature_check_vars.values():
                        var.set(False)
                    for feature in value:
                        if feature in self.feature_check_vars:
                            self.feature_check_vars[feature].set(True)
            elif key == 'selected_target':
                if hasattr(self, 'selected_target_var'):
                    self.selected_target_var.set(value)
            elif key in self.config_widgets:
                widget = self.config_widgets[key]
                if isinstance(widget, ttk.Combobox):
                    widget.set(str(value))
                else:
                    widget.delete(0, tk.END)
                    widget.insert(0, str(value))
        
        if hasattr(self, 'update_trees_label'):
            self.update_trees_label()
    
    def save_model_settings(self):
        """Save model settings to a JSON file"""
        # Create model_settings directory if it doesn't exist
        settings_dir = "model_settings"
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
        
        # Get filename from user
        filename = filedialog.asksaveasfilename(
            initialdir=settings_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Model Settings"
        )
        
        if not filename:
            return
        
        # Collect all model settings
        settings = {}
        
        # Get all config widgets from Model Tester tab
        for key, widget in self.config_widgets.items():
            value = widget.get()
            settings[key] = value
        
        # Also save feature selection if available
        if hasattr(self, 'feature_check_vars'):
            selected_features = [f for f, var in self.feature_check_vars.items() if var.get()]
            settings['selected_features'] = selected_features
        
        # Save selected target
        if hasattr(self, 'selected_target_var'):
            settings['selected_target'] = self.selected_target_var.get()
        
        # Save to JSON file
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)
            
            # Update current settings label
            self.current_model_settings_label.config(text=os.path.basename(filename))
            
            # Save as most recent model settings file
            self.save_recent_files('model', filename)
            
            messagebox.showinfo("Success", f"Model settings saved to:\n{os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings:\n{str(e)}")
    
    def save_model_settings_to_file(self, filename):
        """Save model settings to a specific file (for auto-save)"""
        # Collect all model settings
        settings = {}
        
        # Get all config widgets from Model Tester tab
        for key, widget in self.config_widgets.items():
            value = widget.get()
            settings[key] = value
        
        # Also save feature selection if available
        if hasattr(self, 'feature_check_vars'):
            selected_features = [f for f, var in self.feature_check_vars.items() if var.get()]
            settings['selected_features'] = selected_features
        
        # Save selected target
        if hasattr(self, 'selected_target_var'):
            settings['selected_target'] = self.selected_target_var.get()
        
        # Save to JSON file
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            raise Exception(f"Failed to save model settings: {str(e)}")
    
    def load_model_settings(self):
        """Load model settings from a JSON file"""
        # Default to model_settings directory
        settings_dir = "model_settings"
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
        
        # Get filename from user
        filename = filedialog.askopenfilename(
            initialdir=settings_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Model Settings"
        )
        
        if not filename:
            return
        
        try:
            import json
            with open(filename, 'r') as f:
                settings = json.load(f)
            
            # Apply settings to config widgets
            for key, value in settings.items():
                if key == 'selected_features':
                    # Handle feature selection
                    if hasattr(self, 'feature_check_vars'):
                        # Clear all first
                        for var in self.feature_check_vars.values():
                            var.set(False)
                        # Set selected ones
                        for feature in value:
                            if feature in self.feature_check_vars:
                                self.feature_check_vars[feature].set(True)
                
                elif key == 'selected_target':
                    # Handle target selection
                    if hasattr(self, 'selected_target_var'):
                        self.selected_target_var.set(value)
                
                elif key in self.config_widgets:
                    # Apply to widget
                    widget = self.config_widgets[key]
                    if isinstance(widget, ttk.Combobox):
                        widget.set(str(value))
                    else:
                        widget.delete(0, tk.END)
                        widget.insert(0, str(value))
            
            # Update any dependent UI elements
            if hasattr(self, 'update_trees_label'):
                self.update_trees_label()
            
            # Update current settings label
            self.current_model_settings_label.config(text=os.path.basename(filename))
            
            # Save as most recent model settings file
            self.save_recent_files('model', filename)
            
            messagebox.showinfo("Success", f"Model settings loaded from:\n{os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings:\n{str(e)}")
    
    def save_project(self):
        """Save current project with all configurations"""
        # First save current configurations to history
        data_config_id = self.save_data_config_to_history()
        model_config_id = self.save_model_config_to_history()
        
        if data_config_id and model_config_id:
            project_name = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Gather additional settings including validation parameters
            additional_settings = {
                'source_file': self.data_file_var.get(),
                'validation_start': self.validation_start_var.get(),
                'validation_end': self.validation_end_var.get(),
                'selected_features': [f for f, var in self.feature_check_vars.items() if var.get()],
                'selected_target': self.selected_target_var.get(),
                'train_size': self.config_widgets.get('validation.train_size').get() if 'validation.train_size' in self.config_widgets else '1000',
                'test_size': self.config_widgets.get('validation.test_size').get() if 'validation.test_size' in self.config_widgets else '100',
                'step_size': self.config_widgets.get('validation.step_size').get() if 'validation.step_size' in self.config_widgets else '50',
                'bootstrap_fraction': self.config_widgets.get('model.bootstrap_fraction').get() if 'model.bootstrap_fraction' in self.config_widgets else '0.8',
            }
            
            self.config_manager.save_project(
                project_name,
                data_config_id,
                model_config_id,
                additional_settings
            )
            messagebox.showinfo("Success", f"Project saved as: {project_name}")
        else:
            messagebox.showwarning("Incomplete Configuration", 
                                  "Please complete data and model configuration before saving project.")
    
    def save_project_as(self):
        """Save project with custom name"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="projects"
        )
        if filename:
            # First save current configurations to history
            data_config_id = self.save_data_config_to_history()
            model_config_id = self.save_model_config_to_history()
            
            if data_config_id and model_config_id:
                project_name = os.path.splitext(os.path.basename(filename))[0]
                
                # Gather additional settings including validation parameters
                additional_settings = {
                    'source_file': self.data_file_var.get(),
                    'validation_start': self.validation_start_var.get(),
                    'validation_end': self.validation_end_var.get(),
                    'selected_features': [f for f, var in self.feature_check_vars.items() if var.get()],
                    'selected_target': self.selected_target_var.get(),
                    'train_size': self.config_widgets.get('validation.train_size').get() if 'validation.train_size' in self.config_widgets else '1000',
                    'test_size': self.config_widgets.get('validation.test_size').get() if 'validation.test_size' in self.config_widgets else '100',
                    'step_size': self.config_widgets.get('validation.step_size').get() if 'validation.step_size' in self.config_widgets else '50',
                    'bootstrap_fraction': self.config_widgets.get('model.bootstrap_fraction').get() if 'model.bootstrap_fraction' in self.config_widgets else '0.8',
                }
                
                self.config_manager.save_project(
                    project_name,
                    data_config_id,
                    model_config_id,
                    additional_settings
                )
                messagebox.showinfo("Success", f"Project saved: {project_name}")
            else:
                messagebox.showwarning("Incomplete Configuration", 
                                      "Please complete data and model configuration before saving project.")
    
    def load_project(self):
        """Load a project with all configurations"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="projects"
        )
        if filename:
            try:
                project_data = self.config_manager.load_project(filename)
                
                if project_data['data_config']:
                    # Apply data configuration
                    config = project_data['data_config']
                    self.data_file_var.set(config['csv_file'])
                    self.validation_start_var.set(config['validation_start'])
                    self.validation_end_var.set(config['validation_end'])
                    
                    # Load data file if exists
                    if os.path.exists(config['csv_file']):
                        self.load_data_file()
                    
                    # Set features and targets
                    self.feature_listbox.delete(0, tk.END)
                    for feature in config['features']:
                        self.feature_listbox.insert(tk.END, feature)
                    
                    self.target_listbox.delete(0, tk.END)
                    for target in config['targets']:
                        self.target_listbox.insert(tk.END, target)
                    
                    self.update_feature_target_selection()
                
                if project_data['model_config']:
                    # Apply model configuration
                    config = project_data['model_config']
                    
                    # Set selected features
                    for feature, var in self.feature_check_vars.items():
                        var.set(feature in config['selected_features'])
                    
                    # Set target
                    self.selected_target_var.set(config['selected_target'])
                    
                    # Set model parameters
                    param_mapping = {
                        'model.model_type': 'model_type',
                        'model.n_trees': 'n_trees',
                        'model.max_depth': 'max_depth',
                        'model.bootstrap_fraction': 'bootstrap_fraction',
                        'model.min_leaf_fraction': 'min_leaf_fraction',
                        'model.target_threshold': 'target_threshold',
                        'model.vote_threshold': 'vote_threshold',
                        'preprocessing.normalize_features': 'normalize_features',
                        'preprocessing.normalize_target': 'normalize_target',
                        'preprocessing.vol_window': 'vol_window',
                    }
                    
                    for widget_key, config_key in param_mapping.items():
                        if widget_key in self.config_widgets and config_key in config:
                            widget = self.config_widgets[widget_key]
                            value = config[config_key]
                            
                            if config_key in ['normalize_features', 'normalize_target']:
                                value = 'true' if value else 'false'
                            
                            widget.set(str(value))
                
                # Load validation parameters from additional settings
                if 'project' in project_data and 'additional_settings' in project_data['project']:
                    additional = project_data['project']['additional_settings']
                    
                    # Load validation parameters
                    if 'train_size' in additional and 'validation.train_size' in self.config_widgets:
                        self.config_widgets['validation.train_size'].set(str(additional['train_size']))
                    if 'test_size' in additional and 'validation.test_size' in self.config_widgets:
                        self.config_widgets['validation.test_size'].set(str(additional['test_size']))
                    if 'step_size' in additional and 'validation.step_size' in self.config_widgets:
                        self.config_widgets['validation.step_size'].set(str(additional['step_size']))
                    if 'bootstrap_fraction' in additional and 'model.bootstrap_fraction' in self.config_widgets:
                        self.config_widgets['model.bootstrap_fraction'].set(str(additional['bootstrap_fraction']))
                
                # Update timeline
                self.update_timeline()
                
                messagebox.showinfo("Success", f"Project loaded successfully!\n\nSource file: {project_data['project'].get('name', 'Unknown')}")
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load project:\n{str(e)}")
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background="#ffffe0", 
                           relief="solid", borderwidth=1, font=('Arial', 11))
            label.pack()
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)
    
    def create_history_context_menu(self):
        """Create context menu for configuration history"""
        self.history_menu = tk.Menu(self.root, tearoff=0)
        self.history_menu.add_command(label="Load Configuration", command=self.load_selected_config)
        self.history_menu.add_command(label="Delete", command=self.delete_selected_config)
        
        # Bind right-click
        self.data_history_tree.bind("<Button-3>", self.show_history_menu)
    
    # History menu methods removed - not needed
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", 
                          "OMtree Trading Model\nVersion 3.0\n\n" +
                          "Enhanced configuration and analysis system\n" +
                          "for walk-forward validation of trading strategies.")
    
    def show_user_guide(self):
        """Show user guide"""
        guide = """
USER GUIDE

1. Data & Fields Tab:
   - Load your CSV data file
   - Select features and targets for analysis
   - Set validation date range

2. Model Tester Tab:
   - Configure model parameters
   - Select specific features and target for testing
   - Run walk-forward validation
   - View console output and results

3. Performance Stats Tab:
   - View detailed performance metrics
   - Analyze cumulative equity curve

4. Charts Tab:
   - View various analysis charts
   - Auto-resizing for optimal display

5. Regression Analysis Tab:
   - Perform regression analysis between variables
   - View correlation matrices

Tips:
- Double-click history items to load configurations
- Use Project menu to save/load complete setups
- Validation End Date controls out-of-sample split
"""
        messagebox.showinfo("User Guide", guide)
    
    def sort_permute_results(self, column):
        """Sort the permute results table by the selected column"""
        if not self.permute_results_data:
            return
        
        # Toggle sort direction for this column
        self.permute_sort_reverse[column] = not self.permute_sort_reverse.get(column, False)
        reverse = self.permute_sort_reverse[column]
        
        # Get column index
        columns = list(self.permute_summary_tree['columns'])
        col_idx = columns.index(column)
        
        # Sort data - handle percentage and numeric columns
        def get_sort_value(item):
            value = item[col_idx]
            if isinstance(value, str):
                # Remove % sign if present and convert to float
                if value.endswith('%'):
                    try:
                        return float(value[:-1])
                    except:
                        return 0
                try:
                    return float(value)
                except:
                    return value
            return value
        
        self.permute_results_data.sort(key=get_sort_value, reverse=reverse)
        
        # Clear and repopulate tree
        for item in self.permute_summary_tree.get_children():
            self.permute_summary_tree.delete(item)
        
        for row in self.permute_results_data:
            self.permute_summary_tree.insert('', 'end', values=row)
    
    # === PermuteAlpha Methods ===
    
    def update_permute_selections(self):
        """Update PermuteAlpha listboxes based on loaded data"""
        if not hasattr(self, 'df_unfiltered') or self.df_unfiltered is None:
            return
            
        # Clear listboxes
        self.permute_ticker_listbox.delete(0, tk.END)
        self.permute_target_listbox.delete(0, tk.END)
        self.permute_hour_listbox.delete(0, tk.END)
        self.permute_feature_listbox.delete(0, tk.END)
        
        # Use unfiltered data to get ALL tickers
        if 'Ticker' in self.df_unfiltered.columns:
            unique_tickers = sorted(self.df_unfiltered['Ticker'].unique())
            for ticker in unique_tickers:
                self.permute_ticker_listbox.insert(tk.END, ticker)
            # Update frame title to indicate required
            self.ticker_frame.config(text="Tickers *")
        else:
            self.ticker_frame.config(text="Tickers (N/A)")
        
        # Populate targets from target listbox
        for i in range(self.target_listbox.size()):
            target = self.target_listbox.get(i)
            self.permute_target_listbox.insert(tk.END, target)
        
        # Populate features from feature listbox
        for i in range(self.feature_listbox.size()):
            feature = self.feature_listbox.get(i)
            self.permute_feature_listbox.insert(tk.END, feature)
        
        # Use unfiltered data to get ALL hours  
        if 'Hour' in self.df_unfiltered.columns:
            unique_hours = sorted(self.df_unfiltered['Hour'].unique())
            for hour in unique_hours:
                self.permute_hour_listbox.insert(tk.END, str(hour))
            # Update frame title to indicate required
            self.hour_frame.config(text="Hours *")
        else:
            self.hour_frame.config(text="Hours (N/A)")
    
    def select_all_permute_items(self, listbox_type, select):
        """Select or deselect all items in a permute listbox"""
        if listbox_type == 'tickers':
            listbox = self.permute_ticker_listbox
        elif listbox_type == 'targets':
            listbox = self.permute_target_listbox
        elif listbox_type == 'hours':
            listbox = self.permute_hour_listbox
        elif listbox_type == 'features':
            listbox = self.permute_feature_listbox
        else:
            return
            
        if select:
            listbox.select_set(0, tk.END)
        else:
            listbox.select_clear(0, tk.END)
        
        # Update count after selection change
        self.update_permute_count()
    
    def use_model_tab_features(self):
        """Copy feature selection from Model tab to PermuteAlpha tab"""
        # Clear current selection
        self.permute_feature_listbox.select_clear(0, tk.END)
        
        # Get selected features from Model tab checkboxes
        if hasattr(self, 'feature_check_vars'):
            for i in range(self.permute_feature_listbox.size()):
                feature = self.permute_feature_listbox.get(i)
                if feature in self.feature_check_vars and self.feature_check_vars[feature].get():
                    self.permute_feature_listbox.select_set(i)
        
        # Update count after selection change
        self.update_permute_count()
    
    def update_parallel_settings(self):
        """Update parallel processing settings"""
        if self.permute_use_parallel_var.get():
            self.workers_spin.config(state='normal')
            workers = self.num_workers_var.get()
            # With 20% overhead, effective speedup is workers / 1.2
            effective_speedup = workers / 1.2
            self.speedup_label.config(text=f"Effective speedup: ~{effective_speedup:.1f}x (20% overhead)")
        else:
            self.workers_spin.config(state='disabled')
            self.speedup_label.config(text="Sequential execution (1x speed)")
        
        # Update time estimate with parallel speedup
        self.update_permute_count()
    
    def time_single_iteration(self):
        """Run a single iteration to get actual timing"""
        # Get first selected values for timing test
        selected_tickers = [self.permute_ticker_listbox.get(i) 
                           for i in self.permute_ticker_listbox.curselection()]
        selected_targets = [self.permute_target_listbox.get(i)
                           for i in self.permute_target_listbox.curselection()]
        selected_hours = [self.permute_hour_listbox.get(i)
                         for i in self.permute_hour_listbox.curselection()]
        
        if not selected_tickers or not selected_targets or not selected_hours:
            messagebox.showwarning("No Selection", 
                "Please select at least one ticker, target, and hour to time")
            return
        
        # Disable button during timing
        self.time_single_button.config(state='disabled')
        self.timing_status_label.config(text="Running timing test...")
        
        # Run timing in a thread
        import threading
        thread = threading.Thread(target=self._run_timing_test, 
                                 args=(selected_tickers[0], selected_targets[0], 
                                      selected_hours[0]))
        thread.daemon = True
        thread.start()
    
    def _run_timing_test(self, ticker, target, hour):
        """Run a single iteration for timing"""
        try:
            from datetime import datetime
            import tempfile
            import shutil
            import os
            
            # Get direction
            direction = self.permute_direction_var.get()
            if direction == 'both':
                direction = 'longonly'  # Use longonly for timing test
            
            # Track start time
            start_time = datetime.now()
            
            # Create temporary directory for this test
            temp_dir = tempfile.mkdtemp(prefix='timing_test_')
            
            try:
                # Copy config and prepare
                base_config = 'OMtree_config.ini'
                temp_config = os.path.join(temp_dir, 'temp_config.ini')
                shutil.copy(base_config, temp_config)
                
                # Update config for this single test
                import configparser
                config = configparser.ConfigParser(inline_comment_prefixes='#')
                config.read(temp_config)
                config['data']['ticker_filter'] = ticker
                config['data']['hour_filter'] = str(hour)
                config['data']['target_column'] = target
                config['model']['model_type'] = direction
                
                with open(temp_config, 'w') as f:
                    config.write(f)
                
                # Run validation (simplified version)
                from src.OMtree_validation import DirectionalValidator
                validator = DirectionalValidator(temp_config)
                validator.run_validation(verbose=False)
                
                # Calculate elapsed time
                elapsed = (datetime.now() - start_time).total_seconds()
                
                # Store the measured time
                self.actual_seconds_per_permutation = elapsed
                
                # Update GUI in main thread
                self.root.after(0, self._update_timing_complete, elapsed)
                
            finally:
                # Cleanup
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
        except Exception as e:
            self.root.after(0, self._update_timing_error, str(e))
    
    def _update_timing_complete(self, elapsed_seconds):
        """Update GUI after timing test completes"""
        self.time_single_button.config(state='normal')
        self.timing_status_label.config(text=f"Measured: {elapsed_seconds:.1f}s", foreground='green')
        self.permute_per_second_label.config(text=f"(~{elapsed_seconds:.1f} seconds per permutation)")
        
        # Recalculate estimate with new timing
        self.update_permute_count()
    
    def _update_timing_error(self, error_msg):
        """Update GUI after timing test error"""
        self.time_single_button.config(state='normal')
        self.timing_status_label.config(text="Timing failed", foreground='red')
        messagebox.showerror("Timing Error", f"Failed to run timing test:\n{error_msg}")
    
    def update_permute_count(self):
        """Update the permutation count and time estimate based on current selections"""
        # Get counts
        ticker_count = len(self.permute_ticker_listbox.curselection())
        target_count = len(self.permute_target_listbox.curselection())
        hour_count = len(self.permute_hour_listbox.curselection())
        
        # Check if listboxes have items
        has_tickers = self.permute_ticker_listbox.size() > 0
        has_hours = self.permute_hour_listbox.size() > 0
        
        # For counting purposes:
        # If no tickers in data, count as 1 (will use ALL)
        # If tickers exist but none selected, still show the count as 0 (but don't block calculation)
        effective_ticker_count = ticker_count if has_tickers else 1
        if has_tickers and ticker_count == 0:
            effective_ticker_count = max(1, ticker_count)  # Show at least 1 for calculation
            
        effective_hour_count = hour_count if has_hours else 1
        if has_hours and hour_count == 0:
            effective_hour_count = max(1, hour_count)  # Show at least 1 for calculation
        
        # Get direction count
        direction_choice = self.permute_direction_var.get()
        direction_count = 2 if direction_choice == 'both' else 1
        
        # Calculate total permutations - always show what WOULD be calculated
        display_ticker = ticker_count if ticker_count > 0 else (1 if has_tickers else 1)
        display_hour = hour_count if hour_count > 0 else (1 if has_hours else 1)
        display_target = target_count if target_count > 0 else 0
        
        total_permutations = display_ticker * display_target * display_hour * direction_count
        
        # Build informative message
        parts = []
        if has_tickers:
            parts.append(f"Tickers: {ticker_count}")
        if True:  # Targets always shown
            parts.append(f"Targets: {target_count}")
        if has_hours:
            parts.append(f"Hours: {hour_count}")
        parts.append(f"Directions: {direction_count}")
        
        selection_info = " × ".join(parts)
        
        # Update count label
        self.permute_count_label.config(
            text=f"Total Permutations: {total_permutations} ({selection_info})",
            foreground='black' if total_permutations > 0 else 'red'
        )
        
        # Calculate time estimate
        seconds_per_perm = getattr(self, 'actual_seconds_per_permutation', self.seconds_per_permutation)
        
        # Adjust for parallel processing using the formula:
        # ((time_per_iteration * parallel_factor) * total_iterations) / workers
        if self.permute_use_parallel_var.get():
            workers = self.num_workers_var.get()
            # Apply 20% overhead for parallelism (factor of 1.2)
            parallel_factor = 1.2
            total_seconds = ((seconds_per_perm * parallel_factor) * total_permutations) / workers
            speedup_text = f" ({workers} workers, {parallel_factor:.0%} overhead)"
        else:
            total_seconds = total_permutations * seconds_per_perm
            speedup_text = " (sequential)"
        
        if total_seconds < 60:
            time_str = f"{int(total_seconds)} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds / 60
            time_str = f"{minutes:.1f} minutes"
        else:
            hours = total_seconds / 3600
            time_str = f"{hours:.1f} hours"
        
        self.permute_time_label.config(text=f"Estimated Time: {time_str}")
        
        # Update per-permutation label based on actual timing if available
        if hasattr(self, 'actual_seconds_per_permutation'):
            self.permute_per_second_label.config(
                text=f"(~{self.actual_seconds_per_permutation:.1f} seconds per permutation)"
            )
    
    def run_permute_alpha(self):
        """Run the PermuteAlpha permutations"""
        if self.permute_running:
            messagebox.showwarning("Already Running", "Permutation is already in progress")
            return
            
        # Get selections
        selected_tickers = [self.permute_ticker_listbox.get(i) 
                           for i in self.permute_ticker_listbox.curselection()]
        selected_targets = [self.permute_target_listbox.get(i)
                           for i in self.permute_target_listbox.curselection()]
        selected_hours = [self.permute_hour_listbox.get(i)
                         for i in self.permute_hour_listbox.curselection()]
        selected_features = [self.permute_feature_listbox.get(i)
                            for i in self.permute_feature_listbox.curselection()]
        
        # Get direction choice
        direction_choice = self.permute_direction_var.get()
        if direction_choice == 'both':
            directions = ['longonly', 'shortonly']
        else:
            directions = [direction_choice]
        
        # Validate selections
        # For features - always required
        if not selected_features:
            messagebox.showerror("No Features", "Please select at least one feature for the model")
            return
            
        # For targets - always required
        if not selected_targets:
            messagebox.showerror("No Targets", "Please select at least one target")
            return
        
        # For tickers - if listbox empty, use ALL, otherwise require selection
        if self.permute_ticker_listbox.size() > 0:
            if not selected_tickers:
                messagebox.showerror("No Tickers", "Please select at least one ticker\n(or use Select All)")
                return
        else:
            selected_tickers = ['ALL']
            
        # For hours - if listbox empty, use ALL, otherwise require selection  
        if self.permute_hour_listbox.size() > 0:
            if not selected_hours:
                messagebox.showerror("No Hours", "Please select at least one hour\n(or use Select All)")
                return
        else:
            selected_hours = ['ALL']
        
        # Check if parallel processing is enabled
        if self.permute_use_parallel_var.get():
            # Use parallel processing
            self.permute_thread = threading.Thread(
                target=self._run_permute_alpha_parallel,
                args=(selected_tickers, selected_targets, selected_hours, selected_features, directions)
            )
        else:
            # Use sequential processing
            self.permute_thread = threading.Thread(
                target=self._run_permute_alpha_thread,
                args=(selected_tickers, selected_targets, selected_hours, selected_features, directions)
            )
        
        self.permute_thread.daemon = True
        self.permute_thread.start()
    
    def _run_permute_alpha_parallel(self, tickers, targets, hours, features, directions):
        """Thread function to run permutations in parallel"""
        import itertools
        from datetime import datetime
        from src.parallel_permute_simple import SimpleParallelPermute
        
        # Track start time for elapsed time calculation
        start_time = datetime.now()
        
        # IMPORTANT: Save current Model Tester settings to config file
        # This ensures permutations use the current GUI settings
        self.apply_selections_to_config()
        
        self.permute_running = True
        
        # Update GUI
        self.permute_go_button.config(state='disabled')
        self.permute_stop_button.config(state='normal')
        
        # Clear reports
        for item in self.permute_summary_tree.get_children():
            self.permute_summary_tree.delete(item)
        self.permute_results_data = []
        self.permute_failed_text.delete(1.0, tk.END)
        
        # Reset progress
        self.permute_individual_progress['value'] = 0
        self.permute_step_label.config(text="Starting parallel processing...")
        
        # Create timestamped subfolder for this permutation run
        base_output_dir = self.permute_output_path.get()
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
        
        # Create subfolder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"permute_{timestamp}")
        os.makedirs(output_dir)
        
        # Store for later use
        self.current_permute_output_dir = output_dir
        
        # Generate all combinations
        combinations = list(itertools.product(tickers, targets, hours, directions))
        total_combinations = len(combinations)
        
        self.permute_progress['maximum'] = total_combinations
        self.permute_progress['value'] = 0
        
        # Get number of workers
        n_workers = self.num_workers_var.get()
        
        # Progress callback
        def update_progress(completed, total):
            self.permute_progress['value'] = completed
            self.permute_progress_label.config(
                text=f"Processing {completed}/{total} ({n_workers} workers)"
            )
            self.permute_progress.update()
        
        # Log callback
        def log_message(msg):
            self.permute_step_label.config(text=msg)
            self.permute_step_label.update()
        
        # Auto-save all settings to the output directory
        try:
            # Save Data settings
            data_settings_path = os.path.join(output_dir, 'data_settings.json')
            self.save_data_settings_to_file(data_settings_path)
            
            # Save Model settings
            model_settings_path = os.path.join(output_dir, 'model_settings.json')
            self.save_model_settings_to_file(model_settings_path)
            
            # Save Permute settings
            permute_settings_path = os.path.join(output_dir, 'permute_settings.json')
            self.save_permute_settings_to_file(permute_settings_path)
            
            log_message(f"Settings saved to {output_dir}")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            log_message(f"Warning: Could not save settings: {str(e)}")
            print(f"Error saving settings: {error_details}")
        
        # Callback to add results to GUI as they complete
        all_annual_returns = []
        all_sharpes = []
        all_profit_to_dd = []
        
        def add_result_to_gui(result):
            """Add a single result to the GUI as it completes"""
            if not self.permute_running:
                return
            
            metrics = result['metrics']
            
            # Format for table display
            table_row = (
                result['ticker'],
                result['direction'],
                result['target'],
                str(result['hour']),
                str(metrics.get('num_observations', 0)),
                f"{metrics.get('years_of_data', 0):.2f}",
                str(metrics.get('num_trades', 0)),
                f"{metrics.get('trade_frequency_pct', 0):.1f}",
                f"{metrics.get('win_pct', 0):.1f}",
                f"{metrics.get('avg_loss_pct', 0):.2f}",
                f"{metrics.get('avg_profit_pct', 0):.2f}",
                f"{metrics.get('avg_pnl_pct', 0):.2f}",
                f"{metrics.get('expectancy', 0):.2f}",
                f"{metrics.get('best_day_pct', 0):.2f}",
                f"{metrics.get('worst_day_pct', 0):.2f}",
                f"{metrics.get('avg_annual_pct', 0):.2f}",
                f"{metrics.get('max_draw_pct', 0):.2f}",
                f"{metrics.get('sharpe', 0):.2f}",
                f"{metrics.get('profit_dd_ratio', 0):.2f}",
                f"{metrics.get('upi', 0):.2f}"
            )
            
            # Add to tree view (thread-safe)
            self.permute_summary_tree.insert('', 'end', values=table_row)
            self.permute_results_data.append(table_row)
            
            # Update tree view
            self.permute_summary_tree.update()
            
            # Store for overall statistics
            all_annual_returns.append(metrics.get('avg_annual_pct', 0))
            all_sharpes.append(metrics.get('sharpe', 0))
            profit_dd = metrics.get('profit_dd_ratio', 0)
            if profit_dd != 0 and profit_dd != float('inf'):
                all_profit_to_dd.append(profit_dd)
        
        # Create simplified parallel executor and store as instance variable
        self.parallel_engine = SimpleParallelPermute(n_workers=n_workers)
        
        # Define check_stop callback
        def check_stop():
            return not self.permute_running
        
        try:
            # Run permutations in parallel
            results, failed = self.parallel_engine.run(
                combinations, features, self.config_file, output_dir,
                progress_callback=update_progress,
                result_callback=add_result_to_gui,
                check_stop=check_stop
            )
            
            # Results have already been added to GUI via callback
            # No need to process again
            
            # Add failed permutations to log
            for failed_msg in failed:
                self.permute_failed_text.insert(tk.END, f"{failed_msg}\n")
            
            # Update overall statistics
            if all_annual_returns:
                avg_annual = np.mean(all_annual_returns)
                median_annual = np.median(all_annual_returns)
                avg_sharpe = np.mean(all_sharpes)
                avg_profit_dd = np.mean(all_profit_to_dd) if all_profit_to_dd else 0
                
                stats_text = f"OVERALL STATISTICS (Parallel Execution)\n"
                stats_text += f"{'='*40}\n"
                stats_text += f"Total Permutations: {total_combinations}\n"
                stats_text += f"Successful: {len(results)}\n"
                stats_text += f"Failed: {len(failed)}\n"
                stats_text += f"Workers Used: {n_workers}\n"
                stats_text += f"\nPerformance Metrics:\n"
                stats_text += f"Average Annual Return: {avg_annual:.2f}%\n"
                stats_text += f"Median Annual Return: {median_annual:.2f}%\n"
                stats_text += f"Average Sharpe Ratio: {avg_sharpe:.3f}\n"
                stats_text += f"Average Profit/DD: {avg_profit_dd:.2f}\n"
                
                self.overall_stats_text.config(state='normal')
                self.overall_stats_text.delete(1.0, tk.END)
                self.overall_stats_text.insert(1.0, stats_text)
                self.overall_stats_text.config(state='disabled')
            
            # Export all results and statistics
            if results:
                # 1. Export the Summary Statistics table to CSV
                summary_table_file = os.path.join(output_dir, 'summary_statistics_table.csv')
                self.export_treeview_to_csv(self.permute_summary_tree, summary_table_file)
                
                # 2. Export Overall Statistics to CSV
                overall_stats_file = os.path.join(output_dir, 'overall_statistics.csv')
                overall_stats_data = {
                    'Metric': ['Total Permutations', 'Successful', 'Failed', 'Workers Used',
                              'Average Annual Return (%)', 'Median Annual Return (%)', 
                              'Average Sharpe Ratio', 'Average Profit/DD'],
                    'Value': [total_combinations, len(results), len(failed), n_workers,
                             avg_annual if all_annual_returns else 0,
                             median_annual if all_annual_returns else 0,
                             avg_sharpe if all_sharpes else 0,
                             avg_profit_dd if all_profit_to_dd else 0]
                }
                import pandas as pd
                pd.DataFrame(overall_stats_data).to_csv(overall_stats_file, index=False)
                
                # Calculate elapsed time
                from datetime import datetime
                elapsed_time = datetime.now() - start_time
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                if hours > 0:
                    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                elif minutes > 0:
                    time_str = f"{int(minutes)}m {int(seconds)}s"
                else:
                    time_str = f"{int(seconds)}s"
                
                messagebox.showinfo("Complete", 
                    f"Parallel permutation complete!\n"
                    f"Successful: {len(results)}\n"
                    f"Failed: {len(failed)}\n"
                    f"Workers: {n_workers}\n"
                    f"Time taken: {time_str}\n"
                    f"Results saved to: {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Parallel execution failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up parallel engine
            if hasattr(self, 'parallel_engine'):
                self.parallel_engine = None
            
            # Reset GUI
            self.permute_running = False
            self.permute_go_button.config(state='normal')
            self.permute_stop_button.config(state='disabled')
            self.permute_progress_label.config(text="Complete")
            self.permute_step_label.config(text="")
    
    def _run_permute_alpha_thread(self, tickers, targets, hours, features, directions):
        """Thread function to run permutations"""
        import itertools
        import tempfile
        import shutil
        from datetime import datetime
        
        # IMPORTANT: Save current Model Tester settings to config file
        # This ensures permutations use the current GUI settings
        self.apply_selections_to_config()
        
        self.permute_running = True
        
        # Update GUI
        self.permute_go_button.config(state='disabled')
        self.permute_stop_button.config(state='normal')
        
        # Clear reports
        for item in self.permute_summary_tree.get_children():
            self.permute_summary_tree.delete(item)
        self.permute_results_data = []
        self.permute_failed_text.delete(1.0, tk.END)
        
        # Reset individual progress
        self.permute_individual_progress['value'] = 0
        self.permute_step_label.config(text="")
        
        # Create timestamped subfolder for this permutation run
        base_output_dir = self.permute_output_path.get()
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
        
        # Create subfolder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"permute_{timestamp}")
        os.makedirs(output_dir)
        
        # Store for later use
        self.current_permute_output_dir = output_dir
        
        # Auto-save all settings to the output directory
        try:
            # Save Data settings
            data_settings_path = os.path.join(output_dir, 'data_settings.json')
            self.save_data_settings_to_file(data_settings_path)
            
            # Save Model settings
            model_settings_path = os.path.join(output_dir, 'model_settings.json')
            self.save_model_settings_to_file(model_settings_path)
            
            # Save Permute settings
            permute_settings_path = os.path.join(output_dir, 'permute_settings.json')
            self.save_permute_settings_to_file(permute_settings_path)
            
            self.permute_step_label.config(text=f"Settings saved to {output_dir}")
            self.permute_step_label.update()
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.permute_step_label.config(text=f"Warning: Could not save settings: {str(e)}")
            print(f"Error saving settings: {error_details}")
            self.permute_step_label.update()
        
        # Generate all combinations
        combinations = list(itertools.product(tickers, targets, hours, directions))
        total_combinations = len(combinations)
        
        self.permute_progress['maximum'] = total_combinations
        self.permute_progress['value'] = 0
        
        results = []
        failed = []
        
        # Track timing
        start_time = datetime.now()
        times_per_permutation = []
        
        # Overall statistics will be updated at the end
        all_annual_returns = []
        all_sharpes = []
        all_profit_to_dd = []
        
        for idx, (ticker, target, hour, direction) in enumerate(combinations, 1):
            if not self.permute_running:
                break
            
            permutation_start = datetime.now()
            
            # Calculate and update ETA
            if times_per_permutation:
                avg_time = sum(times_per_permutation) / len(times_per_permutation)
                self.actual_seconds_per_permutation = avg_time
                remaining = total_combinations - idx + 1
                eta_seconds = remaining * avg_time
                
                if eta_seconds < 60:
                    eta_str = f"{int(eta_seconds)}s"
                elif eta_seconds < 3600:
                    eta_str = f"{eta_seconds/60:.1f}min"
                else:
                    eta_str = f"{eta_seconds/3600:.1f}hr"
                
                # Update progress with ETA
                self.permute_progress_label.config(
                    text=f"Processing {idx}/{total_combinations}: {ticker}_{direction}_{target}_{hour} (ETA: {eta_str})"
                )
                
                # Update the estimate labels
                self.permute_per_second_label.config(
                    text=f"(~{avg_time:.1f} seconds per permutation)"
                )
            else:
                # Update progress without ETA for first iteration
                self.permute_progress_label.config(
                    text=f"Processing {idx}/{total_combinations}: {ticker}_{direction}_{target}_{hour}"
                )
            
            try:
                # Create temporary config for this combination
                temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
                
                # Copy current config
                config = configparser.ConfigParser()
                config.read(self.config_file)
                
                # Update for this combination
                config['data']['target_column'] = target
                config['data']['selected_features'] = ','.join(features)
                config['model']['model_type'] = direction  # Will be either 'longonly' or 'shortonly'
                
                if ticker != 'ALL':
                    config['data']['ticker_filter'] = ticker
                elif 'ticker_filter' in config['data']:
                    del config['data']['ticker_filter']
                    
                if hour != 'ALL':
                    config['data']['hour_filter'] = hour
                elif 'hour_filter' in config['data']:
                    del config['data']['hour_filter']
                
                # Write temp config
                config.write(temp_config)
                temp_config.close()
                
                # Run validation
                from OMtree_validation import DirectionalValidator
                validator = DirectionalValidator(temp_config.name)
                
                # Get results - pass the config path too
                daily_returns = self._run_single_permutation(validator, ticker, target, hour, direction, temp_config.name)
                
                if daily_returns is not None:
                    # Save to CSV
                    output_file = f"{ticker}_{direction}_{target}_{hour}.csv"
                    output_path = os.path.join(output_dir, output_file)
                    daily_returns.to_csv(output_path, index=False)
                    
                    # Calculate tradestats metrics manually since we have a different data structure
                    total_days = len(daily_returns)
                    num_trades = daily_returns['TradeFlag'].sum()
                    years = total_days / 252.0
                    
                    # Get trade returns only
                    # IMPORTANT: Returns are likely already in percentage form, need to convert to decimal
                    trade_returns = daily_returns[daily_returns['TradeFlag'] == 1]['Return'] / 100.0
                    
                    # Calculate metrics
                    metrics = {}
                    metrics['num_observations'] = total_days
                    metrics['years_of_data'] = years
                    metrics['num_trades'] = int(num_trades)
                    metrics['trade_frequency_pct'] = (num_trades / total_days * 100) if total_days > 0 else 0
                    metrics['avg_trades_pa'] = num_trades / years if years > 0 else 0
                    metrics['avg_trades_pm'] = metrics['avg_trades_pa'] / 12.0
                    
                    if len(trade_returns) > 0:
                        # Trade metrics (trade_returns is now in decimal form)
                        winning_trades = trade_returns[trade_returns > 0]
                        losing_trades = trade_returns[trade_returns < 0]
                        
                        metrics['win_pct'] = (len(winning_trades) / len(trade_returns)) * 100
                        metrics['avg_loss_pct'] = losing_trades.mean() * 100 if len(losing_trades) > 0 else 0
                        metrics['avg_profit_pct'] = winning_trades.mean() * 100 if len(winning_trades) > 0 else 0
                        metrics['avg_pnl_pct'] = trade_returns.mean() * 100
                        
                        # Expectancy
                        win_rate = metrics['win_pct'] / 100
                        metrics['expectancy'] = (win_rate * metrics['avg_profit_pct']) - ((1 - win_rate) * abs(metrics['avg_loss_pct']))
                        
                        # best_day and worst_day are already in percentage after conversion
                        metrics['best_day_pct'] = trade_returns.max() * 100  # Convert back to percentage
                        metrics['worst_day_pct'] = trade_returns.min() * 100  # Convert back to percentage
                    else:
                        metrics.update({
                            'win_pct': 0, 'avg_loss_pct': 0, 'avg_profit_pct': 0,
                            'avg_pnl_pct': 0, 'expectancy': 0, 'best_day_pct': 0, 'worst_day_pct': 0
                        })
                    
                    # Calculate compound returns for model metrics
                    # Convert returns from percentage to decimal for compounding
                    daily_returns_decimal = daily_returns['Return'] / 100.0
                    cumulative_returns = (1 + daily_returns_decimal).cumprod()
                    final_value = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 1
                    
                    # Annual return (CAGR)
                    if years > 0 and final_value > 0:
                        metrics['avg_annual_pct'] = ((final_value ** (1/years)) - 1) * 100
                    else:
                        metrics['avg_annual_pct'] = 0
                    
                    # Maximum Drawdown
                    running_max = cumulative_returns.expanding().max()
                    drawdown_pct = ((cumulative_returns - running_max) / running_max) * 100
                    metrics['max_draw_pct'] = abs(drawdown_pct.min()) if len(drawdown_pct) > 0 else 0
                    
                    # Sharpe Ratio (using decimal returns)
                    if daily_returns_decimal.std() > 0:
                        metrics['sharpe'] = (daily_returns_decimal.mean() * 252) / (daily_returns_decimal.std() * np.sqrt(252))
                    else:
                        metrics['sharpe'] = 0
                    
                    # Profit to DD Ratio
                    if metrics['max_draw_pct'] > 0:
                        metrics['profit_dd_ratio'] = metrics['avg_annual_pct'] / metrics['max_draw_pct']
                    else:
                        metrics['profit_dd_ratio'] = 0 if metrics['avg_annual_pct'] <= 0 else float('inf')
                    
                    # UPI (simplified calculation)
                    if len(drawdown_pct) > 0:
                        rms_dd = np.sqrt(np.mean(drawdown_pct**2))
                        metrics['upi'] = metrics['avg_annual_pct'] / rms_dd if rms_dd > 0 else 0
                    else:
                        metrics['upi'] = 0
                    
                    # Format for table display matching new column headers:
                    # 'Ticker', 'Direction', 'Target', 'Hour', 'NumObs', 'Years', 'NumTrades',
                    # 'TradeFreq', 'Win%', 'AvgLoss', 'AvgProfit', 'AvgPnL', 'Expectancy',
                    # 'BestDay', 'WorstDay', 'Annual%', 'MaxDD%', 'Sharpe', 'ProfitDD', 'UPI'
                    table_row = (
                        ticker,
                        direction,
                        target,
                        str(hour),
                        str(metrics.get('num_observations', 0)),
                        f"{metrics.get('years_of_data', 0):.2f}",
                        str(metrics.get('num_trades', 0)),
                        f"{metrics.get('trade_frequency_pct', 0):.1f}",
                        f"{metrics.get('win_pct', 0):.1f}",
                        f"{metrics.get('avg_loss_pct', 0):.2f}",
                        f"{metrics.get('avg_profit_pct', 0):.2f}",
                        f"{metrics.get('avg_pnl_pct', 0):.2f}",
                        f"{metrics.get('expectancy', 0):.2f}",
                        f"{metrics.get('best_day_pct', 0):.2f}",
                        f"{metrics.get('worst_day_pct', 0):.2f}",
                        f"{metrics.get('avg_annual_pct', 0):.2f}",
                        f"{metrics.get('max_draw_pct', 0):.2f}",
                        f"{metrics.get('sharpe', 0):.2f}",
                        f"{metrics.get('profit_dd_ratio', 0):.2f}",
                        f"{metrics.get('upi', 0):.2f}"
                    )
                    
                    # Add to tree view
                    self.permute_summary_tree.insert('', 'end', values=table_row)
                    self.permute_results_data.append(table_row)
                    
                    # Store for overall statistics
                    all_annual_returns.append(metrics.get('avg_annual_pct', 0))
                    all_sharpes.append(metrics.get('sharpe', 0))
                    profit_dd = metrics.get('profit_dd_ratio', 0)
                    if profit_dd != 0 and profit_dd != float('inf'):
                        all_profit_to_dd.append(profit_dd)
                    
                    # Store result for CSV export with all tradestats metrics
                    result = {
                        'Ticker': ticker,
                        'Direction': direction,
                        'Target': target,
                        'Hour': hour,
                        'NumObservations': metrics.get('num_observations', 0),
                        'Years': metrics.get('years_of_data', 0),
                        'NumTrades': metrics.get('num_trades', 0),
                        'TradeFreq%': metrics.get('trade_frequency_pct', 0),
                        'Win%': metrics.get('win_pct', 0),
                        'AvgLoss%': metrics.get('avg_loss_pct', 0),
                        'AvgProfit%': metrics.get('avg_profit_pct', 0),
                        'AvgPnL%': metrics.get('avg_pnl_pct', 0),
                        'Expectancy': metrics.get('expectancy', 0),
                        'BestDay%': metrics.get('best_day_pct', 0),
                        'WorstDay%': metrics.get('worst_day_pct', 0),
                        'Annual%': metrics.get('avg_annual_pct', 0),
                        'MaxDD%': metrics.get('max_draw_pct', 0),
                        'Sharpe': metrics.get('sharpe', 0),
                        'ProfitDD': metrics.get('profit_dd_ratio', 0),
                        'UPI': metrics.get('upi', 0),
                        'File': output_file
                    }
                    results.append(result)
                else:
                    raise Exception("No results returned")
                    
                # Clean up temp file
                os.unlink(temp_config.name)
                
            except Exception as e:
                failed.append(f"{ticker}_{direction}_{target}_{hour}: {str(e)}")
                self.permute_failed_text.insert(tk.END, f"{failed[-1]}\n")
                self.permute_failed_text.see(tk.END)
                self.permute_failed_text.update()
                
                # Clean up temp file if exists
                try:
                    if 'temp_config' in locals():
                        os.unlink(temp_config.name)
                except:
                    pass
            
            # Track timing for this permutation
            permutation_time = (datetime.now() - permutation_start).total_seconds()
            times_per_permutation.append(permutation_time)
            
            # Update progress
            self.permute_progress['value'] = idx
            self.permute_progress.update()
        
        # Save summary CSV
        if results:
            summary_df = pd.DataFrame(results)
            summary_path = os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            summary_df.to_csv(summary_path, index=False)
            
            # Calculate and display overall statistics
            if all_annual_returns:
                avg_annual_return = np.mean(all_annual_returns)
                std_annual_return = np.std(all_annual_returns)
                avg_sharpe = np.mean(all_sharpes)
                
                # Portfolio Sharpe (average return / std of returns)
                if std_annual_return > 0:
                    portfolio_sharpe = avg_annual_return / std_annual_return
                else:
                    portfolio_sharpe = 0
                
                avg_profit_dd = np.mean(all_profit_to_dd) if all_profit_to_dd else 0
                
                # Update overall statistics display
                self.overall_stats_text.config(state='normal')
                self.overall_stats_text.delete('1.0', tk.END)
                
                stats_text = f"Overall Portfolio Statistics ({len(results)} successful combinations):\n"
                stats_text += f"─" * 60 + "\n"
                stats_text += f"Average Annual Return: {avg_annual_return:.2f}%\n"
                stats_text += f"Std Dev of Returns: {std_annual_return:.2f}%\n"
                stats_text += f"Portfolio Sharpe Ratio: {portfolio_sharpe:.3f} (avg return / std dev)\n"
                stats_text += f"Average Individual Sharpe: {avg_sharpe:.3f}\n"
                stats_text += f"Average Profit/DD Ratio: {avg_profit_dd:.2f}"
                
                self.overall_stats_text.insert('1.0', stats_text)
                self.overall_stats_text.config(state='disabled')
            
            # Completion summary
            total_time = (datetime.now() - start_time).total_seconds()
            if total_time < 60:
                time_str = f"{int(total_time)} seconds"
            elif total_time < 3600:
                time_str = f"{total_time/60:.1f} minutes"
            else:
                time_str = f"{total_time/3600:.1f} hours"
            
            completion_msg = f"Completed {len(results)}/{total_combinations} in {time_str}"
            if failed:
                completion_msg += f" ({len(failed)} failed)"
            
            self.permute_progress_label.config(text=completion_msg)
        
        # Reset GUI
        self.permute_running = False
        self.permute_go_button.config(state='normal')
        self.permute_stop_button.config(state='disabled')
        self.permute_progress_label.config(text="Complete")
        
        # Export all results and statistics
        if results:
            # 1. Export the Summary Statistics table to CSV
            summary_table_file = os.path.join(output_dir, 'summary_statistics_table.csv')
            self.export_treeview_to_csv(self.permute_summary_tree, summary_table_file)
            
            # 2. Export Overall Statistics to CSV
            overall_stats_file = os.path.join(output_dir, 'overall_statistics.csv')
            overall_stats_data = {
                'Metric': ['Total Permutations', 'Successful', 'Failed',
                          'Average Annual Return (%)', 'Median Annual Return (%)', 
                          'Portfolio Annual Return (%)', 'Portfolio Sharpe Ratio',
                          'Average Sharpe Ratio', 'Average Profit/DD'],
                'Value': [total_combinations, len(results), len(failed),
                         avg_annual if all_annual_returns else 0,
                         median_annual if all_annual_returns else 0,
                         portfolio_annual if results else 0,
                         portfolio_sharpe if results else 0,
                         avg_sharpe if all_sharpes else 0,
                         avg_profit_dd if all_profit_to_dd else 0]
            }
            import pandas as pd
            pd.DataFrame(overall_stats_data).to_csv(overall_stats_file, index=False)
        
        # Calculate elapsed time for the messagebox
        total_time = (datetime.now() - start_time).total_seconds()
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            time_str = f"{int(minutes)}m {int(seconds)}s"
        else:
            time_str = f"{int(seconds)}s"
        
        messagebox.showinfo("Complete", 
            f"Permutation complete!\n"
            f"Processed: {len(results)} combinations\n"
            f"Failed: {len(failed)}\n"
            f"Time taken: {time_str}\n"
            f"Results saved to: {output_dir}")
    
    def _run_single_permutation(self, validator, ticker, target, hour, direction, config_path):
        """Run walk-forward for a single permutation and return daily returns"""
        try:
            # Run the actual walk-forward validation (same as manual validation)
            results_df = validator.run_validation(verbose=False)
            
            if results_df is None or len(results_df) == 0:
                raise Exception("Walk-forward validation returned no results")
            
            # The run_validation returns a DataFrame from the results list
            # The results list has dictionaries with keys: date, prediction, target_raw, test_y_norm
            # We need to load the walkforward_results CSV for full details
            
            # Load the detailed predictions that were saved
            import os
            results_file = f'results/walkforward_results_{validator.model_type}.csv'
            if os.path.exists(results_file):
                detailed_results = pd.read_csv(results_file)
                
                # The CSV has columns: date, time, prediction, actual, signal, pnl
                # We need: Date, Return, TradeFlag, FeatureReturn
                results_df = pd.DataFrame({
                    'Date': detailed_results['date'],
                    'Return': detailed_results['pnl'],  # pnl is the return when trading
                    'TradeFlag': detailed_results['signal'].astype(int),  # signal is the trade flag
                    'FeatureReturn': detailed_results['actual']  # actual is the raw target value
                })
            else:
                # Fallback: use the summary results  
                # Extract what we can from the results DataFrame
                # This won't have raw target values but we can work with what we have
                raise Exception("Detailed walk-forward results file not found")
            
            # Convert dates and aggregate to daily
            results_df['Date'] = pd.to_datetime(results_df['Date'], errors='coerce')
            
            # Remove any NaT dates
            results_df = results_df.dropna(subset=['Date'])
            
            if len(results_df) == 0:
                raise Exception("No valid dates in results")
            
            # Aggregate to daily level
            daily_df = results_df.groupby('Date').agg({
                'Return': 'sum',
                'TradeFlag': lambda x: 1 if x.any() else 0,  # 1 if any trade that day
                'FeatureReturn': 'mean'
            }).reset_index()
            
            return daily_df
            
        except Exception as e:
            import traceback
            error_msg = f"Error in {ticker}_{direction}_{target}_{hour}: {str(e)}\n"
            error_msg += traceback.format_exc()
            print(error_msg)
            raise Exception(str(e))
    
    def stop_permute_alpha(self):
        """Stop the permutation process"""
        self.permute_running = False
        self.permute_progress_label.config(text="Stopping...")
        
        # Stop the parallel engine if it exists
        if hasattr(self, 'parallel_engine') and self.parallel_engine:
            self.parallel_engine.stop()
    
    def browse_permute_output_path(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(
            initialdir=self.permute_output_path.get(),
            title="Select Output Directory"
        )
        if directory:
            self.permute_output_path.set(directory)
    
    def open_permute_results_folder(self):
        """Open the PermuteAlpha results folder"""
        output_dir = self.permute_output_path.get()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if os.name == 'nt':  # Windows
            os.startfile(output_dir)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.call(['open' if sys.platform == 'darwin' else 'xdg-open', output_dir])
    
    def on_closing(self):
        """Handle window closing"""
        if self.validation_process:
            if messagebox.askokcancel("Quit", "A validation is running. Stop it and quit?"):
                self.stop_validation()
                self.root.destroy()
        else:
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = OMtreeGUI(root)
    
    # Set close protocol
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()