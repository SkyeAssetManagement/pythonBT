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

# Import modules
from regression_gui_module import RegressionAnalysisTab
from timeline_component import TimelineVisualization
from config_manager import ConfigurationManager

class OMtreeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OMtree Trading Model - Enhanced Configuration & Analysis")
        self.root.geometry("1900x950")
        
        # Initialize configuration manager
        self.config_manager = ConfigurationManager()
        
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
        self.notebook.add(self.run_tab, text='Run Walk Forward')  # Renamed
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
        
        # File menu - renamed to Project operations
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_command(label="Save Project As...", command=self.save_project_as)
        file_menu.add_command(label="Load Project", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Regression Analysis", command=lambda: self.notebook.select(5))
        tools_menu.add_command(label="Generate Feature Importance", command=self.generate_feature_importance)
        tools_menu.add_command(label="View Results", command=self.load_results)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def setup_data_tab(self):
        """Setup the data loading and field selection tab with config history"""
        # Create main paned window
        paned = ttk.PanedWindow(self.data_tab, orient='vertical')
        paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Top frame for data configuration
        top_frame = ttk.Frame(paned)
        paned.add(top_frame, weight=3)
        
        main_frame = ttk.Frame(top_frame)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
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
        
        # Validation period settings
        validation_frame = ttk.LabelFrame(main_frame, text="Validation Period Settings", padding=8)
        validation_frame.pack(fill='x', pady=(5, 5))
        
        # Date controls frame
        dates_frame = ttk.Frame(validation_frame)
        dates_frame.pack(fill='x')
        
        # Validation Start Date
        start_frame = ttk.Frame(dates_frame)
        start_frame.pack(side='left', padx=20)
        ttk.Label(start_frame, text="Validation Start Date:").pack(anchor='w')
        self.validation_start_var = tk.StringVar(value="2010-01-01")
        self.validation_start_entry = ttk.Entry(start_frame, textvariable=self.validation_start_var, width=15)
        self.validation_start_entry.pack()
        
        # Validation End Date
        end_frame = ttk.Frame(dates_frame)
        end_frame.pack(side='left', padx=20)
        ttk.Label(end_frame, text="Validation End Date:").pack(anchor='w')
        self.validation_end_var = tk.StringVar(value="2015-12-31")
        self.validation_end_entry = ttk.Entry(end_frame, textvariable=self.validation_end_var, width=15)
        self.validation_end_entry.pack()
        
        # Timeline visualization (compact)
        timeline_frame = ttk.Frame(validation_frame, height=120)
        timeline_frame.pack(fill='x', pady=(5, 0))
        self.timeline = TimelineVisualization(timeline_frame)
        
        # Field selection section
        field_frame = ttk.LabelFrame(main_frame, text="Field Selection", padding=8)
        field_frame.pack(fill='both', expand=True)
        
        # Create three columns
        columns_frame = ttk.Frame(field_frame)
        columns_frame.pack(fill='both', expand=True)
        
        # Available columns
        avail_frame = ttk.LabelFrame(columns_frame, text="Available", padding=3)
        avail_frame.grid(row=0, column=0, sticky='nsew', padx=2)
        
        self.available_listbox = tk.Listbox(avail_frame, selectmode='multiple', height=10, exportselection=False)
        self.available_listbox.pack(fill='both', expand=True)
        
        # Buttons column
        button_frame = ttk.Frame(columns_frame)
        button_frame.grid(row=0, column=1, padx=2)
        
        ttk.Button(button_frame, text="Add to Features →", command=self.move_to_features).pack(pady=3)
        ttk.Button(button_frame, text="← Remove Features", command=self.remove_from_features).pack(pady=3)
        ttk.Button(button_frame, text="Add to Targets →", command=self.move_to_targets).pack(pady=3)
        ttk.Button(button_frame, text="← Remove Targets", command=self.remove_from_targets).pack(pady=3)
        
        # Feature columns
        feature_frame = ttk.LabelFrame(columns_frame, text="Features", padding=3)
        feature_frame.grid(row=0, column=2, sticky='nsew', padx=2)
        
        self.feature_listbox = tk.Listbox(feature_frame, selectmode='multiple', height=10, exportselection=False)
        self.feature_listbox.pack(fill='both', expand=True)
        
        # Target columns
        target_frame = ttk.LabelFrame(columns_frame, text="Targets", padding=3)
        target_frame.grid(row=0, column=3, sticky='nsew', padx=2)
        
        self.target_listbox = tk.Listbox(target_frame, selectmode='multiple', height=10, exportselection=False)
        self.target_listbox.pack(fill='both', expand=True)
        
        # Configure grid weights
        columns_frame.columnconfigure(0, weight=1)
        columns_frame.columnconfigure(2, weight=1)
        columns_frame.columnconfigure(3, weight=1)
        
        # Save button for data config
        ttk.Button(field_frame, text="Save Data Configuration", 
                  command=self.save_data_config_to_history, style='Accent.TButton').pack(pady=5)
        
        # Bottom frame for configuration history
        bottom_frame = ttk.LabelFrame(paned, text="Data Configuration History", padding=5)
        paned.add(bottom_frame, weight=1)
        
        # Create treeview for data config history
        columns = ('CSV File', 'Val Start', 'Val End', 'Features', 'Targets', 'Saved')
        self.data_history_tree = ttk.Treeview(bottom_frame, columns=columns, show='headings', height=5)
        
        for col in columns:
            self.data_history_tree.heading(col, text=col)
            self.data_history_tree.column(col, width=120)
        
        self.data_history_tree.pack(fill='both', expand=True, side='left')
        
        # Scrollbar
        data_scroll = ttk.Scrollbar(bottom_frame, orient='vertical', command=self.data_history_tree.yview)
        data_scroll.pack(side='right', fill='y')
        self.data_history_tree.configure(yscrollcommand=data_scroll.set)
        
        # Bind double-click to load config
        self.data_history_tree.bind('<Double-Button-1>', self.load_data_config_from_history)
        
        # Load existing history
        self.refresh_data_history()
    
    def setup_config_tab(self):
        """Setup the model configuration tab with history"""
        # Create main paned window
        paned = ttk.PanedWindow(self.config_tab, orient='vertical')
        paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Top frame for model configuration
        top_frame = ttk.Frame(paned)
        paned.add(top_frame, weight=3)
        
        # Create scrollable frame
        canvas = tk.Canvas(top_frame)
        scrollbar = ttk.Scrollbar(top_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Store config widgets
        self.config_widgets = {}
        
        # Feature/Target Selection Frame
        selection_frame = ttk.LabelFrame(scrollable_frame, text="Model Features & Target Selection", padding=10)
        selection_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        # Feature selection
        feature_select_frame = ttk.Frame(selection_frame)
        feature_select_frame.pack(fill='x', pady=5)
        
        ttk.Label(feature_select_frame, text="Select Features for Model:").pack(anchor='w')
        
        # Feature checkboxes frame
        self.feature_check_frame = ttk.Frame(feature_select_frame)
        self.feature_check_frame.pack(fill='x', padx=20, pady=5)
        self.feature_check_vars = {}
        
        # Target selection
        target_select_frame = ttk.Frame(selection_frame)
        target_select_frame.pack(fill='x', pady=(10, 5))
        
        ttk.Label(target_select_frame, text="Select Target for Model:").pack(anchor='w')
        
        # Target radio buttons frame
        self.target_radio_frame = ttk.Frame(target_select_frame)
        self.target_radio_frame.pack(fill='x', padx=20, pady=5)
        self.selected_target_var = tk.StringVar()
        
        # Update button
        ttk.Button(selection_frame, text="Load Available Features/Targets", 
                  command=self.update_feature_target_selection).pack(pady=5)
        
        # Model parameters
        self.config_structure = {
            'Model Parameters': {
                'section': 'model',
                'fields': [
                    ('model_type', 'Model Type', 'combo', ['longonly', 'shortonly']),
                    ('n_trees', 'Number of Trees', 'spinbox', (10, 1000, 10)),
                    ('max_depth', 'Max Depth', 'spinbox', (1, 10, 1)),
                    ('min_leaf_fraction', 'Min Leaf Fraction', 'spinbox', (0.01, 0.5, 0.01)),
                    ('target_threshold', 'Target Threshold', 'spinbox', (0.0, 0.5, 0.01)),
                    ('vote_threshold', 'Vote Threshold', 'spinbox', (0.5, 1.0, 0.05)),
                ]
            },
            'Preprocessing': {
                'section': 'preprocessing',
                'fields': [
                    ('normalize_features', 'Normalize Features', 'combo', ['true', 'false']),
                    ('normalize_target', 'Normalize Target', 'combo', ['true', 'false']),
                    ('vol_window', 'Volatility Window', 'spinbox', (10, 500, 10)),
                ]
            }
        }
        
        # Create config sections
        row = 1
        for section_name, section_info in self.config_structure.items():
            frame = ttk.LabelFrame(scrollable_frame, text=section_name, padding=10)
            frame.grid(row=row, column=0, sticky='ew', padx=5, pady=5)
            
            for i, field_info in enumerate(section_info['fields']):
                key, label, widget_type = field_info[:3]
                options = field_info[3] if len(field_info) > 3 else None
                
                ttk.Label(frame, text=label + ':').grid(row=i, column=0, sticky='w', padx=5, pady=2)
                
                if widget_type == 'combo':
                    widget = ttk.Combobox(frame, values=options, width=20)
                elif widget_type == 'spinbox':
                    min_val, max_val, increment = options
                    widget = ttk.Spinbox(frame, from_=min_val, to=max_val, increment=increment, width=20)
                else:
                    widget = ttk.Entry(frame, width=30)
                
                widget.grid(row=i, column=1, padx=5, pady=2)
                self.config_widgets[f"{section_info['section']}.{key}"] = widget
            
            row += 1
        
        # Save button
        ttk.Button(scrollable_frame, text="Save Model Configuration", 
                  command=self.save_model_config_to_history, style='Accent.TButton').grid(row=row, column=0, pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bottom frame for configuration history
        bottom_frame = ttk.LabelFrame(paned, text="Model Configuration History", padding=5)
        paned.add(bottom_frame, weight=1)
        
        # Create treeview for model config history
        columns = ('Features', 'Target', 'Model', 'Trees', 'Threshold', 'Saved')
        self.model_history_tree = ttk.Treeview(bottom_frame, columns=columns, show='headings', height=5)
        
        for col in columns:
            self.model_history_tree.heading(col, text=col)
            self.model_history_tree.column(col, width=120)
        
        self.model_history_tree.pack(fill='both', expand=True, side='left')
        
        # Scrollbar
        model_scroll = ttk.Scrollbar(bottom_frame, orient='vertical', command=self.model_history_tree.yview)
        model_scroll.pack(side='right', fill='y')
        self.model_history_tree.configure(yscrollcommand=model_scroll.set)
        
        # Bind double-click to load config
        self.model_history_tree.bind('<Double-Button-1>', self.load_model_config_from_history)
        
        # Load existing history
        self.refresh_model_history()
    
    def setup_run_tab(self):
        """Setup the run walk forward tab"""
        main_frame = ttk.Frame(self.run_tab, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Walk Forward Control", padding=10)
        control_frame.pack(fill='x')
        
        # Run button - renamed
        self.run_button = ttk.Button(control_frame, text="Run Walk Forward", 
                                    command=self.run_walkforward, style='Accent.TButton')
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
        """Setup performance statistics tab"""
        # Create main container
        main_frame = ttk.Frame(self.results_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create paned window for stats and chart
        paned = ttk.PanedWindow(main_frame, orient='horizontal')
        paned.pack(fill='both', expand=True)
        
        # Left side - Statistics
        stats_frame = ttk.LabelFrame(paned, text="Performance Metrics", padding=10)
        paned.add(stats_frame, weight=1)
        
        # Summary statistics
        summary_frame = ttk.Frame(stats_frame)
        summary_frame.pack(fill='x', pady=(0, 10))
        
        self.stats_labels = {}
        stats_layout = [
            ('total_obs', 'Total Observations:', 0, 0),
            ('total_trades', 'Total Trades:', 1, 0),
            ('trade_freq', 'Trading Frequency:', 2, 0),
            ('hit_rate', 'Hit Rate:', 3, 0),
            ('edge', 'Edge:', 4, 0),
            ('cumulative_pnl', 'Cumulative P&L:', 5, 0),
        ]
        
        for key, label, row, col in stats_layout:
            ttk.Label(summary_frame, text=label).grid(row=row, column=col, sticky='w', padx=5, pady=2)
            value_label = ttk.Label(summary_frame, text="-")
            value_label.grid(row=row, column=col+1, sticky='w', padx=5, pady=2)
            self.stats_labels[key] = value_label
        
        # Right side - Equity curve
        chart_frame = ttk.LabelFrame(paned, text="Cumulative Equity Curve", padding=10)
        paned.add(chart_frame, weight=2)
        
        self.equity_figure = plt.Figure(figsize=(8, 6), facecolor='white')
        self.equity_canvas = FigureCanvasTkAgg(self.equity_figure, master=chart_frame)
        self.equity_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_charts_tab(self):
        """Setup charts display tab"""
        main_frame = ttk.Frame(self.charts_tab)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Chart selection
        select_frame = ttk.Frame(main_frame)
        select_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(select_frame, text="Select Chart:").pack(side='left', padx=5)
        
        self.chart_var = tk.StringVar()
        chart_options = [
            ('Comprehensive', 'comprehensive'),
            ('Progression', 'progression'),
            ('Feature Importance', 'feature_importance'),
        ]
        
        for label, value in chart_options:
            ttk.Radiobutton(select_frame, text=label, variable=self.chart_var, 
                           value=value, command=self.load_chart).pack(side='left', padx=10)
        
        self.chart_var.set('comprehensive')
        
        # Chart display frame with auto-resize
        chart_display_frame = ttk.LabelFrame(main_frame, text="Chart Display", padding=5)
        chart_display_frame.pack(fill='both', expand=True)
        
        # Create label for chart with auto-resize binding
        self.chart_label = ttk.Label(chart_display_frame, text="No chart loaded")
        self.chart_label.pack(fill='both', expand=True)
        
        # Bind resize event
        chart_display_frame.bind('<Configure>', self.resize_chart)
        self.current_chart_path = None
    
    # Helper Methods
    
    def save_data_config_to_history(self):
        """Save current data configuration to history"""
        features = list(self.feature_listbox.get(0, tk.END))
        targets = list(self.target_listbox.get(0, tk.END))
        
        if not features or not targets:
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
        
        self.refresh_data_history()
        return config_id
    
    def save_model_config_to_history(self):
        """Save current model configuration to history"""
        # Get selected features
        selected_features = []
        for feature, var in self.feature_check_vars.items():
            if var.get():
                selected_features.append(feature)
        
        if not selected_features or not self.selected_target_var.get():
            return
        
        # Get model parameters
        model_params = {}
        for key, widget in self.config_widgets.items():
            if isinstance(widget, ttk.Combobox):
                value = widget.get()
            else:
                value = widget.get()
            
            # Parse the key
            section, param = key.split('.')
            if param == 'normalize_features' or param == 'normalize_target':
                value = value.lower() == 'true'
            elif param != 'model_type':
                try:
                    value = float(value)
                except:
                    pass
            
            model_params[param] = value
        
        config_id = self.config_manager.save_model_config(
            selected_features=selected_features,
            selected_target=self.selected_target_var.get(),
            model_params=model_params
        )
        
        self.refresh_model_history()
        return config_id
    
    def refresh_data_history(self):
        """Refresh data configuration history display"""
        self.data_history_tree.delete(*self.data_history_tree.get_children())
        
        df = self.config_manager.get_data_configs_df()
        for _, row in df.iterrows():
            self.data_history_tree.insert('', 'end', values=tuple(row))
    
    def refresh_model_history(self):
        """Refresh model configuration history display"""
        self.model_history_tree.delete(*self.model_history_tree.get_children())
        
        df = self.config_manager.get_model_configs_df()
        for _, row in df.iterrows():
            self.model_history_tree.insert('', 'end', values=tuple(row))
    
    def load_data_config_from_history(self, event):
        """Load a data configuration from history"""
        selection = self.data_history_tree.selection()
        if selection:
            item = self.data_history_tree.item(selection[0])
            index = self.data_history_tree.index(selection[0])
            
            config = self.config_manager.get_data_config(index)
            if config:
                # Apply the configuration
                self.data_file_var.set(config['csv_file'])
                self.validation_start_var.set(config['validation_start'])
                self.validation_end_var.set(config['validation_end'])
                
                # Update listboxes
                self.feature_listbox.delete(0, tk.END)
                for feature in config.get('features', []):
                    self.feature_listbox.insert(tk.END, feature)
                
                self.target_listbox.delete(0, tk.END)
                for target in config.get('targets', []):
                    self.target_listbox.insert(tk.END, target)
                
                self.update_timeline()
    
    def load_model_config_from_history(self, event):
        """Load a model configuration from history"""
        selection = self.model_history_tree.selection()
        if selection:
            index = self.model_history_tree.index(selection[0])
            
            config = self.config_manager.get_model_config(index)
            if config:
                # Update feature/target selection
                self.update_feature_target_selection()
                
                # Apply selected features
                for feature, var in self.feature_check_vars.items():
                    var.set(feature in config['selected_features'])
                
                # Set target
                self.selected_target_var.set(config['selected_target'])
                
                # Apply model parameters
                param_mapping = {
                    'model.model_type': 'model_type',
                    'model.n_trees': 'n_trees',
                    'model.max_depth': 'max_depth',
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
                        
                        if isinstance(widget, ttk.Combobox):
                            widget.set(str(value))
                        else:
                            widget.delete(0, tk.END)
                            widget.insert(0, str(value))
    
    def run_walkforward(self):
        """Run walk forward validation with auto-save"""
        # Auto-save configurations to history
        data_config_id = self.save_data_config_to_history()
        model_config_id = self.save_model_config_to_history()
        
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
        self.status_label.config(text="Running...")
        
        # Start validation in separate thread
        thread = threading.Thread(target=self._run_validation_thread)
        thread.daemon = True
        thread.start()
    
    def apply_selections_to_config(self):
        """Apply current selections to config file for walkforward script"""
        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        # Get selected features from checkboxes
        selected_features = []
        for feature, var in self.feature_check_vars.items():
            if var.get():
                selected_features.append(feature)
        
        # Update config with selections
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
            
            if isinstance(widget, ttk.Combobox):
                config[section][param] = widget.get()
            else:
                config[section][param] = widget.get()
        
        # Save to file
        with open(self.config_file, 'w') as f:
            config.write(f)
    
    def resize_chart(self, event=None):
        """Auto-resize chart to fit window"""
        if self.current_chart_path and os.path.exists(self.current_chart_path):
            try:
                # Get current frame size
                width = self.chart_label.winfo_width()
                height = self.chart_label.winfo_height()
                
                if width > 1 and height > 1:  # Valid size
                    # Open and resize image
                    img = Image.open(self.current_chart_path)
                    img.thumbnail((width, height), Image.Resampling.LANCZOS)
                    
                    # Update display
                    photo = ImageTk.PhotoImage(img)
                    self.chart_label.config(image=photo, text="")
                    self.chart_label.image = photo
            except:
                pass
    
    def load_chart(self):
        """Load and display selected chart"""
        chart_type = self.chart_var.get()
        
        # Get model type from config
        config = configparser.ConfigParser()
        config.read(self.config_file)
        model_type = config['model'].get('model_type', 'longonly')
        
        # Map chart types
        chart_files = {
            'comprehensive': f'OMtree_comprehensive_{model_type}.png',
            'progression': f'OMtree_progression_{model_type}.png',
            'feature_importance': 'feature_importance.png',
        }
        
        chart_file = chart_files.get(chart_type)
        
        if chart_file and os.path.exists(chart_file):
            self.current_chart_path = chart_file
            self.resize_chart()
        else:
            self.chart_label.config(text=f"Chart not found: {chart_file}")
            self.current_chart_path = None
    
    def generate_feature_importance(self):
        """Generate feature importance chart"""
        try:
            subprocess.run(['python', 'generate_feature_importance.py'], check=True)
            self.chart_var.set('feature_importance')
            self.load_chart()
        except:
            pass
    
    def save_project(self):
        """Save current project"""
        # Get latest configs
        data_config, model_config = self.config_manager.get_latest_configs()
        
        if data_config and model_config:
            project_name = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.config_manager.save_project(
                project_name,
                data_config['id'],
                model_config['id']
            )
    
    def save_project_as(self):
        """Save project with custom name"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="projects"
        )
        if filename:
            data_config, model_config = self.config_manager.get_latest_configs()
            if data_config and model_config:
                project_name = os.path.splitext(os.path.basename(filename))[0]
                self.config_manager.save_project(
                    project_name,
                    data_config['id'],
                    model_config['id']
                )
    
    def load_project(self):
        """Load a project"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="projects"
        )
        if filename:
            project_data = self.config_manager.load_project(filename)
            # Apply configurations
            # (Implementation would apply the loaded configs)
    
    # Simplified/stub methods for remaining functionality
    def browse_data_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.data_file_var.set(filename)
    
    def load_data_file(self):
        try:
            filename = self.data_file_var.get()
            self.df = pd.read_csv(filename)
            self.available_columns = list(self.df.columns)
            
            self.available_listbox.delete(0, tk.END)
            for col in self.available_columns:
                self.available_listbox.insert(tk.END, col)
            
            # Auto-detect features/targets
            for col in self.available_columns:
                if col.startswith('Ret_fwd'):
                    self.target_listbox.insert(tk.END, col)
                elif col.startswith('Ret_') or col in ['DiffClose@Obs', 'NoneClose@Obs']:
                    self.feature_listbox.insert(tk.END, col)
            
            self.data_info_label.config(text=f"Loaded: {len(self.df)} rows × {len(self.df.columns)} columns")
            
            # Set date columns for saving
            self.date_column_var = tk.StringVar(value="Date")
            self.time_column_var = tk.StringVar(value="Time")
            
            # Update timeline if dates available
            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                self.data_start_date = self.df['Date'].min()
                self.data_end_date = self.df['Date'].max()
                self.update_timeline()
            
        except Exception as e:
            self.data_info_label.config(text=f"Error: {str(e)}")
    
    def update_timeline(self):
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
            except:
                pass
    
    def move_to_features(self):
        selected = self.available_listbox.curselection()
        for idx in reversed(selected):
            col = self.available_listbox.get(idx)
            if col not in self.feature_listbox.get(0, tk.END):
                self.feature_listbox.insert(tk.END, col)
    
    def remove_from_features(self):
        selected = self.feature_listbox.curselection()
        for idx in reversed(selected):
            self.feature_listbox.delete(idx)
    
    def move_to_targets(self):
        selected = self.available_listbox.curselection()
        for idx in reversed(selected):
            col = self.available_listbox.get(idx)
            if col not in self.target_listbox.get(0, tk.END):
                self.target_listbox.insert(tk.END, col)
    
    def remove_from_targets(self):
        selected = self.target_listbox.curselection()
        for idx in reversed(selected):
            self.target_listbox.delete(idx)
    
    def update_feature_target_selection(self):
        """Update feature/target selection in Model Config"""
        for widget in self.feature_check_frame.winfo_children():
            widget.destroy()
        for widget in self.target_radio_frame.winfo_children():
            widget.destroy()
        
        features = list(self.feature_listbox.get(0, tk.END))
        targets = list(self.target_listbox.get(0, tk.END))
        
        if features:
            for i, feature in enumerate(features):
                var = tk.BooleanVar(value=True)
                self.feature_check_vars[feature] = var
                cb = ttk.Checkbutton(self.feature_check_frame, text=feature, variable=var)
                cb.grid(row=i//3, column=i%3, sticky='w', padx=5, pady=2)
        
        if targets:
            for i, target in enumerate(targets):
                rb = ttk.Radiobutton(self.target_radio_frame, text=target, 
                                   variable=self.selected_target_var, value=target)
                rb.grid(row=i//3, column=i%3, sticky='w', padx=5, pady=2)
            if targets:
                self.selected_target_var.set(targets[0])
    
    def load_config(self):
        """Load initial configuration"""
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
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
                        for field_info in section_info['fields']:
                            key = field_info[0]
                            widget_key = f"{section}.{key}"
                            if widget_key in self.config_widgets and key in config[section]:
                                widget = self.config_widgets[widget_key]
                                value = config[section][key]
                                
                                if isinstance(widget, ttk.Combobox):
                                    widget.set(value)
                                else:
                                    widget.delete(0, tk.END)
                                    widget.insert(0, value)
        except:
            pass
    
    def _run_validation_thread(self):
        """Run validation in separate thread"""
        try:
            process = subprocess.Popen(
                ['python', 'OMtree_walkforward.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.validation_process = process
            
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
        self.console_text.insert(tk.END, text)
        self.console_text.see(tk.END)
    
    def _validation_complete(self, success):
        self.progress_bar.stop()
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        if success:
            self.status_label.config(text="Completed")
            self.console_text.insert(tk.END, "\nWalk forward completed successfully!\n")
            self.load_results()
        else:
            self.status_label.config(text="Failed")
            self.console_text.insert(tk.END, "\nWalk forward failed!\n")
    
    def _validation_error(self, error):
        self.progress_bar.stop()
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Error")
        self.console_text.insert(tk.END, f"\nError: {error}\n")
    
    def stop_validation(self):
        if self.validation_process:
            self.validation_process.terminate()
            self.validation_process = None
            self.progress_bar.stop()
            self.run_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_label.config(text="Stopped")
    
    def load_results(self):
        """Load and display results"""
        try:
            if not os.path.exists('OMtree_results.csv'):
                return
            
            df = pd.read_csv('OMtree_results.csv')
            df['date'] = pd.to_datetime(df['date'])
            
            trades = df[df['prediction'] == 1]
            hit_rate = trades['actual_profitable'].mean() if len(trades) > 0 else 0
            total_trades = len(trades)
            total_observations = len(df)
            
            config = configparser.ConfigParser()
            config.read('OMtree_config.ini')
            model_type = config['model']['model_type']
            base_rate = float(config['validation']['base_rate'])
            
            if len(trades) > 0:
                if model_type == 'longonly':
                    trades['pnl'] = trades['target_value']
                else:
                    trades['pnl'] = -trades['target_value']
                cumulative_pnl = trades['pnl'].sum()
            else:
                cumulative_pnl = 0
            
            edge = hit_rate - base_rate
            
            # Update labels
            self.stats_labels['total_obs'].config(text=f"{total_observations:,}")
            self.stats_labels['total_trades'].config(text=f"{total_trades:,}")
            self.stats_labels['trade_freq'].config(text=f"{total_trades/total_observations:.1%}")
            self.stats_labels['hit_rate'].config(text=f"{hit_rate:.2%}")
            self.stats_labels['edge'].config(text=f"{edge:+.2%}")
            self.stats_labels['cumulative_pnl'].config(text=f"{cumulative_pnl:.2f}")
            
            # Plot equity curve
            if len(trades) > 0:
                self.plot_equity_curve(trades)
        except:
            pass
    
    def plot_equity_curve(self, trades):
        try:
            self.equity_figure.clear()
            ax = self.equity_figure.add_subplot(111)
            
            trades_sorted = trades.sort_values('date')
            cumulative_pnl = trades_sorted['pnl'].cumsum()
            
            ax.plot(trades_sorted['date'], cumulative_pnl, 'b-', linewidth=2)
            ax.fill_between(trades_sorted['date'], 0, cumulative_pnl, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_title('Cumulative Equity Curve')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative P&L')
            ax.grid(True, alpha=0.3)
            
            self.equity_figure.tight_layout()
            self.equity_canvas.draw()
        except:
            pass
    
    def show_about(self):
        messagebox.showinfo("About", "OMtree Trading Model\nVersion 2.1\n\nEnhanced Configuration Management")

def main():
    root = tk.Tk()
    app = OMtreeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()