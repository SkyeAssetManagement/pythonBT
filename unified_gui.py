#!/usr/bin/env python3
"""
Unified Trading System GUI - Combined OMtree + ABtoPython
=========================================================
Integrates ML decision trees with advanced trade visualization
Following safety-first principles with feature flag protection
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

# Core imports
from datetime import datetime
import pandas as pd
import numpy as np
import threading
import json

# Feature flag system
from feature_flags import get_feature_flags, feature_flag

# OMtree components
from src.config_manager import ConfigurationManager
from src.performance_stats import calculate_performance_stats, format_stats_for_display
from src.date_parser import FlexibleDateParser

# Trading components (protected by feature flags)
try:
    from src.trading.data.trade_data import TradeData, TradeCollection
    import src.trading.data.trade_data_extended
    TRADE_DATA_AVAILABLE = True
except ImportError:
    TRADE_DATA_AVAILABLE = False
    print("Warning: Trade data modules not available")

# Original OMtree GUI import
try:
    from OMtree_gui import OMtreeGUI
    OMTREE_AVAILABLE = True
except ImportError:
    OMTREE_AVAILABLE = False
    print("Warning: Original OMtree GUI not available")


class UnifiedTradingGUI(tk.Tk):
    """
    Unified GUI combining OMtree ML and ABtoPython visualization
    
    Safety principles:
    - All new features behind flags
    - Gradual rollout of functionality
    - Fallback to original behavior if issues
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize feature flags
        self.flags = get_feature_flags()
        
        # Window setup
        self.title("Unified Trading System - OMtree + Trade Visualization")
        self.geometry("1400x800")
        
        # Configuration
        self.config_manager = ConfigurationManager()
        self.current_config = None
        
        # Data
        self.trade_collection = None
        self.ml_results = None
        
        # Setup UI
        self.setup_ui()
        
        # Load default configuration
        self.load_default_config()
        
    def setup_ui(self):
        """Setup the main UI with tabs"""
        # Create menu bar
        self.create_menu()
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Original OMtree tabs
        self.create_configuration_tab()
        self.create_walkforward_tab()
        self.create_performance_tab()
        
        # New tabs (protected by feature flags)
        if self.flags.is_enabled('show_trade_visualization_tab'):
            self.create_trade_visualization_tab()
        
        # Status bar
        self.create_status_bar()
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Configuration", command=self.load_configuration)
        file_menu.add_command(label="Save Configuration", command=self.save_configuration)
        file_menu.add_separator()
        
        if self.flags.is_enabled('enable_vbt_integration'):
            file_menu.add_command(label="Import VectorBT Trades", command=self.import_vbt_trades)
            file_menu.add_separator()
        
        file_menu.add_command(label="Exit", command=self.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Walk-Forward", command=self.run_walkforward)
        analysis_menu.add_command(label="Calculate Statistics", command=self.calculate_stats)
        
        # Feature Flags menu (for testing)
        if self.flags.is_enabled('debug_mode'):
            flags_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Feature Flags", menu=flags_menu)
            flags_menu.add_command(label="View Flags", command=self.show_feature_flags)
            flags_menu.add_command(label="Enable Low Risk", command=self.enable_low_risk)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        
    def create_configuration_tab(self):
        """Create configuration tab"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Configuration")
        
        # Create sections
        # Data configuration
        data_frame = ttk.LabelFrame(config_frame, text="Data Settings", padding=10)
        data_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        
        ttk.Label(data_frame, text="CSV File:").grid(row=0, column=0, sticky='w')
        self.csv_path_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.csv_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="Browse", command=self.browse_csv).grid(row=0, column=2)
        
        # Model configuration
        model_frame = ttk.LabelFrame(config_frame, text="Model Settings", padding=10)
        model_frame.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        
        ttk.Label(model_frame, text="Model Type:").grid(row=0, column=0, sticky='w')
        self.model_type_var = tk.StringVar(value='longonly')
        ttk.Combobox(model_frame, textvariable=self.model_type_var, 
                     values=['longonly', 'shortonly'], width=20).grid(row=0, column=1, padx=5)
        
        ttk.Label(model_frame, text="Trees:").grid(row=1, column=0, sticky='w')
        self.n_trees_var = tk.IntVar(value=100)
        ttk.Spinbox(model_frame, from_=10, to=1000, textvariable=self.n_trees_var, 
                    width=20).grid(row=1, column=1, padx=5)
        
        ttk.Label(model_frame, text="Max Depth:").grid(row=2, column=0, sticky='w')
        self.max_depth_var = tk.IntVar(value=3)
        ttk.Spinbox(model_frame, from_=1, to=10, textvariable=self.max_depth_var, 
                    width=20).grid(row=2, column=1, padx=5)
        
        # Validation configuration
        val_frame = ttk.LabelFrame(config_frame, text="Validation Settings", padding=10)
        val_frame.grid(row=2, column=0, padx=5, pady=5, sticky='ew')
        
        ttk.Label(val_frame, text="Train Size:").grid(row=0, column=0, sticky='w')
        self.train_size_var = tk.IntVar(value=2000)
        ttk.Spinbox(val_frame, from_=100, to=10000, increment=100, 
                    textvariable=self.train_size_var, width=20).grid(row=0, column=1, padx=5)
        
        ttk.Label(val_frame, text="Test Size:").grid(row=1, column=0, sticky='w')
        self.test_size_var = tk.IntVar(value=100)
        ttk.Spinbox(val_frame, from_=10, to=1000, increment=10, 
                    textvariable=self.test_size_var, width=20).grid(row=1, column=1, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        ttk.Button(button_frame, text="Load Config", command=self.load_configuration).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_configuration).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Apply Changes", command=self.apply_configuration).pack(side='left', padx=5)
        
    def create_walkforward_tab(self):
        """Create walk-forward validation tab"""
        wf_frame = ttk.Frame(self.notebook)
        self.notebook.add(wf_frame, text="Walk-Forward")
        
        # Control panel
        control_frame = ttk.Frame(wf_frame)
        control_frame.pack(side='top', fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Run Walk-Forward", 
                  command=self.run_walkforward).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Stop", 
                  command=self.stop_walkforward).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Export Results", 
                  command=self.export_results).pack(side='left', padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(wf_frame, variable=self.progress_var)
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(wf_frame, height=30)
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
    def create_performance_tab(self):
        """Create performance analysis tab"""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="Performance")
        
        # Metrics display
        metrics_frame = ttk.LabelFrame(perf_frame, text="Performance Metrics", padding=10)
        metrics_frame.pack(side='top', fill='x', padx=5, pady=5)
        
        # Create metric labels
        self.metric_labels = {}
        metrics = ['Sharpe Ratio', 'Win Rate', 'Max Drawdown', 'Total Return', 'Calmar Ratio']
        
        for i, metric in enumerate(metrics):
            ttk.Label(metrics_frame, text=f"{metric}:").grid(row=i//3, column=(i%3)*2, sticky='w', padx=5)
            self.metric_labels[metric] = ttk.Label(metrics_frame, text="N/A")
            self.metric_labels[metric].grid(row=i//3, column=(i%3)*2+1, sticky='w', padx=5)
        
        # Chart area (placeholder for now)
        chart_frame = ttk.LabelFrame(perf_frame, text="Equity Curve", padding=10)
        chart_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Placeholder for chart
        self.chart_canvas = tk.Canvas(chart_frame, bg='white')
        self.chart_canvas.pack(fill='both', expand=True)
        
    @feature_flag('show_trade_visualization_tab')
    def create_trade_visualization_tab(self):
        """Create trade visualization tab (new feature)"""
        trade_frame = ttk.Frame(self.notebook)
        self.notebook.add(trade_frame, text="Trade Visualization")
        
        # Control panel
        control_frame = ttk.Frame(trade_frame)
        control_frame.pack(side='top', fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Load Trades", 
                  command=self.load_trades).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Analyze Trades", 
                  command=self.analyze_trades).pack(side='left', padx=5)
        
        if self.flags.is_enabled('enable_range_bar_charts'):
            ttk.Button(control_frame, text="Show Range Bars", 
                      command=self.show_range_bars).pack(side='left', padx=5)
        
        # Trade list
        trade_list_frame = ttk.LabelFrame(trade_frame, text="Trades", padding=10)
        trade_list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Treeview for trades
        columns = ('ID', 'Time', 'Type', 'Price', 'Size', 'P&L')
        self.trade_tree = ttk.Treeview(trade_list_frame, columns=columns, show='headings')
        
        for col in columns:
            self.trade_tree.heading(col, text=col)
            self.trade_tree.column(col, width=100)
        
        self.trade_tree.pack(side='left', fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(trade_list_frame, orient='vertical', 
                                 command=self.trade_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.trade_tree.configure(yscrollcommand=scrollbar.set)
        
    def create_status_bar(self):
        """Create status bar"""
        self.status_frame = ttk.Frame(self)
        self.status_frame.pack(side='bottom', fill='x')
        
        self.status_label = ttk.Label(self.status_frame, text="Ready", relief='sunken')
        self.status_label.pack(side='left', fill='x', expand=True, padx=2)
        
        # Feature flag indicator
        if self.flags.is_enabled('debug_mode'):
            flags_enabled = sum(1 for v in self.flags.get_all_flags().values() if v)
            flag_label = ttk.Label(self.status_frame, 
                                  text=f"Features: {flags_enabled}/{len(self.flags.get_all_flags())}")
            flag_label.pack(side='right', padx=5)
    
    # Action methods
    def load_configuration(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
        )
        if filename:
            self.update_status(f"Loading configuration: {filename}")
            # Load configuration logic here
            
    def save_configuration(self):
        """Save current configuration"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".ini",
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
        )
        if filename:
            self.update_status(f"Saving configuration: {filename}")
            # Save configuration logic here
            
    def apply_configuration(self):
        """Apply current configuration"""
        self.update_status("Configuration applied")
        
    def browse_csv(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path_var.set(filename)
            
    def run_walkforward(self):
        """Run walk-forward validation"""
        self.update_status("Running walk-forward validation...")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Starting walk-forward validation...\n")
        
        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=self._run_walkforward_thread)
        thread.daemon = True
        thread.start()
        
    def _run_walkforward_thread(self):
        """Walk-forward thread worker"""
        try:
            # Simulation of walk-forward
            import time
            for i in range(101):
                time.sleep(0.01)
                self.progress_var.set(i)
                if i % 20 == 0:
                    self.results_text.insert(tk.END, f"Progress: {i}%\n")
                    self.results_text.see(tk.END)
            
            self.results_text.insert(tk.END, "\nWalk-forward completed!\n")
            self.update_status("Walk-forward validation completed")
            
        except Exception as e:
            self.results_text.insert(tk.END, f"\nError: {str(e)}\n")
            self.update_status(f"Error: {str(e)}")
            
    def stop_walkforward(self):
        """Stop walk-forward validation"""
        self.update_status("Stopping walk-forward...")
        
    def export_results(self):
        """Export results to file"""
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.update_status(f"Exporting results to {filename}")
            
    def calculate_stats(self):
        """Calculate performance statistics"""
        self.update_status("Calculating statistics...")
        
        # Update metric displays
        sample_metrics = {
            'Sharpe Ratio': '1.45',
            'Win Rate': '58.3%',
            'Max Drawdown': '-12.4%',
            'Total Return': '34.2%',
            'Calmar Ratio': '2.76'
        }
        
        for metric, value in sample_metrics.items():
            if metric in self.metric_labels:
                self.metric_labels[metric].config(text=value)
                
    @feature_flag('use_new_trade_data')
    def load_trades(self):
        """Load trades using new trade data system"""
        self.update_status("Loading trades...")
        
        if not TRADE_DATA_AVAILABLE:
            messagebox.showerror("Error", "Trade data modules not available")
            return
            
        # Clear existing trades
        for item in self.trade_tree.get_children():
            self.trade_tree.delete(item)
            
        # Load sample trades
        try:
            from trading.data.trade_data import create_sample_trades
            self.trade_collection = create_sample_trades(100)
            
            # Display in tree
            for trade in self.trade_collection.trades[:100]:  # Limit display
                self.trade_tree.insert('', 'end', values=(
                    trade.trade_id,
                    trade.timestamp.strftime('%H:%M:%S'),
                    trade.trade_type,
                    f"{trade.price:.2f}",
                    f"{trade.size:.0f}",
                    f"{trade.pnl:.2f}" if trade.pnl else "N/A"
                ))
                
            self.update_status(f"Loaded {len(self.trade_collection)} trades")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load trades: {str(e)}")
            
    def analyze_trades(self):
        """Analyze loaded trades"""
        if not self.trade_collection:
            messagebox.showwarning("Warning", "No trades loaded")
            return
            
        stats = self.trade_collection.calculate_statistics()
        
        message = f"Trade Statistics:\n\n"
        message += f"Total Trades: {stats['total_trades']}\n"
        message += f"Win Rate: {stats['win_rate']:.1f}%\n"
        message += f"Total P&L: ${stats['total_pnl']:.2f}\n"
        message += f"Avg P&L: ${stats['avg_pnl']:.2f}\n"
        
        messagebox.showinfo("Trade Analysis", message)
        
    @feature_flag('enable_range_bar_charts')
    def show_range_bars(self):
        """Show range bar chart visualization"""
        messagebox.showinfo("Range Bars", "Range bar visualization not yet implemented")
        
    @feature_flag('enable_vbt_integration')
    def import_vbt_trades(self):
        """Import trades from VectorBT"""
        messagebox.showinfo("VectorBT Import", "VectorBT integration not yet implemented")
        
    def show_feature_flags(self):
        """Show current feature flag status"""
        flags = self.flags.get_all_flags()
        message = "Feature Flags Status:\n\n"
        
        for flag, enabled in flags.items():
            status = "ENABLED" if enabled else "disabled"
            message += f"{flag}: {status}\n"
            
        messagebox.showinfo("Feature Flags", message)
        
    def enable_low_risk(self):
        """Enable all low-risk features"""
        self.flags.enable_low_risk_features()
        messagebox.showinfo("Feature Flags", "Low-risk features enabled. Restart GUI to see changes.")
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
Unified Trading System
Version 1.0.0

Combining:
- OMtree ML Decision Trees
- ABtoPython Trade Visualization

Following safety-first refactoring principles
        """
        messagebox.showinfo("About", about_text)
        
    def show_docs(self):
        """Show documentation"""
        messagebox.showinfo("Documentation", "See HOW-TO-GUIDE.md and CODE-DOCUMENTATION.md")
        
    def load_default_config(self):
        """Load default configuration on startup"""
        self.csv_path_var.set("data/sample_trading_data.csv")
        self.update_status("Default configuration loaded")
        
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.update_idletasks()


def main():
    """Main entry point"""
    # Check if we should use the unified GUI
    flags = get_feature_flags()
    
    if flags.is_enabled('use_unified_gui'):
        print("Starting Unified Trading GUI...")
        app = UnifiedTradingGUI()
    elif OMTREE_AVAILABLE:
        print("Starting original OMtree GUI...")
        root = tk.Tk()
        app = OMtreeGUI(root)
    else:
        print("Starting basic Unified GUI...")
        app = UnifiedTradingGUI()
        
    app.mainloop()


if __name__ == "__main__":
    main()