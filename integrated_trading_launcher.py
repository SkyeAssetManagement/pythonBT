#!/usr/bin/env python3
"""
Integrated Trading System Launcher
===================================
Main entry point for all trading system functionality:
- Range Bar Creation
- VectorBT Pro Backtesting  
- Data Visualization & Charting
- OMtree ML Trading Models
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os
import subprocess
import threading
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add paths for all modules
sys.path.insert(0, 'src')
sys.path.insert(0, 'tradingCode')
sys.path.insert(0, 'createRangeBars')

class IntegratedTradingLauncher(tk.Tk):
    """
    Main launcher GUI for integrated trading system
    Provides access to all functionality through a unified interface
    """
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("SAM - Integrated Backtesting System")
        self.geometry("1000x700")
        self.configure(bg='#1e1e1e')
        
        # Style configuration
        self.setup_styles()
        
        # Current process tracking
        self.running_processes = {}
        
        # Setup UI
        self.setup_ui()
        
        # Center window
        self.center_window()
        
    def setup_styles(self):
        """Configure modern dark theme"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Dark theme colors
        bg_color = '#1e1e1e'
        fg_color = '#ffffff'
        select_color = '#404040'
        button_color = '#2d2d2d'
        
        self.style.configure('Title.TLabel', 
                           background=bg_color, 
                           foreground=fg_color,
                           font=('Arial', 24, 'bold'))
        
        self.style.configure('Heading.TLabel',
                           background=bg_color,
                           foreground=fg_color,
                           font=('Arial', 14, 'bold'))
                           
        self.style.configure('Card.TFrame',
                           background=button_color,
                           borderwidth=2,
                           relief='raised')
                           
        self.style.configure('Launch.TButton',
                           background=button_color,
                           foreground=fg_color,
                           font=('Arial', 11, 'bold'),
                           borderwidth=1,
                           focuscolor='none')
        
        self.style.map('Launch.TButton',
                      background=[('active', select_color)])
    
    def center_window(self):
        """Center the window on screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_ui(self):
        """Setup the main UI layout"""
        # Title
        title_frame = tk.Frame(self, bg='#1e1e1e', pady=20)
        title_frame.pack(fill='x')
        
        title_label = ttk.Label(title_frame, 
                               text="SAM - Integrated Backtesting System",
                               style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame,
                                 text="Complete Trading Solution Suite",
                                 style='Heading.TLabel')
        subtitle_label.pack(pady=(5, 0))
        
        # Main content area with cards
        content_frame = tk.Frame(self, bg='#1e1e1e')
        content_frame.pack(fill='both', expand=True, padx=40, pady=20)
        
        # Create feature cards
        self.create_feature_cards(content_frame)
        
        # Status bar
        self.create_status_bar()
    
    def create_feature_cards(self, parent):
        """Create cards for each major feature"""
        
        # Grid layout for cards
        cards = [
            {
                'title': 'Data Preparation',
                'icon': '📊',
                'description': 'Create Range Bars from Tick Data',
                'details': [
                    '• Convert tick data to range bars',
                    '• ATR-based dynamic ranges',
                    '• Multiple timeframe support',
                    '• Parallel processing'
                ],
                'action': self.launch_range_bars
            },
            {
                'title': 'ML Backtesting',
                'icon': '🤖',
                'description': 'OMtree Decision Tree Models',
                'details': [
                    '• Machine learning models',
                    '• Walk-forward analysis',
                    '• Feature selection',
                    '• Performance metrics'
                ],
                'action': self.launch_omtree
            },
            {
                'title': 'VectorBT Pro',
                'icon': '📈',
                'description': 'High-Performance Backtesting',
                'details': [
                    '• Vectorized backtesting',
                    '• Strategy optimization',
                    '• Portfolio analysis',
                    '• Risk metrics'
                ],
                'action': self.launch_vectorbt
            },
            {
                'title': 'Data Visualization',
                'icon': '📉',
                'description': 'Advanced Charting Dashboard',
                'details': [
                    '• Interactive charts',
                    '• Trade visualization',
                    '• Technical indicators',
                    '• Real-time updates'
                ],
                'action': self.launch_visualization
            },
            {
                'title': 'Quick Chart',
                'icon': '👁',
                'description': 'View Parquet Data Files',
                'details': [
                    '• Quick data viewer',
                    '• Parquet file browser',
                    '• Basic charting',
                    '• Data validation'
                ],
                'action': self.launch_quick_chart
            },
            {
                'title': 'Unified System',
                'icon': '🔧',
                'description': 'Integrated Trading GUI',
                'details': [
                    '• All features combined',
                    '• Workflow automation',
                    '• Custom pipelines',
                    '• Advanced mode'
                ],
                'action': self.launch_unified
            }
        ]
        
        # Create 2x3 grid
        for i, card_info in enumerate(cards):
            row = i // 3
            col = i % 3
            
            card = self.create_card(parent, card_info)
            card.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        
        # Configure grid weights
        for i in range(3):
            parent.columnconfigure(i, weight=1)
        for i in range(2):
            parent.rowconfigure(i, weight=1)
    
    def create_card(self, parent, info):
        """Create a single feature card"""
        card_frame = tk.Frame(parent, bg='#2d2d2d', relief='raised', bd=1)
        
        # Icon and title
        header_frame = tk.Frame(card_frame, bg='#2d2d2d')
        header_frame.pack(fill='x', padx=15, pady=(15, 5))
        
        icon_label = tk.Label(header_frame, 
                             text=info['icon'],
                             font=('Arial', 28),
                             bg='#2d2d2d',
                             fg='white')
        icon_label.pack(side='left')
        
        title_label = tk.Label(header_frame,
                              text=info['title'],
                              font=('Arial', 14, 'bold'),
                              bg='#2d2d2d',
                              fg='white')
        title_label.pack(side='left', padx=(10, 0))
        
        # Description
        desc_label = tk.Label(card_frame,
                             text=info['description'],
                             font=('Arial', 10),
                             bg='#2d2d2d',
                             fg='#a0a0a0',
                             anchor='w')
        desc_label.pack(fill='x', padx=15, pady=(0, 10))
        
        # Details
        details_frame = tk.Frame(card_frame, bg='#2d2d2d')
        details_frame.pack(fill='both', expand=True, padx=15)
        
        for detail in info['details']:
            detail_label = tk.Label(details_frame,
                                   text=detail,
                                   font=('Arial', 9),
                                   bg='#2d2d2d',
                                   fg='#808080',
                                   anchor='w')
            detail_label.pack(anchor='w')
        
        # Launch button
        button_frame = tk.Frame(card_frame, bg='#2d2d2d')
        button_frame.pack(fill='x', padx=15, pady=(10, 15))
        
        launch_btn = tk.Button(button_frame,
                              text='LAUNCH',
                              command=info['action'],
                              bg='#0d7377',
                              fg='white',
                              font=('Arial', 10, 'bold'),
                              relief='raised',
                              bd=1,
                              cursor='hand2',
                              activebackground='#14a085')
        launch_btn.pack(fill='x')
        
        # Hover effects
        def on_enter(e):
            card_frame.configure(bg='#404040')
            for widget in card_frame.winfo_children():
                if isinstance(widget, (tk.Label, tk.Frame)):
                    widget.configure(bg='#404040')
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Label):
                            child.configure(bg='#404040')
        
        def on_leave(e):
            card_frame.configure(bg='#2d2d2d')
            for widget in card_frame.winfo_children():
                if isinstance(widget, (tk.Label, tk.Frame)):
                    widget.configure(bg='#2d2d2d')
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Label):
                            child.configure(bg='#2d2d2d')
        
        card_frame.bind("<Enter>", on_enter)
        card_frame.bind("<Leave>", on_leave)
        
        return card_frame
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = tk.Frame(self, bg='#161616', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame,
                                    text="Ready",
                                    bg='#161616',
                                    fg='#808080',
                                    anchor='w',
                                    padx=10)
        self.status_label.pack(side='left', fill='x', expand=True)
        
        # System info
        info_label = tk.Label(status_frame,
                             text=f"Python {sys.version.split()[0]} | SAM v2.0",
                             bg='#161616',
                             fg='#606060',
                             anchor='e',
                             padx=10)
        info_label.pack(side='right')
    
    def update_status(self, message, color='#808080'):
        """Update status bar message"""
        self.status_label.configure(text=message, fg=color)
        self.update_idletasks()
    
    def launch_range_bars(self):
        """Launch range bar creation tool"""
        self.update_status("Launching Range Bar Creator...", '#00ff00')
        
        def run():
            try:
                # Try to import and launch the range bar GUI
                from createRangeBars.main import RangeBarCreatorGUI
                
                # Create new window for range bar tool
                range_window = tk.Toplevel(self)
                range_window.title("Range Bar Creator")
                range_window.geometry("900x600")
                
                # Initialize the range bar GUI
                app = RangeBarCreatorGUI(range_window)
                
                self.update_status("Range Bar Creator launched successfully", '#00ff00')
            except ImportError:
                # Fallback to command line
                try:
                    subprocess.Popen([sys.executable, 'createRangeBars/main.py'])
                    self.update_status("Range Bar Creator launched (external)", '#00ff00')
                except Exception as e:
                    self.update_status(f"Error: {str(e)}", '#ff0000')
                    messagebox.showerror("Launch Error", 
                                       f"Could not launch Range Bar Creator:\n{str(e)}")
        
        threading.Thread(target=run, daemon=True).start()
    
    def launch_omtree(self):
        """Launch OMtree ML GUI"""
        self.update_status("Launching OMtree ML System...", '#00ff00')
        
        def run():
            try:
                subprocess.Popen([sys.executable, 'OMtree_gui.py'])
                self.update_status("OMtree ML System launched successfully", '#00ff00')
            except Exception as e:
                self.update_status(f"Error: {str(e)}", '#ff0000')
                messagebox.showerror("Launch Error", 
                                   f"Could not launch OMtree:\n{str(e)}")
        
        threading.Thread(target=run, daemon=True).start()
    
    def launch_vectorbt(self):
        """Launch VectorBT backtesting"""
        self.update_status("Launching VectorBT Pro Backtesting...", '#00ff00')
        
        def run():
            try:
                # Check if we can use the trading code's VectorBT integration
                subprocess.Popen([sys.executable, 'tradingCode/vectorbt_optimizer_modular.py'])
                self.update_status("VectorBT Pro launched successfully", '#00ff00')
            except Exception as e:
                self.update_status(f"Error: {str(e)}", '#ff0000')
                messagebox.showerror("Launch Error", 
                                   f"Could not launch VectorBT:\n{str(e)}")
        
        threading.Thread(target=run, daemon=True).start()
    
    def launch_visualization(self):
        """Launch advanced visualization dashboard"""
        self.update_status("Launching Trading Dashboard...", '#00ff00')
        
        def run():
            try:
                subprocess.Popen([sys.executable, 'tradingCode/main.py'])
                self.update_status("Trading Dashboard launched successfully", '#00ff00')
            except Exception as e:
                self.update_status(f"Error: {str(e)}", '#ff0000')
                messagebox.showerror("Launch Error", 
                                   f"Could not launch Dashboard:\n{str(e)}")
        
        threading.Thread(target=run, daemon=True).start()
    
    def launch_quick_chart(self):
        """Launch quick chart viewer for parquet data"""
        self.update_status("Opening Quick Chart Viewer...", '#00ff00')
        
        # Create a simple file browser and chart viewer
        chart_window = tk.Toplevel(self)
        chart_window.title("Quick Chart - Parquet Data Viewer")
        chart_window.geometry("800x600")
        
        QuickChartViewer(chart_window)
        
        self.update_status("Quick Chart Viewer opened", '#00ff00')
    
    def launch_unified(self):
        """Launch unified GUI with all features"""
        self.update_status("Launching Unified Trading System...", '#00ff00')
        
        def run():
            try:
                subprocess.Popen([sys.executable, 'unified_gui.py'])
                self.update_status("Unified System launched successfully", '#00ff00')
            except Exception as e:
                self.update_status(f"Error: {str(e)}", '#ff0000')
                messagebox.showerror("Launch Error", 
                                   f"Could not launch Unified System:\n{str(e)}")
        
        threading.Thread(target=run, daemon=True).start()


class QuickChartViewer(tk.Frame):
    """Simple viewer for parquet data files"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.pack(fill='both', expand=True)
        
        self.setup_ui()
        self.load_available_files()
    
    def setup_ui(self):
        """Setup the quick chart UI"""
        # File selection frame
        select_frame = ttk.Frame(self)
        select_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(select_frame, text="Select Data File:").pack(side='left', padx=(0, 10))
        
        self.file_combo = ttk.Combobox(select_frame, width=50)
        self.file_combo.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        ttk.Button(select_frame, text="Load", command=self.load_file).pack(side='left')
        ttk.Button(select_frame, text="Browse...", command=self.browse_file).pack(side='left', padx=(5, 0))
        
        # Data display
        self.text_display = tk.Text(self, wrap='none', height=20)
        self.text_display.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(self.text_display, orient='vertical', command=self.text_display.yview)
        v_scroll.pack(side='right', fill='y')
        h_scroll = ttk.Scrollbar(self.text_display, orient='horizontal', command=self.text_display.xview)
        h_scroll.pack(side='bottom', fill='x')
        
        self.text_display.config(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
    
    def load_available_files(self):
        """Load list of available parquet files"""
        parquet_files = []
        
        # Check common data directories
        data_dirs = ['parquetData', 'dataParquet', 'data', 'tradingCode/data']
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                for root, dirs, files in os.walk(data_dir):
                    for file in files:
                        if file.endswith(('.parquet', '.csv')):
                            rel_path = os.path.relpath(os.path.join(root, file))
                            parquet_files.append(rel_path)
        
        self.file_combo['values'] = parquet_files
        if parquet_files:
            self.file_combo.set(parquet_files[0])
    
    def browse_file(self):
        """Browse for a data file"""
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("Data files", "*.parquet *.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_combo.set(filename)
            self.load_file()
    
    def load_file(self):
        """Load and display the selected file"""
        filepath = self.file_combo.get()
        if not filepath:
            return
        
        try:
            # Clear display
            self.text_display.delete('1.0', tk.END)
            
            # Load data
            if filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath)
            
            # Display info
            info = f"File: {filepath}\n"
            info += f"Shape: {df.shape}\n"
            info += f"Columns: {', '.join(df.columns)}\n"
            info += f"\nFirst 100 rows:\n"
            info += "-" * 80 + "\n"
            info += df.head(100).to_string()
            info += f"\n\nLast 100 rows:\n"
            info += "-" * 80 + "\n"
            info += df.tail(100).to_string()
            
            self.text_display.insert('1.0', info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{str(e)}")


def main():
    """Main entry point"""
    app = IntegratedTradingLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()