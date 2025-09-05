import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import configparser
import subprocess
import threading
import pandas as pd
import os
from datetime import datetime
from PIL import Image, ImageTk
import json

class OMtreeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OMtree Trading Model - Configuration & Analysis")
        self.root.geometry("1400x900")
        
        # Store config file path
        self.config_file = 'OMtree_config.ini'
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.config_tab = ttk.Frame(self.notebook)
        self.run_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.charts_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.config_tab, text='Configuration')
        self.notebook.add(self.run_tab, text='Run Validation')
        self.notebook.add(self.results_tab, text='Performance Stats')
        self.notebook.add(self.charts_tab, text='Charts')
        
        # Initialize tabs
        self.setup_config_tab()
        self.setup_run_tab()
        self.setup_results_tab()
        self.setup_charts_tab()
        
        # Load initial config
        self.load_config()
        
        # Process tracking
        self.validation_process = None
        
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
        
        # Define config structure with widget types and constraints
        self.config_structure = {
            'Data Settings': {
                'section': 'data',
                'fields': [
                    ('csv_file', 'Data File', 'Input CSV file with trading signals', 'entry', None),
                    ('target_column', 'Target Column', 'Column containing forward returns', 'entry', None),
                    ('selected_features', 'Features', 'Comma-separated list of features to use', 'entry', None),
                    ('date_column', 'Date Column', 'Column containing timestamps', 'entry', None),
                ]
            },
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
                    ('model_type', 'Model Type', 'Trading direction', 'combo', ['longonly', 'shortonly']),
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
                    ('train_size', 'Training Size', 'Training window (observations)', 'spinbox', (100, 5000, 100)),
                    ('test_size', 'Test Size', 'Test window (observations)', 'spinbox', (10, 500, 10)),
                    ('step_size', 'Step Size', 'Days between retraining', 'spinbox', (1, 200, 10)),
                    ('base_rate', 'Base Rate', 'Expected profitable rate', 'spinbox', (0.3, 0.7, 0.01)),
                    ('validation_start_date', 'Start Date', 'Out-of-sample start (YYYY-MM-DD)', 'entry', None),
                ]
            }
        }
        
        # Create config sections
        row = 0
        for section_name, section_info in self.config_structure.items():
            # Section header
            header = ttk.LabelFrame(scrollable_frame, text=section_name, padding=10)
            header.grid(row=row, column=0, columnspan=3, sticky='ew', padx=10, pady=5)
            
            section_row = 0
            section = section_info['section']
            
            for field_data in section_info['fields']:
                field_key = field_data[0]
                field_label = field_data[1]
                field_desc = field_data[2]
                widget_type = field_data[3] if len(field_data) > 3 else 'entry'
                widget_params = field_data[4] if len(field_data) > 4 else None
                
                # Label
                label = ttk.Label(header, text=field_label + ':')
                label.grid(row=section_row, column=0, sticky='w', padx=5, pady=2)
                
                # Create appropriate widget based on type
                if widget_type == 'combo':
                    # Dropdown/Combobox for constrained choices
                    widget = ttk.Combobox(header, width=27, state='readonly')
                    widget['values'] = widget_params if widget_params else []
                    widget.grid(row=section_row, column=1, padx=5, pady=2)
                    
                elif widget_type == 'spinbox':
                    # Spinbox for numeric values with range
                    if widget_params:
                        min_val, max_val, increment = widget_params
                        # Determine if we need decimal precision
                        if isinstance(increment, float) and increment < 1:
                            widget = ttk.Spinbox(header, from_=min_val, to=max_val, 
                                               increment=increment, width=28,
                                               format="%.2f")
                        else:
                            widget = ttk.Spinbox(header, from_=min_val, to=max_val, 
                                               increment=increment, width=28)
                    else:
                        widget = ttk.Spinbox(header, from_=0, to=1000, width=28)
                    widget.grid(row=section_row, column=1, padx=5, pady=2)
                    
                else:  # 'entry'
                    # Regular text entry
                    widget = ttk.Entry(header, width=30)
                    widget.grid(row=section_row, column=1, padx=5, pady=2)
                
                # Description
                desc = ttk.Label(header, text=field_desc, font=('Arial', 8), foreground='gray')
                desc.grid(row=section_row, column=2, sticky='w', padx=5, pady=2)
                
                # Store widget reference
                self.config_widgets[f"{section}.{field_key}"] = widget
                
                section_row += 1
            
            row += 1
        
        # Buttons frame
        button_frame = ttk.Frame(self.config_tab)
        button_frame.pack(side='bottom', fill='x', padx=10, pady=10)
        
        ttk.Button(button_frame, text="Load Config", command=self.load_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_config).pack(side='left', padx=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def setup_run_tab(self):
        """Setup the validation runner tab"""
        # Top frame for controls
        control_frame = ttk.LabelFrame(self.run_tab, text="Validation Control", padding=10)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Run button
        self.run_button = ttk.Button(control_frame, text="Run Walk-Forward Validation", 
                                     command=self.run_validation, style="Accent.TButton")
        self.run_button.pack(side='left', padx=5)
        
        # Stop button
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_validation, 
                                      state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           maximum=100, length=300)
        self.progress_bar.pack(side='left', padx=20)
        
        # Progress label
        self.progress_label = ttk.Label(control_frame, text="Ready")
        self.progress_label.pack(side='left', padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(self.run_tab, text="Validation Output", padding=10)
        output_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Text output with scrollbar
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, 
                                                    width=100, height=30,
                                                    font=('Consolas', 10))
        self.output_text.pack(fill='both', expand=True)
        
    def setup_results_tab(self):
        """Setup the results display tab"""
        # Main frame
        main_frame = ttk.Frame(self.results_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Refresh button
        refresh_frame = ttk.Frame(main_frame)
        refresh_frame.pack(fill='x', pady=5)
        ttk.Button(refresh_frame, text="Refresh Results", 
                  command=self.load_results).pack(side='left')
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD,
                                                     width=100, height=35,
                                                     font=('Consolas', 10))
        self.results_text.pack(fill='both', expand=True)
        
    def setup_charts_tab(self):
        """Setup the charts viewer tab"""
        # Control frame
        control_frame = ttk.Frame(self.charts_tab)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(control_frame, text="Select Chart:").pack(side='left', padx=5)
        
        # Chart selector
        self.chart_var = tk.StringVar()
        self.chart_combo = ttk.Combobox(control_frame, textvariable=self.chart_var, 
                                       width=40, state='readonly')
        self.chart_combo.pack(side='left', padx=5)
        self.chart_combo.bind('<<ComboboxSelected>>', self.load_chart)
        
        # Refresh button
        ttk.Button(control_frame, text="Refresh", command=self.refresh_charts).pack(side='left', padx=5)
        
        # Canvas for image display
        self.chart_canvas = tk.Canvas(self.charts_tab, bg='white')
        self.chart_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize chart list
        self.refresh_charts()
        
    def load_config(self):
        """Load configuration from INI file"""
        try:
            config = configparser.ConfigParser(inline_comment_prefixes='#')
            config.read(self.config_file)
            
            # Populate widgets
            for section_name, section_info in self.config_structure.items():
                section = section_info['section']
                for field_data in section_info['fields']:
                    field_key = field_data[0]
                    widget_type = field_data[3] if len(field_data) > 3 else 'entry'
                    widget_key = f"{section}.{field_key}"
                    
                    if widget_key in self.config_widgets:
                        widget = self.config_widgets[widget_key]
                        try:
                            value = config[section][field_key]
                            # Remove inline comments
                            if '#' in value:
                                value = value.split('#')[0].strip()
                            
                            # Set value based on widget type
                            if isinstance(widget, ttk.Combobox):
                                widget.set(value)
                            elif isinstance(widget, ttk.Spinbox):
                                widget.delete(0, tk.END)
                                widget.insert(0, value)
                            else:  # Entry widget
                                widget.delete(0, tk.END)
                                widget.insert(0, value)
                        except:
                            pass
            
            messagebox.showinfo("Success", "Configuration loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {str(e)}")
            
    def save_config(self):
        """Save configuration to INI file"""
        try:
            # Read existing config to preserve comments and other sections
            config = configparser.ConfigParser(inline_comment_prefixes='#')
            config.read(self.config_file)
            
            # Update values from widgets
            for section_name, section_info in self.config_structure.items():
                section = section_info['section']
                if section not in config:
                    config.add_section(section)
                    
                for field_data in section_info['fields']:
                    field_key = field_data[0]
                    widget_key = f"{section}.{field_key}"
                    if widget_key in self.config_widgets:
                        widget = self.config_widgets[widget_key]
                        # Get value from any widget type
                        value = widget.get()
                        config[section][field_key] = value
            
            # Write to file
            with open(self.config_file, 'w') as f:
                config.write(f)
            
            messagebox.showinfo("Success", "Configuration saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {str(e)}")
            
    def reset_config(self):
        """Reset configuration to defaults"""
        if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
            # Default values
            defaults = {
                'data.csv_file': 'DTSnnData.csv',
                'data.target_column': 'RetForward',
                'data.selected_features': 'Overnight,3day,Last10,First20',
                'data.date_column': 'Date/Time',
                'preprocessing.normalize_features': 'true',
                'preprocessing.normalize_target': 'true',
                'preprocessing.vol_window': '50',
                'preprocessing.smoothing_type': 'exponential',
                'preprocessing.smoothing_alpha': '0.1',
                'model.model_type': 'longonly',
                'model.n_trees': '200',
                'model.max_depth': '1',
                'model.bootstrap_fraction': '0.8',
                'model.min_leaf_fraction': '0.2',
                'model.target_threshold': '0.05',
                'model.vote_threshold': '0.6',
                'validation.train_size': '1000',
                'validation.test_size': '100',
                'validation.step_size': '50',
                'validation.base_rate': '0.42',
                'validation.validation_start_date': '2010-01-01',
            }
            
            for key, value in defaults.items():
                if key in self.config_widgets:
                    widget = self.config_widgets[key]
                    if isinstance(widget, ttk.Combobox):
                        widget.set(value)
                    elif isinstance(widget, ttk.Spinbox):
                        widget.delete(0, tk.END)
                        widget.insert(0, value)
                    else:  # Entry widget
                        widget.delete(0, tk.END)
                        widget.insert(0, value)
                    
    def run_validation(self):
        """Run the walk-forward validation in a separate thread"""
        # Save current config first
        self.save_config()
        
        # Clear output
        self.output_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.progress_label.config(text="Starting validation...")
        
        # Disable run button, enable stop
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Run in separate thread
        thread = threading.Thread(target=self.run_validation_thread)
        thread.daemon = True
        thread.start()
        
    def run_validation_thread(self):
        """Thread function to run validation"""
        try:
            # Run the validation script
            process = subprocess.Popen(
                ['python', 'OMtree_walkforward.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.validation_process = process
            
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Update output text
                    self.root.after(0, self.update_output, line)
                    
                    # Parse progress if present
                    if 'Progress:' in line and '%' in line:
                        try:
                            pct_str = line.split('%')[0].split()[-1]
                            pct = float(pct_str)
                            self.root.after(0, self.update_progress, pct)
                        except:
                            pass
                            
            # Wait for process to complete
            process.wait()
            
            # Update UI
            self.root.after(0, self.validation_complete, process.returncode)
            
        except Exception as e:
            self.root.after(0, self.validation_error, str(e))
            
    def stop_validation(self):
        """Stop the running validation"""
        if self.validation_process:
            self.validation_process.terminate()
            self.validation_process = None
            self.output_text.insert(tk.END, "\n\n[STOPPED BY USER]\n")
            self.run_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.progress_label.config(text="Stopped")
            
    def update_output(self, text):
        """Update output text widget"""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        
    def update_progress(self, percent):
        """Update progress bar"""
        self.progress_var.set(percent)
        self.progress_label.config(text=f"Progress: {percent:.1f}%")
        
    def validation_complete(self, return_code):
        """Handle validation completion"""
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        if return_code == 0:
            self.progress_var.set(100)
            self.progress_label.config(text="Validation Complete!")
            messagebox.showinfo("Success", "Validation completed successfully!\nCheck the Results and Charts tabs.")
            
            # Auto-refresh results and charts
            self.load_results()
            self.refresh_charts()
        else:
            self.progress_label.config(text="Validation Failed")
            messagebox.showerror("Error", "Validation failed. Check the output for details.")
            
    def validation_error(self, error_msg):
        """Handle validation error"""
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress_label.config(text="Error")
        messagebox.showerror("Error", f"Validation error: {error_msg}")
        
    def load_results(self):
        """Load and display performance results"""
        try:
            # Clear text
            self.results_text.delete(1.0, tk.END)
            
            # Check if results file exists
            if not os.path.exists('OMtree_results.csv'):
                self.results_text.insert(tk.END, "No results available. Please run validation first.")
                return
                
            # Load results
            df = pd.read_csv('OMtree_results.csv')
            
            # Calculate statistics
            stats_text = "="*80 + "\n"
            stats_text += "PERFORMANCE STATISTICS SUMMARY\n"
            stats_text += "="*80 + "\n\n"
            
            # Basic info
            stats_text += f"Total Observations: {len(df):,}\n"
            stats_text += f"Date Range: {df['date'].min()} to {df['date'].max()}\n\n"
            
            # Trading stats
            trades = df[df['prediction'] == 1]
            stats_text += f"TRADING METRICS:\n"
            stats_text += f"  Total Trades: {len(trades):,}\n"
            stats_text += f"  Trading Frequency: {len(trades)/len(df):.1%}\n"
            
            if len(trades) > 0:
                stats_text += f"  Hit Rate: {trades['actual_profitable'].mean():.1%}\n"
                stats_text += f"  Average P&L per Trade: {trades['target_value'].mean():+.4f}\n"
                stats_text += f"  Total P&L: {trades['target_value'].sum():.2f}\n"
                stats_text += f"  Win/Loss Ratio: {trades['actual_profitable'].sum()}/{len(trades)-trades['actual_profitable'].sum()}\n"
                
                # Monthly analysis
                df['date'] = pd.to_datetime(df['date'])
                df['month'] = df['date'].dt.to_period('M')
                
                monthly_trades = trades.copy()
                monthly_trades['date'] = pd.to_datetime(monthly_trades['date'])
                monthly_trades['month'] = monthly_trades['date'].dt.to_period('M')
                
                monthly_pnl = monthly_trades.groupby('month')['target_value'].sum()
                
                if len(monthly_pnl) > 0:
                    stats_text += f"\nMONTHLY PERFORMANCE:\n"
                    stats_text += f"  Average Monthly P&L: {monthly_pnl.mean():+.2f}\n"
                    stats_text += f"  Std Dev Monthly P&L: {monthly_pnl.std():.2f}\n"
                    
                    # Annualized Sharpe
                    annual_return = monthly_pnl.mean() * 12
                    annual_vol = monthly_pnl.std() * (12**0.5)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    stats_text += f"  Annualized Sharpe Ratio: {sharpe:.3f}\n"
                    
                    stats_text += f"  Best Month: {monthly_pnl.max():+.2f}\n"
                    stats_text += f"  Worst Month: {monthly_pnl.min():+.2f}\n"
                    stats_text += f"  Positive Months: {(monthly_pnl > 0).sum()}/{len(monthly_pnl)} ({(monthly_pnl > 0).mean():.1%})\n"
                    
                # Yearly breakdown
                yearly_trades = trades.copy()
                yearly_trades['year'] = pd.to_datetime(yearly_trades['date']).dt.year
                
                stats_text += f"\nYEARLY BREAKDOWN:\n"
                stats_text += f"{'Year':<8} {'Trades':<10} {'Hit Rate':<12} {'Total P&L':<12}\n"
                stats_text += "-"*50 + "\n"
                
                for year in sorted(yearly_trades['year'].unique()):
                    year_data = yearly_trades[yearly_trades['year'] == year]
                    stats_text += f"{year:<8} {len(year_data):<10} {year_data['actual_profitable'].mean():<12.1%} {year_data['target_value'].sum():<12.2f}\n"
                    
            # Display stats
            self.results_text.insert(tk.END, stats_text)
            
            # Also check for performance log
            if os.path.exists('OMtree_performance.csv'):
                perf_df = pd.read_csv('OMtree_performance.csv')
                if len(perf_df) > 0:
                    latest = perf_df.iloc[-1]
                    stats_text = "\n" + "="*80 + "\n"
                    stats_text += "LATEST CONFIGURATION PERFORMANCE\n"
                    stats_text += "="*80 + "\n"
                    stats_text += f"Timestamp: {latest['timestamp']}\n"
                    stats_text += f"Features: {latest['features']}\n"
                    stats_text += f"Trees: {latest['n_trees']}, Depth: {latest['max_depth']}\n"
                    stats_text += f"Vote Threshold: {latest['vote_threshold']}\n"
                    stats_text += f"Sharpe Ratio: {latest['sharpe_ratio']:.3f}\n"
                    self.results_text.insert(tk.END, stats_text)
                    
        except Exception as e:
            self.results_text.insert(tk.END, f"Error loading results: {str(e)}")
            
    def refresh_charts(self):
        """Refresh the list of available charts"""
        chart_files = []
        
        # Look for PNG files
        for file in os.listdir('.'):
            if file.endswith('.png') and 'OMtree' in file:
                chart_files.append(file)
                
        self.chart_combo['values'] = chart_files
        if chart_files:
            self.chart_combo.set(chart_files[0])
            self.load_chart()
            
    def load_chart(self, event=None):
        """Load and display selected chart"""
        try:
            chart_file = self.chart_var.get()
            if not chart_file or not os.path.exists(chart_file):
                return
                
            # Load image
            img = Image.open(chart_file)
            
            # Resize to fit canvas
            canvas_width = self.chart_canvas.winfo_width()
            canvas_height = self.chart_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate scaling
                img_width, img_height = img.size
                scale_x = canvas_width / img_width
                scale_y = canvas_height / img_height
                scale = min(scale_x, scale_y, 1.0)  # Don't upscale
                
                if scale < 1.0:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.chart_image = ImageTk.PhotoImage(img)
            
            # Clear canvas and display image
            self.chart_canvas.delete("all")
            self.chart_canvas.create_image(
                canvas_width//2 if canvas_width > 1 else 400,
                canvas_height//2 if canvas_height > 1 else 300,
                image=self.chart_image,
                anchor='center'
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load chart: {str(e)}")

def main():
    root = tk.Tk()
    app = OMtreeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()