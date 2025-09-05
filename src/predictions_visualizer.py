import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import stats
import tkinter as tk
from tkinter import ttk
import os

class PredictionsVisualizer:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.df = None  # Store loaded data
        self.setup_ui()
        
    def setup_ui(self):
        # Create main container
        main_container = ttk.Frame(self.parent_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Top section - Controls
        controls_frame = ttk.LabelFrame(main_container, text="Analysis Controls", padding=10)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        # Lookback control (in number of trades)
        ttk.Label(controls_frame, text="Lookback (# trades):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.lookback_var = tk.IntVar(value=50)
        lookback_spinbox = ttk.Spinbox(controls_frame, from_=10, to=500, increment=10, width=10,
                                       textvariable=self.lookback_var)
        lookback_spinbox.grid(row=0, column=1, padx=5, pady=2)
        
        # Chart type selection
        ttk.Label(controls_frame, text="Chart Type:").grid(row=0, column=2, sticky='w', padx=(20, 5), pady=2)
        self.chart_type_var = tk.StringVar(value='IC')
        chart_combo = ttk.Combobox(controls_frame, textvariable=self.chart_type_var, 
                                   values=['IC', 'P-Value', 'Both'], state='readonly', width=10)
        chart_combo.grid(row=0, column=3, padx=5, pady=2)
        
        # Update button
        ttk.Button(controls_frame, text="Update Charts", 
                  command=self.update_charts).grid(row=0, column=4, padx=20, pady=2)
        
        # Load button
        ttk.Button(controls_frame, text="Load Predictions", 
                  command=self.load_predictions).grid(row=0, column=5, padx=5, pady=2)
        
        # Export button
        ttk.Button(controls_frame, text="Export Data", 
                  command=self.export_data).grid(row=0, column=6, padx=5, pady=2)
        
        # Statistics display
        stats_row = 1
        ttk.Label(controls_frame, text="Latest Statistics:", font=('Arial', 10, 'bold')).grid(
            row=stats_row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 2))
        
        self.stats_labels = {}
        stats_to_show = [
            ('last_ic', 'Last IC:', 1, 2),
            ('last_pval', 'Last P-val:', 1, 4),
            ('n_trades', 'Total Trades:', 2, 0),
            ('avg_ic', 'Avg IC:', 2, 2),
            ('ic_stability', 'IC Stability:', 2, 4)
        ]
        
        for key, label, row, col in stats_to_show:
            ttk.Label(controls_frame, text=label).grid(row=row, column=col, sticky='w', padx=5, pady=2)
            value_label = ttk.Label(controls_frame, text='N/A', font=('Arial', 10))
            value_label.grid(row=row, column=col+1, sticky='w', padx=5, pady=2)
            self.stats_labels[key] = value_label
        
        # Middle section - Charts
        chart_frame = ttk.LabelFrame(main_container, text="Rolling Analysis", padding=10)
        chart_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Bottom section - Table
        table_frame = ttk.LabelFrame(main_container, text="Trade Details (Last 100)", padding=10)
        table_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create Treeview with scrollbars
        tree_container = ttk.Frame(table_frame)
        tree_container.pack(fill='both', expand=True)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_container, orient="vertical")
        hsb = ttk.Scrollbar(tree_container, orient="horizontal")
        
        # Treeview
        columns = ('Date', 'Time', 'Prediction', 'Actual', 'Signal', 'IC', 'P-Value')
        self.tree = ttk.Treeview(tree_container, columns=columns, show='headings',
                                 yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                                 height=8)
        
        # Configure scrollbars
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        
        # Configure columns
        for col in columns:
            self.tree.heading(col, text=col)
            width = 100 if col not in ['Date', 'Time'] else 90
            self.tree.column(col, width=width)
        
        # Pack treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
    
    def calculate_rolling_ic(self, df, lookback_trades):
        """Calculate rolling IC based on number of trades (not time)"""
        # Filter to only trades (signal == 1)
        trades_df = df[df['signal'] == 1].copy()
        
        if len(trades_df) < lookback_trades:
            return None, None, None
        
        rolling_dates = []
        rolling_ic = []
        rolling_pval = []
        
        for i in range(lookback_trades, len(trades_df) + 1):
            # Get the last 'lookback_trades' trades
            window = trades_df.iloc[i-lookback_trades:i]
            
            # Calculate IC for this window
            ic, pval = stats.spearmanr(window['prediction'].values, 
                                       window['actual'].values)
            
            rolling_dates.append(window['date'].iloc[-1])
            rolling_ic.append(ic)
            rolling_pval.append(pval)
        
        return rolling_dates, rolling_ic, rolling_pval
    
    def update_charts(self):
        """Update the rolling charts based on current settings"""
        if self.df is None:
            print("No data loaded. Please load predictions first.")
            return
        
        lookback = self.lookback_var.get()
        chart_type = self.chart_type_var.get()
        
        # Calculate rolling IC for trades only
        dates, ic_values, pval_values = self.calculate_rolling_ic(self.df, lookback)
        
        if dates is None:
            print(f"Not enough trades for lookback period of {lookback}")
            return
        
        # Clear the figure
        self.figure.clear()
        
        if chart_type == 'Both':
            # Create two subplots
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212)
            
            # Plot IC
            ax1.plot(dates, ic_values, color='blue', linewidth=1.5)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.fill_between(dates, ic_values, 0, where=np.array(ic_values) > 0, 
                            alpha=0.3, color='green', label='Positive IC')
            ax1.fill_between(dates, ic_values, 0, where=np.array(ic_values) <= 0, 
                            alpha=0.3, color='red', label='Negative IC')
            ax1.set_ylabel('Information Coefficient')
            ax1.set_title(f'Rolling IC (Last {lookback} Trades)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot P-value
            ax2.plot(dates, pval_values, color='purple', linewidth=1.5)
            ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% Significance')
            ax2.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.5, label='1% Significance')
            ax2.fill_between(dates, pval_values, 0, where=np.array(pval_values) <= 0.05, 
                            alpha=0.3, color='green', label='Significant')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('P-Value')
            ax2.set_title(f'Rolling P-Value (Last {lookback} Trades)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
        else:
            # Single plot
            ax = self.figure.add_subplot(111)
            
            if chart_type == 'IC':
                ax.plot(dates, ic_values, color='blue', linewidth=1.5)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.fill_between(dates, ic_values, 0, where=np.array(ic_values) > 0, 
                               alpha=0.3, color='green', label='Positive IC')
                ax.fill_between(dates, ic_values, 0, where=np.array(ic_values) <= 0, 
                               alpha=0.3, color='red', label='Negative IC')
                ax.set_ylabel('Information Coefficient')
                ax.set_title(f'Rolling IC (Last {lookback} Trades)')
            else:  # P-Value
                ax.plot(dates, pval_values, color='purple', linewidth=1.5)
                ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% Significance')
                ax.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.5, label='1% Significance')
                ax.fill_between(dates, pval_values, 0, where=np.array(pval_values) <= 0.05, 
                               alpha=0.3, color='green', label='Significant')
                ax.set_ylabel('P-Value')
                ax.set_title(f'Rolling P-Value (Last {lookback} Trades)')
            
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Update statistics
        self.update_statistics(ic_values, pval_values)
    
    def update_statistics(self, ic_values, pval_values):
        """Update the statistics labels"""
        if ic_values and pval_values:
            # Latest values
            self.stats_labels['last_ic'].config(text=f"{ic_values[-1]:.4f}")
            self.stats_labels['last_pval'].config(text=f"{pval_values[-1]:.4f}")
            
            # Average IC
            avg_ic = np.mean(ic_values)
            self.stats_labels['avg_ic'].config(text=f"{avg_ic:.4f}")
            
            # IC Stability (standard deviation)
            ic_std = np.std(ic_values)
            self.stats_labels['ic_stability'].config(text=f"{ic_std:.4f}")
            
            # Total trades
            trades_df = self.df[self.df['signal'] == 1]
            self.stats_labels['n_trades'].config(text=str(len(trades_df)))
    
    def load_predictions(self):
        """Load predictions from walk-forward results"""
        import configparser
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read('OMtree_config.ini')
        model_type = config.get('model', 'model_type', fallback='longonly')
        
        # Try to load the walk-forward results file
        results_file = f'walkforward_results_{model_type}.csv'
        if not os.path.exists(results_file):
            print(f"Results file {results_file} not found")
            return
        
        try:
            # Load the results
            self.df = pd.read_csv(results_file)
            
            # Ensure we have the required columns
            required_cols = ['date', 'prediction', 'actual', 'signal']
            if not all(col in self.df.columns for col in required_cols):
                print(f"Missing required columns in {results_file}")
                return
            
            # Convert date to datetime
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Sort by date
            self.df = self.df.sort_values('date')
            
            # Update charts
            self.update_charts()
            
            # Update table with trade details
            self.update_table()
            
            print(f"Loaded {len(self.df)} predictions, {len(self.df[self.df['signal'] == 1])} trades")
            
        except Exception as e:
            print(f"Error loading predictions: {str(e)}")
    
    def update_table(self):
        """Update the table with rolling IC for each trade"""
        if self.df is None:
            return
        
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get trades only
        trades_df = self.df[self.df['signal'] == 1].copy()
        lookback = self.lookback_var.get()
        
        # Show last 100 trades with their rolling IC
        for idx, (_, row) in enumerate(trades_df.tail(100).iterrows()):
            # Calculate IC for this trade's window if we have enough history
            trade_position = len(trades_df) - len(trades_df.tail(100)) + idx
            
            ic_str = ''
            pval_str = ''
            if trade_position >= lookback:
                # Get window of trades ending at this trade
                window = trades_df.iloc[trade_position-lookback+1:trade_position+1]
                ic, pval = stats.spearmanr(window['prediction'].values, 
                                          window['actual'].values)
                ic_str = f"{ic:.4f}" if not np.isnan(ic) else 'N/A'
                pval_str = f"{pval:.4f}" if not np.isnan(pval) else 'N/A'
            
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else ''
            time_str = row['time'] if 'time' in row and pd.notna(row['time']) else ''
            pred_str = f"{row['prediction']:.6f}" if pd.notna(row['prediction']) else ''
            actual_str = f"{row['actual']:.6f}" if pd.notna(row['actual']) else ''
            signal_str = 'TRADE' if row['signal'] == 1 else 'HOLD'
            
            # Insert into tree
            self.tree.insert('', 'end', values=(
                date_str, time_str, pred_str, actual_str, 
                signal_str, ic_str, pval_str
            ))
    
    def export_data(self):
        """Export rolling IC data to CSV"""
        if self.df is None:
            print("No data to export")
            return
        
        lookback = self.lookback_var.get()
        dates, ic_values, pval_values = self.calculate_rolling_ic(self.df, lookback)
        
        if dates:
            export_df = pd.DataFrame({
                'date': dates,
                'rolling_ic': ic_values,
                'rolling_pvalue': pval_values
            })
            
            filename = f'rolling_ic_analysis_{lookback}trades.csv'
            export_df.to_csv(filename, index=False)
            print(f"Data exported to {filename}")
        else:
            print("Not enough data to export")