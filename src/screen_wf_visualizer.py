import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import stats
import tkinter as tk
from tkinter import ttk
import os

class ScreenWFVisualizer:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.df = None  # Store loaded data
        self.setup_ui()
        
    def setup_ui(self):
        # Create main container
        main_container = ttk.Frame(self.parent_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Top section - Controls
        controls_frame = ttk.LabelFrame(main_container, text="IC Screen Controls", padding=10)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        # IC threshold control
        ttk.Label(controls_frame, text="IC Threshold:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.ic_threshold_var = tk.DoubleVar(value=0.0)
        ic_spinbox = ttk.Spinbox(controls_frame, from_=-1.0, to=1.0, increment=0.05, width=10,
                                 textvariable=self.ic_threshold_var, format="%.2f")
        ic_spinbox.grid(row=0, column=1, padx=5, pady=2)
        
        # Lookback for IC calculation
        ttk.Label(controls_frame, text="IC Lookback (# trades):").grid(row=0, column=2, sticky='w', padx=(20, 5), pady=2)
        self.lookback_var = tk.IntVar(value=30)
        lookback_spinbox = ttk.Spinbox(controls_frame, from_=10, to=200, increment=10, width=10,
                                       textvariable=self.lookback_var)
        lookback_spinbox.grid(row=0, column=3, padx=5, pady=2)
        
        # Update button
        ttk.Button(controls_frame, text="Apply Screen", 
                  command=self.apply_screen).grid(row=0, column=4, padx=20, pady=2)
        
        # Load button
        ttk.Button(controls_frame, text="Load Data", 
                  command=self.load_data).grid(row=0, column=5, padx=5, pady=2)
        
        # Export button
        ttk.Button(controls_frame, text="Export Results", 
                  command=self.export_results).grid(row=0, column=6, padx=5, pady=2)
        
        # Statistics comparison
        stats_frame = ttk.LabelFrame(main_container, text="Performance Comparison", padding=10)
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        # Create statistics grid
        headers = ['Metric', 'Original', 'Screened', 'Change']
        for col, header in enumerate(headers):
            ttk.Label(stats_frame, text=header, font=('Arial', 10, 'bold')).grid(
                row=0, column=col, padx=10, pady=2, sticky='w')
        
        self.stats_labels = {}
        metrics = [
            ('total_trades', 'Total Trades'),
            ('total_pnl', 'Total P&L'),
            ('avg_pnl', 'Avg P&L/Trade'),
            ('win_rate', 'Win Rate'),
            ('sharpe', 'Sharpe Ratio'),
            ('max_dd', 'Max Drawdown'),
            ('avg_ic', 'Avg IC'),
            ('trades_removed', 'Trades Removed')
        ]
        
        for idx, (key, label) in enumerate(metrics, 1):
            ttk.Label(stats_frame, text=label).grid(row=idx, column=0, padx=10, pady=2, sticky='w')
            
            # Original value
            orig_label = ttk.Label(stats_frame, text='N/A')
            orig_label.grid(row=idx, column=1, padx=10, pady=2, sticky='w')
            self.stats_labels[f'{key}_orig'] = orig_label
            
            # Screened value
            screen_label = ttk.Label(stats_frame, text='N/A')
            screen_label.grid(row=idx, column=2, padx=10, pady=2, sticky='w')
            self.stats_labels[f'{key}_screen'] = screen_label
            
            # Change value
            change_label = ttk.Label(stats_frame, text='N/A')
            change_label.grid(row=idx, column=3, padx=10, pady=2, sticky='w')
            self.stats_labels[f'{key}_change'] = change_label
        
        # Middle section - Equity Curves
        chart_frame = ttk.LabelFrame(main_container, text="Equity Curves", padding=10)
        chart_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Bottom section - Trade Details
        table_frame = ttk.LabelFrame(main_container, text="Screened Trades Detail", padding=10)
        table_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create Treeview
        tree_container = ttk.Frame(table_frame)
        tree_container.pack(fill='both', expand=True)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_container, orient="vertical")
        hsb = ttk.Scrollbar(tree_container, orient="horizontal")
        
        # Treeview
        columns = ('Date', 'Actual Return', 'Rolling IC', 'Included', 'Cum P&L')
        self.tree = ttk.Treeview(tree_container, columns=columns, show='headings',
                                 yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                                 height=8)
        
        # Configure scrollbars
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        
        # Configure columns
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120)
        
        # Pack treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
    
    def calculate_rolling_ic_for_screening(self, df, lookback_trades):
        """
        Calculate rolling IC for each trade using previous trades only (no look-ahead)
        Returns IC as of the PREVIOUS trade to avoid look-ahead bias
        """
        # Filter to only trades
        trades_df = df[df['signal'] == 1].copy().reset_index(drop=True)
        
        # Initialize IC column
        trades_df['rolling_ic'] = np.nan
        
        # Calculate rolling IC for each trade using only prior trades
        for i in range(len(trades_df)):
            if i >= lookback_trades:
                # Use trades from i-lookback to i-1 (excluding current trade)
                window = trades_df.iloc[i-lookback_trades:i]
                
                if len(window) >= 2:
                    # Calculate IC on the window
                    ic, _ = stats.spearmanr(window['prediction'].values, 
                                           window['actual'].values)
                    trades_df.loc[i, 'rolling_ic'] = ic
        
        return trades_df
    
    def apply_screen(self):
        """Apply IC screen to trades and update visualizations"""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        ic_threshold = self.ic_threshold_var.get()
        lookback = self.lookback_var.get()
        
        # Calculate rolling IC for screening
        trades_df = self.calculate_rolling_ic_for_screening(self.df, lookback)
        
        # Apply screen: include trade only if previous rolling IC >= threshold
        trades_df['included'] = trades_df['rolling_ic'] >= ic_threshold
        # First lookback trades always included (no IC history yet)
        trades_df.loc[:lookback-1, 'included'] = True
        
        # Calculate cumulative P&L for both original and screened
        trades_df['cum_pnl_orig'] = trades_df['actual'].cumsum()
        
        screened_returns = trades_df['actual'].copy()
        screened_returns[~trades_df['included']] = 0
        trades_df['cum_pnl_screen'] = screened_returns.cumsum()
        
        # Update charts
        self.update_equity_curves(trades_df)
        
        # Update statistics
        self.update_statistics(trades_df)
        
        # Update table
        self.update_table(trades_df)
        
        print(f"Screen applied: {(~trades_df['included']).sum()} trades removed out of {len(trades_df)}")
    
    def update_equity_curves(self, trades_df):
        """Update the equity curve comparison chart"""
        self.figure.clear()
        
        # Create two subplots
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        # Plot cumulative P&L comparison
        ax1.plot(trades_df['date'], trades_df['cum_pnl_orig'], 
                label='Original', color='blue', linewidth=1.5)
        ax1.plot(trades_df['date'], trades_df['cum_pnl_screen'], 
                label=f'Screened (IC >= {self.ic_threshold_var.get():.2f})', 
                color='green', linewidth=1.5, alpha=0.8)
        
        # Shade removed trade regions
        for i in range(len(trades_df)):
            if not trades_df.iloc[i]['included']:
                ax1.axvline(x=trades_df.iloc[i]['date'], color='red', alpha=0.2, linewidth=0.5)
        
        ax1.set_ylabel('Cumulative P&L')
        ax1.set_title('Equity Curves: Original vs IC-Screened')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown comparison
        dd_orig = self.calculate_drawdown(trades_df['cum_pnl_orig'].values)
        dd_screen = self.calculate_drawdown(trades_df['cum_pnl_screen'].values)
        
        ax2.fill_between(trades_df['date'], dd_orig, 0, 
                         color='blue', alpha=0.3, label='Original DD')
        ax2.fill_between(trades_df['date'], dd_screen, 0, 
                         color='green', alpha=0.3, label='Screened DD')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown')
        ax2.set_title('Drawdown Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def calculate_drawdown(self, cum_returns):
        """Calculate drawdown series"""
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = cum_returns - running_max
        return drawdown
    
    def update_statistics(self, trades_df):
        """Update performance statistics comparison"""
        # Original statistics
        orig_trades = len(trades_df)
        orig_pnl = trades_df['cum_pnl_orig'].iloc[-1]
        orig_avg_pnl = trades_df['actual'].mean()
        orig_win_rate = (trades_df['actual'] > 0).mean()
        orig_sharpe = self.calculate_sharpe(trades_df['actual'])
        orig_max_dd = self.calculate_drawdown(trades_df['cum_pnl_orig'].values).min()
        
        # Screened statistics
        screened_trades = trades_df['included'].sum()
        screen_pnl = trades_df['cum_pnl_screen'].iloc[-1]
        screened_returns = trades_df.loc[trades_df['included'], 'actual']
        screen_avg_pnl = screened_returns.mean() if len(screened_returns) > 0 else 0
        screen_win_rate = (screened_returns > 0).mean() if len(screened_returns) > 0 else 0
        screen_sharpe = self.calculate_sharpe(screened_returns) if len(screened_returns) > 0 else 0
        screen_max_dd = self.calculate_drawdown(trades_df['cum_pnl_screen'].values).min()
        
        # Average IC of included trades
        included_ic = trades_df.loc[trades_df['included'], 'rolling_ic'].mean()
        
        # Update labels
        self.stats_labels['total_trades_orig'].config(text=f"{orig_trades}")
        self.stats_labels['total_trades_screen'].config(text=f"{screened_trades}")
        self.stats_labels['total_trades_change'].config(text=f"{screened_trades - orig_trades}")
        
        self.stats_labels['total_pnl_orig'].config(text=f"{orig_pnl:.4f}")
        self.stats_labels['total_pnl_screen'].config(text=f"{screen_pnl:.4f}")
        self.stats_labels['total_pnl_change'].config(text=f"{screen_pnl - orig_pnl:+.4f}")
        
        self.stats_labels['avg_pnl_orig'].config(text=f"{orig_avg_pnl:.6f}")
        self.stats_labels['avg_pnl_screen'].config(text=f"{screen_avg_pnl:.6f}")
        self.stats_labels['avg_pnl_change'].config(text=f"{screen_avg_pnl - orig_avg_pnl:+.6f}")
        
        self.stats_labels['win_rate_orig'].config(text=f"{orig_win_rate:.2%}")
        self.stats_labels['win_rate_screen'].config(text=f"{screen_win_rate:.2%}")
        self.stats_labels['win_rate_change'].config(text=f"{(screen_win_rate - orig_win_rate)*100:+.1f}%")
        
        self.stats_labels['sharpe_orig'].config(text=f"{orig_sharpe:.3f}")
        self.stats_labels['sharpe_screen'].config(text=f"{screen_sharpe:.3f}")
        self.stats_labels['sharpe_change'].config(text=f"{screen_sharpe - orig_sharpe:+.3f}")
        
        self.stats_labels['max_dd_orig'].config(text=f"{orig_max_dd:.4f}")
        self.stats_labels['max_dd_screen'].config(text=f"{screen_max_dd:.4f}")
        self.stats_labels['max_dd_change'].config(text=f"{screen_max_dd - orig_max_dd:+.4f}")
        
        self.stats_labels['avg_ic_orig'].config(text="N/A")
        self.stats_labels['avg_ic_screen'].config(text=f"{included_ic:.4f}" if not np.isnan(included_ic) else "N/A")
        self.stats_labels['avg_ic_change'].config(text="N/A")
        
        trades_removed = orig_trades - screened_trades
        removal_pct = (trades_removed / orig_trades * 100) if orig_trades > 0 else 0
        self.stats_labels['trades_removed_orig'].config(text="0")
        self.stats_labels['trades_removed_screen'].config(text=f"{trades_removed}")
        self.stats_labels['trades_removed_change'].config(text=f"{removal_pct:.1f}%")
    
    def calculate_sharpe(self, returns):
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        mean_return = returns.mean()
        std_return = returns.std()
        if std_return == 0:
            return 0
        # Annualized Sharpe (assuming daily returns, 252 trading days)
        return np.sqrt(252) * mean_return / std_return
    
    def update_table(self, trades_df):
        """Update the table with screened trade details"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Show last 100 trades
        for _, row in trades_df.tail(100).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else ''
            actual_str = f"{row['actual']:.6f}"
            ic_str = f"{row['rolling_ic']:.4f}" if pd.notna(row['rolling_ic']) else 'N/A'
            included_str = 'YES' if row['included'] else 'NO'
            cum_pnl_str = f"{row['cum_pnl_screen']:.4f}"
            
            # Color code excluded trades
            tags = () if row['included'] else ('excluded',)
            
            # Insert into tree
            item = self.tree.insert('', 'end', values=(
                date_str, actual_str, ic_str, included_str, cum_pnl_str
            ), tags=tags)
        
        # Configure tag colors
        self.tree.tag_configure('excluded', background='#ffcccc')
    
    def load_data(self):
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
            
            print(f"Loaded {len(self.df)} predictions, {len(self.df[self.df['signal'] == 1])} trades")
            
            # Auto-apply screen with current settings
            self.apply_screen()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
    
    def export_results(self):
        """Export screened results to CSV"""
        if self.df is None:
            print("No data to export")
            return
        
        lookback = self.lookback_var.get()
        ic_threshold = self.ic_threshold_var.get()
        
        # Get screened trades
        trades_df = self.calculate_rolling_ic_for_screening(self.df, lookback)
        trades_df['included'] = trades_df['rolling_ic'] >= ic_threshold
        trades_df.loc[:lookback-1, 'included'] = True
        
        # Calculate cumulative P&L
        trades_df['cum_pnl_orig'] = trades_df['actual'].cumsum()
        screened_returns = trades_df['actual'].copy()
        screened_returns[~trades_df['included']] = 0
        trades_df['cum_pnl_screen'] = screened_returns.cumsum()
        
        # Export
        filename = f'ic_screened_results_threshold{ic_threshold:.2f}_lookback{lookback}.csv'
        trades_df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")