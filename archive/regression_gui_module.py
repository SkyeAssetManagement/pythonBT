import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RegressionAnalysisTab:
    def __init__(self, parent_notebook):
        self.parent = parent_notebook
        self.df = None
        self.selected_x_vars = []
        self.selected_y_vars = []
        
        # Create regression tab
        self.regression_frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.regression_frame, text='Regression Analysis')
        
        self.setup_regression_tab()
    
    def setup_regression_tab(self):
        """Setup the regression analysis tab"""
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
        self.file_entry.insert(0, "DTSmlDATA7x7.csv")
        ttk.Button(file_frame, text="Load", command=self.load_data).pack(side='left')
        
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
                                    yscrollcommand=x_scroll.set, height=10, exportselection=False)
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
                                    yscrollcommand=y_scroll.set, height=10, exportselection=False)
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
    
    def load_data(self):
        """Load data from CSV file"""
        try:
            filename = self.file_entry.get()
            self.df = pd.read_csv(filename)
            
            # Try to load from config first
            import configparser
            config = configparser.ConfigParser()
            config_loaded = False
            
            if config.read('OMtree_config.ini'):
                # Check for validation_end_date and filter data
                if 'validation' in config and 'validation_end_date' in config['validation']:
                    validation_end_date = config['validation']['validation_end_date']
                    
                    # Find date column
                    date_col = None
                    if 'data' in config and 'date_column' in config['data']:
                        date_col = config['data']['date_column']
                    else:
                        # Try common date column names
                        for col in ['Date', 'date', 'DateTime', 'datetime', 'Date/Time']:
                            if col in self.df.columns:
                                date_col = col
                                break
                    
                    if date_col:
                        self.df[date_col] = pd.to_datetime(self.df[date_col])
                        validation_end_date = pd.to_datetime(validation_end_date)
                        original_len = len(self.df)
                        self.df = self.df[self.df[date_col] <= validation_end_date].copy()
                        filtered_len = len(self.df)
                        if original_len > filtered_len:
                            messagebox.showinfo("Data Filtered", 
                                              f"Data filtered to validation_end_date ({validation_end_date.strftime('%Y-%m-%d')})\n"
                                              f"Using {filtered_len} of {original_len} rows\n"
                                              f"{original_len - filtered_len} rows reserved for out-of-sample testing")
            
            # Identify columns
            self.all_columns = list(self.df.columns)
            
            if config.read('OMtree_config.ini'):
                try:
                    # Try to get features and targets from config
                    if 'data' in config:
                        if 'feature_columns' in config['data']:
                            self.feature_columns = [col.strip() for col in config['data']['feature_columns'].split(',')]
                            config_loaded = True
                        if 'all_targets' in config['data']:
                            self.target_columns = [col.strip() for col in config['data']['all_targets'].split(',')]
                            config_loaded = True
                        elif 'target_column' in config['data']:
                            # Fallback to single target
                            self.target_columns = [config['data']['target_column'].strip()]
                            config_loaded = True
                except:
                    pass
            
            # If not loaded from config, try to identify based on naming convention
            if not config_loaded:
                self.feature_columns = [col for col in self.all_columns 
                                       if col.startswith('Ret_') and 'fwd' not in col]
                self.target_columns = [col for col in self.all_columns 
                                     if col.startswith('Ret_fwd')]
            
            # If no clear pattern, let user decide
            if not self.feature_columns and not self.target_columns:
                # Exclude obvious non-numeric columns
                exclude = ['Date', 'Time', 'DateTime', 'Ticker']
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                self.feature_columns = [col for col in numeric_cols if col not in exclude]
                self.target_columns = self.feature_columns.copy()
            
            self.update_variable_lists()
            self.status_var.set(f"Loaded {len(self.df)} rows with {len(self.all_columns)} columns")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Failed to load data")
    
    def select_all_x(self):
        """Select all X variables"""
        self.x_listbox.select_set(0, tk.END)
        self.update_selection_info()
    
    def deselect_all_x(self):
        """Deselect all X variables"""
        self.x_listbox.select_clear(0, tk.END)
        self.update_selection_info()
    
    def select_all_y(self):
        """Select all Y variables"""
        self.y_listbox.select_set(0, tk.END)
        self.update_selection_info()
    
    def deselect_all_y(self):
        """Deselect all Y variables"""
        self.y_listbox.select_clear(0, tk.END)
        self.update_selection_info()
    
    def update_selection_info(self, event=None):
        """Update the selection info label"""
        x_count = len(self.x_listbox.curselection())
        y_count = len(self.y_listbox.curselection())
        total_regressions = x_count * y_count
        
        if total_regressions > 0:
            self.selection_info.config(
                text=f"Selected: {x_count} X variables × {y_count} Y variables = {total_regressions} regressions"
            )
        else:
            self.selection_info.config(text="Please select variables for analysis")
    
    def update_variable_lists(self):
        """Update the variable selection lists based on analysis type"""
        if self.df is None:
            return
        
        analysis_type = self.analysis_type.get()
        
        # Clear lists
        self.x_listbox.delete(0, tk.END)
        self.y_listbox.delete(0, tk.END)
        
        if analysis_type == "features_to_targets":
            # X = features, Y = targets
            for col in self.feature_columns:
                self.x_listbox.insert(tk.END, col)
            for col in self.target_columns:
                self.y_listbox.insert(tk.END, col)
        elif analysis_type == "features_to_features":
            # Both X and Y = features
            for col in self.feature_columns:
                self.x_listbox.insert(tk.END, col)
                self.y_listbox.insert(tk.END, col)
        elif analysis_type == "targets_to_targets":
            # Both X and Y = targets
            for col in self.target_columns:
                self.x_listbox.insert(tk.END, col)
                self.y_listbox.insert(tk.END, col)
        else:  # custom
            # All columns available for both
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                self.x_listbox.insert(tk.END, col)
                self.y_listbox.insert(tk.END, col)
        
        # Auto-select all items in both listboxes
        self.x_listbox.select_set(0, tk.END)
        self.y_listbox.select_set(0, tk.END)
        
        # Update selection info
        self.update_selection_info()
    
    def run_regression(self):
        """Run regression analysis on selected variables"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        # Get selected variables
        x_indices = self.x_listbox.curselection()
        y_indices = self.y_listbox.curselection()
        
        if not x_indices or not y_indices:
            messagebox.showwarning("Warning", "Please select at least one X and one Y variable")
            return
        
        self.selected_x_vars = [self.x_listbox.get(i) for i in x_indices]
        self.selected_y_vars = [self.y_listbox.get(i) for i in y_indices]
        
        self.status_var.set("Running regression analysis...")
        
        try:
            # Get unique columns (avoid duplicates if same column selected in both X and Y)
            all_vars = list(set(self.selected_x_vars + self.selected_y_vars))
            
            # Prepare data - select only numeric columns
            df_subset = self.df[all_vars].copy()
            
            # Convert to numeric, forcing errors to NaN
            for col in all_vars:
                df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
            
            # Handle zeros as NaN if they represent missing data
            df_clean = df_subset.replace(0.0, np.nan)
            df_clean = df_clean.dropna()
            
            if len(df_clean) < 10:
                messagebox.showwarning("Warning", "Not enough valid data points for analysis")
                return
            
            # Run regression analysis
            self.perform_regression_analysis(df_clean)
            
            self.status_var.set(f"Analysis complete: {len(self.selected_x_vars)} X vars × "
                              f"{len(self.selected_y_vars)} Y vars = "
                              f"{len(self.selected_x_vars) * len(self.selected_y_vars)} regressions")
            
        except Exception as e:
            import traceback
            error_msg = f"Regression analysis failed:\n{str(e)}\n\nDetails:\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Analysis failed")
    
    def perform_regression_analysis(self, df_clean):
        """Perform the actual regression analysis"""
        n_x = len(self.selected_x_vars)
        n_y = len(self.selected_y_vars)
        
        # Initialize matrices
        r2_matrix = np.zeros((n_x, n_y))
        coef_matrix = np.zeros((n_x, n_y))
        pvalue_matrix = np.zeros((n_x, n_y))
        
        # Perform regressions
        for i, x_var in enumerate(self.selected_x_vars):
            for j, y_var in enumerate(self.selected_y_vars):
                try:
                    # Skip if same variable (perfect correlation)
                    if x_var == y_var:
                        r2_matrix[i, j] = 1.0
                        coef_matrix[i, j] = 1.0
                        pvalue_matrix[i, j] = 0.0
                        continue
                    
                    # Ensure we have the columns in our cleaned dataframe
                    if x_var not in df_clean.columns or y_var not in df_clean.columns:
                        r2_matrix[i, j] = np.nan
                        coef_matrix[i, j] = np.nan
                        pvalue_matrix[i, j] = np.nan
                        continue
                    
                    # Get data as numpy arrays
                    X = df_clean[x_var].values.reshape(-1, 1)
                    y = df_clean[y_var].values
                    
                    # Check for constant values
                    if np.std(X) == 0 or np.std(y) == 0:
                        r2_matrix[i, j] = np.nan
                        coef_matrix[i, j] = np.nan
                        pvalue_matrix[i, j] = np.nan
                        continue
                    
                    # Linear regression
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    
                    r2_matrix[i, j] = r2_score(y, y_pred)
                    coef_matrix[i, j] = model.coef_[0]
                    
                    # Calculate p-value
                    _, _, _, p_value, _ = stats.linregress(X.flatten(), y)
                    pvalue_matrix[i, j] = p_value
                    
                except Exception as e:
                    print(f"Error in regression {x_var} -> {y_var}: {str(e)}")
                    r2_matrix[i, j] = np.nan
                    coef_matrix[i, j] = np.nan
                    pvalue_matrix[i, j] = np.nan
        
        # Create visualizations
        self.create_matrix_plots(r2_matrix, coef_matrix, pvalue_matrix, df_clean)
        
        # Generate summary statistics
        self.generate_summary(r2_matrix, coef_matrix, pvalue_matrix, df_clean)
    
    def create_matrix_plots(self, r2_matrix, coef_matrix, pvalue_matrix, df_clean):
        """Create matrix visualization plots"""
        # Clear previous plots
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        
        # Create figure with subplots
        fig = plt.Figure(figsize=(14, 10), tight_layout=True)
        
        # R² heatmap
        ax1 = fig.add_subplot(2, 2, 1)
        sns.heatmap(r2_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=self.selected_y_vars,
                   yticklabels=self.selected_x_vars,
                   cbar_kws={'label': 'R² Score'},
                   ax=ax1, vmin=0, vmax=1,
                   annot_kws={'size': 8})
        ax1.set_title('R² Values', fontweight='bold', fontsize=11)
        ax1.set_xlabel('Y Variables (Dependent)', fontsize=9)
        ax1.set_ylabel('X Variables (Independent)', fontsize=9)
        ax1.tick_params(axis='both', labelsize=8)
        
        # Coefficient heatmap
        ax2 = fig.add_subplot(2, 2, 2)
        sns.heatmap(coef_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   xticklabels=self.selected_y_vars,
                   yticklabels=self.selected_x_vars,
                   center=0, cbar_kws={'label': 'Coefficient'},
                   ax=ax2, annot_kws={'size': 8})
        ax2.set_title('Regression Coefficients', fontweight='bold', fontsize=11)
        ax2.set_xlabel('Y Variables (Dependent)', fontsize=9)
        ax2.set_ylabel('X Variables (Independent)', fontsize=9)
        ax2.tick_params(axis='both', labelsize=8)
        
        # Correlation heatmap
        ax3 = fig.add_subplot(2, 2, 3)
        # Get unique columns for correlation calculation
        x_vars_in_df = [v for v in self.selected_x_vars if v in df_clean.columns]
        y_vars_in_df = [v for v in self.selected_y_vars if v in df_clean.columns]
        all_vars_unique = list(set(x_vars_in_df + y_vars_in_df))
        
        if all_vars_unique:
            correlation_matrix = df_clean[all_vars_unique].corr()
            # Create subset with proper indexing
            correlation_subset = pd.DataFrame(np.nan, 
                                            index=x_vars_in_df, 
                                            columns=y_vars_in_df)
            for x_var in x_vars_in_df:
                for y_var in y_vars_in_df:
                    if x_var in correlation_matrix.index and y_var in correlation_matrix.columns:
                        correlation_subset.loc[x_var, y_var] = correlation_matrix.loc[x_var, y_var]
            
            sns.heatmap(correlation_subset, annot=True, fmt='.3f', cmap='coolwarm',
                       center=0, cbar_kws={'label': 'Correlation'},
                       ax=ax3, vmin=-1, vmax=1, annot_kws={'size': 8})
        else:
            ax3.text(0.5, 0.5, 'No valid correlations', ha='center', va='center')
        ax3.set_title('Correlation Matrix', fontweight='bold', fontsize=11)
        ax3.set_xlabel('Y Variables', fontsize=9)
        ax3.set_ylabel('X Variables', fontsize=9)
        ax3.tick_params(axis='both', labelsize=8)
        
        # Significance heatmap
        ax4 = fig.add_subplot(2, 2, 4)
        sig_matrix = np.where(pvalue_matrix < 0.001, 3,
                            np.where(pvalue_matrix < 0.01, 2,
                            np.where(pvalue_matrix < 0.05, 1, 0)))
        sns.heatmap(sig_matrix, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=self.selected_y_vars,
                   yticklabels=self.selected_x_vars,
                   cbar_kws={'label': 'Significance Level'},
                   ax=ax4, vmin=0, vmax=3, annot_kws={'size': 8})
        ax4.set_title('Statistical Significance\n(0=NS, 1=p<0.05, 2=p<0.01, 3=p<0.001)', 
                     fontweight='bold', fontsize=10)
        ax4.set_xlabel('Y Variables (Dependent)', fontsize=9)
        ax4.set_ylabel('X Variables (Independent)', fontsize=9)
        ax4.tick_params(axis='both', labelsize=8)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.matrix_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def generate_summary(self, r2_matrix, coef_matrix, pvalue_matrix, df_clean):
        """Generate summary statistics text"""
        self.summary_text.delete('1.0', tk.END)
        
        # Header
        self.summary_text.insert(tk.END, "="*80 + "\n")
        self.summary_text.insert(tk.END, "REGRESSION ANALYSIS SUMMARY\n")
        self.summary_text.insert(tk.END, "="*80 + "\n\n")
        
        # Data info
        self.summary_text.insert(tk.END, f"Data points used: {len(df_clean)}\n")
        self.summary_text.insert(tk.END, f"X variables: {', '.join(self.selected_x_vars)}\n")
        self.summary_text.insert(tk.END, f"Y variables: {', '.join(self.selected_y_vars)}\n")
        self.summary_text.insert(tk.END, f"Total regressions: {r2_matrix.size}\n\n")
        
        # Top relationships by R²
        self.summary_text.insert(tk.END, "Top 10 Relationships by R² Score:\n")
        self.summary_text.insert(tk.END, "-"*50 + "\n")
        self.summary_text.insert(tk.END, f"{'X Variable':<15} {'Y Variable':<15} {'R²':<10} {'Coef':<10} {'P-value':<10}\n")
        self.summary_text.insert(tk.END, "-"*50 + "\n")
        
        # Create sorted list of all relationships
        relationships = []
        for i, x_var in enumerate(self.selected_x_vars):
            for j, y_var in enumerate(self.selected_y_vars):
                relationships.append({
                    'x': x_var,
                    'y': y_var,
                    'r2': r2_matrix[i, j],
                    'coef': coef_matrix[i, j],
                    'pvalue': pvalue_matrix[i, j]
                })
        
        relationships.sort(key=lambda x: x['r2'], reverse=True)
        
        for rel in relationships[:10]:
            self.summary_text.insert(tk.END, 
                f"{rel['x']:<15} {rel['y']:<15} {rel['r2']:<10.4f} "
                f"{rel['coef']:<10.4f} {rel['pvalue']:<10.4e}\n")
        
        # Average R² by X variable
        self.summary_text.insert(tk.END, "\n\nAverage R² by X Variable:\n")
        self.summary_text.insert(tk.END, "-"*30 + "\n")
        for i, x_var in enumerate(self.selected_x_vars):
            avg_r2 = np.mean(r2_matrix[i, :])
            self.summary_text.insert(tk.END, f"{x_var}: {avg_r2:.4f}\n")
        
        # Average R² by Y variable
        self.summary_text.insert(tk.END, "\n\nAverage R² by Y Variable:\n")
        self.summary_text.insert(tk.END, "-"*30 + "\n")
        for j, y_var in enumerate(self.selected_y_vars):
            avg_r2 = np.mean(r2_matrix[:, j])
            self.summary_text.insert(tk.END, f"{y_var}: {avg_r2:.4f}\n")
        
        # Overall statistics
        self.summary_text.insert(tk.END, "\n\nOverall Statistics:\n")
        self.summary_text.insert(tk.END, "-"*30 + "\n")
        self.summary_text.insert(tk.END, f"Overall Average R²: {np.mean(r2_matrix):.4f}\n")
        self.summary_text.insert(tk.END, f"Maximum R²: {np.max(r2_matrix):.4f}\n")
        self.summary_text.insert(tk.END, f"Minimum R²: {np.min(r2_matrix):.4f}\n")
        
        # Significance summary
        total_tests = r2_matrix.size
        sig_001 = np.sum(pvalue_matrix < 0.001)
        sig_01 = np.sum(pvalue_matrix < 0.01)
        sig_05 = np.sum(pvalue_matrix < 0.05)
        
        self.summary_text.insert(tk.END, "\n\nStatistical Significance:\n")
        self.summary_text.insert(tk.END, "-"*30 + "\n")
        self.summary_text.insert(tk.END, f"p < 0.001: {sig_001}/{total_tests} ({100*sig_001/total_tests:.1f}%)\n")
        self.summary_text.insert(tk.END, f"p < 0.01:  {sig_01}/{total_tests} ({100*sig_01/total_tests:.1f}%)\n")
        self.summary_text.insert(tk.END, f"p < 0.05:  {sig_05}/{total_tests} ({100*sig_05/total_tests:.1f}%)\n")