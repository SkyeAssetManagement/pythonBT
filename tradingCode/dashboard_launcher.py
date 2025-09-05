#!/usr/bin/env python3
"""
Trading Dashboard Launcher
==========================
GUI launcher for the trading visualization dashboard
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import sys
import os
from pathlib import Path

class DashboardLauncher(tk.Tk):
    """GUI launcher for trading dashboard with parameter selection"""
    
    def __init__(self):
        super().__init__()
        
        self.title("Trading Dashboard Launcher")
        self.geometry("500x450")
        
        self.setup_ui()
        self.load_defaults()
        
    def setup_ui(self):
        """Setup the launcher UI"""
        # Title
        title_label = ttk.Label(self, text="Trading Dashboard Configuration", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Symbol selection
        ttk.Label(main_frame, text="Symbol:").grid(row=0, column=0, sticky='w', pady=5)
        self.symbol_var = tk.StringVar(value="ES")
        symbol_combo = ttk.Combobox(main_frame, textvariable=self.symbol_var, 
                                    values=["ES", "NQ", "YM", "RTY", "CL", "GC", "SI"], 
                                    width=20)
        symbol_combo.grid(row=0, column=1, pady=5, padx=10)
        
        # Strategy selection
        ttk.Label(main_frame, text="Strategy:").grid(row=1, column=0, sticky='w', pady=5)
        self.strategy_var = tk.StringVar(value="simpleSMA")
        strategy_combo = ttk.Combobox(main_frame, textvariable=self.strategy_var,
                                      values=["simpleSMA", "momentum", "meanReversion"], 
                                      width=20)
        strategy_combo.grid(row=1, column=1, pady=5, padx=10)
        
        # Date range
        ttk.Label(main_frame, text="Start Date:").grid(row=2, column=0, sticky='w', pady=5)
        self.start_date_var = tk.StringVar(value="2024-01-01")
        ttk.Entry(main_frame, textvariable=self.start_date_var, width=22).grid(row=2, column=1, pady=5, padx=10)
        
        ttk.Label(main_frame, text="End Date:").grid(row=3, column=0, sticky='w', pady=5)
        self.end_date_var = tk.StringVar(value="2024-12-31")
        ttk.Entry(main_frame, textvariable=self.end_date_var, width=22).grid(row=3, column=1, pady=5, padx=10)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Display Options", padding="10")
        options_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky='ew')
        
        self.use_defaults = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use Default Settings", 
                       variable=self.use_defaults).pack(anchor='w')
        
        self.mplfinance = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Use Matplotlib Finance", 
                       variable=self.mplfinance).pack(anchor='w')
        
        self.plotly = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Use Plotly Dashboard", 
                       variable=self.plotly).pack(anchor='w')
        
        self.intraday = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Intraday Mode", 
                       variable=self.intraday).pack(anchor='w')
        
        # Screenshot options
        screenshot_frame = ttk.LabelFrame(main_frame, text="Screenshot Options", padding="10")
        screenshot_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky='ew')
        
        self.auto_screenshot = tk.BooleanVar(value=False)
        ttk.Checkbutton(screenshot_frame, text="Auto Screenshot", 
                       variable=self.auto_screenshot,
                       command=self.toggle_screenshot_options).pack(anchor='w')
        
        self.screenshot_delay = tk.IntVar(value=2)
        delay_frame = ttk.Frame(screenshot_frame)
        delay_frame.pack(anchor='w', pady=2)
        ttk.Label(delay_frame, text="Delay (seconds):").pack(side='left')
        self.delay_spin = ttk.Spinbox(delay_frame, from_=1, to=60, 
                                      textvariable=self.screenshot_delay, 
                                      width=10, state='disabled')
        self.delay_spin.pack(side='left', padx=5)
        
        # Launch buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Launch Dashboard", 
                  command=self.launch_dashboard).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Launch Headless", 
                  command=self.launch_headless).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Cancel", 
                  command=self.destroy).pack(side='left', padx=5)
    
    def toggle_screenshot_options(self):
        """Enable/disable screenshot options"""
        if self.auto_screenshot.get():
            self.delay_spin.config(state='normal')
        else:
            self.delay_spin.config(state='disabled')
    
    def load_defaults(self):
        """Load default configuration if available"""
        # Could load from a config file here
        pass
    
    def build_command(self, headless=False):
        """Build command line arguments"""
        cmd = [sys.executable, 'main.py']
        
        # Add required arguments
        cmd.append(self.symbol_var.get())
        cmd.append(self.strategy_var.get())
        
        # Add optional arguments
        if self.use_defaults.get():
            cmd.append('--useDefaults')
        
        if self.start_date_var.get():
            cmd.extend(['--start_date', self.start_date_var.get()])
        
        if self.end_date_var.get():
            cmd.extend(['--end_date', self.end_date_var.get()])
        
        if self.mplfinance.get():
            cmd.append('--mplfinance')
        
        if self.plotly.get():
            cmd.append('--plotly')
        
        if self.intraday.get():
            cmd.append('--intraday')
        
        if self.auto_screenshot.get():
            cmd.extend(['--screenshot-delay', str(self.screenshot_delay.get())])
        
        if headless:
            cmd.append('--headless')
            cmd.append('--no-show')
        
        return cmd
    
    def launch_dashboard(self):
        """Launch the dashboard with selected parameters"""
        try:
            cmd = self.build_command()
            
            # Change to tradingCode directory
            os.chdir('tradingCode')
            
            # Launch the dashboard
            subprocess.Popen(cmd)
            
            messagebox.showinfo("Success", "Dashboard launched successfully!")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch dashboard:\n{str(e)}")
    
    def launch_headless(self):
        """Launch dashboard in headless mode"""
        try:
            cmd = self.build_command(headless=True)
            
            # Change to tradingCode directory
            os.chdir('tradingCode')
            
            # Launch the dashboard
            subprocess.Popen(cmd)
            
            messagebox.showinfo("Success", "Headless dashboard launched successfully!")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch headless dashboard:\n{str(e)}")


def main():
    """Main entry point"""
    app = DashboardLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()