import tkinter as tk
from tkinter import ttk
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

class TimelineVisualization:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.canvas = None
        self.figure = None
        
    def update_timeline(self, data_start, data_end, validation_start, validation_end):
        """Update timeline - wrapper for compatibility"""
        self.create_timeline(data_start, data_end, validation_start, validation_end)
        
    def create_timeline(self, data_start, data_end, validation_start, validation_end):
        """Create a compact visual timeline showing data split"""
        
        # Clear previous figure if exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Create figure with very compact size
        self.figure = plt.Figure(figsize=(10, 1.5), facecolor='white', tight_layout={'pad': 0.5})
        ax = self.figure.add_subplot(111)
        
        # Convert dates to datetime if they're strings
        if isinstance(data_start, str):
            data_start = pd.to_datetime(data_start)
        if isinstance(data_end, str):
            data_end = pd.to_datetime(data_end)
        if isinstance(validation_start, str):
            validation_start = pd.to_datetime(validation_start)
        if isinstance(validation_end, str):
            validation_end = pd.to_datetime(validation_end)
        
        # Calculate durations
        total_days = (data_end - data_start).days
        train_days = (validation_end - data_start).days
        test_days = (data_end - validation_end).days
        
        # Create simple bar visualization
        bar_height = 0.6
        bar_y = 0.5
        
        # Full data range (light gray background)
        ax.barh(bar_y, mdates.date2num(data_end) - mdates.date2num(data_start),
                left=mdates.date2num(data_start), height=bar_height,
                color='#E0E0E0', edgecolor='black', linewidth=1.5,
                label=f'Full Data ({total_days} days)')
        
        # Training data (green)
        ax.barh(bar_y, mdates.date2num(validation_end) - mdates.date2num(data_start),
                left=mdates.date2num(data_start), height=bar_height,
                color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=1,
                label=f'Training ({train_days} days)')
        
        # Out-of-sample data (red)
        if test_days > 0:
            ax.barh(bar_y, mdates.date2num(data_end) - mdates.date2num(validation_end),
                    left=mdates.date2num(validation_end), height=bar_height,
                    color='#FF5252', alpha=0.7, edgecolor='black', linewidth=1,
                    label=f'Out-of-Sample ({test_days} days)')
        
        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
        
        # Minimal styling
        ax.set_ylim(0, 1)
        ax.set_xlim(mdates.date2num(data_start), mdates.date2num(data_end))
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Compact legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                 ncol=3, frameon=False, fontsize=8)
        
        # Rotate x-axis labels for space
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=8)
        
        # Create and embed canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)