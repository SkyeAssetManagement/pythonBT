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
        
    def create_timeline(self, data_start, data_end, validation_start, validation_end):
        """Create a visual timeline showing data split"""
        
        # Clear previous figure if exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Create figure - compact for efficient space use
        self.figure = plt.Figure(figsize=(12, 3), facecolor='white', tight_layout=True)
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
        validation_days = (validation_end - validation_start).days
        
        # Create the timeline bars - compact
        bar_height = 0.4
        bar_y = 0.5
        
        # Full data range (light gray background)
        full_rect = Rectangle((mdates.date2num(data_start), bar_y - bar_height/2),
                              mdates.date2num(data_end) - mdates.date2num(data_start),
                              bar_height,
                              facecolor='#E0E0E0',
                              edgecolor='black',
                              linewidth=2)
        ax.add_patch(full_rect)
        
        # Training data (green)
        train_rect = Rectangle((mdates.date2num(data_start), bar_y - bar_height/2),
                               mdates.date2num(validation_end) - mdates.date2num(data_start),
                               bar_height,
                               facecolor='#4CAF50',
                               edgecolor='black',
                               linewidth=2,
                               alpha=0.8)
        ax.add_patch(train_rect)
        
        # Out-of-sample data (red)
        if validation_end < data_end:
            oos_rect = Rectangle((mdates.date2num(validation_end), bar_y - bar_height/2),
                                 mdates.date2num(data_end) - mdates.date2num(validation_end),
                                 bar_height,
                                 facecolor='#FF5252',
                                 edgecolor='black',
                                 linewidth=2,
                                 alpha=0.8)
            ax.add_patch(oos_rect)
        
        # Set axis properties - adjusted limits for better spacing
        ax.set_xlim(mdates.date2num(data_start) - 100, mdates.date2num(data_end) + 100)
        ax.set_ylim(0, 1.5)  # Increased height for labels
        
        # Format x-axis as dates - simplified for less crowding
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Every 2 years
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        
        # Rotate date labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Remove y-axis
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add title with more padding
        ax.set_title('Data Timeline & Validation Period', fontsize=12, fontweight='bold', pad=15)
        
        # Add vertical lines for key dates
        ax.axvline(mdates.date2num(validation_start), color='blue', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvline(mdates.date2num(validation_end), color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Add text annotations with dates - positioned above the bar
        ax.text(mdates.date2num(validation_start), 0.85, f'Validation Start\n{validation_start.strftime("%Y-%m-%d")}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='blue')
        ax.text(mdates.date2num(validation_end), 0.85, f'Validation End\n{validation_end.strftime("%Y-%m-%d")}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')
        
        # Add data start and end labels
        ax.text(mdates.date2num(data_start), 0.25, f'{data_start.strftime("%Y-%m-%d")}', 
                ha='left', va='center', fontsize=8, color='black')
        ax.text(mdates.date2num(data_end), 0.25, f'{data_end.strftime("%Y-%m-%d")}', 
                ha='right', va='center', fontsize=8, color='black')
        
        # Add legend - positioned better
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4CAF50', alpha=0.8, label=f'Training/Validation ({train_days:,} days)'),
            Patch(facecolor='#FF5252', alpha=0.8, label=f'Out-of-Sample ({test_days:,} days)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=9)
        
        # Add percentage labels - positioned below the bar
        train_pct = (train_days / total_days) * 100
        test_pct = (test_days / total_days) * 100
        
        info_text = f"Total: {total_days:,} days | Training: {train_pct:.1f}% | Out-of-Sample: {test_pct:.1f}%"
        ax.text(0.5, 0.15, info_text, transform=ax.transAxes, 
                ha='center', fontsize=10, fontweight='bold')
        
        # Adjust layout
        self.figure.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        return train_days, test_days, validation_days
    
    def update_timeline(self, data_start, data_end, validation_start, validation_end):
        """Update existing timeline with new dates"""
        return self.create_timeline(data_start, data_end, validation_start, validation_end)