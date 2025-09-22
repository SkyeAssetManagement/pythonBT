#!/usr/bin/env python3
"""
Simple test to verify timestamps are displayed correctly
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')

import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg

def test_chart_with_timestamps():
    """Test chart with proper timestamps"""

    # Load CSV data
    csv_file = r"dataRaw\range-ATR30x0.05\ES\diffAdjusted\ES-DIFF-range-ATR30x0.05-dailyATR.csv"
    print(f"Loading: {csv_file}")

    df = pd.read_csv(csv_file, nrows=100)  # Load first 100 rows

    # Combine Date and Time
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    print(f"\nFirst 5 DateTime values:")
    for i in range(5):
        dt = df['DateTime'].iloc[i]
        print(f"  [{i}] {dt} -> hour={dt.hour}, minute={dt.minute}, second={dt.second}")

    # Create simple PyQtGraph app
    app = QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(show=True, title="Timestamp Test")
    win.resize(1000, 600)

    plot = win.addPlot(title="ES Data with Timestamps")

    # Plot candlesticks
    x = np.arange(len(df))

    # Create simple candlestick plot
    plot.plot(x, df['Close'].values, pen='w', name='Close')

    # Format x-axis with timestamps
    timestamps = df['DateTime'].values

    # Create axis labels
    num_labels = 10
    step = len(df) // num_labels
    x_ticks = []

    for i in range(0, len(df), step):
        ts = pd.Timestamp(timestamps[i])
        time_str = ts.strftime('%H:%M:%S')
        date_str = ts.strftime('%Y-%m-%d')
        label = f"{time_str}\n{date_str}"
        x_ticks.append((i, label))
        print(f"Tick at x={i}: {label.replace(chr(10), ' ')}")

    # Set the ticks
    x_axis = plot.getAxis('bottom')
    x_axis.setTicks([x_ticks])

    print("\nChart displayed. Check if timestamps show correctly on x-axis.")

    # Show window
    win.show()

    # Don't run event loop, just check setup
    QtWidgets.QApplication.processEvents()

    return df

if __name__ == "__main__":
    df = test_chart_with_timestamps()
    print("\nTest complete. If running interactively, app.exec_() to see chart.")