"""
Replicate the EXACT black blob issue from user's screenshot
Focus: Same date range (2020-01-03 to 2020-01-14), same zoom level
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add src path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def replicate_exact_issue():
    """Replicate the exact black blob issue from the screenshot"""
    print("="*70)
    print("REPLICATING EXACT BLACK BLOB ISSUE")
    print("="*70)
    print("Target: Same date range as user screenshot (2020-01-03 to 2020-01-14)")
    print("Expected: Black blobs like in the screenshot")
    print("="*70)
    
    try:
        from PyQt5 import QtWidgets, QtCore
        from dashboard.chart_widget import TradingChart
        from dashboard.data_structures import ChartDataBuffer
        
        # Create application
        app = QtWidgets.QApplication([])
        
        # Load ES data
        es_file = Path(__file__).parent.parent / "dataRaw" / "1m" / "ES" / "Current" / "ES-NONE-1m-EST-NoPad.csv"
        
        if not es_file.exists():
            print(f"ES data file not found: {es_file}")
            return False
        
        print(f"Loading ES data from: {es_file}")
        df = pd.read_csv(es_file)
        
        # Convert Date column to datetime for filtering
        df['DateTime'] = pd.to_datetime(df['Date'])
        
        # Filter to exact same date range as user's screenshot: 2020-01-03 to 2020-01-14
        start_date = pd.Timestamp('2020-01-03')
        end_date = pd.Timestamp('2020-01-14 23:59:59')
        
        df_filtered = df[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)]
        
        if len(df_filtered) == 0:
            print("No data found for the specified date range")
            print(f"Available date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
            return False
        
        print(f"Filtered data: {len(df_filtered)} bars from {df_filtered['DateTime'].min()} to {df_filtered['DateTime'].max()}")
        
        # Create data buffer with exact same data as user sees
        timestamps = (df_filtered['DateTime'].astype('int64') // 10**6).values  # Convert to nanoseconds
        
        data_buffer = ChartDataBuffer(
            timestamps=timestamps,
            open=df_filtered['Open'].values.astype(np.float64),
            high=df_filtered['High'].values.astype(np.float64),
            low=df_filtered['Low'].values.astype(np.float64),
            close=df_filtered['Close'].values.astype(np.float64),
            volume=df_filtered['Volume'].values.astype(np.float64)
        )
        
        print(f"Created data buffer with {len(data_buffer)} bars")
        print(f"Price range: ${data_buffer.low.min():.2f} - ${data_buffer.high.max():.2f}")
        
        # Create chart widget (this should replicate the exact issue)
        chart = TradingChart()
        chart.setWindowTitle("REPLICATING BLACK BLOB ISSUE - Same Date Range")
        chart.resize(1400, 900)
        
        # Set data
        chart.set_data(data_buffer)
        
        # Show chart
        chart.show()
        
        # Wait for rendering
        app.processEvents()
        time.sleep(2)
        app.processEvents()
        
        # Take screenshot to compare
        print("Taking screenshot to compare with user's...")
        pixmap = chart.grab()
        
        screenshot_file = Path(__file__).parent / "REPLICATED_BLACK_BLOB_ISSUE.png"
        success = pixmap.save(str(screenshot_file))
        
        if success:
            print(f"SUCCESS: Screenshot saved to {screenshot_file}")
            print("\\nCOMPARE THIS SCREENSHOT TO USER'S:")
            print("1. Should show black blobs (confirming replication)")
            print("2. Same date range: 2020-01-03 to 2020-01-14")
            print("3. Same blob appearance")
            print("\\nIf this matches user's screenshot, we've replicated the issue!")
        
        # Keep dashboard open for manual inspection
        print("\\nDashboard is open - you can manually inspect:")
        print("- Do you see black blobs?")
        print("- Same as user's screenshot?")
        print("- Press Ctrl+C to continue with fix")
        
        try:
            app.exec_()
        except KeyboardInterrupt:
            print("\\nContinuing with analysis...")
        
        return True
        
    except Exception as e:
        print(f"Error replicating issue: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_rendering_issue():
    """Analyze what's causing the black blob rendering"""
    print("\\n" + "="*70)
    print("ANALYZING BLACK BLOB RENDERING ISSUE")
    print("="*70)
    
    print("From user's screenshot, the issue appears to be:")
    print("1. Candlesticks render as solid BLACK rectangles")
    print("2. No visible outlines or borders")
    print("3. No distinction between up/down candles")
    print("4. No visible wicks")
    print("5. Just solid black blobs")
    
    print("\\nPossible causes:")
    print("1. Brush color is black instead of white/red")
    print("2. Pen color is wrong")
    print("3. Rectangle drawing is filling entire area")
    print("4. Antialiasing or rendering mode issue")
    print("5. Color format issue (RGBA vs RGB)")
    
    print("\\nNext steps:")
    print("1. Examine actual color values being used")
    print("2. Check if brush/pen are being applied correctly")
    print("3. Test different color formats")
    print("4. Add debug output for color values")

def main():
    """Main replication function"""
    print("Attempting to replicate the exact black blob issue...")
    
    success = replicate_exact_issue()
    
    if success:
        analyze_rendering_issue()
        print("\\nReplication attempt complete - check the screenshot!")
    else:
        print("\\nFailed to replicate - check error messages above")

if __name__ == "__main__":
    main()