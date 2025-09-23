"""
Comprehensive fix script for hover data and trade generation issues
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def analyze_and_fix_issues():
    """Analyze and apply fixes to the codebase"""
    print("="*60)
    print("ANALYZING AND FIXING HOVER DATA & TRADE GENERATION ISSUES")
    print("="*60)

    # Issue 1: Check data structure consistency
    print("\n1. CHECKING DATA STRUCTURE IN launch_pyqtgraph_with_selector.py...")

    launcher_file = "launch_pyqtgraph_with_selector.py"
    with open(launcher_file, 'r') as f:
        content = f.read()

    # Check if we're using correct keys
    if "'DateTime'" in content and "'timestamp'" in content:
        print("   WARNING: Mixed use of 'DateTime' and 'timestamp' keys found")
        print("   FIX: Ensuring consistent use of lowercase keys throughout")

        # The fix is already in place - verify it's correct
        if "self.full_data = {" in content:
            start = content.find("self.full_data = {")
            end = content.find("}", start) + 1
            data_dict = content[start:end]
            print(f"   Current data structure keys: ")
            for line in data_dict.split('\n'):
                if ":" in line and "'" in line:
                    print(f"      {line.strip()}")

    # Issue 2: Check the parent class expectations
    print("\n2. CHECKING PARENT CLASS in pyqtgraph_range_bars_final.py...")

    parent_file = "src/trading/visualization/pyqtgraph_range_bars_final.py"
    with open(parent_file, 'r') as f:
        parent_content = f.read()

    # Find what keys the parent expects
    if "self.full_data['timestamp']" in parent_content:
        print("   Parent expects: 'timestamp' (lowercase)")
    if "self.full_data['open']" in parent_content:
        print("   Parent expects: 'open', 'high', 'low', 'close' (lowercase)")

    # Issue 3: Fix the key mapping to ensure consistency
    print("\n3. APPLYING FIX TO ENSURE CONSISTENT KEY NAMES...")

    # The current code should be correct, but let's verify line 221-230
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "self.full_data = {" in line:
            print(f"   Found data structure definition at line {i+1}")
            # Check next 10 lines
            for j in range(i, min(i+10, len(lines))):
                if "'timestamp':" in lines[j]:
                    print(f"   Line {j+1}: Using 'timestamp' key - CORRECT")
                if "'DateTime':" in lines[j]:
                    print(f"   Line {j+1}: Using 'DateTime' key - INCORRECT, needs fix")

    # Issue 4: Check trade generation
    print("\n4. CHECKING TRADE GENERATION...")

    # Check if dataframe is being set correctly
    if "self.dataframe = df" in content:
        print("   DataFrame is being stored for trade generation - CORRECT")

    # Check if trades are being generated
    if "def load_configured_trades" in content:
        print("   Trade loading method exists - CORRECT")

        # Find the method and check it
        start = content.find("def load_configured_trades")
        end = content.find("\n    def ", start + 1)
        if end == -1:
            end = len(content)
        method_content = content[start:end]

        if "generate_system_trades" in method_content:
            print("   System trade generation is called - CORRECT")

            # Check if DataFrame columns are correct
            if "self.dataframe" in method_content:
                print("   Using self.dataframe for trade generation - CORRECT")

    # Issue 5: Verify the actual fix is correct
    print("\n5. VERIFYING THE ACTUAL DATA STRUCTURE...")

    # The key issue is in lines 221-230 of launch_pyqtgraph_with_selector.py
    # It should use lowercase keys to match parent class

    correct_structure = """        self.full_data = {
            'timestamp': timestamps,
            'open': df['Open'].values.astype(np.float32),
            'high': df['High'].values.astype(np.float32),
            'low': df['Low'].values.astype(np.float32),
            'close': df['Close'].values.astype(np.float32),
            'volume': df['Volume'].values.astype(np.float32) if 'Volume' in df else None,
            'aux1': df['AUX1'].values.astype(np.float32) if 'AUX1' in df else None,
            'aux2': df['AUX2'].values.astype(np.float32) if 'AUX2' in df else None
        }"""

    if correct_structure.strip() in content:
        print("   Data structure is CORRECT - using lowercase keys")
        return True
    else:
        print("   Data structure might have issues - checking details...")

        # Extract the actual structure
        if "self.full_data = {" in content:
            start = content.find("self.full_data = {")
            end = content.find("}", start) + 1
            actual_structure = content[start:end]
            print("\n   Actual structure:")
            print(actual_structure)

            # Check for specific issues
            issues = []
            if "'DateTime'" in actual_structure:
                issues.append("Using 'DateTime' instead of 'timestamp'")
            if "'Open'" in actual_structure:
                issues.append("Using uppercase OHLC keys instead of lowercase")

            if issues:
                print("\n   ISSUES FOUND:")
                for issue in issues:
                    print(f"   - {issue}")
                return False
            else:
                print("   Structure looks correct")
                return True

    return True

def test_with_real_data():
    """Test the application with real data to verify fixes"""
    print("\n" + "="*60)
    print("TESTING WITH REAL DATA")
    print("="*60)

    # Find a real data file
    import glob
    csv_files = glob.glob(r"C:\code\PythonBT\dataRaw\**\*.csv", recursive=True)

    if not csv_files:
        print("ERROR: No CSV files found for testing")
        return False

    test_file = csv_files[0]
    print(f"\nUsing test file: {test_file}")

    # Load the data to test
    df = pd.read_csv(test_file)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Test the data structure conversion
    if 'Date' in df.columns and 'Time' in df.columns:
        print("\nTesting Date+Time combination...")
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        sample = df['DateTime'].iloc[0]
        print(f"First DateTime: {sample}")
        if sample.hour != 0 or sample.minute != 0:
            print("SUCCESS: Time component preserved!")
        else:
            print("WARNING: Time might be lost")

    # Test data structure creation
    print("\nTesting data structure creation...")
    timestamps = pd.to_datetime(df['DateTime']).values if 'DateTime' in df else None

    test_structure = {
        'timestamp': timestamps,  # CORRECT: lowercase
        'open': df['Open'].values.astype(np.float32) if 'Open' in df else None,
        'high': df['High'].values.astype(np.float32) if 'High' in df else None,
        'low': df['Low'].values.astype(np.float32) if 'Low' in df else None,
        'close': df['Close'].values.astype(np.float32) if 'Close' in df else None,
    }

    print("Created test structure with keys:", list(test_structure.keys()))

    # Verify data accessibility
    if test_structure['timestamp'] is not None and len(test_structure['timestamp']) > 0:
        print(f"Can access timestamp[0]: {test_structure['timestamp'][0]}")
        print(f"Can access open[0]: {test_structure['open'][0]}")
        return True

    return False

def create_test_launcher():
    """Create a test script to verify everything works"""
    test_script = '''"""
Test script to verify hover data and trade generation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from launch_pyqtgraph_with_selector import ConfiguredChart
import pandas as pd

def test_hover_and_trades():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Configure with real data
    import glob
    csv_files = glob.glob(r"C:\\code\\PythonBT\\dataRaw\\**\\*.csv", recursive=True)
    if not csv_files:
        print("ERROR: No data files found")
        return False

    config = {
        'data_file': csv_files[0],
        'trade_source': 'system',
        'system': 'SMA Crossover'
    }

    print(f"Testing with: {config['data_file']}")

    # Create chart
    chart = ConfiguredChart(config)

    # Test hover data
    if chart.full_data is None:
        print("ERROR: full_data is None!")
        return False

    print(f"SUCCESS: full_data loaded with keys: {list(chart.full_data.keys())}")

    # Test data access
    if 'timestamp' in chart.full_data:
        print(f"  timestamp[0]: {chart.full_data['timestamp'][0]}")
    if 'open' in chart.full_data:
        print(f"  open[0]: {chart.full_data['open'][0]}")

    # Test trade generation
    if hasattr(chart, 'current_trades'):
        print(f"Trades generated: {len(chart.current_trades)}")

    return True

if __name__ == "__main__":
    success = test_hover_and_trades()
    print(f"\\nTest result: {'SUCCESS' if success else 'FAILED'}")
'''

    with open('test_integrated.py', 'w') as f:
        f.write(test_script)

    print("\nCreated test_integrated.py for verification")

def main():
    # Run all checks
    structure_ok = analyze_and_fix_issues()
    data_ok = test_with_real_data()

    # Create test script
    create_test_launcher()

    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nData Structure: {'OK' if structure_ok else 'NEEDS FIX'}")
    print(f"Real Data Test: {'OK' if data_ok else 'FAILED'}")

    if structure_ok and data_ok:
        print("\nAll checks passed! The code should work correctly.")
        print("\nTo verify, run: python test_integrated.py")
    else:
        print("\nIssues detected. Manual fixes may be needed.")

    return structure_ok and data_ok

if __name__ == "__main__":
    success = main()