#!/usr/bin/env python3
"""
Test script to verify the data selector workflow
"""

import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading/visualization')

from PyQt5 import QtWidgets
from pyqtgraph_data_selector import DataSelectorDialog

def main():
    """Test the selector workflow"""
    print("="*60)
    print("Testing Data Selector Workflow")
    print("="*60)
    
    # Create Qt Application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create and show selector
    selector = DataSelectorDialog()
    
    # Check that data files are found
    print(f"\nData files found: {selector.data_list.count()}")
    if selector.data_list.count() > 0:
        print("Sample files:")
        for i in range(min(5, selector.data_list.count())):
            item = selector.data_list.item(i)
            print(f"  - {item.text()}")
    
    # Check trade files
    print(f"\nTrade files found: {selector.trade_list.count()}")
    
    # Check indicators
    print(f"\nIndicators available: {len(selector.indicator_checkboxes)}")
    checked = [k for k, cb in selector.indicator_checkboxes.items() if cb.isChecked()]
    print(f"Pre-selected: {checked}")
    
    # Show dialog
    print("\nShowing dialog...")
    result = selector.exec_()
    
    if result == QtWidgets.QDialog.Accepted:
        config = selector.get_configuration()
        print("\n" + "="*60)
        print("Configuration Selected:")
        print("="*60)
        print(f"Data file: {config['data_file']}")
        print(f"Trade source: {config['trade_source']}")
        print(f"Trade file: {config['trade_file']}")
        print(f"System: {config['system']}")
        print(f"Indicators: {config['indicators']}")
        print("\nWorkflow test completed successfully!")
    else:
        print("\nDialog cancelled by user")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())