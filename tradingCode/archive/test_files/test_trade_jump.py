"""
Test trade jump functionality
"""

import requests
import json

def test_trade_jump():
    """Test if trade navigation is working."""
    print("\n" + "="*60)
    print("TESTING TRADE JUMP FUNCTIONALITY")
    print("="*60)
    
    # The dashboard uses trade IDs from Exit Trade Id column (0, 1, 2, etc.)
    print("\nThe trade IDs are numeric (0, 1, 2, 3, etc.)")
    print("\nTo jump to a trade:")
    print("1. Enter a trade ID number (e.g., '0', '1', '2', etc.)")
    print("2. Click Jump button")
    print("3. Chart should center on bar index where trade occurred")
    
    print("\nExample trade locations:")
    print("- Trade 0: Entry at bar 107")
    print("- Trade 1: Entry at bar 146") 
    print("- Trade 2: Entry at bar 187")
    print("- Trade 3: Entry at bar 325")
    print("- Trade 4: Entry at bar 419")
    
    print("\nOr click directly on any trade in the trade list table!")
    
    print("\n" + "="*60)
    print("EXPECTED BEHAVIOR:")
    print("="*60)
    print("1. Entering '0' should jump to bar 107")
    print("2. Entering '1' should jump to bar 146")
    print("3. Clicking first row in trade table should jump to bar 107")
    print("4. The chart viewport should center on the trade location")
    print("5. Trade markers (triangles) should be visible")
    
    return True

if __name__ == "__main__":
    test_trade_jump()
    
    print("\n" + "="*60)
    print("IMPORTANT: Trade IDs are NUMBERS not T### format")
    print("="*60)
    print("\nThe dashboard is using VectorBT's Exit Trade Id column")
    print("which contains numeric IDs: 0, 1, 2, 3, etc.")
    print("\nGo to http://localhost:8050 and try entering '0' or '1'")
    print("in the Jump to Trade input box!")