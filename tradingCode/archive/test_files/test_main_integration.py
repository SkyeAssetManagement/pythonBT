#!/usr/bin/env python3
"""
Test main.py integration with vectorized strategy
"""

import sys
import importlib
from pathlib import Path

# Add strategies to path
strategies_path = Path(__file__).parent / "strategies"
if str(strategies_path) not in sys.path:
    sys.path.insert(0, str(strategies_path))

def test_strategy_import():
    """Test if strategy can be imported by main.py"""
    
    print("=== TESTING STRATEGY IMPORT ===")
    
    try:
        # Test the import mechanism used by main.py
        module = importlib.import_module('strategies.time_window_strategy_vectorized')
        print(f"PASS: Successfully imported module: {module}")
        
        # Check available classes
        import inspect
        from strategies.base_strategy import BaseStrategy
        
        strategy_classes = []
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseStrategy) and 
                obj != BaseStrategy):
                strategy_classes.append((name, obj))
        
        print(f"Found {len(strategy_classes)} strategy classes:")
        for name, cls in strategy_classes:
            print(f"  - {name}: {cls}")
        
        # Test instantiation
        if strategy_classes:
            strategy_class = strategy_classes[0][1]  # Take first class
            strategy = strategy_class()
            print(f"PASS: Successfully created strategy instance: {strategy.name}")
            
            # Test parameter combinations
            combos = strategy.get_parameter_combinations()
            print(f"PASS: Strategy has {len(combos)} parameter combinations")
            
            return True
        else:
            print("FAIL: No strategy classes found")
            return False
            
    except Exception as e:
        print(f"FAIL: Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_selection():
    """Test strategy selection logic from main.py"""
    
    print("\n=== TESTING STRATEGY SELECTION LOGIC ===")
    
    try:
        module = importlib.import_module('strategies.time_window_strategy_vectorized')
        
        import inspect
        from strategies.base_strategy import BaseStrategy
        
        strategy_classes = []
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseStrategy) and 
                obj != BaseStrategy):
                strategy_classes.append((name, obj))
        
        print(f"Available classes: {[name for name, cls in strategy_classes]}")
        
        # Use main.py's selection logic
        strategy_class = None
        for name, cls in strategy_classes:
            if 'Strategy' in name and 'Sweep' not in name:
                print(f"Selected strategy: {name}")
                strategy_class = cls
                break
        
        # Fall back to first class if no single strategy found
        if strategy_class is None:
            strategy_class = strategy_classes[0][1]
            print(f"Using fallback strategy: {strategy_classes[0][0]}")
        
        strategy = strategy_class()
        print(f"PASS: Final strategy: {strategy.name}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Strategy selection failed: {e}")
        return False

if __name__ == "__main__":
    print("MAIN.PY INTEGRATION TEST")
    print("=" * 50)
    
    success = True
    
    # Test 1: Import
    if not test_strategy_import():
        success = False
    
    # Test 2: Selection logic
    if not test_strategy_selection():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Strategy ready for main.py integration")
        print("  Usage: python main.py SYMBOL time_window_strategy_vectorized")
    else:
        print("FAILURE: Integration issues need to be resolved")