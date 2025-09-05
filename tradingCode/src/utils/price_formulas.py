"""
Price formula evaluation utilities for OHLC-based execution pricing.
Supports mathematical expressions using O, H, L, C variables.
"""

import numpy as np
import re
from typing import Dict, Union


class PriceFormulaEvaluator:
    """Evaluates mathematical formulas using OHLC price data."""
    
    def __init__(self):
        """Initialize the formula evaluator."""
        # Allowed operators and functions for safety
        self.allowed_operators = {
            '+', '-', '*', '/', '(', ')', ' ',
            'O', 'H', 'L', 'C',  # OHLC variables
            'max', 'min', 'sqrt', 'abs',  # Basic functions
            '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'  # Numbers
        }
        
    def validate_formula(self, formula: str) -> bool:
        """
        Validate that formula contains only safe characters and operators.
        
        Args:
            formula: Mathematical formula string
            
        Returns:
            True if formula is safe to evaluate
        """
        if not formula or not isinstance(formula, str):
            return False
            
        # Check for allowed characters only
        formula_chars = set(formula.upper())
        if not formula_chars.issubset(self.allowed_operators):
            return False
            
        # Check for balanced parentheses
        paren_count = 0
        for char in formula:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count < 0:
                    return False
        
        return paren_count == 0
    
    def parse_formula(self, formula: str, ohlc_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Parse and evaluate OHLC formula to get price array.
        
        Args:
            formula: Mathematical formula string (e.g., "(H + L + C) / 3")
            ohlc_data: Dictionary with 'open', 'high', 'low', 'close' arrays
            
        Returns:
            Array of calculated prices
            
        Raises:
            ValueError: If formula is invalid or evaluation fails
        """
        if not self.validate_formula(formula):
            raise ValueError(f"Invalid or unsafe formula: {formula}")
        
        # Create variable mapping
        variables = {
            'O': ohlc_data['open'],
            'H': ohlc_data['high'], 
            'L': ohlc_data['low'],
            'C': ohlc_data['close']
        }
        
        # Prepare formula for evaluation
        eval_formula = formula.upper()
        
        # Add numpy functions to allowed namespace
        safe_dict = {
            'O': variables['O'],
            'H': variables['H'],
            'L': variables['L'], 
            'C': variables['C'],
            'max': np.maximum,
            'min': np.minimum,
            'sqrt': np.sqrt,
            'abs': np.abs,
            '__builtins__': {}  # Restrict built-in functions for security
        }
        
        try:
            # Evaluate the formula
            result = eval(eval_formula, safe_dict)
            
            # Ensure result is a numpy array
            if not isinstance(result, np.ndarray):
                result = np.full(len(ohlc_data['close']), result)
                
            return result
            
        except Exception as e:
            raise ValueError(f"Error evaluating formula '{formula}': {e}")
    
    def get_execution_prices(self, formula: str, ohlc_data: Dict[str, np.ndarray], 
                           signal_type: str = "buy") -> np.ndarray:
        """
        Get execution prices based on formula and signal type.
        
        Args:
            formula: Price formula string
            ohlc_data: OHLC data dictionary
            signal_type: "buy" or "sell" for context
            
        Returns:
            Array of execution prices
        """
        try:
            prices = self.parse_formula(formula, ohlc_data)
            
            # Validate prices are reasonable (within OHLC bounds)
            high_prices = ohlc_data['high']
            low_prices = ohlc_data['low']
            
            # Clip prices to valid OHLC range
            prices = np.clip(prices, low_prices, high_prices)
            
            return prices
            
        except Exception as e:
            # Fallback to close prices if formula fails
            print(f"Warning: Formula evaluation failed for {signal_type} signals: {e}")
            print(f"Falling back to close prices")
            return ohlc_data['close']


def create_common_formulas() -> Dict[str, str]:
    """
    Create dictionary of common price formulas.
    
    Returns:
        Dictionary of formula names and expressions
    """
    return {
        'close': 'C',
        'open': 'O', 
        'high': 'H',
        'low': 'L',
        'typical': '(H + L + C) / 3',          # Typical price
        'median': '(H + L) / 2',               # Median price
        'weighted_close': '(H + L + 2*C) / 4', # Weighted close
        'ohlc4': '(O + H + L + C) / 4',        # OHLC4 average
        'hlc3': '(H + L + C) / 3',             # HLC3 (same as typical)
        'hl2': '(H + L) / 2',                  # HL2 (same as median)
        'conservative_buy': 'H',                # Conservative buy (high price)
        'aggressive_buy': 'L',                  # Aggressive buy (low price)
        'conservative_sell': 'L',               # Conservative sell (low price)
        'aggressive_sell': 'H'                  # Aggressive sell (high price)
    }


# Example usage and validation
if __name__ == "__main__":
    # Test the formula evaluator
    evaluator = PriceFormulaEvaluator()
    
    # Create sample OHLC data
    n_bars = 10
    np.random.seed(42)
    
    sample_data = {
        'open': np.random.uniform(100, 110, n_bars),
        'high': np.random.uniform(110, 120, n_bars),
        'low': np.random.uniform(90, 100, n_bars),
        'close': np.random.uniform(100, 110, n_bars)
    }
    
    # Test common formulas
    formulas = create_common_formulas()
    
    print("Testing Price Formula Evaluator")
    print("=" * 40)
    
    for name, formula in formulas.items():
        try:
            prices = evaluator.get_execution_prices(formula, sample_data, "test")
            print(f"{name:15} | {formula:20} | {prices[0]:.2f}")
        except Exception as e:
            print(f"{name:15} | {formula:20} | ERROR: {e}")
    
    print("\nFormula evaluation test completed!")