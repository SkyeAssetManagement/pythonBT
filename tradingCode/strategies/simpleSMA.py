import numpy as np
import vectorbtpro as vbt
from typing import Dict, Tuple, List
from .base_strategy import BaseStrategy


class SimpleSMAStrategy(BaseStrategy):
    """Simple SMA crossover strategy with 20/100 moving averages."""
    
    def __init__(self):
        super().__init__("SimpleSMA_20_100")
        
    def generate_signals(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate signals using 100/20 SMA crossover."""
        return self._generate_signals_for_params(data, {'fast_period': 20, 'slow_period': 100})
    
    def get_parameter_combinations(self, use_defaults_only: bool = False) -> List[Dict]:
        """
        Get parameter combinations for testing.
        For now, just the base 100/20 combination, but can be expanded.
        
        Args:
            use_defaults_only: If True, return only default parameters (single combination)
        """
        # Start with single combination, can expand to multiple variations
        return [
            {'fast_period': 20, 'slow_period': 100}
        ]
    
    def _generate_signals_for_params(self, data: Dict[str, np.ndarray], 
                                   params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate SMA crossover signals for given parameters.
        
        Args:
            data: Dictionary with OHLCV arrays
            params: Dictionary with 'fast_period' and 'slow_period'
            
        Returns:
            Tuple of (entries, exits) as boolean arrays
        """
        close_prices = data['close']
        fast_period = params['fast_period']
        slow_period = params['slow_period']
        
        # Calculate moving averages using vectorBT
        fast_ma = vbt.MA.run(close_prices, fast_period).ma.values
        slow_ma = vbt.MA.run(close_prices, slow_period).ma.values
        
        # Generate crossover signals
        # Entry: Fast MA crosses above Slow MA
        entries = np.zeros(len(close_prices), dtype=bool)
        exits = np.zeros(len(close_prices), dtype=bool)
        
        # Use vectorized operations for crossover detection
        fast_above_slow = fast_ma > slow_ma
        fast_above_slow_prev = np.roll(fast_above_slow, 1)
        fast_above_slow_prev[0] = False
        
        # Entry when fast crosses above slow
        entries = fast_above_slow & ~fast_above_slow_prev
        
        # Exit when fast crosses below slow
        exits = ~fast_above_slow & fast_above_slow_prev
        
        # Ensure we don't have signals before we have enough data
        min_period = max(fast_period, slow_period)
        entries[:min_period] = False
        exits[:min_period] = False
        
        return entries, exits


# Extended version with multiple parameter combinations
class SimpleSMAParameterSweep(BaseStrategy):
    """Extended SMA strategy with multiple parameter combinations."""
    
    def __init__(self):
        super().__init__("SimpleSMA_ParameterSweep")
        
    def generate_signals(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate signals using default 100/20 combination."""
        return self._generate_signals_for_params(data, {'fast_period': 20, 'slow_period': 100})
    
    def get_parameter_combinations(self) -> List[Dict]:
        """
        Get all parameter combinations for testing.
        This creates hundreds of combinations as requested.
        """
        combinations = []
        
        # Fast MA periods: 5 to 50 in steps of 5
        fast_periods = list(range(5, 55, 5))
        
        # Slow MA periods: 50 to 200 in steps of 10
        slow_periods = list(range(50, 210, 10))
        
        # Generate all combinations where fast < slow
        for fast in fast_periods:
            for slow in slow_periods:
                if fast < slow:
                    combinations.append({
                        'fast_period': fast,
                        'slow_period': slow
                    })
        
        return combinations
    
    def _generate_signals_for_params(self, data: Dict[str, np.ndarray], 
                                   params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Generate SMA crossover signals for given parameters."""
        close_prices = data['close']
        fast_period = params['fast_period']
        slow_period = params['slow_period']
        
        # Calculate moving averages using vectorBT
        fast_ma = vbt.MA.run(close_prices, fast_period).ma.values
        slow_ma = vbt.MA.run(close_prices, slow_period).ma.values
        
        # Generate crossover signals
        entries = np.zeros(len(close_prices), dtype=bool)
        exits = np.zeros(len(close_prices), dtype=bool)
        
        # Use vectorized operations for crossover detection
        fast_above_slow = fast_ma > slow_ma
        fast_above_slow_prev = np.roll(fast_above_slow, 1)
        fast_above_slow_prev[0] = False
        
        # Entry when fast crosses above slow
        entries = fast_above_slow & ~fast_above_slow_prev
        
        # Exit when fast crosses below slow
        exits = ~fast_above_slow & fast_above_slow_prev
        
        # Ensure we don't have signals before we have enough data
        min_period = max(fast_period, slow_period)
        entries[:min_period] = False
        exits[:min_period] = False
        
        return entries, exits