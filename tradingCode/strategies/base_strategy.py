import numpy as np
import vectorbtpro as vbt
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def generate_signals(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate entry and exit signals.
        
        Args:
            data: Dictionary with OHLCV arrays
            
        Returns:
            Tuple of (entries, exits) as boolean arrays
        """
        pass
    
    @abstractmethod
    def get_parameter_combinations(self) -> List[Dict]:
        """
        Get all parameter combinations to test.
        
        Returns:
            List of parameter dictionaries
        """
        pass
    
    def run_vectorized_backtest(self, data: Dict[str, np.ndarray], 
                              config: Dict) -> vbt.Portfolio:
        """
        Run vectorized backtest for all parameter combinations.
        
        Args:
            data: Dictionary with OHLCV arrays
            config: Backtest configuration
            
        Returns:
            VectorBT Portfolio object
        """
        close_prices = data['close']
        param_combinations = self.get_parameter_combinations()
        n_bars = len(close_prices)
        n_combos = len(param_combinations)
        
        # Get execution price array based on config
        execution_price_type = config.get('execution_price', 'close')
        
        # Handle formula-based execution pricing
        if execution_price_type == 'formula':
            # Use formula evaluator similar to VectorBT engine
            from src.utils.price_formulas import PriceFormulaEvaluator
            evaluator = PriceFormulaEvaluator()
            
            buy_formula = config.get('buy_execution_formula', 'C')
            sell_formula = config.get('sell_execution_formula', 'C')
            
            # For now, use buy formula as default (can be enhanced later)
            execution_prices = evaluator.get_execution_prices(buy_formula, data, "buy")
        else:
            execution_prices = data[execution_price_type]
        
        # Apply signal lag if configured
        signal_lag = config.get('signal_lag', 0)
        
        # Handle single parameter case
        if n_combos == 1:
            entries, exits = self._generate_signals_for_params(data, param_combinations[0])
            
            # Apply signal lag if configured
            if signal_lag > 0:
                entries = np.roll(entries, signal_lag)
                exits = np.roll(exits, signal_lag)
                entries[:signal_lag] = False
                exits[:signal_lag] = False
                
            # Run single backtest using config parameters
            pf = vbt.Portfolio.from_signals(
                close=execution_prices,
                entries=entries,
                exits=exits,
                size=config.get('position_size', 1),
                size_type=config.get('position_size_type', 'value'),
                direction=config.get('direction', 'both'),
                init_cash=config.get('initial_cash', 100000),
                min_size=config.get('min_size', 1e-10),
                fees=config.get('fees', 0.001),
                fixed_fees=config.get('fixed_fees', 1.0),
                slippage=config.get('slippage', 0.0005),
                freq=config.get('freq', '5T'),
                call_seq=config.get('call_seq', 'auto')
            )
        else:
            # For multiple parameters, generate signals for each combination
            n_bars = len(close_prices)
            entries_list = []
            exits_list = []
            
            for params in param_combinations:
                entry_signals, exit_signals = self._generate_signals_for_params(data, params)
                entries_list.append(entry_signals)
                exits_list.append(exit_signals)
            
            # Stack signals into 2D arrays (bars x parameters)
            entries = np.column_stack(entries_list)
            exits = np.column_stack(exits_list)
            
            # Apply signal lag if configured
            if signal_lag > 0:
                entries = np.roll(entries, signal_lag, axis=0)
                exits = np.roll(exits, signal_lag, axis=0)
                entries[:signal_lag] = False
                exits[:signal_lag] = False
            
            # Run vectorized backtest for all parameter combinations using config parameters
            pf = vbt.Portfolio.from_signals(
                close=execution_prices,
                entries=entries,
                exits=exits,
                size=config.get('position_size', 1),
                size_type=config.get('position_size_type', 'value'),
                direction=config.get('direction', 'both'),
                init_cash=config.get('initial_cash', 100000),
                min_size=config.get('min_size', 1e-10),
                fees=config.get('fees', 0.001),
                fixed_fees=config.get('fixed_fees', 1.0),
                slippage=config.get('slippage', 0.0005),
                freq=config.get('freq', '5T'),
                call_seq=config.get('call_seq', 'auto')
            )
        
        return pf
    
    @abstractmethod
    def _generate_signals_for_params(self, data: Dict[str, np.ndarray], 
                                   params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate signals for specific parameter combination.
        
        Args:
            data: Dictionary with OHLCV arrays
            params: Parameter dictionary
            
        Returns:
            Tuple of (entries, exits) as boolean arrays
        """
        pass