"""
Phased Trading Engine - Distributes trades across multiple bars
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhasedConfig:
    """Configuration for phased trading"""
    enabled: bool = False
    entry_bars: int = 1
    exit_bars: int = 1
    entry_distribution: str = "linear"  # linear, exponential, front_loaded, back_loaded
    exit_distribution: str = "linear"
    consolidate_trades: bool = True  # Whether to consolidate in tradelist


class PhasedTradingEngine:
    """
    Phased trading engine that distributes position entry/exit across multiple bars.
    
    This engine takes a single entry/exit signal and spreads it across multiple bars
    according to the configured distribution pattern.
    """
    
    def __init__(self, config: PhasedConfig):
        """Initialize with configuration."""
        self.config = config
        self.entry_weights = self._calculate_weights(config.entry_bars, config.entry_distribution)
        self.exit_weights = self._calculate_weights(config.exit_bars, config.exit_distribution)
        
    def _calculate_weights(self, n_bars: int, distribution: str) -> np.ndarray:
        """Calculate weight distribution for phased execution."""
        if n_bars <= 0:
            return np.array([1.0])
            
        if distribution == "exponential":
            # Exponentially increasing weights
            exp_values = np.exp(np.linspace(0, 2, n_bars))
            weights = exp_values / exp_values.sum()
        elif distribution == "front_loaded":
            # More weight at the beginning
            weights = np.linspace(2, 1, n_bars)
            weights = weights / weights.sum()
        elif distribution == "back_loaded":
            # More weight at the end
            weights = np.linspace(1, 2, n_bars)
            weights = weights / weights.sum()
        else:  # linear
            weights = np.ones(n_bars) / n_bars
            
        return weights
    
    def process_signals(self, 
                       entries: np.ndarray,
                       exits: np.ndarray,
                       data: Dict[str, np.ndarray],
                       position_size: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Process signals and apply phasing.
        
        Args:
            entries: Entry signals from strategy
            exits: Exit signals from strategy
            data: OHLCV data dictionary
            position_size: Total position size to distribute
            
        Returns:
            Dictionary with phased signals and sizes
        """
        if not self.config.enabled:
            # Return original signals if phasing disabled
            return {
                'entries': entries,
                'exits': exits,
                'entry_sizes': np.where(entries, position_size, 0),
                'exit_sizes': np.where(exits, position_size, 0)
            }
        
        n_bars = len(entries)
        
        # Initialize output arrays
        phased_entries = np.zeros(n_bars, dtype=bool)
        phased_exits = np.zeros(n_bars, dtype=bool)
        entry_sizes = np.zeros(n_bars, dtype=np.float64)
        exit_sizes = np.zeros(n_bars, dtype=np.float64)
        
        # Process entry signals
        entry_indices = np.where(entries)[0]
        for entry_idx in entry_indices:
            # Spread entry across multiple bars
            for phase_num in range(self.config.entry_bars):
                phase_idx = entry_idx + phase_num
                if phase_idx < n_bars:
                    phased_entries[phase_idx] = True
                    entry_sizes[phase_idx] = position_size * self.entry_weights[phase_num]
        
        # Process exit signals
        exit_indices = np.where(exits)[0]
        for exit_idx in exit_indices:
            # Spread exit across multiple bars
            for phase_num in range(self.config.exit_bars):
                phase_idx = exit_idx + phase_num
                if phase_idx < n_bars:
                    phased_exits[phase_idx] = True
                    exit_sizes[phase_idx] = position_size * self.exit_weights[phase_num]
        
        # Calculate execution prices if using formula-based pricing
        execution_prices = self._calculate_execution_prices(data, phased_entries, phased_exits)
        
        # Log phasing results
        n_original_entries = np.sum(entries)
        n_phased_entries = np.sum(phased_entries)
        n_original_exits = np.sum(exits)
        n_phased_exits = np.sum(phased_exits)
        
        logger.info(f"Phased trading: {n_original_entries} entries -> {n_phased_entries} phased entries")
        logger.info(f"Phased trading: {n_original_exits} exits -> {n_phased_exits} phased exits")
        
        return {
            'entries': phased_entries,
            'exits': phased_exits,
            'entry_sizes': entry_sizes,
            'exit_sizes': exit_sizes,
            'execution_prices': execution_prices
        }
    
    def _calculate_execution_prices(self,
                                   data: Dict[str, np.ndarray],
                                   entries: np.ndarray,
                                   exits: np.ndarray) -> np.ndarray:
        """
        Calculate execution prices for phased trades using (H+L+C)/3 formula.
        
        Args:
            data: OHLCV data
            entries: Phased entry signals
            exits: Phased exit signals
            
        Returns:
            Array of execution prices
        """
        # Calculate (H+L+C)/3 for each bar
        hlc3_prices = (data['high'] + data['low'] + data['close']) / 3.0
        
        # For entries and exits, use the HLC3 price
        execution_prices = data['close'].copy()  # Default to close
        
        # Apply HLC3 prices where we have signals
        signal_mask = entries | exits
        execution_prices[signal_mask] = hlc3_prices[signal_mask]
        
        return execution_prices
    
    def calculate_weighted_average_prices(self,
                                         data: Dict[str, np.ndarray],
                                         entries: np.ndarray,
                                         exits: np.ndarray,
                                         entry_sizes: np.ndarray,
                                         exit_sizes: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the weighted average entry and exit prices for reporting.
        
        Args:
            data: OHLCV data
            entries: Phased entry signals
            exits: Phased exit signals
            entry_sizes: Entry position sizes
            exit_sizes: Exit position sizes
            
        Returns:
            Tuple of (weighted_avg_entry_price, weighted_avg_exit_price)
        """
        # Calculate HLC3 prices
        hlc3_prices = (data['high'] + data['low'] + data['close']) / 3.0
        
        # Calculate weighted average entry price
        entry_indices = np.where(entries)[0]
        if len(entry_indices) > 0:
            entry_prices = hlc3_prices[entry_indices]
            entry_weights = entry_sizes[entry_indices]
            weighted_entry_price = np.sum(entry_prices * entry_weights) / np.sum(entry_weights)
        else:
            weighted_entry_price = 0.0
        
        # Calculate weighted average exit price
        exit_indices = np.where(exits)[0]
        if len(exit_indices) > 0:
            exit_prices = hlc3_prices[exit_indices]
            exit_weights = exit_sizes[exit_indices]
            weighted_exit_price = np.sum(exit_prices * exit_weights) / np.sum(exit_weights)
        else:
            weighted_exit_price = 0.0
        
        return weighted_entry_price, weighted_exit_price


def create_config_from_yaml(yaml_config: Dict) -> PhasedConfig:
    """
    Create PhasedConfig from YAML configuration.
    
    Args:
        yaml_config: Dictionary from YAML file
        
    Returns:
        PhasedConfig object
    """
    backtest = yaml_config.get('backtest', {})
    
    return PhasedConfig(
        enabled=backtest.get('phased_trading_enabled', False),
        entry_bars=backtest.get('phased_entry_bars', 1),
        exit_bars=backtest.get('phased_exit_bars', 1),
        entry_distribution=backtest.get('phased_entry_distribution', 'linear'),
        exit_distribution=backtest.get('phased_exit_distribution', 'linear'),
        consolidate_trades=backtest.get('consolidate_phased_trades', True)
    )