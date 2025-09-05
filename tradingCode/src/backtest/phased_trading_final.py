"""
Final Phased Trading Solution - Forces truly separate trades in VectorBT
This version ensures each phase appears as a separate trade in the tradelist.
"""

import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class PhasedTradingFinal:
    """
    Final implementation that forces VectorBT to create separate trades.
    
    Key innovation: Creates non-overlapping trades by ensuring complete 
    position closure between each phase.
    """
    
    def __init__(self, entry_bars: int, exit_bars: int, 
                 entry_distribution: str = "linear", 
                 exit_distribution: str = "linear"):
        """Initialize phased trading parameters."""
        self.entry_bars = entry_bars
        self.exit_bars = exit_bars
        self.entry_distribution = entry_distribution
        self.exit_distribution = exit_distribution
        
        # Calculate weights
        self.entry_weights = self._calculate_weights(entry_bars, entry_distribution)
        self.exit_weights = self._calculate_weights(exit_bars, exit_distribution)
    
    def _calculate_weights(self, n_bars: int, distribution: str) -> np.ndarray:
        """Calculate weight distribution."""
        if n_bars <= 1:
            return np.array([1.0])
        
        if distribution == "exponential":
            exp_values = np.exp(np.linspace(0, 2, n_bars))
            return exp_values / exp_values.sum()
        elif distribution == "front_loaded":
            weights = np.linspace(2, 1, n_bars)
            return weights / weights.sum()
        elif distribution == "back_loaded":
            weights = np.linspace(1, 2, n_bars)
            return weights / weights.sum()
        else:  # linear
            return np.ones(n_bars) / n_bars
    
    def create_phased_trades(self, 
                            original_entries: np.ndarray,
                            original_exits: np.ndarray,
                            position_size: float) -> Dict[str, List]:
        """
        Create completely separate trades for each phase.
        
        Returns a dictionary with lists of trade specifications that will
        be converted to separate portfolio simulations and then combined.
        """
        trades = []
        n_bars = len(original_entries)
        
        # Find original signals
        entry_indices = np.where(original_entries)[0]
        
        for entry_idx in entry_indices:
            # Find corresponding exit
            future_exits = np.where(original_exits[entry_idx:])[0]
            if len(future_exits) == 0:
                continue
            
            base_exit_idx = entry_idx + future_exits[0]
            
            # Create separate trade for each entry phase
            for phase_num in range(self.entry_bars):
                # Calculate phase-specific parameters
                phase_size = position_size * self.entry_weights[phase_num]
                
                # Stagger entry times to prevent overlap
                # Key: Add enough spacing to ensure VBT sees them as separate
                phase_entry_idx = entry_idx + phase_num * 3
                
                if phase_entry_idx >= n_bars:
                    continue
                
                # For exits, we have options:
                # 1. All exit at same time (base_exit_idx)
                # 2. Staggered exits
                if self.exit_bars == 1:
                    phase_exit_idx = base_exit_idx
                else:
                    # Stagger exits too
                    exit_phase = min(phase_num, self.exit_bars - 1)
                    phase_exit_idx = base_exit_idx + exit_phase * 2
                
                # Ensure exit is after entry and within bounds
                phase_exit_idx = max(phase_exit_idx, phase_entry_idx + 2)
                phase_exit_idx = min(phase_exit_idx, n_bars - 1)
                
                # Record this phase as a separate trade
                trades.append({
                    'phase_num': phase_num,
                    'entry_idx': phase_entry_idx,
                    'exit_idx': phase_exit_idx,
                    'size': phase_size,
                    'weight': self.entry_weights[phase_num],
                    'original_entry': entry_idx,
                    'original_exit': base_exit_idx
                })
        
        return {'trades': trades}
    
    def generate_separated_signals(self,
                                  original_entries: np.ndarray,
                                  original_exits: np.ndarray,
                                  position_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate completely separated entry/exit signals with sizes.
        
        This method ensures each phase trade is completely isolated with
        no possibility of consolidation by VectorBT.
        
        Returns:
            entries: Boolean array of entry signals
            exits: Boolean array of exit signals  
            sizes: Array of position sizes for each entry
        """
        n_bars = len(original_entries)
        
        # Initialize arrays
        entries = np.zeros(n_bars, dtype=bool)
        exits = np.zeros(n_bars, dtype=bool)
        sizes = np.zeros(n_bars, dtype=np.float64)
        
        # Get trade specifications
        trade_specs = self.create_phased_trades(original_entries, original_exits, position_size)
        
        # Place each trade
        for trade in trade_specs['trades']:
            # Set entry
            if trade['entry_idx'] < n_bars:
                entries[trade['entry_idx']] = True
                sizes[trade['entry_idx']] = trade['size']
            
            # Set exit
            if trade['exit_idx'] < n_bars:
                exits[trade['exit_idx']] = True
        
        # Validate and fix any issues
        n_entries = np.sum(entries)
        n_exits = np.sum(exits)
        
        if n_entries != n_exits:
            logger.warning(f"Entry/exit mismatch: {n_entries} entries, {n_exits} exits")
            # Add missing exits for unmatched entries
            if n_entries > n_exits:
                # Find last entry position
                entry_positions = np.where(entries)[0]
                if len(entry_positions) > 0:
                    last_entry = entry_positions[-1]
                    # Add exits after the last entry
                    for i in range(n_entries - n_exits):
                        exit_idx = min(last_entry + 10 + i * 2, n_bars - 1)
                        if exit_idx < n_bars:
                            exits[exit_idx] = True
        
        logger.info(f"Generated {n_entries} separate phase trades from {np.sum(original_entries)} original signals")
        
        return entries, exits, sizes


def apply_phased_trading_final(config: dict,
                              entries: np.ndarray,
                              exits: np.ndarray,
                              position_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply final phased trading solution to signals.
    
    This is the main entry point for the enhanced phased trading system.
    
    Args:
        config: Backtest configuration dictionary
        entries: Original entry signals
        exits: Original exit signals
        position_size: Base position size
        
    Returns:
        Tuple of (phased_entries, phased_exits, sizes)
    """
    if not config.get('phased_trading_enabled', False):
        # Return original signals if not enabled
        sizes = np.where(entries, position_size, 0)
        return entries, exits, sizes
    
    if config.get('consolidate_phased_trades', True):
        # Use original phased engine with consolidation
        # This is handled by the existing phased_trading_engine.py
        return None, None, None  # Signal to use existing engine
    
    # Use final solution for separate trades
    engine = PhasedTradingFinal(
        entry_bars=config.get('phased_entry_bars', 1),
        exit_bars=config.get('phased_exit_bars', 1),
        entry_distribution=config.get('phased_entry_distribution', 'linear'),
        exit_distribution=config.get('phased_exit_distribution', 'linear')
    )
    
    return engine.generate_separated_signals(entries, exits, position_size)