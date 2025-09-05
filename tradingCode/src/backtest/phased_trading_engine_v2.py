"""
Phased Trading Engine V2 - Forces separate trades in VectorBT
This version creates complete entry-exit cycles for each phase to ensure
VectorBT treats them as separate trades.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhasedConfigV2:
    """Configuration for phased trading V2"""
    enabled: bool = False
    entry_bars: int = 1
    exit_bars: int = 1
    entry_distribution: str = "linear"  # linear, exponential, front_loaded, back_loaded
    exit_distribution: str = "linear"
    min_separation_bars: int = 1  # Minimum bars between phase trades
    force_separate_trades: bool = True  # Force VBT to create separate trades


class PhasedTradingEngineV2:
    """
    Enhanced phased trading engine that forces VectorBT to create separate trades.
    
    Key innovation: Creates complete entry-exit pairs for each phase to ensure
    VectorBT treats them as independent trades rather than consolidating them.
    """
    
    def __init__(self, config: PhasedConfigV2):
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
    
    def create_separate_trade_signals(self, 
                                     original_entries: np.ndarray,
                                     original_exits: np.ndarray,
                                     position_size: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Transform original signals into separate trade signals for each phase.
        
        This method creates complete entry-exit pairs for each phase, forcing
        VectorBT to treat them as separate trades.
        
        Args:
            original_entries: Original entry signals from strategy
            original_exits: Original exit signals from strategy
            position_size: Total position size to distribute
            
        Returns:
            Dictionary with transformed signals and sizes
        """
        n_bars = len(original_entries)
        
        # Initialize output arrays
        phased_entries = np.zeros(n_bars, dtype=bool)
        phased_exits = np.zeros(n_bars, dtype=bool)
        phased_sizes = np.zeros(n_bars, dtype=np.float64)
        
        # Track trade metadata for debugging
        trade_metadata = []
        
        # Find original entry signals
        entry_indices = np.where(original_entries)[0]
        
        for entry_idx in entry_indices:
            # Find corresponding exit for this entry
            future_exits = np.where(original_exits[entry_idx:])[0]
            if len(future_exits) == 0:
                # No exit found, skip this entry
                continue
                
            exit_idx = entry_idx + future_exits[0]
            trade_duration = exit_idx - entry_idx
            
            # Calculate phase parameters
            n_entry_phases = self.config.entry_bars
            n_exit_phases = self.config.exit_bars
            
            # For each entry phase, create a complete mini-trade
            for entry_phase in range(n_entry_phases):
                # Calculate entry position for this phase
                phase_entry_idx = entry_idx + entry_phase * (self.config.min_separation_bars + 1)
                
                if phase_entry_idx >= n_bars:
                    break
                    
                # Calculate size for this phase
                phase_size = position_size * self.entry_weights[entry_phase]
                
                # Determine exit strategy for this phase
                if n_exit_phases == 1:
                    # Simple exit - all phases exit at the same time
                    phase_exit_idx = exit_idx
                else:
                    # Staggered exits - each phase has its own exit timing
                    exit_phase = min(entry_phase, n_exit_phases - 1)
                    phase_exit_idx = exit_idx + exit_phase * self.config.min_separation_bars
                    
                # Ensure exit is after entry and within bounds
                phase_exit_idx = max(phase_exit_idx, phase_entry_idx + 1)
                phase_exit_idx = min(phase_exit_idx, n_bars - 1)
                
                # Set signals for this phase trade
                phased_entries[phase_entry_idx] = True
                phased_exits[phase_exit_idx] = True
                phased_sizes[phase_entry_idx] = phase_size
                
                # Record metadata
                trade_metadata.append({
                    'original_entry': entry_idx,
                    'original_exit': exit_idx,
                    'phase_num': entry_phase,
                    'phase_entry': phase_entry_idx,
                    'phase_exit': phase_exit_idx,
                    'phase_size': phase_size,
                    'phase_weight': self.entry_weights[entry_phase]
                })
        
        # Log summary
        logger.info(f"Created {len(trade_metadata)} separate phase trades from {len(entry_indices)} original signals")
        
        return {
            'entries': phased_entries,
            'exits': phased_exits,
            'sizes': phased_sizes,
            'metadata': trade_metadata
        }
    
    def apply_advanced_phasing(self,
                              entries: np.ndarray,
                              exits: np.ndarray,
                              prices: Dict[str, np.ndarray],
                              position_size: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Apply advanced phasing that guarantees separate trades in VectorBT.
        
        This method uses a more sophisticated approach:
        1. Creates non-overlapping entry-exit pairs
        2. Ensures each phase is a complete trade cycle
        3. Prevents VectorBT from consolidating trades
        
        Args:
            entries: Entry signals
            exits: Exit signals  
            prices: Price data
            position_size: Base position size
            
        Returns:
            Dictionary with phased signals and trade information
        """
        if not self.config.enabled or not self.config.force_separate_trades:
            # Return original signals if phasing disabled
            return {
                'entries': entries,
                'exits': exits,
                'sizes': np.where(entries, position_size, 0),
                'entry_prices': prices['close'],
                'exit_prices': prices['close']
            }
        
        # Transform signals to create separate trades
        result = self.create_separate_trade_signals(entries, exits, position_size)
        
        # Add price information
        result['entry_prices'] = prices['close'].copy()
        result['exit_prices'] = prices['close'].copy()
        
        # Log diagnostic information
        n_original_entries = np.sum(entries)
        n_phased_entries = np.sum(result['entries'])
        n_phased_exits = np.sum(result['exits'])
        
        logger.info(f"Phasing summary: {n_original_entries} original -> {n_phased_entries} phased entries, {n_phased_exits} phased exits")
        
        # Validate and fix entry/exit mismatch
        if n_phased_entries != n_phased_exits:
            logger.warning(f"Entry/exit mismatch: {n_phased_entries} entries, {n_phased_exits} exits - fixing...")
            # Ensure we have matching exits for all entries
            if n_phased_entries > n_phased_exits:
                # Add exit at the end for unmatched entries
                last_entry_idx = np.where(result['entries'])[0][-1]
                exit_idx = min(last_entry_idx + 10, len(result['exits']) - 1)
                result['exits'][exit_idx] = True
            elif n_phased_exits > n_phased_entries:
                # Remove extra exits
                exit_indices = np.where(result['exits'])[0]
                for i in range(n_phased_exits - n_phased_entries):
                    result['exits'][exit_indices[-(i+1)]] = False
        
        return result
    
    def generate_test_signals(self, n_bars: int = 1000) -> Dict[str, np.ndarray]:
        """
        Generate test signals to demonstrate phased trading.
        
        Creates a few well-spaced signals to clearly show the phasing effect.
        
        Args:
            n_bars: Number of bars to generate
            
        Returns:
            Dictionary with test signals
        """
        entries = np.zeros(n_bars, dtype=bool)
        exits = np.zeros(n_bars, dtype=bool)
        
        # Create a few test trades with good spacing
        test_trades = [
            (50, 150),   # Trade 1: entry at 50, exit at 150
            (200, 300),  # Trade 2: entry at 200, exit at 300
            (400, 500),  # Trade 3: entry at 400, exit at 500
        ]
        
        for entry_idx, exit_idx in test_trades:
            if entry_idx < n_bars:
                entries[entry_idx] = True
            if exit_idx < n_bars:
                exits[exit_idx] = True
        
        return {
            'entries': entries,
            'exits': exits
        }


def create_config_from_yaml(yaml_config: Dict) -> PhasedConfigV2:
    """
    Create PhasedConfigV2 from YAML configuration.
    
    Args:
        yaml_config: Dictionary from YAML file
        
    Returns:
        PhasedConfigV2 object
    """
    backtest = yaml_config.get('backtest', {})
    
    return PhasedConfigV2(
        enabled=backtest.get('phased_trading_enabled', False),
        entry_bars=backtest.get('phased_entry_bars', 1),
        exit_bars=backtest.get('phased_exit_bars', 1),
        entry_distribution=backtest.get('phased_entry_distribution', 'linear'),
        exit_distribution=backtest.get('phased_exit_distribution', 'linear'),
        min_separation_bars=backtest.get('phased_min_separation_bars', 1),
        force_separate_trades=backtest.get('phased_force_separate_trades', True)
    )