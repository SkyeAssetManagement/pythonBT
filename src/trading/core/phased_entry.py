"""
Phased Entry System - Core Implementation
Handles pyramid/scaling entry strategies with risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import yaml
import os


@dataclass
class PhasedEntryConfig:
    """Configuration for phased entry strategies"""
    enabled: bool = False
    max_phases: int = 3
    initial_size_percent: float = 33.33
    phase_trigger_type: str = "percent"
    phase_trigger_value: float = 2.0
    phase_size_type: str = "equal"
    phase_size_multiplier: float = 1.0
    max_adverse_move: float = 5.0
    require_profit: bool = True
    time_limit_bars: int = 50
    adapt_stop_loss: bool = True
    use_average_price: bool = True

    @classmethod
    def from_yaml(cls, config_path: str = None) -> 'PhasedEntryConfig':
        """Load phased entry configuration from YAML file"""
        if config_path is None:
            possible_paths = [
                "C:\\code\\PythonBT\\tradingCode\\config.yaml",
                "tradingCode\\config.yaml",
                "config.yaml"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                phased_config = config.get('phased_entries', {})

                if not phased_config:
                    return cls()

                # Extract nested configurations
                trigger_config = phased_config.get('phase_trigger', {})
                sizing_config = phased_config.get('phase_sizing', {})
                risk_config = phased_config.get('risk_management', {})
                stop_config = phased_config.get('stop_loss', {})

                return cls(
                    enabled=phased_config.get('enabled', False),
                    max_phases=phased_config.get('max_phases', 3),
                    initial_size_percent=phased_config.get('initial_size_percent', 33.33),
                    phase_trigger_type=trigger_config.get('type', 'percent'),
                    phase_trigger_value=trigger_config.get('value', 2.0),
                    phase_size_type=sizing_config.get('type', 'equal'),
                    phase_size_multiplier=sizing_config.get('multiplier', 1.0),
                    max_adverse_move=risk_config.get('max_adverse_move', 5.0),
                    require_profit=risk_config.get('require_profit', True),
                    time_limit_bars=risk_config.get('time_limit_bars', 50),
                    adapt_stop_loss=stop_config.get('adapt_to_phases', True),
                    use_average_price=stop_config.get('use_average_price', True)
                )

        return cls()


@dataclass
class TradePhase:
    """Individual phase within a phased entry"""
    phase_number: int
    trigger_price: float
    execution_price: float = 0.0
    size: float = 0.0
    bar_index: int = -1
    timestamp: Optional[datetime] = None
    trigger_met: bool = False
    executed: bool = False
    pnl: float = 0.0
    fees: float = 0.0


class PhasedEntry:
    """Manages a single phased entry position"""

    def __init__(self, config: PhasedEntryConfig, initial_signal_bar: int,
                 initial_price: float, is_long: bool, total_target_size: float):
        self.config = config
        self.initial_signal_bar = initial_signal_bar
        self.initial_price = initial_price
        self.is_long = is_long
        self.total_target_size = total_target_size
        self.phases: List[TradePhase] = []
        self.is_complete = False
        self.is_active = True
        self.stop_scaling = False

        # Create initial phase
        initial_size = total_target_size * (config.initial_size_percent / 100.0)
        self.phases.append(TradePhase(
            phase_number=1,
            trigger_price=initial_price,
            execution_price=0.0,  # Will be set when executed
            size=initial_size
        ))

        # Pre-calculate phase triggers and sizes
        self._setup_phases()

    def _setup_phases(self):
        """Pre-calculate all phase triggers and sizes"""
        if self.config.max_phases <= 1:
            return

        remaining_size = self.total_target_size - self.phases[0].size
        phases_remaining = self.config.max_phases - 1

        for i in range(1, self.config.max_phases):
            # Calculate trigger price for this phase
            if self.config.phase_trigger_type == "percent":
                if self.is_long:
                    trigger_price = self.initial_price * (1 + (self.config.phase_trigger_value * i / 100))
                else:
                    trigger_price = self.initial_price * (1 - (self.config.phase_trigger_value * i / 100))
            elif self.config.phase_trigger_type == "points":
                if self.is_long:
                    trigger_price = self.initial_price + (self.config.phase_trigger_value * i)
                else:
                    trigger_price = self.initial_price - (self.config.phase_trigger_value * i)
            else:
                # Default to percent
                if self.is_long:
                    trigger_price = self.initial_price * (1 + (self.config.phase_trigger_value * i / 100))
                else:
                    trigger_price = self.initial_price * (1 - (self.config.phase_trigger_value * i / 100))

            # Calculate size for this phase
            if self.config.phase_size_type == "equal":
                phase_size = remaining_size / phases_remaining
            elif self.config.phase_size_type == "decreasing":
                phase_size = (remaining_size / phases_remaining) * (self.config.phase_size_multiplier ** (i-1))
            elif self.config.phase_size_type == "increasing":
                phase_size = (remaining_size / phases_remaining) / (self.config.phase_size_multiplier ** (i-1))
            else:
                # Default to equal
                phase_size = remaining_size / phases_remaining

            self.phases.append(TradePhase(
                phase_number=i + 1,
                trigger_price=trigger_price,
                size=phase_size
            ))

            remaining_size -= phase_size
            phases_remaining -= 1

    def check_phase_triggers(self, current_price: float, current_bar: int) -> List[TradePhase]:
        """Check if any phases should be triggered"""
        if self.stop_scaling or self.is_complete:
            return []

        # Check time limit
        if current_bar - self.initial_signal_bar > self.config.time_limit_bars:
            self.stop_scaling = True
            return []

        # Check adverse move limit
        if self._check_adverse_move(current_price):
            self.stop_scaling = True
            return []

        # Check profit requirement
        if self.config.require_profit and not self._is_profitable(current_price):
            return []

        triggered_phases = []
        for phase in self.phases:
            if not phase.trigger_met and not phase.executed:
                if self._is_trigger_met(phase, current_price):
                    phase.trigger_met = True
                    triggered_phases.append(phase)

        return triggered_phases

    def _check_adverse_move(self, current_price: float) -> bool:
        """Check if adverse move limit is exceeded"""
        if self.config.max_adverse_move <= 0:
            return False

        if self.is_long:
            adverse_pct = ((self.initial_price - current_price) / self.initial_price) * 100
        else:
            adverse_pct = ((current_price - self.initial_price) / self.initial_price) * 100

        return adverse_pct > self.config.max_adverse_move

    def _is_profitable(self, current_price: float) -> bool:
        """Check if position is currently profitable"""
        avg_price = self.get_average_entry_price()
        if avg_price == 0:
            return True  # No executed phases yet

        if self.is_long:
            return current_price > avg_price
        else:
            return current_price < avg_price

    def _is_trigger_met(self, phase: TradePhase, current_price: float) -> bool:
        """Check if phase trigger is met"""
        if self.is_long:
            return current_price >= phase.trigger_price
        else:
            return current_price <= phase.trigger_price

    def execute_phase(self, phase: TradePhase, execution_price: float,
                     bar_index: int, timestamp: Optional[datetime] = None, fees: float = 0.0):
        """Execute a triggered phase"""
        phase.executed = True
        phase.execution_price = execution_price
        phase.bar_index = bar_index
        phase.timestamp = timestamp
        phase.fees = fees

        # Check if all phases are complete
        executed_phases = sum(1 for p in self.phases if p.executed)
        if executed_phases >= self.config.max_phases:
            self.is_complete = True

    def get_executed_phases(self) -> List[TradePhase]:
        """Get all executed phases"""
        return [p for p in self.phases if p.executed]

    def get_total_executed_size(self) -> float:
        """Get total size of executed phases"""
        return sum(p.size for p in self.phases if p.executed)

    def get_average_entry_price(self) -> float:
        """Calculate weighted average entry price of executed phases"""
        executed = self.get_executed_phases()
        if not executed:
            return 0.0

        total_value = sum(p.size * p.execution_price for p in executed)
        total_size = sum(p.size for p in executed)

        return total_value / total_size if total_size > 0 else 0.0

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate current unrealized P&L across all executed phases"""
        total_pnl = 0.0
        for phase in self.get_executed_phases():
            if self.is_long:
                phase_pnl = (current_price - phase.execution_price) * phase.size
            else:
                phase_pnl = (phase.execution_price - current_price) * phase.size
            total_pnl += phase_pnl - phase.fees

        return total_pnl

    def calculate_phase_pnl(self, phase: TradePhase, exit_price: float) -> float:
        """Calculate P&L for a specific phase"""
        if not phase.executed:
            return 0.0

        if self.is_long:
            gross_pnl = (exit_price - phase.execution_price) * phase.size
        else:
            gross_pnl = (phase.execution_price - exit_price) * phase.size

        return gross_pnl - phase.fees


class PhasedPositionTracker:
    """Tracks all phased positions and their states"""

    def __init__(self, config: PhasedEntryConfig):
        self.config = config
        self.active_positions: Dict[str, PhasedEntry] = {}
        self.completed_positions: List[PhasedEntry] = []
        self.position_history: List[Dict] = []

    def start_phased_entry(self, symbol: str, signal_bar: int, initial_price: float,
                          is_long: bool, target_size: float) -> PhasedEntry:
        """Start a new phased entry"""
        if not self.config.enabled:
            return None

        entry = PhasedEntry(
            config=self.config,
            initial_signal_bar=signal_bar,
            initial_price=initial_price,
            is_long=is_long,
            total_target_size=target_size
        )

        self.active_positions[symbol] = entry
        return entry

    def get_active_entry(self, symbol: str) -> Optional[PhasedEntry]:
        """Get active phased entry for symbol"""
        return self.active_positions.get(symbol)

    def complete_entry(self, symbol: str) -> Optional[PhasedEntry]:
        """Mark a phased entry as complete and move to history"""
        if symbol in self.active_positions:
            entry = self.active_positions.pop(symbol)
            entry.is_active = False
            self.completed_positions.append(entry)
            return entry
        return None

    def check_all_triggers(self, symbol: str, current_price: float,
                          current_bar: int) -> List[TradePhase]:
        """Check triggers for all active positions"""
        if symbol not in self.active_positions:
            return []

        return self.active_positions[symbol].check_phase_triggers(current_price, current_bar)

    def get_total_position_size(self, symbol: str) -> float:
        """Get total executed position size for symbol"""
        if symbol not in self.active_positions:
            return 0.0
        return self.active_positions[symbol].get_total_executed_size()

    def get_statistics(self) -> Dict:
        """Get phased entry statistics"""
        if not self.completed_positions:
            return {
                'total_phased_positions': 0,
                'avg_phases_per_position': 0.0,
                'completion_rate': 0.0,
                'most_common_phases': 0
            }

        total_positions = len(self.completed_positions)
        total_phases = sum(len(entry.get_executed_phases()) for entry in self.completed_positions)
        avg_phases = total_phases / total_positions if total_positions > 0 else 0

        complete_entries = sum(1 for entry in self.completed_positions if entry.is_complete)
        completion_rate = (complete_entries / total_positions) * 100 if total_positions > 0 else 0

        # Find most common number of phases
        phase_counts = [len(entry.get_executed_phases()) for entry in self.completed_positions]
        most_common = max(set(phase_counts), key=phase_counts.count) if phase_counts else 0

        return {
            'total_phased_positions': total_positions,
            'avg_phases_per_position': avg_phases,
            'completion_rate': completion_rate,
            'most_common_phases': most_common,
            'active_positions': len(self.active_positions)
        }