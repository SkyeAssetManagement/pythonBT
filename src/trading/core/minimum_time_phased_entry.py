"""
Minimum Execution Time Phased Entry System
==========================================
Spreads phases dynamically across a minimum execution timeframe
based on actual bars elapsed since signal + lag.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import yaml
import os


@dataclass
class MinimumTimePhasedConfig:
    """Configuration for minimum execution time phased entry"""
    enabled: bool = False
    max_phases: int = 3

    # Minimum execution time approach
    minimum_execution_minutes: float = 5.0      # Minimum time to complete all phases (minutes)
    spread_method: str = "equal"                # "equal", "weighted_front", "weighted_back"

    # Dynamic calculation based on data frequency
    data_frequency_minutes: float = 5.0         # Data bar frequency in minutes

    # Risk management
    max_adverse_move: float = 3.0
    require_profit: bool = True
    allow_early_completion: bool = True         # Complete early if all phases trigger

    # Size allocation
    initial_size_percent: float = 33.33
    phase_size_type: str = "equal"              # "equal", "decreasing", "increasing"
    phase_size_multiplier: float = 1.0

    @classmethod
    def from_yaml(cls, config_path: str = None) -> 'MinimumTimePhasedConfig':
        """Load minimum time phased entry configuration from YAML"""
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
                phased_config = config.get('minimum_time_phased_entries', {})

                if phased_config:
                    return cls(**phased_config)

        return cls()

    def calculate_phase_bars(self, signal_lag: int) -> List[int]:
        """
        Calculate the bar offsets for each phase based on minimum execution time

        Args:
            signal_lag: Number of bars to lag the signal

        Returns:
            List of bar offsets from signal bar (including lag)
        """
        # Calculate total bars needed for minimum execution time
        total_bars_needed = int(self.minimum_execution_minutes / self.data_frequency_minutes)

        if total_bars_needed < self.max_phases:
            total_bars_needed = self.max_phases  # Ensure minimum bars for phases

        # Calculate phase execution bars based on spread method
        if self.spread_method == "equal":
            # Spread phases equally across the timeframe
            phase_intervals = total_bars_needed // self.max_phases
            phase_bars = [signal_lag + (i * phase_intervals) for i in range(self.max_phases)]

        elif self.spread_method == "weighted_front":
            # More phases early, fewer later
            weights = [0.5, 0.3, 0.2][:self.max_phases]
            cumulative_bars = 0
            phase_bars = [signal_lag]

            for i in range(1, self.max_phases):
                bars_for_this_phase = int(total_bars_needed * weights[i])
                cumulative_bars += bars_for_this_phase
                phase_bars.append(signal_lag + cumulative_bars)

        elif self.spread_method == "weighted_back":
            # Fewer phases early, more later
            weights = [0.2, 0.3, 0.5][:self.max_phases]
            cumulative_bars = 0
            phase_bars = [signal_lag]

            for i in range(1, self.max_phases):
                bars_for_this_phase = int(total_bars_needed * weights[i])
                cumulative_bars += bars_for_this_phase
                phase_bars.append(signal_lag + cumulative_bars)
        else:
            # Default to equal spacing
            phase_intervals = total_bars_needed // self.max_phases
            phase_bars = [signal_lag + (i * phase_intervals) for i in range(self.max_phases)]

        return phase_bars


@dataclass
class MinimumTimePhase:
    """Individual phase within a minimum time phased entry"""
    phase_number: int
    scheduled_bar: int                  # Bar when this phase should execute
    size: float
    executed: bool = False
    execution_bar: Optional[int] = None
    execution_price: Optional[float] = None
    execution_timestamp: Optional[datetime] = None
    fees: float = 0.0


class MinimumTimePhasedEntry:
    """Manages a single minimum time phased entry position"""

    def __init__(self, config: MinimumTimePhasedConfig, signal_bar: int,
                 signal_price: float, is_long: bool, target_size: float, signal_lag: int):
        self.config = config
        self.signal_bar = signal_bar
        self.signal_price = signal_price
        self.is_long = is_long
        self.target_size = target_size
        self.signal_lag = signal_lag

        # Calculate when execution starts (signal + lag)
        self.execution_start_bar = signal_bar + signal_lag

        # Calculate phase execution bars
        self.phase_bars = self.config.calculate_phase_bars(signal_lag)

        # Create phases
        self.phases: List[MinimumTimePhase] = []
        self._generate_phases()

        # State tracking
        self.stop_scaling = False
        self.is_complete = False
        self.completion_reason = None

    def _generate_phases(self):
        """Generate phases with sizes and scheduled bars"""
        remaining_size = self.target_size

        for i in range(self.config.max_phases):
            # Calculate phase size based on configuration
            if self.config.phase_size_type == "equal":
                phase_size = self.target_size / self.config.max_phases
            elif self.config.phase_size_type == "decreasing":
                # First phase largest, subsequent phases smaller
                phase_multiplier = self.config.phase_size_multiplier ** i
                phase_size = (self.target_size / self.config.max_phases) * phase_multiplier
            elif self.config.phase_size_type == "increasing":
                # First phase smallest, subsequent phases larger
                phase_multiplier = self.config.phase_size_multiplier ** i
                phase_size = (self.target_size / self.config.max_phases) / phase_multiplier
            else:
                phase_size = self.target_size / self.config.max_phases

            # Adjust last phase to use remaining size exactly
            if i == self.config.max_phases - 1:
                phase_size = remaining_size

            self.phases.append(MinimumTimePhase(
                phase_number=i + 1,
                scheduled_bar=self.phase_bars[i],
                size=phase_size
            ))

            remaining_size -= phase_size

    def check_phase_execution(self, current_bar: int, current_price: float) -> List[MinimumTimePhase]:
        """
        Check if any phases should be executed at current bar

        Returns:
            List of phases ready for execution
        """
        if self.stop_scaling or self.is_complete:
            return []

        # Check risk management
        if self._check_adverse_move(current_price):
            self.stop_scaling = True
            self.completion_reason = "adverse_move_limit"
            return []

        if self.config.require_profit and not self._is_profitable(current_price):
            return []

        # Find phases ready for execution
        ready_phases = []
        for phase in self.phases:
            if not phase.executed and current_bar >= phase.scheduled_bar:
                ready_phases.append(phase)

        return ready_phases

    def execute_phase(self, phase: MinimumTimePhase, execution_price: float,
                     execution_bar: int, timestamp: Optional[datetime] = None,
                     fees: float = 0.0):
        """Execute a specific phase"""
        phase.executed = True
        phase.execution_price = execution_price
        phase.execution_bar = execution_bar
        phase.execution_timestamp = timestamp
        phase.fees = fees

        # Check if all phases complete
        if all(p.executed for p in self.phases):
            self.is_complete = True
            self.completion_reason = "all_phases_executed"

    def _check_adverse_move(self, current_price: float) -> bool:
        """Check if adverse move limit is exceeded"""
        if self.config.max_adverse_move <= 0:
            return False

        # Calculate move from initial signal price
        if self.is_long:
            adverse_move_pct = ((self.signal_price - current_price) / self.signal_price) * 100
        else:
            adverse_move_pct = ((current_price - self.signal_price) / self.signal_price) * 100

        return adverse_move_pct > self.config.max_adverse_move

    def _is_profitable(self, current_price: float) -> bool:
        """Check if current position is profitable"""
        executed_phases = self.get_executed_phases()
        if not executed_phases:
            return True  # No position yet, allow first phase

        avg_entry_price = self.get_average_entry_price()
        if avg_entry_price is None:
            return True

        if self.is_long:
            return current_price > avg_entry_price
        else:
            return current_price < avg_entry_price

    def get_executed_phases(self) -> List[MinimumTimePhase]:
        """Get list of executed phases"""
        return [p for p in self.phases if p.executed]

    def get_total_executed_size(self) -> float:
        """Get total size of executed phases"""
        return sum(p.size for p in self.phases if p.executed)

    def get_average_entry_price(self) -> Optional[float]:
        """Calculate average entry price of executed phases"""
        executed = self.get_executed_phases()
        if not executed:
            return None

        total_value = sum(p.execution_price * p.size for p in executed)
        total_size = sum(p.size for p in executed)

        return total_value / total_size if total_size > 0 else None

    def get_status_summary(self) -> Dict:
        """Get summary of phased entry status"""
        executed_phases = self.get_executed_phases()

        return {
            'signal_bar': self.signal_bar,
            'execution_start_bar': self.execution_start_bar,
            'total_phases': len(self.phases),
            'executed_phases': len(executed_phases),
            'completion_percentage': (len(executed_phases) / len(self.phases)) * 100,
            'is_complete': self.is_complete,
            'completion_reason': self.completion_reason,
            'average_entry_price': self.get_average_entry_price(),
            'total_executed_size': self.get_total_executed_size(),
            'next_phase_bar': min((p.scheduled_bar for p in self.phases if not p.executed), default=None)
        }


class MinimumTimePhasedTracker:
    """Tracks multiple minimum time phased entries"""

    def __init__(self, config: MinimumTimePhasedConfig):
        self.config = config
        self.active_entries: Dict[str, MinimumTimePhasedEntry] = {}
        self.completed_entries: List[MinimumTimePhasedEntry] = []

    def start_phased_entry(self, symbol: str, signal_bar: int, signal_price: float,
                          is_long: bool, target_size: float, signal_lag: int) -> MinimumTimePhasedEntry:
        """Start a new minimum time phased entry"""
        if symbol in self.active_entries:
            # Complete existing entry first
            self.complete_entry(symbol)

        entry = MinimumTimePhasedEntry(
            self.config, signal_bar, signal_price, is_long, target_size, signal_lag
        )

        self.active_entries[symbol] = entry
        return entry

    def get_active_entry(self, symbol: str) -> Optional[MinimumTimePhasedEntry]:
        """Get active minimum time phased entry for symbol"""
        return self.active_entries.get(symbol)

    def complete_entry(self, symbol: str) -> Optional[MinimumTimePhasedEntry]:
        """Mark entry as complete and move to history"""
        entry = self.active_entries.pop(symbol, None)
        if entry:
            entry.is_complete = True
            if not entry.completion_reason:
                entry.completion_reason = "manual_completion"
            self.completed_entries.append(entry)
        return entry

    def get_statistics(self) -> Dict:
        """Get statistics about minimum time phased entries"""
        if not self.completed_entries:
            return {}

        total_entries = len(self.completed_entries)
        total_phases = sum(len(entry.get_executed_phases()) for entry in self.completed_entries)

        completion_reasons = {}
        for entry in self.completed_entries:
            reason = entry.completion_reason or "unknown"
            completion_reasons[reason] = completion_reasons.get(reason, 0) + 1

        avg_phases_per_entry = total_phases / total_entries if total_entries > 0 else 0

        return {
            'total_entries': total_entries,
            'total_phases_executed': total_phases,
            'average_phases_per_entry': avg_phases_per_entry,
            'completion_reasons': completion_reasons,
            'active_entries': len(self.active_entries)
        }