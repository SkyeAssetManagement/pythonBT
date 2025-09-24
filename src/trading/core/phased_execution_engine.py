"""
Enhanced Execution Engine with Phased Entry Support
Extends the standalone execution engine to handle pyramid/scaling entries
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import yaml
import os

from .standalone_execution import StandaloneExecutionEngine, ExecutionConfig
from .phased_entry import PhasedEntryConfig, PhasedEntry, TradePhase, PhasedPositionTracker


class PhasedExecutionConfig(ExecutionConfig):
    """Extended execution configuration with phased entry support"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phased_config = PhasedEntryConfig.from_yaml()

    @classmethod
    def from_yaml(cls, config_path: str = None) -> 'PhasedExecutionConfig':
        """Load configuration from YAML file with phased entry support"""
        # Load base config
        base_config = super(PhasedExecutionConfig, cls).from_yaml(config_path)

        # Create enhanced config with base parameters
        enhanced_config = cls(
            signal_lag=base_config.signal_lag,
            execution_price=base_config.execution_price,
            buy_execution_formula=base_config.buy_execution_formula,
            sell_execution_formula=base_config.sell_execution_formula,
            position_size=base_config.position_size,
            position_size_type=base_config.position_size_type,
            initial_cash=base_config.initial_cash,
            fees=base_config.fees,
            fixed_fees=base_config.fixed_fees,
            slippage=base_config.slippage
        )

        return enhanced_config


class PhasedExecutionEngine(StandaloneExecutionEngine):
    """
    Enhanced execution engine supporting phased entries
    Maintains backward compatibility with single-phase trades
    """

    def __init__(self, config: PhasedExecutionConfig = None):
        """Initialize with phased execution configuration"""
        if config is None:
            config = PhasedExecutionConfig.from_yaml()

        super().__init__(config)
        self.phased_config = config.phased_config
        self.position_tracker = PhasedPositionTracker(self.phased_config)

        print(f"[PHASED_ENGINE] Phased entries: {'ENABLED' if self.phased_config.enabled else 'DISABLED'}")
        if self.phased_config.enabled:
            print(f"[PHASED_ENGINE] Max phases: {self.phased_config.max_phases}")
            print(f"[PHASED_ENGINE] Initial size: {self.phased_config.initial_size_percent}%")
            print(f"[PHASED_ENGINE] Trigger: {self.phased_config.phase_trigger_value}% {self.phased_config.phase_trigger_type}")

    def execute_signals_with_phases(self, signals: pd.Series, df: pd.DataFrame,
                                  symbol: str = "DEFAULT") -> List[Dict]:
        """
        Execute trading signals with phased entry support
        Falls back to standard execution if phased entries disabled
        """
        if not self.phased_config.enabled:
            return self.execute_signals(signals, df)

        return self._execute_phased_signals(signals, df, symbol)

    def _execute_phased_signals(self, signals: pd.Series, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Execute signals with phased entry logic"""
        trades = []
        position = 0
        trade_id = 0
        current_cash = self.config.initial_cash

        # Check for DateTime column
        has_datetime = 'DateTime' in df.columns or 'timestamp' in df.columns

        print(f"[PHASED_ENGINE] Processing {len(signals)} signals for {symbol}")

        for i in range(len(signals)):
            signal = signals.iloc[i]
            current_price = self._get_price(df, i, 'Close')

            # Check for phase triggers on existing positions
            if symbol in self.position_tracker.active_positions:
                triggered_phases = self.position_tracker.check_all_triggers(symbol, current_price, i)
                for phase in triggered_phases:
                    phase_trades = self._execute_phase(phase, df, i, symbol, has_datetime, trade_id)
                    trades.extend(phase_trades)
                    trade_id += len(phase_trades)

            # Handle new signals
            if signal != position:
                # Exit current position if needed
                if position != 0:
                    exit_trades = self._execute_position_exit(
                        df, i, position, symbol, has_datetime, trade_id
                    )
                    trades.extend(exit_trades)
                    trade_id += len(exit_trades)
                    position = 0

                # Enter new position if signal is non-zero
                if signal != 0:
                    entry_trades = self._execute_position_entry(
                        df, i, signal, symbol, has_datetime, trade_id
                    )
                    trades.extend(entry_trades)
                    trade_id += len(entry_trades)
                    position = signal

        # Close any remaining positions
        if position != 0:
            final_trades = self._execute_position_exit(
                df, len(df) - 1, position, symbol, has_datetime, trade_id
            )
            trades.extend(final_trades)

        print(f"[PHASED_ENGINE] Generated {len(trades)} total trade records")
        return trades

    def _execute_position_entry(self, df: pd.DataFrame, signal_bar: int, signal: int,
                               symbol: str, has_datetime: bool, trade_id: int) -> List[Dict]:
        """Execute a new position entry with phased entry logic"""
        is_long = signal > 0
        current_price = self._get_price(df, signal_bar, 'Close')

        # Start phased entry
        phased_entry = self.position_tracker.start_phased_entry(
            symbol=symbol,
            signal_bar=signal_bar,
            initial_price=current_price,
            is_long=is_long,
            target_size=self.config.position_size
        )

        if phased_entry is None:
            # Fallback to single entry
            return self._execute_single_entry(df, signal_bar, signal, has_datetime, trade_id)

        # Execute first phase immediately
        first_phase = phased_entry.phases[0]
        return self._execute_phase(first_phase, df, signal_bar, symbol, has_datetime, trade_id)

    def _execute_phase(self, phase: TradePhase, df: pd.DataFrame, bar_index: int,
                      symbol: str, has_datetime: bool, trade_id: int) -> List[Dict]:
        """Execute a specific phase of a phased entry"""
        phased_entry = self.position_tracker.get_active_entry(symbol)
        if not phased_entry:
            return []

        is_buy = phased_entry.is_long
        trade_type = 'BUY' if is_buy else 'SHORT'

        # Calculate execution price and apply lag
        exec_price, exec_bar, formula = self.calculate_execution_price(df, bar_index, is_buy)
        exec_price_adjusted, fees = self.apply_friction(exec_price, phase.size, is_buy)

        # Execute the phase
        timestamp = self._get_timestamp(df, exec_bar, has_datetime)
        phased_entry.execute_phase(phase, exec_price_adjusted, exec_bar, timestamp, fees)

        # Create trade record with phase information
        trade = {
            'trade_id': trade_id,
            'signal_bar': phased_entry.initial_signal_bar,
            'execution_bar': exec_bar,
            'lag': exec_bar - bar_index,
            'trade_type': trade_type,
            'signal_price': phased_entry.initial_price,
            'execution_price': exec_price,
            'execution_price_adjusted': exec_price_adjusted,
            'formula': formula,
            'size': phase.size,
            'fees': fees,
            'timestamp': timestamp,
            'phase_number': phase.phase_number,
            'total_phases': len(phased_entry.phases),
            'is_phased_entry': True,
            'phase_trigger_price': phase.trigger_price,
            'average_entry_price': phased_entry.get_average_entry_price(),
            'total_position_size': phased_entry.get_total_executed_size()
        }

        print(f"[PHASED_ENGINE] Executed phase {phase.phase_number} for {symbol}: "
              f"Size={phase.size:.4f}, Price={exec_price_adjusted:.2f}")

        return [trade]

    def _execute_position_exit(self, df: pd.DataFrame, bar_index: int, position: int,
                              symbol: str, has_datetime: bool, trade_id: int) -> List[Dict]:
        """Execute position exit, handling phased positions"""
        phased_entry = self.position_tracker.get_active_entry(symbol)

        if phased_entry:
            # Exit all executed phases of phased position
            return self._execute_phased_exit(df, bar_index, phased_entry, symbol, has_datetime, trade_id)
        else:
            # Single position exit
            return self._execute_single_exit(df, bar_index, position, has_datetime, trade_id)

    def _execute_phased_exit(self, df: pd.DataFrame, bar_index: int, phased_entry: PhasedEntry,
                            symbol: str, has_datetime: bool, trade_id: int) -> List[Dict]:
        """Exit all phases of a phased position"""
        trades = []
        executed_phases = phased_entry.get_executed_phases()

        if not executed_phases:
            return trades

        is_buy = not phased_entry.is_long  # Exit is opposite of entry
        trade_type = 'SELL' if phased_entry.is_long else 'COVER'

        # Calculate total exit
        total_size = sum(p.size for p in executed_phases)
        exec_price, exec_bar, formula = self.calculate_execution_price(df, bar_index, is_buy)
        exec_price_adjusted, total_fees = self.apply_friction(exec_price, total_size, is_buy)

        # Calculate P&L for each phase and create trade records
        for i, phase in enumerate(executed_phases):
            # Calculate phase-specific P&L
            phase_pnl_points = phased_entry.calculate_phase_pnl(phase, exec_price_adjusted)

            # Calculate percentage P&L based on phase entry
            if phase.execution_price > 0:
                if phased_entry.is_long:
                    phase_pnl_percent = ((exec_price_adjusted / phase.execution_price) - 1) * 100
                else:
                    phase_pnl_percent = (1 - (exec_price_adjusted / phase.execution_price)) * 100
            else:
                phase_pnl_percent = 0.0

            # Allocate fees proportionally
            phase_fees = (phase.size / total_size) * total_fees
            timestamp = self._get_timestamp(df, exec_bar, has_datetime)

            trade = {
                'trade_id': trade_id + i,
                'signal_bar': bar_index,
                'execution_bar': exec_bar,
                'lag': exec_bar - bar_index,
                'trade_type': trade_type,
                'signal_price': self._get_price(df, bar_index, 'Close'),
                'execution_price': exec_price,
                'execution_price_adjusted': exec_price_adjusted,
                'formula': formula,
                'size': phase.size,
                'fees': phase_fees,
                'pnl_points': phase_pnl_points,
                'pnl_percent': phase_pnl_percent,
                'timestamp': timestamp,
                'phase_number': phase.phase_number,
                'phase_entry_price': phase.execution_price,
                'is_phased_exit': True,
                'total_phases': len(executed_phases),
                'average_entry_price': phased_entry.get_average_entry_price()
            }
            trades.append(trade)

        # Complete the phased entry
        self.position_tracker.complete_entry(symbol)

        print(f"[PHASED_ENGINE] Exited {len(executed_phases)} phases for {symbol}")
        return trades

    def _execute_single_entry(self, df: pd.DataFrame, signal_bar: int, signal: int,
                             has_datetime: bool, trade_id: int) -> List[Dict]:
        """Execute single entry (fallback for non-phased)"""
        is_buy = signal > 0
        trade_type = 'BUY' if is_buy else 'SHORT'

        exec_price, exec_bar, formula = self.calculate_execution_price(df, signal_bar, is_buy)
        exec_price_adjusted, fees = self.apply_friction(exec_price, self.config.position_size, is_buy)
        timestamp = self._get_timestamp(df, exec_bar, has_datetime)

        trade = {
            'trade_id': trade_id,
            'signal_bar': signal_bar,
            'execution_bar': exec_bar,
            'lag': exec_bar - signal_bar,
            'trade_type': trade_type,
            'signal_price': self._get_price(df, signal_bar, 'Close'),
            'execution_price': exec_price,
            'execution_price_adjusted': exec_price_adjusted,
            'formula': formula,
            'size': self.config.position_size,
            'fees': fees,
            'timestamp': timestamp,
            'is_phased_entry': False
        }

        return [trade]

    def _execute_single_exit(self, df: pd.DataFrame, bar_index: int, position: int,
                            has_datetime: bool, trade_id: int) -> List[Dict]:
        """Execute single exit (fallback for non-phased)"""
        is_buy = False  # Exits are sells
        trade_type = 'SELL' if position > 0 else 'COVER'

        exec_price, exec_bar, formula = self.calculate_execution_price(df, bar_index, is_buy)
        exec_price_adjusted, fees = self.apply_friction(exec_price, abs(position), is_buy)
        timestamp = self._get_timestamp(df, exec_bar, has_datetime)

        trade = {
            'trade_id': trade_id,
            'signal_bar': bar_index,
            'execution_bar': exec_bar,
            'lag': exec_bar - bar_index,
            'trade_type': trade_type,
            'signal_price': self._get_price(df, bar_index, 'Close'),
            'execution_price': exec_price,
            'execution_price_adjusted': exec_price_adjusted,
            'formula': formula,
            'size': abs(position),
            'fees': fees,
            'timestamp': timestamp,
            'is_phased_entry': False
        }

        return [trade]

    def get_phased_statistics(self) -> Dict:
        """Get statistics about phased entry performance"""
        return self.position_tracker.get_statistics()