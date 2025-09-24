"""
Phased Entry Strategy Base Class
Extends the trading strategy base to support phased/pyramid entries
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import sys
import os
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.trade_data import TradeData, TradeCollection
from .enhanced_base import EnhancedTradingStrategy
from ..core.phased_entry import PhasedEntryConfig
from ..core.phased_execution_engine import PhasedExecutionEngine, PhasedExecutionConfig


class PhasedTradingStrategy(EnhancedTradingStrategy):
    """Enhanced trading strategy base class with phased entry support"""

    def __init__(self, name: str = "PhasedStrategy", config_path: str = None):
        super().__init__(name, config_path)

        # Load phased entry configuration
        self.phased_config = PhasedEntryConfig.from_yaml(config_path)

        # Initialize phased execution engine
        exec_config = PhasedExecutionConfig.from_yaml(config_path)
        self.execution_engine = PhasedExecutionEngine(exec_config)

        print(f"[PHASED_STRATEGY] {name} initialized with phased entries: "
              f"{'ENABLED' if self.phased_config.enabled else 'DISABLED'}")

    def generate_phased_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate both primary signals and phase strength indicators

        Returns:
            Tuple of (primary_signals, phase_strength)
            - primary_signals: Standard 1/-1/0 signals
            - phase_strength: Float values indicating signal confidence/strength
        """
        # Default implementation - subclasses should override for advanced phasing
        primary_signals = self.generate_signals(df)
        phase_strength = pd.Series(1.0, index=primary_signals.index)  # Full strength by default

        return primary_signals, phase_strength

    def should_use_phased_entry(self, signal_bar: int, df: pd.DataFrame, signal_strength: float) -> bool:
        """
        Determine if a signal should use phased entry
        Override in subclasses for custom logic
        """
        if not self.phased_config.enabled:
            return False

        # Default: use phased entry for strong signals only
        return signal_strength >= 0.8

    def calculate_dynamic_phase_size(self, phase_number: int, current_price: float,
                                   entry_price: float, df: pd.DataFrame, bar_index: int) -> float:
        """
        Calculate dynamic phase sizing based on market conditions
        Override for advanced sizing logic
        """
        # Default implementation uses config settings
        if self.phased_config.phase_size_type == "equal":
            return 1.0  # Equal weighting
        elif self.phased_config.phase_size_type == "decreasing":
            return 1.0 / (phase_number * self.phased_config.phase_size_multiplier)
        elif self.phased_config.phase_size_type == "increasing":
            return phase_number * self.phased_config.phase_size_multiplier
        else:
            return 1.0

    def calculate_adaptive_triggers(self, df: pd.DataFrame, entry_bar: int, is_long: bool) -> List[float]:
        """
        Calculate adaptive phase triggers based on market conditions
        Override for volatility-adjusted or indicator-based triggers
        """
        entry_price = df['Close'].iloc[entry_bar] if 'Close' in df.columns else df['close'].iloc[entry_bar]
        triggers = []

        for i in range(1, self.phased_config.max_phases):
            if self.phased_config.phase_trigger_type == "percent":
                trigger_pct = self.phased_config.phase_trigger_value * i
                if is_long:
                    trigger = entry_price * (1 + trigger_pct / 100)
                else:
                    trigger = entry_price * (1 - trigger_pct / 100)
            elif self.phased_config.phase_trigger_type == "atr":
                # ATR-based triggers (if ATR is available)
                atr_trigger = self._calculate_atr_trigger(df, entry_bar, i, is_long)
                trigger = atr_trigger if atr_trigger is not None else entry_price
            else:
                # Points-based triggers
                if is_long:
                    trigger = entry_price + (self.phased_config.phase_trigger_value * i)
                else:
                    trigger = entry_price - (self.phased_config.phase_trigger_value * i)

            triggers.append(trigger)

        return triggers

    def _calculate_atr_trigger(self, df: pd.DataFrame, entry_bar: int,
                              phase_num: int, is_long: bool) -> Optional[float]:
        """Calculate ATR-based trigger levels"""
        try:
            # Calculate ATR if not present
            if 'ATR' not in df.columns and 'atr' not in df.columns:
                atr_period = 14
                high_col = 'High' if 'High' in df.columns else 'high'
                low_col = 'Low' if 'Low' in df.columns else 'low'
                close_col = 'Close' if 'Close' in df.columns else 'close'

                df_temp = df.copy()
                df_temp['tr'] = np.maximum(
                    df_temp[high_col] - df_temp[low_col],
                    np.maximum(
                        abs(df_temp[high_col] - df_temp[close_col].shift(1)),
                        abs(df_temp[low_col] - df_temp[close_col].shift(1))
                    )
                )
                df_temp['ATR'] = df_temp['tr'].rolling(window=atr_period).mean()
                atr = df_temp['ATR'].iloc[entry_bar]
            else:
                atr_col = 'ATR' if 'ATR' in df.columns else 'atr'
                atr = df[atr_col].iloc[entry_bar]

            if pd.isna(atr) or atr <= 0:
                return None

            entry_price = df['Close'].iloc[entry_bar] if 'Close' in df.columns else df['close'].iloc[entry_bar]
            atr_multiplier = self.phased_config.phase_trigger_value * phase_num

            if is_long:
                return entry_price + (atr * atr_multiplier)
            else:
                return entry_price - (atr * atr_multiplier)

        except Exception as e:
            print(f"[PHASED_STRATEGY] Error calculating ATR trigger: {e}")
            return None

    def signals_to_phased_trades(self, signals: pd.Series, df: pd.DataFrame,
                                start_bar: int = 0, symbol: str = "DEFAULT") -> TradeCollection:
        """
        Convert signals to trades using phased execution engine
        """
        if not self.phased_config.enabled:
            # Fallback to standard processing
            return self.signals_to_trades_with_lag(signals, df, start_bar)

        # Use phased execution engine
        trade_records = self.execution_engine.execute_signals_with_phases(
            signals=signals,
            df=df,
            symbol=symbol
        )

        # Convert to TradeData objects
        trades = []
        for record in trade_records:
            trade = TradeData(
                bar_index=record.get('execution_bar', 0),
                trade_type=record.get('trade_type', 'UNKNOWN'),
                price=record.get('execution_price_adjusted', 0.0),
                trade_id=record.get('trade_id', 0),
                timestamp=record.get('timestamp'),
                pnl=record.get('pnl_percent', 0.0),
                strategy=self.name
            )

            # Add phased entry specific attributes
            trade.phase_number = record.get('phase_number', 1)
            trade.is_phased = record.get('is_phased_entry', False) or record.get('is_phased_exit', False)
            trade.total_phases = record.get('total_phases', 1)
            trade.average_entry_price = record.get('average_entry_price', trade.price)
            trade.phase_trigger_price = record.get('phase_trigger_price')
            trade.total_position_size = record.get('total_position_size', record.get('size', 0))

            trades.append(trade)

        return TradeCollection(trades if trades else [])

    def get_phased_performance_metrics(self) -> Dict:
        """Get performance metrics specific to phased entries"""
        base_stats = self.execution_engine.get_phased_statistics()

        # Add strategy-specific metrics
        enhanced_stats = {
            **base_stats,
            'strategy_name': self.name,
            'phased_enabled': self.phased_config.enabled,
            'max_phases_configured': self.phased_config.max_phases,
            'trigger_type': self.phased_config.phase_trigger_type,
            'trigger_value': self.phased_config.phase_trigger_value
        }

        return enhanced_stats

    def run_backtest_with_phases(self, df: pd.DataFrame, symbol: str = "DEFAULT") -> Tuple[TradeCollection, Dict]:
        """
        Run a complete backtest with phased entry support

        Returns:
            Tuple of (TradeCollection, performance_metrics)
        """
        # Generate signals
        signals = self.generate_signals(df)

        # Convert to trades using phased logic
        trades = self.signals_to_phased_trades(signals, df, symbol=symbol)

        # Get performance metrics
        performance = self.get_phased_performance_metrics()

        return trades, performance

    def optimize_phase_parameters(self, df: pd.DataFrame, param_ranges: Dict) -> Dict:
        """
        Optimize phased entry parameters using grid search

        Args:
            df: Price data for optimization
            param_ranges: Dictionary of parameter ranges to test
                {
                    'phase_trigger_value': [1.0, 1.5, 2.0, 2.5],
                    'max_phases': [2, 3, 4],
                    'initial_size_percent': [25, 33, 40, 50]
                }

        Returns:
            Dictionary with optimal parameters and performance metrics
        """
        best_performance = -np.inf
        best_params = {}
        results = []

        # Generate all parameter combinations
        import itertools
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        for param_combo in itertools.product(*param_values):
            # Update configuration
            old_config = {}
            for name, value in zip(param_names, param_combo):
                old_config[name] = getattr(self.phased_config, name)
                setattr(self.phased_config, name, value)

            try:
                # Run backtest with current parameters
                trades, metrics = self.run_backtest_with_phases(df)

                # Calculate performance score (customize as needed)
                if trades.trades:
                    total_return = sum(t.pnl or 0 for t in trades.trades)
                    win_rate = len([t for t in trades.trades if (t.pnl or 0) > 0]) / len(trades.trades)
                    performance_score = total_return * win_rate  # Simple score
                else:
                    performance_score = -np.inf

                result = {
                    'params': dict(zip(param_names, param_combo)),
                    'performance_score': performance_score,
                    'total_trades': len(trades.trades),
                    'metrics': metrics
                }
                results.append(result)

                # Track best performance
                if performance_score > best_performance:
                    best_performance = performance_score
                    best_params = dict(zip(param_names, param_combo))

            except Exception as e:
                print(f"[PHASED_STRATEGY] Error in optimization: {e}")
            finally:
                # Restore original configuration
                for name, value in old_config.items():
                    setattr(self.phased_config, name, value)

        return {
            'best_params': best_params,
            'best_performance': best_performance,
            'all_results': results
        }