"""
Adapter layer between existing strategy_runner.py and new unified execution engine
Allows gradual migration without breaking existing functionality
"""

import sys
import os
import yaml
from typing import Dict, Optional, Any, Union
import pandas as pd

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.standalone_execution import ExecutionConfig
from core.trade_types import TradeRecord, TradeRecordCollection
from strategies.strategy_wrapper import StrategyFactory, WrappedStrategy
from strategies.base import TradingStrategy
from data.trade_data import TradeData, TradeCollection


class StrategyRunnerAdapter:
    """
    Adapter that routes strategy execution through either:
    1. New unified execution engine (if enabled)
    2. Legacy execution path (default)
    """

    def __init__(self, config_path: str = None):
        """Initialize adapter with configuration"""
        self.config = self._load_config(config_path)
        self.use_unified_engine = self.config.get('use_unified_engine', False)
        self.execution_config = None

        if self.use_unified_engine:
            self.execution_config = ExecutionConfig.from_yaml(config_path)
            print("[ADAPTER] Using unified execution engine")
        else:
            print("[ADAPTER] Using legacy execution path")

    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration with unified engine flag"""
        if config_path is None:
            # Try common locations
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
                # Check for unified engine flag
                return config
        else:
            return {'use_unified_engine': False}

    def run_strategy(self,
                    strategy_name: str,
                    parameters: Dict[str, Any],
                    df: pd.DataFrame) -> Union[TradeCollection, TradeRecordCollection]:
        """
        Run strategy with appropriate execution engine

        Args:
            strategy_name: Name of strategy to run
            parameters: Strategy parameters
            df: Price data DataFrame

        Returns:
            TradeCollection (legacy) or TradeRecordCollection (unified)
        """
        if self.use_unified_engine:
            return self._run_unified(strategy_name, parameters, df)
        else:
            return self._run_legacy(strategy_name, parameters, df)

    def _run_unified(self,
                    strategy_name: str,
                    parameters: Dict[str, Any],
                    df: pd.DataFrame) -> TradeRecordCollection:
        """Run strategy using unified execution engine"""
        print(f"[ADAPTER] Running {strategy_name} with unified engine")

        # Create wrapped strategy
        wrapped_strategy = StrategyFactory.create_from_name(
            strategy_name,
            parameters,
            self.execution_config
        )

        # Execute trades with unified engine
        trades = wrapped_strategy.execute_trades(df)

        # Calculate indicators for display
        indicators = wrapped_strategy.calculate_indicators(df)

        # Store for later retrieval
        self.last_indicators = indicators
        self.last_metadata = wrapped_strategy.metadata

        print(f"[ADAPTER] Generated {len(trades)} trades with unified engine")
        return trades

    def _run_legacy(self,
                   strategy_name: str,
                   parameters: Dict[str, Any],
                   df: pd.DataFrame) -> TradeCollection:
        """Run strategy using legacy execution path"""
        print(f"[ADAPTER] Running {strategy_name} with legacy engine")

        # Import legacy strategies
        from strategies.sma_crossover import SMACrossoverStrategy
        from strategies.rsi_momentum import RSIMomentumStrategy
        from strategies.enhanced_sma_crossover import EnhancedSMACrossoverStrategy

        # Create strategy instance
        strategy = None
        if strategy_name.lower() in ['sma', 'sma_crossover']:
            strategy = SMACrossoverStrategy(
                fast_period=parameters.get('fast_period', 10),
                slow_period=parameters.get('slow_period', 30),
                long_only=parameters.get('long_only', True)
            )
        elif strategy_name.lower() in ['rsi', 'rsi_momentum']:
            strategy = RSIMomentumStrategy(
                period=parameters.get('rsi_period', 14),
                oversold=parameters.get('oversold', 30),
                overbought=parameters.get('overbought', 70),
                long_only=parameters.get('long_only', True)
            )
        elif strategy_name.lower() in ['enhanced_sma']:
            strategy = EnhancedSMACrossoverStrategy(
                fast_period=parameters.get('fast_period', 10),
                slow_period=parameters.get('slow_period', 30),
                long_only=parameters.get('long_only', True)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Generate signals
        signals = strategy.generate_signals(df)

        # Convert to trades based on strategy type
        if hasattr(strategy, 'signals_to_trades_with_lag'):
            # Enhanced strategy with lag support
            trades = strategy.signals_to_trades_with_lag(signals, df)
        else:
            # Standard strategy
            trades = strategy.signals_to_trades(signals, df)

        print(f"[ADAPTER] Generated {len(trades)} trades with legacy engine")
        return trades

    def convert_to_legacy(self, trades: TradeRecordCollection) -> TradeCollection:
        """
        Convert unified trade records to legacy format for compatibility

        Args:
            trades: TradeRecordCollection from unified engine

        Returns:
            TradeCollection compatible with existing UI
        """
        if isinstance(trades, TradeCollection):
            return trades  # Already legacy format

        # Convert each TradeRecord to legacy TradeData
        legacy_trades = []
        for trade in trades:
            legacy_trade = TradeData(
                bar_index=trade.bar_index,
                trade_type=trade.trade_type,
                price=trade.price,
                trade_id=trade.trade_id,
                timestamp=trade.timestamp,
                pnl=trade.pnl_points,  # Legacy uses points
                strategy=trade.strategy
            )

            # Add percentage P&L as custom attribute for enhanced display
            if hasattr(trade, 'pnl_percent'):
                legacy_trade.pnl_percent = trade.pnl_percent

            legacy_trades.append(legacy_trade)

        return TradeCollection(legacy_trades)

    def get_indicators(self) -> Dict:
        """Get calculated indicators from last run"""
        return getattr(self, 'last_indicators', {})

    def get_metadata(self) -> Optional[Any]:
        """Get strategy metadata from last run"""
        return getattr(self, 'last_metadata', None)

    def toggle_unified_engine(self, enable: bool):
        """
        Enable or disable unified execution engine

        Args:
            enable: True to use unified engine, False for legacy
        """
        self.use_unified_engine = enable
        if enable and not self.execution_config:
            self.execution_config = ExecutionConfig.from_yaml()

        print(f"[ADAPTER] Unified engine {'enabled' if enable else 'disabled'}")


class StrategyRunnerWidget:
    """
    Widget adapter that can be used as drop-in replacement for existing StrategyRunner
    Routes execution through the adapter layer
    """

    def __init__(self, original_widget=None):
        """Initialize with reference to original widget if available"""
        self.original_widget = original_widget
        self.adapter = StrategyRunnerAdapter()

    def run_strategy_with_adapter(self, strategy_name: str, parameters: Dict, df: pd.DataFrame):
        """Run strategy through adapter and emit appropriate signals"""
        # Run through adapter
        trades = self.adapter.run_strategy(strategy_name, parameters, df)

        # Convert to legacy format if needed for UI compatibility
        if isinstance(trades, TradeRecordCollection):
            legacy_trades = self.adapter.convert_to_legacy(trades)
        else:
            legacy_trades = trades

        # Emit signal if widget available
        if self.original_widget and hasattr(self.original_widget, 'trades_generated'):
            self.original_widget.trades_generated.emit(legacy_trades)

        # Emit indicators if available
        indicators = self.adapter.get_indicators()
        if indicators and self.original_widget and hasattr(self.original_widget, 'indicators_calculated'):
            self.original_widget.indicators_calculated.emit(indicators)

        return legacy_trades

    def set_unified_engine(self, enable: bool):
        """Enable or disable unified engine"""
        self.adapter.toggle_unified_engine(enable)