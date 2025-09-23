"""
Strategy wrapper that adds metadata and indicator definitions to existing strategies
Provides a uniform interface while maintaining backward compatibility
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import sys
import os

# Import base strategies
try:
    from .base import TradingStrategy
    from .sma_crossover import SMACrossoverStrategy
    from .rsi_momentum import RSIMomentumStrategy
except ImportError:
    # Handle both module and direct import styles
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from base import TradingStrategy
    from sma_crossover import SMACrossoverStrategy
    from rsi_momentum import RSIMomentumStrategy

# Import execution engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.standalone_execution import StandaloneExecutionEngine, ExecutionConfig
from core.trade_types import TradeRecord, TradeRecordCollection


@dataclass
class IndicatorDefinition:
    """Definition for a strategy indicator"""
    name: str
    display_name: str
    plot_type: str  # 'line', 'scatter', 'bar', 'area'
    color: str  # Hex color or name
    line_style: str = 'solid'  # 'solid', 'dashed', 'dotted'
    line_width: float = 1.0
    opacity: float = 1.0
    y_axis: str = 'price'  # 'price' or 'indicator'
    visible: bool = True
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategyMetadata:
    """Metadata for a trading strategy"""
    name: str
    display_name: str
    description: str
    category: str  # 'trend', 'momentum', 'volatility', 'volume'
    parameters: Dict[str, Any]
    indicators: List[IndicatorDefinition]
    requires_volume: bool = False
    min_bars_required: int = 1
    supports_long: bool = True
    supports_short: bool = True


class WrappedStrategy:
    """
    Wrapper for existing strategies that adds metadata and unified interface
    Maintains backward compatibility while enabling new features
    """

    def __init__(self,
                 strategy: TradingStrategy,
                 metadata: StrategyMetadata,
                 execution_config: ExecutionConfig = None):
        self.strategy = strategy
        self.metadata = metadata
        self.execution_config = execution_config or ExecutionConfig()
        self.execution_engine = StandaloneExecutionEngine(self.execution_config)

        # Cache for indicator values
        self._indicator_cache = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals - delegates to underlying strategy"""
        return self.strategy.generate_signals(df)

    def execute_trades(self, df: pd.DataFrame) -> TradeRecordCollection:
        """
        Generate and execute trades using the unified execution engine

        Args:
            df: Price data DataFrame

        Returns:
            TradeRecordCollection with executed trades
        """
        # Generate signals
        signals = self.generate_signals(df)

        # Execute with unified engine
        trade_records = self.execution_engine.execute_signals(signals, df)

        # Convert to TradeRecord objects
        trades = []
        for record_dict in trade_records:
            trade = TradeRecord(
                trade_id=record_dict['trade_id'],
                bar_index=record_dict['execution_bar'],
                trade_type=record_dict['trade_type'],
                price=record_dict['execution_price'],
                signal_bar=record_dict['signal_bar'],
                execution_bar=record_dict['execution_bar'],
                lag=record_dict['lag'],
                signal_price=record_dict['signal_price'],
                execution_formula=record_dict['formula'],
                size=record_dict.get('size', 1.0),
                pnl_points=record_dict.get('pnl_points'),
                pnl_percent=record_dict.get('pnl_percent'),
                timestamp=record_dict.get('timestamp'),
                strategy=self.metadata.name,
                strategy_params=self.metadata.parameters
            )
            trades.append(trade)

        return TradeRecordCollection(trades)

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all indicators defined in metadata

        Returns:
            Dictionary mapping indicator names to their values
        """
        indicators = {}

        # Get close prices
        close = df['Close'] if 'Close' in df.columns else df['close']

        # Calculate based on strategy type
        if isinstance(self.strategy, SMACrossoverStrategy):
            # Calculate SMAs
            fast_sma, slow_sma = self.strategy.calculate_smas(df)
            indicators['sma_fast'] = fast_sma
            indicators['sma_slow'] = slow_sma

        elif hasattr(self.strategy, 'calculate_indicators'):
            # Use strategy's own indicator calculation if available
            indicators = self.strategy.calculate_indicators(df)

        # Cache for reuse
        self._indicator_cache = indicators
        return indicators

    def get_indicator_values(self, indicator_name: str) -> Optional[pd.Series]:
        """Get cached indicator values by name"""
        return self._indicator_cache.get(indicator_name)

    def get_performance_metrics(self, trades: TradeRecordCollection) -> Dict:
        """Calculate strategy-specific performance metrics"""
        base_metrics = trades.get_metrics()

        # Add strategy-specific metrics
        base_metrics['strategy_name'] = self.metadata.name
        base_metrics['parameters'] = self.metadata.parameters

        return base_metrics

    def to_legacy_strategy(self) -> TradingStrategy:
        """Return the underlying legacy strategy for backward compatibility"""
        return self.strategy


class StrategyFactory:
    """Factory for creating wrapped strategies with metadata"""

    @staticmethod
    def create_sma_crossover(fast_period: int = 10,
                            slow_period: int = 30,
                            long_only: bool = True,
                            execution_config: ExecutionConfig = None) -> WrappedStrategy:
        """Create wrapped SMA crossover strategy"""

        # Create base strategy
        strategy = SMACrossoverStrategy(fast_period, slow_period, long_only)

        # Define metadata
        metadata = StrategyMetadata(
            name=f"SMA_{fast_period}_{slow_period}",
            display_name=f"SMA Crossover ({fast_period}/{slow_period})",
            description=f"Simple moving average crossover strategy with {fast_period} and {slow_period} period SMAs",
            category='trend',
            parameters={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'long_only': long_only
            },
            indicators=[
                IndicatorDefinition(
                    name='sma_fast',
                    display_name=f'SMA {fast_period}',
                    plot_type='line',
                    color='#00FF00',  # Green
                    line_style='solid',
                    line_width=1.5,
                    y_axis='price'
                ),
                IndicatorDefinition(
                    name='sma_slow',
                    display_name=f'SMA {slow_period}',
                    plot_type='line',
                    color='#FF0000',  # Red
                    line_style='solid',
                    line_width=1.5,
                    y_axis='price'
                )
            ],
            requires_volume=False,
            min_bars_required=max(fast_period, slow_period),
            supports_long=True,
            supports_short=not long_only
        )

        return WrappedStrategy(strategy, metadata, execution_config)

    @staticmethod
    def create_rsi_momentum(rsi_period: int = 14,
                           oversold: float = 30,
                           overbought: float = 70,
                           long_only: bool = True,
                           execution_config: ExecutionConfig = None) -> WrappedStrategy:
        """Create wrapped RSI momentum strategy"""

        # Create base strategy
        strategy = RSIMomentumStrategy(period=rsi_period, oversold=oversold,
                                      overbought=overbought, long_only=long_only)

        # Define metadata
        metadata = StrategyMetadata(
            name=f"RSI_{rsi_period}",
            display_name=f"RSI Momentum ({rsi_period})",
            description=f"RSI momentum strategy with {oversold}/{overbought} levels",
            category='momentum',
            parameters={
                'rsi_period': rsi_period,
                'oversold': oversold,
                'overbought': overbought,
                'long_only': long_only
            },
            indicators=[
                IndicatorDefinition(
                    name='rsi',
                    display_name=f'RSI {rsi_period}',
                    plot_type='line',
                    color='#FFA500',  # Orange
                    line_style='solid',
                    line_width=2.0,
                    y_axis='indicator'
                ),
                IndicatorDefinition(
                    name='rsi_oversold',
                    display_name='Oversold',
                    plot_type='line',
                    color='#00FF00',  # Green
                    line_style='dashed',
                    line_width=1.0,
                    y_axis='indicator'
                ),
                IndicatorDefinition(
                    name='rsi_overbought',
                    display_name='Overbought',
                    plot_type='line',
                    color='#FF0000',  # Red
                    line_style='dashed',
                    line_width=1.0,
                    y_axis='indicator'
                )
            ],
            requires_volume=False,
            min_bars_required=rsi_period + 1,
            supports_long=True,
            supports_short=not long_only
        )

        return WrappedStrategy(strategy, metadata, execution_config)

    @staticmethod
    def create_from_name(strategy_name: str,
                        parameters: Dict[str, Any] = None,
                        execution_config: ExecutionConfig = None) -> WrappedStrategy:
        """Create strategy by name with parameters"""
        parameters = parameters or {}

        if strategy_name.lower() in ['sma', 'sma_crossover']:
            return StrategyFactory.create_sma_crossover(
                fast_period=parameters.get('fast_period', 10),
                slow_period=parameters.get('slow_period', 30),
                long_only=parameters.get('long_only', True),
                execution_config=execution_config
            )

        elif strategy_name.lower() in ['rsi', 'rsi_momentum']:
            return StrategyFactory.create_rsi_momentum(
                rsi_period=parameters.get('rsi_period', 14),
                oversold=parameters.get('oversold', 30),
                overbought=parameters.get('overbought', 70),
                long_only=parameters.get('long_only', True),
                execution_config=execution_config
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    @staticmethod
    def list_available_strategies() -> List[Dict]:
        """List all available strategies with their metadata"""
        return [
            {
                'name': 'sma_crossover',
                'display_name': 'SMA Crossover',
                'category': 'trend',
                'description': 'Simple moving average crossover strategy',
                'default_parameters': {
                    'fast_period': 10,
                    'slow_period': 30,
                    'long_only': True
                }
            },
            {
                'name': 'rsi_momentum',
                'display_name': 'RSI Momentum',
                'category': 'momentum',
                'description': 'RSI momentum strategy with oversold/overbought levels',
                'default_parameters': {
                    'rsi_period': 14,
                    'oversold': 30,
                    'overbought': 70,
                    'long_only': True
                }
            }
        ]