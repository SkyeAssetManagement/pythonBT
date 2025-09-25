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

# Import TWAP system
try:
    from core.time_based_twap_execution import TimeBasedTWAPConfig
    from core.vectorbt_twap_adapter import VectorBTTWAPAdapter
    from core.optimized_twap_adapter import OptimizedTWAPAdapter, OptimizedTWAPConfig
    TWAP_AVAILABLE = True
    OPTIMIZED_TWAP_AVAILABLE = True
    print("[ADAPTER] TWAP system available (with optimization)")
except ImportError as e:
    try:
        from core.time_based_twap_execution import TimeBasedTWAPConfig
        from core.vectorbt_twap_adapter import VectorBTTWAPAdapter
        TWAP_AVAILABLE = True
        OPTIMIZED_TWAP_AVAILABLE = False
        print("[ADAPTER] TWAP system available (standard only)")
    except ImportError as e2:
        TWAP_AVAILABLE = False
        OPTIMIZED_TWAP_AVAILABLE = False
        print(f"[ADAPTER] TWAP system not available: {e2}")


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
        self.use_pure_array_processing = self.config.get('use_pure_array_processing', True)
        self.execution_config = None

        # Initialize TWAP system if available and enabled
        self.twap_adapter = None
        self.optimized_twap_adapter = None
        self.use_twap = False
        if TWAP_AVAILABLE:
            twap_config = self.config.get('time_based_twap', {})
            if twap_config.get('enabled', False):
                try:
                    # Initialize both standard and optimized adapters
                    self.twap_adapter = VectorBTTWAPAdapter(TimeBasedTWAPConfig.from_dict(twap_config))

                    if OPTIMIZED_TWAP_AVAILABLE:
                        # Create optimized config with large dataset settings
                        optimized_config = OptimizedTWAPConfig.from_dict(twap_config)
                        optimized_config.chunk_size = 5000  # Process 5k signals at a time
                        optimized_config.max_signals_before_chunking = 10000  # Use chunking for 10k+ signals
                        optimized_config.skip_vectorbt_portfolio = False  # Try VectorBT first, fallback if needed

                        self.optimized_twap_adapter = OptimizedTWAPAdapter(optimized_config)
                        print(f"[ADAPTER] Optimized TWAP system enabled (min_time: {twap_config.get('minimum_execution_minutes', 5.0)} minutes, chunk_size: 5000)")
                    else:
                        print(f"[ADAPTER] Standard TWAP system enabled (min_time: {twap_config.get('minimum_execution_minutes', 5.0)} minutes)")

                    self.use_twap = True
                except Exception as e:
                    print(f"[ADAPTER] Failed to initialize TWAP system: {e}")

        if self.use_unified_engine and not self.use_twap:
            self.execution_config = ExecutionConfig.from_yaml(config_path)
            engine_type = "Pure Array (O(1))" if self.use_pure_array_processing else "Standalone (O(n))"
            print(f"[ADAPTER] Using unified execution engine with {engine_type} processing")
        elif not self.use_twap:
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
        if self.use_twap:
            return self._run_twap(strategy_name, parameters, df)
        elif self.use_unified_engine:
            return self._run_unified(strategy_name, parameters, df)
        else:
            return self._run_legacy(strategy_name, parameters, df)

    def _run_unified(self,
                    strategy_name: str,
                    parameters: Dict[str, Any],
                    df: pd.DataFrame) -> TradeRecordCollection:
        """Run strategy using unified execution engine"""
        print(f"[ADAPTER] Running {strategy_name} with unified engine")

        # Create wrapped strategy with pure array processing flag
        wrapped_strategy = StrategyFactory.create_from_name(
            strategy_name,
            parameters,
            self.execution_config,
            use_pure_array=self.use_pure_array_processing
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

        # Use VectorBT for proper P&L calculation
        trades = self._calculate_trades_with_vectorbt(signals, df, strategy_name)

        print(f"[ADAPTER] Generated {len(trades)} trades with vectorized P&L calculation")
        return trades

    def _calculate_trades_with_vectorbt(self, signals: pd.Series, df: pd.DataFrame, strategy_name: str) -> TradeCollection:
        """Calculate trades and P&L using VectorBT for proper vectorized calculations"""
        try:
            import vectorbtpro as vbt
        except ImportError:
            print("[ADAPTER] VectorBT not available, falling back to manual calculation")
            return self._fallback_to_manual_trades(signals, df, strategy_name)

        # Convert signals to entry/exit signals
        entries, exits = self._convert_signals_to_entries_exits(signals)

        # Get close prices
        close = df['Close'] if 'Close' in df.columns else df['close']

        # Create VectorBT portfolio with proper P&L calculation
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            freq='1D',  # Adjust frequency as needed
            init_cash=1.0,  # $1 initial investment for percentage calculation
            fees=0.0,  # No fees for now
            slippage=0.0  # No slippage for now
        )

        # Convert VectorBT trades to our format
        from trading.integration.vbt_integration import VBTTradeLoader
        loader = VBTTradeLoader()

        # Prepare price data for timestamp conversion
        price_data = {
            'datetime': df.index.values if hasattr(df.index, 'values') else None,
            'close': close.values
        }

        trades = loader.load_vbt_trades(portfolio, price_data)

        # Add strategy name to trades
        for trade in trades:
            trade.strategy = strategy_name
            # VectorBT calculates P&L in dollars, convert to percentage
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                # For $1 investment, P&L in dollars equals percentage return
                trade.pnl_percent = trade.pnl

        return trades

    def _convert_signals_to_entries_exits(self, signals: pd.Series) -> tuple:
        """Convert position signals to entry/exit signals"""
        entries = pd.Series(False, index=signals.index)
        exits = pd.Series(False, index=signals.index)

        # Find position changes
        position_changes = signals.diff()

        # Entry signals: going from 0 to 1 (long) or 0 to -1 (short)
        long_entries = (position_changes == 1)
        short_entries = (position_changes == -1)
        entries = long_entries | short_entries

        # Exit signals: going from non-zero to 0, or changing position direction
        position_exits = ((signals.shift(1) != 0) & (signals == 0))
        direction_changes = ((signals.shift(1) > 0) & (signals < 0)) | ((signals.shift(1) < 0) & (signals > 0))
        exits = position_exits | direction_changes

        return entries, exits

    def _fallback_to_manual_trades(self, signals: pd.Series, df: pd.DataFrame, strategy_name: str) -> TradeCollection:
        """Fallback to manual trade calculation if VectorBT is not available"""
        print("[ADAPTER] Using fallback manual trade calculation")
        # This would use the original signals_to_trades method
        # For now, return empty collection
        from data.trade_data import TradeCollection
        return TradeCollection([])

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

    def _run_twap(self,
                 strategy_name: str,
                 parameters: Dict[str, Any],
                 df: pd.DataFrame) -> TradeCollection:
        """Run strategy using volume-weighted TWAP execution"""
        print(f"[ADAPTER] Running {strategy_name} with volume-weighted TWAP execution")

        # Generate signals using legacy strategy
        from strategies.sma_crossover import SMACrossoverStrategy

        if strategy_name.lower() in ['sma', 'sma_crossover']:
            strategy = SMACrossoverStrategy(
                fast_period=parameters.get('fast_period', 10),
                slow_period=parameters.get('slow_period', 30),
                long_only=parameters.get('long_only', False)  # Enable both long and short for TWAP
            )
        else:
            raise ValueError(f"TWAP execution not yet supported for strategy: {strategy_name}")

        # Generate signals
        signals = strategy.generate_signals(df)
        long_signals = (signals > 0)
        short_signals = (signals < 0)

        # Validate DataFrame has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"[ADAPTER] Missing required columns for TWAP: {missing_columns}")
            print(f"[ADAPTER] Available columns: {list(df.columns)}")

            # Try to map common alternative column names
            column_mapping = {
                'Close': 'close', 'CLOSE': 'close',
                'Open': 'open', 'OPEN': 'open',
                'High': 'high', 'HIGH': 'high',
                'Low': 'low', 'LOW': 'low',
                'Volume': 'volume', 'VOLUME': 'volume',
                'Vol': 'volume', 'VOL': 'volume'
            }

            # Create a copy with standardized column names
            df_twap = df.copy()
            for old_name, new_name in column_mapping.items():
                if old_name in df_twap.columns and new_name not in df_twap.columns:
                    df_twap[new_name] = df_twap[old_name]
                    print(f"[ADAPTER] Mapped column '{old_name}' -> '{new_name}'")
        else:
            df_twap = df

        # Execute with TWAP
        signal_lag = self.config.get('backtest', {}).get('signal_lag', 2)
        position_size = self.config.get('backtest', {}).get('position_size', 1.0)
        fees = self.config.get('backtest', {}).get('fees', 0.0)

        try:
            twap_results = self.twap_adapter.execute_portfolio_with_twap(
                df=df_twap,
                long_signals=long_signals,
                short_signals=short_signals,
                signal_lag=signal_lag,
                size=position_size,
                fees=fees
            )
        except Exception as e:
            print(f"[ADAPTER] TWAP execution failed: {e}")
            print(f"[ADAPTER] DataFrame shape: {df_twap.shape}")
            print(f"[ADAPTER] DataFrame columns: {list(df_twap.columns)}")
            if hasattr(df_twap, 'index'):
                print(f"[ADAPTER] DataFrame index type: {type(df_twap.index)}")
            raise

        # Store TWAP metadata for trade panel
        self.last_twap_results = twap_results
        # Calculate indicators if method exists
        if hasattr(strategy, 'calculate_indicators'):
            self.last_indicators = strategy.calculate_indicators(df)
        else:
            # Generate basic indicators for display
            self.last_indicators = {
                f'SMA_{strategy.fast_period}': df['close'].rolling(strategy.fast_period).mean(),
                f'SMA_{strategy.slow_period}': df['close'].rolling(strategy.slow_period).mean()
            }

        # Convert TWAP trade metadata to TradeCollection
        trade_metadata = twap_results['trade_metadata']
        trades = []

        for _, row in trade_metadata.iterrows():
            # Map direction to trade_type
            trade_type = 'BUY' if row['direction'] == 'long' else 'SELL'

            trade = TradeData(
                bar_index=int(row['signal_bar']),
                trade_type=trade_type,
                price=row['twap_price'],
                timestamp=df.index[row['signal_bar']] if row['signal_bar'] < len(df) else df.index[-1],
                size=row['total_position_size'],
                strategy=strategy_name
            )

            # Add TWAP metadata
            trade.metadata = {
                'exec_bars': row['exec_bars'],
                'execution_time_minutes': row['execution_time_minutes'],
                'num_phases': row['num_phases'],
                'total_volume': row['total_volume']
            }
            trades.append(trade)

        trade_collection = TradeCollection(trades)
        print(f"[ADAPTER] Generated {len(trades)} TWAP trades with execBars data")

        return trade_collection

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