"""Enhanced trading strategy base with execution price formulas and signal lag tracking"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import sys
import os
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.trade_data import TradeData, TradeCollection


class EnhancedTradeData(TradeData):
    """Extended trade data with execution details"""
    def __init__(self,
                 bar_index: int,
                 trade_type: str,
                 price: float,
                 signal_bar: int = None,
                 bars_since_signal: int = 0,
                 execution_formula: str = None,
                 signal_price: float = None,
                 **kwargs):
        super().__init__(bar_index, trade_type, price, **kwargs)
        self.signal_bar = signal_bar if signal_bar is not None else bar_index
        self.bars_since_signal = bars_since_signal
        self.execution_formula = execution_formula
        self.signal_price = signal_price
        # Calculate lag: execution bar - signal bar
        self.lag = bar_index - self.signal_bar if signal_bar is not None else 0


class EnhancedTradingStrategy(ABC):
    """Enhanced base class with execution price formulas and lag tracking"""

    def __init__(self, name: str = "Strategy", config_path: str = None):
        self.name = name
        self.parameters = {}
        self.config = self.load_config(config_path)

        # Extract execution settings
        backtest_config = self.config.get('backtest', {})
        self.signal_lag = backtest_config.get('signal_lag', 1)
        self.execution_price_type = backtest_config.get('execution_price', 'close')
        self.buy_execution_formula = backtest_config.get('buy_execution_formula', 'C')
        self.sell_execution_formula = backtest_config.get('sell_execution_formula', 'C')

        print(f"[ENHANCED_STRATEGY] Initialized with:")
        print(f"  - Signal lag: {self.signal_lag} bars")
        print(f"  - Execution type: {self.execution_price_type}")
        print(f"  - Buy formula: {self.buy_execution_formula}")
        print(f"  - Sell formula: {self.sell_execution_formula}")

    def load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file"""
        if config_path is None:
            # Try to find config.yaml in common locations
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
                print(f"[ENHANCED_STRATEGY] Loaded config from {config_path}")
                return config
        else:
            print(f"[ENHANCED_STRATEGY] No config found, using defaults")
            return {
                'backtest': {
                    'signal_lag': 1,
                    'execution_price': 'close',
                    'buy_execution_formula': 'C',
                    'sell_execution_formula': 'C'
                }
            }

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from price data"""
        pass

    def calculate_execution_price(self, df: pd.DataFrame, bar_idx: int,
                                 is_buy: bool, signal_bar_idx: int = None) -> Tuple[float, str, int]:
        """
        Calculate execution price using configured formula

        Returns:
            Tuple of (execution_price, formula_used, bars_since_signal)
        """
        # Handle signal lag
        execution_bar = min(bar_idx + self.signal_lag, len(df) - 1)

        # Get OHLC for execution bar
        O = df['Open'].iloc[execution_bar] if 'Open' in df.columns else df['open'].iloc[execution_bar]
        H = df['High'].iloc[execution_bar] if 'High' in df.columns else df['high'].iloc[execution_bar]
        L = df['Low'].iloc[execution_bar] if 'Low' in df.columns else df['low'].iloc[execution_bar]
        C = df['Close'].iloc[execution_bar] if 'Close' in df.columns else df['close'].iloc[execution_bar]

        # Select formula based on trade direction
        formula = self.buy_execution_formula if is_buy else self.sell_execution_formula

        # Calculate bars since signal
        signal_bar = signal_bar_idx if signal_bar_idx is not None else bar_idx
        bars_since_signal = execution_bar - signal_bar

        # Execute formula
        if self.execution_price_type == 'formula':
            try:
                # Safe evaluation with only OHLC variables
                price = eval(formula, {"__builtins__": {}}, {'O': O, 'H': H, 'L': L, 'C': C})
            except Exception as e:
                print(f"[ENHANCED_STRATEGY] Formula error: {e}, using close price")
                price = C
        elif self.execution_price_type == 'open':
            price = O
        elif self.execution_price_type == 'high':
            price = H
        elif self.execution_price_type == 'low':
            price = L
        else:  # default to close
            price = C

        return price, formula, bars_since_signal

    def signals_to_trades_with_lag(self, signals: pd.Series, df: pd.DataFrame,
                                   start_bar: int = 0) -> TradeCollection:
        """
        Convert signals to trades with proper execution lag and price formulas
        """
        trades = []
        position = 0
        trade_id = 0
        entry_price = None
        entry_signal_bar = None

        # Check for DateTime column
        has_datetime = 'DateTime' in df.columns or 'timestamp' in df.columns

        print(f"[ENHANCED_STRATEGY] Processing {len(signals)} signals with lag={self.signal_lag}")

        for i in range(len(signals)):
            signal = signals.iloc[i]

            # Skip if no change in position
            if signal == position:
                continue

            # Signal detected at this bar
            signal_bar_idx = i

            # Calculate execution bar with lag
            execution_bar_idx = min(i + self.signal_lag, len(df) - 1)

            # Skip if execution would be beyond data
            if execution_bar_idx >= len(df):
                print(f"[ENHANCED_STRATEGY] Skipping signal at bar {i}, execution would be beyond data")
                continue

            # Get timestamp for execution bar
            timestamp = None
            if has_datetime:
                if 'DateTime' in df.columns:
                    timestamp = pd.Timestamp(df['DateTime'].iloc[execution_bar_idx])
                elif 'timestamp' in df.columns:
                    timestamp = pd.Timestamp(df['timestamp'].iloc[execution_bar_idx])

            # Exit current position if needed
            if position != 0:
                is_buy = False  # Exits are sells
                trade_type = 'SELL' if position > 0 else 'COVER'

                # Calculate execution price with formula
                exec_price, formula, bars_since = self.calculate_execution_price(
                    df, signal_bar_idx, is_buy, entry_signal_bar
                )

                # Calculate P&L
                pnl = None
                if entry_price is not None:
                    if position > 0:  # Long position
                        pnl = exec_price - entry_price
                    else:  # Short position
                        pnl = entry_price - exec_price

                # Get signal bar price for reference
                signal_price = df['Close'].iloc[signal_bar_idx] if 'Close' in df.columns else df['close'].iloc[signal_bar_idx]

                trade = EnhancedTradeData(
                    bar_index=start_bar + execution_bar_idx,
                    trade_type=trade_type,
                    price=exec_price,
                    signal_bar=start_bar + signal_bar_idx,
                    bars_since_signal=bars_since,
                    execution_formula=formula,
                    signal_price=signal_price,
                    trade_id=trade_id,
                    timestamp=timestamp,
                    pnl=pnl,
                    strategy=self.name
                )
                trades.append(trade)
                trade_id += 1
                entry_price = None
                entry_signal_bar = None

            # Enter new position if signal is non-zero
            if signal != 0:
                is_buy = signal > 0
                trade_type = 'BUY' if is_buy else 'SHORT'

                # Calculate execution price with formula
                exec_price, formula, bars_since = self.calculate_execution_price(
                    df, signal_bar_idx, is_buy
                )

                # Store for P&L calculation
                entry_price = exec_price
                entry_signal_bar = signal_bar_idx

                # Get signal bar price for reference
                signal_price = df['Close'].iloc[signal_bar_idx] if 'Close' in df.columns else df['close'].iloc[signal_bar_idx]

                trade = EnhancedTradeData(
                    bar_index=start_bar + execution_bar_idx,
                    trade_type=trade_type,
                    price=exec_price,
                    signal_bar=start_bar + signal_bar_idx,
                    bars_since_signal=bars_since,
                    execution_formula=formula,
                    signal_price=signal_price,
                    trade_id=trade_id,
                    timestamp=timestamp,
                    strategy=self.name
                )
                trades.append(trade)
                trade_id += 1

            position = signal

        # Close final position if still open
        if position != 0 and len(df) > 0:
            signal_bar_idx = len(df) - 1 - self.signal_lag
            execution_bar_idx = len(df) - 1

            is_buy = False
            trade_type = 'SELL' if position > 0 else 'COVER'

            # Calculate execution price
            exec_price, formula, bars_since = self.calculate_execution_price(
                df, signal_bar_idx, is_buy, entry_signal_bar
            )

            # Get timestamp
            timestamp = None
            if has_datetime:
                if 'DateTime' in df.columns:
                    timestamp = pd.Timestamp(df['DateTime'].iloc[execution_bar_idx])
                elif 'timestamp' in df.columns:
                    timestamp = pd.Timestamp(df['timestamp'].iloc[execution_bar_idx])

            # Calculate final P&L
            pnl = None
            if entry_price is not None:
                if position > 0:
                    pnl = exec_price - entry_price
                else:
                    pnl = entry_price - exec_price

            signal_price = df['Close'].iloc[signal_bar_idx] if 'Close' in df.columns else df['close'].iloc[signal_bar_idx]

            trade = EnhancedTradeData(
                bar_index=start_bar + execution_bar_idx,
                trade_type=trade_type,
                price=exec_price,
                signal_bar=start_bar + signal_bar_idx,
                bars_since_signal=bars_since,
                execution_formula=formula,
                signal_price=signal_price,
                trade_id=trade_id,
                timestamp=timestamp,
                pnl=pnl,
                strategy=self.name
            )
            trades.append(trade)

        print(f"[ENHANCED_STRATEGY] Generated {len(trades)} trades with execution lag")
        return TradeCollection(trades if trades else [])