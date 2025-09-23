"""
Standalone execution engine for trade execution with config.yaml support
Handles bar lag, execution price formulas, and trade record generation
Independent from visualization - can be tested separately
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import yaml
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExecutionConfig:
    """Configuration for trade execution"""
    signal_lag: int = 1
    execution_price: str = 'close'
    buy_execution_formula: str = 'C'
    sell_execution_formula: str = 'C'
    position_size: float = 1.0
    position_size_type: str = 'value'
    initial_cash: float = 100000.0
    fees: float = 0.0
    fixed_fees: float = 0.0
    slippage: float = 0.0

    @classmethod
    def from_yaml(cls, config_path: str = None) -> 'ExecutionConfig':
        """Load configuration from YAML file"""
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
                backtest_config = config.get('backtest', {})

                return cls(
                    signal_lag=backtest_config.get('signal_lag', 1),
                    execution_price=backtest_config.get('execution_price', 'close'),
                    buy_execution_formula=backtest_config.get('buy_execution_formula', 'C'),
                    sell_execution_formula=backtest_config.get('sell_execution_formula', 'C'),
                    position_size=backtest_config.get('position_size', 1.0),
                    position_size_type=backtest_config.get('position_size_type', 'value'),
                    initial_cash=backtest_config.get('initial_cash', 100000.0),
                    fees=backtest_config.get('fees', 0.0),
                    fixed_fees=backtest_config.get('fixed_fees', 0.0),
                    slippage=backtest_config.get('slippage', 0.0)
                )

        return cls()  # Return defaults if no config


class StandaloneExecutionEngine:
    """
    Execution engine that handles trade execution independently of visualization
    Can be tested and used separately from any UI components
    """

    def __init__(self, config: ExecutionConfig = None):
        """Initialize with execution configuration"""
        self.config = config or ExecutionConfig()
        print(f"[EXECUTION_ENGINE] Initialized with:")
        print(f"  Signal lag: {self.config.signal_lag} bars")
        print(f"  Execution price: {self.config.execution_price}")
        print(f"  Buy formula: {self.config.buy_execution_formula}")
        print(f"  Sell formula: {self.config.sell_execution_formula}")
        print(f"  Position size: {self.config.position_size} ({self.config.position_size_type})")

    def calculate_execution_price(self,
                                 df: pd.DataFrame,
                                 signal_bar: int,
                                 is_buy: bool) -> Tuple[float, int, str]:
        """
        Calculate execution price based on configuration

        Args:
            df: Price data DataFrame with OHLC columns
            signal_bar: Bar index where signal was generated
            is_buy: True for buy/cover, False for sell/short

        Returns:
            Tuple of (execution_price, execution_bar, formula_used)
        """
        # Calculate execution bar with lag
        execution_bar = min(signal_bar + self.config.signal_lag, len(df) - 1)

        # Get OHLC for execution bar
        O = self._get_price(df, execution_bar, 'Open')
        H = self._get_price(df, execution_bar, 'High')
        L = self._get_price(df, execution_bar, 'Low')
        C = self._get_price(df, execution_bar, 'Close')

        # Select formula
        formula = self.config.buy_execution_formula if is_buy else self.config.sell_execution_formula

        # Calculate price based on execution_price type
        if self.config.execution_price == 'formula':
            try:
                # Safe evaluation with only OHLC variables
                price = eval(formula, {"__builtins__": {}}, {'O': O, 'H': H, 'L': L, 'C': C})
            except Exception as e:
                print(f"[EXECUTION_ENGINE] Formula error: {e}, using close")
                price = C
                formula = 'C'
        elif self.config.execution_price == 'open':
            price = O
            formula = 'O'
        elif self.config.execution_price == 'high':
            price = H
            formula = 'H'
        elif self.config.execution_price == 'low':
            price = L
            formula = 'L'
        else:  # default to close
            price = C
            formula = 'C'

        return price, execution_bar, formula

    def _get_price(self, df: pd.DataFrame, bar_idx: int, price_type: str) -> float:
        """Get price from dataframe with column name flexibility"""
        col_variants = [price_type, price_type.lower()]
        for col in col_variants:
            if col in df.columns:
                return df[col].iloc[bar_idx]
        raise ValueError(f"Column {price_type} not found in DataFrame")

    def calculate_position_size(self,
                               price: float,
                               current_cash: float,
                               current_value: float) -> float:
        """
        Calculate position size based on configuration

        Args:
            price: Execution price
            current_cash: Available cash
            current_value: Current portfolio value

        Returns:
            Number of shares/contracts to trade
        """
        if self.config.position_size_type == 'value':
            # Fixed dollar amount
            return self.config.position_size / price
        elif self.config.position_size_type == 'amount':
            # Fixed number of shares
            return self.config.position_size
        elif self.config.position_size_type == 'percent':
            # Percentage of portfolio
            portfolio_value = current_cash + current_value
            value_to_invest = portfolio_value * (self.config.position_size / 100)
            return value_to_invest / price
        else:
            return 1.0  # Default to 1 share

    def apply_friction(self,
                      price: float,
                      size: float,
                      is_buy: bool) -> Tuple[float, float]:
        """
        Apply fees and slippage to execution

        Args:
            price: Base execution price
            size: Position size
            is_buy: Trade direction

        Returns:
            Tuple of (adjusted_price, total_fees)
        """
        # Apply slippage
        if self.config.slippage > 0:
            if is_buy:
                price = price * (1 + self.config.slippage)
            else:
                price = price * (1 - self.config.slippage)

        # Calculate fees
        trade_value = price * size
        percentage_fee = trade_value * self.config.fees
        total_fees = percentage_fee + self.config.fixed_fees

        return price, total_fees

    def execute_signals(self,
                       signals: pd.Series,
                       df: pd.DataFrame) -> List[Dict]:
        """
        Execute trading signals with proper lag and pricing

        Args:
            signals: Series of trading signals (1, -1, 0)
            df: DataFrame with OHLC price data

        Returns:
            List of trade records with full execution details
        """
        trades = []
        position = 0
        entry_price = None
        entry_bar = None
        current_cash = self.config.initial_cash
        trade_id = 0

        # Check for DateTime column
        has_datetime = 'DateTime' in df.columns or 'timestamp' in df.columns

        for i in range(len(signals)):
            signal = signals.iloc[i]

            # Skip if no change
            if signal == position:
                continue

            # Exit current position if needed
            if position != 0:
                is_buy = False  # Exits are sells
                trade_type = 'SELL' if position > 0 else 'COVER'

                # Calculate execution details
                exec_price, exec_bar, formula = self.calculate_execution_price(
                    df, i, is_buy
                )

                # Calculate P&L
                pnl_points = None
                pnl_percent = None
                if entry_price is not None:
                    if position > 0:  # Long position
                        pnl_points = exec_price - entry_price
                    else:  # Short position
                        pnl_points = entry_price - exec_price
                    pnl_percent = (pnl_points / entry_price) * 100

                # Get timestamp if available
                timestamp = self._get_timestamp(df, exec_bar, has_datetime)

                # Create trade record
                trade = {
                    'trade_id': trade_id,
                    'signal_bar': i,
                    'execution_bar': exec_bar,
                    'lag': exec_bar - i,
                    'trade_type': trade_type,
                    'signal_price': self._get_price(df, i, 'Close'),
                    'execution_price': exec_price,
                    'formula': formula,
                    'size': abs(position),
                    'pnl_points': pnl_points,
                    'pnl_percent': pnl_percent,
                    'timestamp': timestamp
                }
                trades.append(trade)
                trade_id += 1
                entry_price = None
                entry_bar = None

            # Enter new position if signal is non-zero
            if signal != 0:
                is_buy = signal > 0
                trade_type = 'BUY' if is_buy else 'SHORT'

                # Calculate execution details
                exec_price, exec_bar, formula = self.calculate_execution_price(
                    df, i, is_buy
                )

                # Apply friction
                exec_price_adjusted, fees = self.apply_friction(
                    exec_price,
                    self.config.position_size,
                    is_buy
                )

                # Store for P&L calculation
                entry_price = exec_price
                entry_bar = i

                # Get timestamp if available
                timestamp = self._get_timestamp(df, exec_bar, has_datetime)

                # Create trade record
                trade = {
                    'trade_id': trade_id,
                    'signal_bar': i,
                    'execution_bar': exec_bar,
                    'lag': exec_bar - i,
                    'trade_type': trade_type,
                    'signal_price': self._get_price(df, i, 'Close'),
                    'execution_price': exec_price,
                    'formula': formula,
                    'size': self.config.position_size,
                    'fees': fees,
                    'timestamp': timestamp
                }
                trades.append(trade)
                trade_id += 1

            position = signal

        # Close final position if open
        if position != 0 and len(df) > 0:
            signal_bar = len(df) - 1 - self.config.signal_lag
            is_buy = False
            trade_type = 'SELL' if position > 0 else 'COVER'

            exec_price, exec_bar, formula = self.calculate_execution_price(
                df, signal_bar, is_buy
            )

            # Calculate final P&L
            pnl_points = None
            pnl_percent = None
            if entry_price is not None:
                if position > 0:
                    pnl_points = exec_price - entry_price
                else:
                    pnl_points = entry_price - exec_price
                pnl_percent = (pnl_points / entry_price) * 100

            timestamp = self._get_timestamp(df, exec_bar, has_datetime)

            trade = {
                'trade_id': trade_id,
                'signal_bar': signal_bar,
                'execution_bar': exec_bar,
                'lag': exec_bar - signal_bar,
                'trade_type': trade_type,
                'signal_price': self._get_price(df, signal_bar, 'Close'),
                'execution_price': exec_price,
                'formula': formula,
                'size': abs(position),
                'pnl_points': pnl_points,
                'pnl_percent': pnl_percent,
                'timestamp': timestamp
            }
            trades.append(trade)

        return trades

    def _get_timestamp(self, df: pd.DataFrame, bar_idx: int, has_datetime: bool):
        """Extract timestamp from DataFrame if available"""
        if has_datetime:
            if 'DateTime' in df.columns:
                return pd.Timestamp(df['DateTime'].iloc[bar_idx])
            elif 'timestamp' in df.columns:
                return pd.Timestamp(df['timestamp'].iloc[bar_idx])
        return None

    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate performance metrics from trade records

        Args:
            trades: List of trade dictionaries

        Returns:
            Dictionary of performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl_percent': 0.0,
                'avg_pnl_percent': 0.0,
                'max_win_percent': 0.0,
                'max_loss_percent': 0.0
            }

        # Extract P&L data
        pnl_percents = [t.get('pnl_percent', 0) for t in trades if t.get('pnl_percent') is not None]

        if not pnl_percents:
            return {
                'total_trades': len(trades),
                'win_rate': 0.0,
                'total_pnl_percent': 0.0,
                'avg_pnl_percent': 0.0,
                'max_win_percent': 0.0,
                'max_loss_percent': 0.0
            }

        wins = [p for p in pnl_percents if p > 0]
        losses = [p for p in pnl_percents if p < 0]

        return {
            'total_trades': len(trades),
            'closed_trades': len(pnl_percents),
            'win_rate': len(wins) / len(pnl_percents) * 100 if pnl_percents else 0,
            'total_pnl_percent': sum(pnl_percents),
            'avg_pnl_percent': sum(pnl_percents) / len(pnl_percents),
            'max_win_percent': max(wins) if wins else 0,
            'max_loss_percent': min(losses) if losses else 0,
            'wins': len(wins),
            'losses': len(losses)
        }