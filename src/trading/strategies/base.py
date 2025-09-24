"""Base trading strategy interface"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.trade_data import TradeData, TradeCollection


class TradingStrategy(ABC):
    """Base class for all trading strategies"""

    def __init__(self, name: str = "Strategy"):
        self.name = name
        self.parameters = {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from price data

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series of signals: 1 (long), -1 (short), 0 (flat)
        """
        pass

    def signals_to_trades(self, signals: pd.Series, df: pd.DataFrame,
                         start_bar: int = 0) -> TradeCollection:
        """
        Convert signals to trades

        Args:
            signals: Series of 1/-1/0 signals
            df: DataFrame with price data
            start_bar: Starting bar index offset

        Returns:
            TradeCollection with generated trades
        """
        trades = []
        position = 0
        trade_id = 0
        entry_price = None  # Track entry price for P&L calculation

        # Check if we have DateTime column
        print(f"[STRATEGY] DataFrame columns: {df.columns.tolist()}")
        has_datetime = 'DateTime' in df.columns or 'timestamp' in df.columns
        print(f"[STRATEGY] Has DateTime: {has_datetime}")

        for i in range(len(signals)):
            signal = signals.iloc[i]

            # Skip if no change
            if signal == position:
                continue

            bar_idx = start_bar + i
            price = df['Close'].iloc[i] if 'Close' in df.columns else df['close'].iloc[i]

            # Get timestamp if available
            timestamp = None
            if has_datetime:
                if 'DateTime' in df.columns:
                    timestamp = pd.Timestamp(df['DateTime'].iloc[i])
                    if i == 0:  # Log first timestamp
                        # ts = pd.Timestamp(timestamp)
                        # print(f"[PHASE4-DEBUG] First trade timestamp: {timestamp}, hour={ts.hour}, minute={ts.minute}, second={ts.second}")
                        print(f"[STRATEGY] First timestamp from DateTime: {timestamp}")
                elif 'timestamp' in df.columns:
                    timestamp = pd.Timestamp(df['timestamp'].iloc[i])
                    if i == 0:  # Log first timestamp
                        print(f"[STRATEGY] First timestamp from 'timestamp': {timestamp}")

            # Exit current position if needed
            if position != 0:
                trade_type = 'SELL' if position > 0 else 'COVER'

                # Calculate P&L for exit trades (percentage based on $1 invested)
                pnl = None
                pnl_percent = None
                if entry_price is not None:
                    if position > 0:  # Long position
                        # For $1 invested: (exit_price/entry_price - 1) * 100
                        pnl_percent = ((price / entry_price) - 1) * 100
                        pnl = price - entry_price  # Keep points for compatibility
                    else:  # Short position
                        # For $1 invested in short: (1 - exit_price/entry_price) * 100
                        pnl_percent = (1 - (price / entry_price)) * 100
                        pnl = entry_price - price  # Keep points for compatibility

                trade = TradeData(
                    bar_index=bar_idx,
                    trade_type=trade_type,
                    price=price,
                    trade_id=trade_id,
                    timestamp=timestamp,
                    pnl=pnl_percent,  # Store percentage as pnl
                    strategy=self.name
                )
                # Also store the percentage explicitly
                trade.pnl_percent = pnl_percent
                trades.append(trade)
                trade_id += 1
                entry_price = None

            # Enter new position if signal is non-zero
            if signal != 0:
                trade_type = 'BUY' if signal > 0 else 'SHORT'
                entry_price = price  # Store entry price for P&L calculation

                trade = TradeData(
                    bar_index=bar_idx,
                    trade_type=trade_type,
                    price=price,
                    trade_id=trade_id,
                    timestamp=timestamp,
                    strategy=self.name
                )
                trades.append(trade)
                trade_id += 1

            position = signal

        # Close final position if still open
        if position != 0 and len(df) > 0:
            bar_idx = start_bar + len(df) - 1
            price = df['Close'].iloc[-1] if 'Close' in df.columns else df['close'].iloc[-1]
            trade_type = 'SELL' if position > 0 else 'COVER'

            # Get final timestamp
            timestamp = None
            if has_datetime:
                if 'DateTime' in df.columns:
                    timestamp = pd.Timestamp(df['DateTime'].iloc[-1])
                elif 'timestamp' in df.columns:
                    timestamp = pd.Timestamp(df['timestamp'].iloc[-1])

            # Calculate final P&L (percentage based on $1 invested)
            pnl = None
            pnl_percent = None
            if entry_price is not None:
                if position > 0:  # Long position
                    # For $1 invested: (exit_price/entry_price - 1) * 100
                    pnl_percent = ((price / entry_price) - 1) * 100
                    pnl = price - entry_price  # Keep points for compatibility
                else:  # Short position
                    # For $1 invested in short: (1 - exit_price/entry_price) * 100
                    pnl_percent = (1 - (price / entry_price)) * 100
                    pnl = entry_price - price  # Keep points for compatibility

            trade = TradeData(
                bar_index=bar_idx,
                trade_type=trade_type,
                price=price,
                trade_id=trade_id,
                timestamp=timestamp,
                pnl=pnl_percent,  # Store percentage as pnl
                strategy=self.name
            )
            # Also store the percentage explicitly
            trade.pnl_percent = pnl_percent
            trades.append(trade)

        return TradeCollection(trades if trades else [])