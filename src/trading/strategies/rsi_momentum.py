"""RSI Momentum Strategy"""

import pandas as pd
import numpy as np
from .base import TradingStrategy


class RSIMomentumStrategy(TradingStrategy):
    """
    RSI-based momentum strategy
    Long when RSI crosses above oversold level
    Short when RSI crosses below overbought level
    """

    def __init__(self, period: int = 14, oversold: float = 30,
                 overbought: float = 70, long_only: bool = True):
        super().__init__(name=f"RSI_{period}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.long_only = long_only
        self.parameters = {
            'period': period,
            'oversold': oversold,
            'overbought': overbought,
            'long_only': long_only
        }

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator"""
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=1).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate RSI momentum signals"""
        # Get close prices
        close = df['Close'] if 'Close' in df.columns else df['close']

        # Calculate RSI
        rsi = self.calculate_rsi(close)

        # Initialize signals
        signals = pd.Series(0, index=df.index, dtype=float)

        # Generate entry/exit signals
        for i in range(1, len(rsi)):
            # Long signal: RSI crosses above oversold
            if rsi.iloc[i] > self.oversold and rsi.iloc[i-1] <= self.oversold:
                signals.iloc[i] = 1

            # Exit long: RSI crosses below overbought
            elif rsi.iloc[i] < self.overbought and rsi.iloc[i-1] >= self.overbought:
                if not self.long_only:
                    signals.iloc[i] = -1  # Short signal
                else:
                    signals.iloc[i] = 0  # Exit to flat

            # Short signal (if not long-only): RSI crosses below overbought
            elif not self.long_only and rsi.iloc[i] < self.overbought and rsi.iloc[i-1] >= self.overbought:
                signals.iloc[i] = -1

            # Cover short: RSI crosses above oversold
            elif not self.long_only and rsi.iloc[i] > self.oversold and rsi.iloc[i-1] <= self.oversold:
                signals.iloc[i] = 1

        # Forward fill signals (maintain position)
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)

        return signals

    def get_rsi_values(self, df: pd.DataFrame) -> pd.Series:
        """Get RSI values for plotting"""
        close = df['Close'] if 'Close' in df.columns else df['close']
        return self.calculate_rsi(close)