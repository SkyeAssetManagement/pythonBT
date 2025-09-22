"""Simple Moving Average Crossover Strategy"""

import pandas as pd
import numpy as np
from typing import Tuple
from .base import TradingStrategy


class SMACrossoverStrategy(TradingStrategy):
    """
    Classic SMA crossover strategy
    Long when fast SMA > slow SMA
    Short when fast SMA < slow SMA
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30,
                 long_only: bool = True):
        super().__init__(name=f"SMA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.long_only = long_only
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'long_only': long_only
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate SMA crossover signals"""
        # Get close prices
        close = df['Close'] if 'Close' in df.columns else df['close']

        # Calculate SMAs
        sma_fast = close.rolling(window=self.fast_period, min_periods=1).mean()
        sma_slow = close.rolling(window=self.slow_period, min_periods=1).mean()

        # Generate signals
        signals = pd.Series(index=df.index, dtype=float)

        # Long signal when fast > slow
        long_signal = (sma_fast > sma_slow).astype(float)

        if self.long_only:
            # Long only: 1 when fast > slow, 0 otherwise
            signals = long_signal
        else:
            # Long/short: 1 when fast > slow, -1 when fast < slow
            short_signal = (sma_fast < sma_slow).astype(float) * -1
            signals = long_signal + short_signal

        # Fill NaN with 0 (no position)
        signals = signals.fillna(0)

        return signals

    def calculate_smas(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate SMA values for plotting"""
        close = df['Close'] if 'Close' in df.columns else df['close']
        sma_fast = close.rolling(window=self.fast_period, min_periods=1).mean()
        sma_slow = close.rolling(window=self.slow_period, min_periods=1).mean()
        return sma_fast, sma_slow