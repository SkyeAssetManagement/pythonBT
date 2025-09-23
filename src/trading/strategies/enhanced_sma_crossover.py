"""Enhanced SMA Crossover Strategy with execution lag and price formulas"""

import pandas as pd
import numpy as np
from typing import Tuple
from .enhanced_base import EnhancedTradingStrategy


class EnhancedSMACrossoverStrategy(EnhancedTradingStrategy):
    """
    SMA crossover strategy with proper execution lag and price formulas
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30,
                 long_only: bool = True, config_path: str = None):
        super().__init__(name=f"Enhanced_SMA_{fast_period}_{slow_period}", config_path=config_path)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.long_only = long_only
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'long_only': long_only,
            'signal_lag': self.signal_lag,
            'buy_formula': self.buy_execution_formula,
            'sell_formula': self.sell_execution_formula
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate SMA crossover signals"""
        # Get close prices
        close = df['Close'] if 'Close' in df.columns else df['close']

        # Calculate SMAs
        sma_fast = close.rolling(window=self.fast_period, min_periods=1).mean()
        sma_slow = close.rolling(window=self.slow_period, min_periods=1).mean()

        # Generate raw signals
        signals = pd.Series(index=df.index, dtype=float)

        # Track crossovers
        fast_above = sma_fast > sma_slow
        fast_below = sma_fast < sma_slow

        # Generate position signals
        if self.long_only:
            signals = fast_above.astype(float)
        else:
            signals = fast_above.astype(float) - fast_below.astype(float)

        # Fill NaN with 0
        signals = signals.fillna(0)

        # Log signal statistics
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        flat_signals = (signals == 0).sum()

        print(f"[ENHANCED_SMA] Generated signals: {long_signals} long, {short_signals} short, {flat_signals} flat")

        return signals

    def calculate_smas(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate SMA values for plotting"""
        close = df['Close'] if 'Close' in df.columns else df['close']
        sma_fast = close.rolling(window=self.fast_period, min_periods=1).mean()
        sma_slow = close.rolling(window=self.slow_period, min_periods=1).mean()
        return sma_fast, sma_slow

    def signals_to_trades(self, signals: pd.Series, df: pd.DataFrame,
                         start_bar: int = 0):
        """Override to use enhanced version with lag"""
        return self.signals_to_trades_with_lag(signals, df, start_bar)