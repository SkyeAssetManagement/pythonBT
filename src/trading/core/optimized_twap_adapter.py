#!/usr/bin/env python3
"""
Optimized TWAP Adapter for Large Datasets
==========================================
Handles large production datasets (100k+ signals) without hanging
by implementing chunked processing and memory optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import warnings
from dataclasses import dataclass

from time_based_twap_execution import TimeBasedTWAPConfig, TimeBasedTWAPEngine


@dataclass
class OptimizedTWAPConfig(TimeBasedTWAPConfig):
    """Extended config for large dataset optimization"""
    chunk_size: int = 10000  # Process signals in chunks
    max_signals_before_chunking: int = 50000  # Threshold for chunked processing
    memory_efficient_mode: bool = True  # Use memory-efficient processing
    skip_vectorbt_portfolio: bool = False  # Skip VectorBT for very large datasets


class OptimizedTWAPAdapter:
    """
    Optimized TWAP adapter that handles large production datasets
    without hanging by processing signals in manageable chunks
    """

    def __init__(self, config: OptimizedTWAPConfig):
        self.config = config
        self.twap_engine = TimeBasedTWAPEngine(config)
        self.last_results = None

    def execute_portfolio_with_twap_chunked(self,
                                          df: pd.DataFrame,
                                          long_signals: pd.Series,
                                          short_signals: pd.Series,
                                          signal_lag: int = 1,
                                          size: float = 1.0,
                                          fees: float = 0.0) -> Dict[str, Any]:
        """
        Execute TWAP with chunked processing for large datasets

        This method processes signals in chunks to avoid memory issues
        and hanging when dealing with large production datasets
        """

        total_long_signals = long_signals.sum() if hasattr(long_signals, 'sum') else 0
        total_short_signals = short_signals.sum() if hasattr(short_signals, 'sum') else 0
        total_signals = total_long_signals + total_short_signals

        print(f"[OPTIMIZED] Processing {total_signals} total signals ({total_long_signals} long, {total_short_signals} short)")

        # Check if we need chunked processing
        if total_signals > self.config.max_signals_before_chunking:
            print(f"[OPTIMIZED] Large dataset detected - using chunked processing (chunk_size: {self.config.chunk_size})")
            return self._execute_chunked_processing(df, long_signals, short_signals, signal_lag, size, fees)
        else:
            print(f"[OPTIMIZED] Small dataset - using normal processing")
            return self._execute_normal_processing(df, long_signals, short_signals, signal_lag, size, fees)

    def _execute_chunked_processing(self,
                                  df: pd.DataFrame,
                                  long_signals: pd.Series,
                                  short_signals: pd.Series,
                                  signal_lag: int,
                                  size: float,
                                  fees: float) -> Dict[str, Any]:
        """Execute TWAP with chunked signal processing"""

        # Find signal indices
        long_signal_indices = long_signals[long_signals].index.tolist()
        short_signal_indices = short_signals[short_signals].index.tolist()

        all_long_execution_data = []
        all_short_execution_data = []
        total_phases = 0

        # Process long signals in chunks
        if long_signal_indices:
            print(f"[OPTIMIZED] Processing {len(long_signal_indices)} long signals in chunks of {self.config.chunk_size}")

            for i in range(0, len(long_signal_indices), self.config.chunk_size):
                chunk_indices = long_signal_indices[i:i + self.config.chunk_size]
                chunk_signals = pd.Series(False, index=df.index)
                chunk_signals.loc[chunk_indices] = True

                print(f"[OPTIMIZED] Processing long chunk {i//self.config.chunk_size + 1}: {len(chunk_indices)} signals")

                # Process chunk
                chunk_results = self.twap_engine.execute_signals_with_volume_weighted_twap(
                    df=df,
                    signals_long=chunk_signals,
                    signals_short=pd.Series(False, index=df.index),  # Empty short signals for this chunk
                    signal_lag=signal_lag,
                    target_position_size=size
                )

                all_long_execution_data.extend(chunk_results['long_execution_data'])
                total_phases += chunk_results['total_natural_phases']

        # Process short signals in chunks
        if short_signal_indices:
            print(f"[OPTIMIZED] Processing {len(short_signal_indices)} short signals in chunks of {self.config.chunk_size}")

            for i in range(0, len(short_signal_indices), self.config.chunk_size):
                chunk_indices = short_signal_indices[i:i + self.config.chunk_size]
                chunk_signals = pd.Series(False, index=df.index)
                chunk_signals.loc[chunk_indices] = True

                print(f"[OPTIMIZED] Processing short chunk {i//self.config.chunk_size + 1}: {len(chunk_indices)} signals")

                # Process chunk
                chunk_results = self.twap_engine.execute_signals_with_volume_weighted_twap(
                    df=df,
                    signals_long=pd.Series(False, index=df.index),  # Empty long signals for this chunk
                    signals_short=chunk_signals,
                    signal_lag=signal_lag,
                    target_position_size=size
                )

                all_short_execution_data.extend(chunk_results['short_execution_data'])
                total_phases += chunk_results['total_natural_phases']

        # Combine results
        combined_results = {
            'long_execution_data': all_long_execution_data,
            'short_execution_data': all_short_execution_data,
            'total_natural_phases': total_phases
        }

        print(f"[OPTIMIZED] Chunked processing complete: {len(all_long_execution_data)} long trades, {len(all_short_execution_data)} short trades")

        # Create trade metadata
        trade_metadata = self._create_optimized_trade_metadata(combined_results)

        # Skip VectorBT portfolio creation for very large datasets
        if self.config.skip_vectorbt_portfolio:
            print("[OPTIMIZED] Skipping VectorBT portfolio creation for large dataset")
            return {
                'execution_results': combined_results,
                'trade_metadata': trade_metadata,
                'twap_summary': self._create_summary_stats(combined_results),
                'vectorbt_data': None  # No VectorBT data for very large datasets
            }

        # Create simplified VectorBT data for large datasets
        vectorbt_data = self._create_lightweight_vectorbt_data(df, combined_results, size, fees)

        return {
            'execution_results': combined_results,
            'trade_metadata': trade_metadata,
            'twap_summary': self._create_summary_stats(combined_results),
            'vectorbt_data': vectorbt_data
        }

    def _execute_normal_processing(self,
                                 df: pd.DataFrame,
                                 long_signals: pd.Series,
                                 short_signals: pd.Series,
                                 signal_lag: int,
                                 size: float,
                                 fees: float) -> Dict[str, Any]:
        """Execute normal TWAP processing for smaller datasets"""

        execution_results = self.twap_engine.execute_signals_with_volume_weighted_twap(
            df=df,
            signals_long=long_signals,
            signals_short=short_signals,
            signal_lag=signal_lag,
            target_position_size=size
        )

        trade_metadata = self._create_optimized_trade_metadata(execution_results)

        # Try to create VectorBT data if not too large
        try:
            from vectorbt_twap_adapter import VectorBTTWAPAdapter
            vbt_adapter = VectorBTTWAPAdapter(self.config)
            vectorbt_data = vbt_adapter._prepare_vectorbt_data(df, execution_results, size, fees)
        except Exception as e:
            print(f"[OPTIMIZED] VectorBT preparation failed: {e}")
            vectorbt_data = self._create_lightweight_vectorbt_data(df, execution_results, size, fees)

        return {
            'execution_results': execution_results,
            'trade_metadata': trade_metadata,
            'twap_summary': self._create_summary_stats(execution_results),
            'vectorbt_data': vectorbt_data
        }

    def _create_optimized_trade_metadata(self, execution_results: Dict[str, Any]) -> pd.DataFrame:
        """Create trade metadata optimized for large datasets"""

        trade_records = []

        # Process long trades
        for execution_data in execution_results['long_execution_data']:
            trade_record = {
                'signal_bar': execution_data['signal_bar'],
                'direction': 'long',
                'twap_price': execution_data['twap_price'],
                'total_position_size': execution_data['total_position_size'],
                'exec_bars': execution_data['bar_count'],
                'execution_time_minutes': execution_data['execution_time_minutes'],
                'num_phases': execution_data['num_phases'],
                'total_volume': execution_data['total_volume']
            }
            trade_records.append(trade_record)

        # Process short trades
        for execution_data in execution_results['short_execution_data']:
            trade_record = {
                'signal_bar': execution_data['signal_bar'],
                'direction': 'short',
                'twap_price': execution_data['twap_price'],
                'total_position_size': execution_data['total_position_size'],
                'exec_bars': execution_data['bar_count'],
                'execution_time_minutes': execution_data['execution_time_minutes'],
                'num_phases': execution_data['num_phases'],
                'total_volume': execution_data['total_volume']
            }
            trade_records.append(trade_record)

        if trade_records:
            return pd.DataFrame(trade_records).sort_values('signal_bar')
        else:
            return pd.DataFrame()

    def _create_lightweight_vectorbt_data(self,
                                        df: pd.DataFrame,
                                        execution_results: Dict[str, Any],
                                        size: float,
                                        fees: float) -> Dict[str, Any]:
        """Create lightweight VectorBT data for large datasets without full portfolio"""

        # Create basic signal arrays
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        sizes = pd.Series(0.0, index=df.index)
        prices = pd.Series(df['close'], index=df.index)  # Use close prices

        # Mark entry points from execution data
        for execution_data in execution_results['long_execution_data']:
            signal_bar = execution_data['signal_bar']
            if signal_bar < len(df):
                entries.iloc[signal_bar] = True
                sizes.iloc[signal_bar] = execution_data['total_position_size']
                prices.iloc[signal_bar] = execution_data['twap_price']

        for execution_data in execution_results['short_execution_data']:
            signal_bar = execution_data['signal_bar']
            if signal_bar < len(df):
                entries.iloc[signal_bar] = True
                sizes.iloc[signal_bar] = -execution_data['total_position_size']  # Negative for short
                prices.iloc[signal_bar] = execution_data['twap_price']

        return {
            'close': df['close'],
            'entries': entries,
            'exits': exits,
            'size': sizes,
            'price': prices,
            'fees': fees,
            'accumulate': False,  # Simplified mode
            'lightweight_mode': True  # Flag indicating this is lightweight data
        }

    def _create_summary_stats(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for execution results"""

        long_data = execution_results['long_execution_data']
        short_data = execution_results['short_execution_data']

        summary = {
            'total_signals': len(long_data) + len(short_data),
            'long_signals': len(long_data),
            'short_signals': len(short_data),
            'total_phases': execution_results['total_natural_phases'],
            'avg_execution_time_minutes': 0,
            'avg_execution_bars': 0,
            'processing_mode': 'chunked' if len(long_data) + len(short_data) > self.config.max_signals_before_chunking else 'normal'
        }

        if long_data or short_data:
            all_data = long_data + short_data

            execution_times = [d['execution_time_minutes'] for d in all_data]
            execution_bars = [d['bar_count'] for d in all_data]

            summary.update({
                'avg_execution_time_minutes': np.mean(execution_times),
                'avg_execution_bars': np.mean(execution_bars),
                'min_execution_time': np.min(execution_times),
                'max_execution_time': np.max(execution_times),
            })

        return summary