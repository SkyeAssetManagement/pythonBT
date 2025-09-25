"""
VectorBT Pro TWAP Adapter
========================
Adapter to integrate time-based TWAP execution with VectorBT Pro Portfolio.from_signals()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from .time_based_twap_execution import TimeBasedTWAPEngine, TimeBasedTWAPConfig


class VectorBTTWAPAdapter:
    """
    Adapter class to integrate TWAP execution with VectorBT Pro

    This class acts as a bridge between our TWAP engine and VectorBT Pro's
    Portfolio.from_signals() method, providing the custom pricing and
    execution data needed for proper P&L calculation.
    """

    def __init__(self, config: TimeBasedTWAPConfig = None):
        self.config = config or TimeBasedTWAPConfig()
        self.twap_engine = TimeBasedTWAPEngine(self.config)
        self.execution_metadata = None

    def execute_portfolio_with_twap(self, df: pd.DataFrame, long_signals: pd.Series,
                                   short_signals: pd.Series = None, signal_lag: int = 1,
                                   size: float = 1.0, fees: float = 0.0,
                                   **vectorbt_kwargs) -> Dict[str, Any]:
        """
        Execute portfolio using TWAP pricing with VectorBT Pro integration

        Args:
            df: DataFrame with OHLC data and timestamps
            long_signals: Boolean series for long entry signals
            short_signals: Boolean series for short entry signals (optional)
            signal_lag: Number of bars to lag signals
            size: Position size
            fees: Trading fees
            **vectorbt_kwargs: Additional arguments passed to VectorBT Pro

        Returns:
            Dictionary containing portfolio result and metadata
        """

        # Ensure we have proper signals
        if short_signals is None:
            short_signals = pd.Series(False, index=df.index)

        # Execute volume-weighted TWAP calculation
        execution_results = self.twap_engine.execute_signals_with_volume_weighted_twap(
            df, long_signals, short_signals, signal_lag, size
        )

        # Store metadata for later use
        self.execution_metadata = execution_results

        # Prepare VectorBT Pro data
        vectorbt_data = self._prepare_vectorbt_data(
            df, execution_results, size, fees, **vectorbt_kwargs
        )

        # Create trade metadata for display using new volume-weighted data
        trade_metadata = self._create_volume_weighted_trade_metadata(execution_results)

        results = {
            'vectorbt_data': vectorbt_data,
            'trade_metadata': trade_metadata,
            'execution_results': execution_results,
            'twap_summary': self._create_twap_summary(execution_results)
        }

        return results

    def _prepare_vectorbt_data(self, df: pd.DataFrame, execution_results: Dict[str, Any],
                              size: float, fees: float, **kwargs) -> Dict[str, Any]:
        """
        Prepare data dictionary for VectorBT Pro Portfolio.from_signals()

        This creates the exact parameter structure needed for VectorBT Pro
        """

        # Base data for VectorBT Pro - updated for volume-weighted phased execution
        vectorbt_params = {
            'close': df['close'],
            'entries': execution_results['phase_signals_long']['entries'],
            'exits': pd.Series(False, index=df.index),  # Will be handled by exit signals
            'short_entries': execution_results['phase_signals_short']['entries'],
            'short_exits': pd.Series(False, index=df.index),  # Will be handled by exit signals
            'price': execution_results['custom_price_array'],  # Our volume-weighted TWAP prices!
            'size': execution_results['phase_signals_long']['sizes'],  # Volume-proportional sizes!
            'fees': fees,
            'accumulate': True,  # Enable accumulation for phased entries
        }

        # Add optional OHLC data if available
        for col in ['open', 'high', 'low']:
            if col in df.columns:
                vectorbt_params[col] = df[col]

        # Merge any additional VectorBT parameters
        vectorbt_params.update(kwargs)

        return vectorbt_params

    def _create_volume_weighted_trade_metadata(self, execution_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create trade metadata from volume-weighted execution results

        This creates the execBars column and other metadata for display
        """
        trade_records = []

        # Process long execution data
        for execution_data in execution_results['long_execution_data']:
            trade_record = {
                'signal_bar': execution_data['signal_bar'],
                'direction': 'long',
                'exec_bars': execution_data['bar_count'],
                'execution_time_minutes': execution_data['execution_time_minutes'],
                'twap_price': execution_data['vwap_price'],
                'total_volume': execution_data['total_volume'],
                'num_phases': execution_data['num_phases'],
                'total_position_size': execution_data['total_position_size']
            }
            trade_records.append(trade_record)

        # Process short execution data
        for execution_data in execution_results['short_execution_data']:
            trade_record = {
                'signal_bar': execution_data['signal_bar'],
                'direction': 'short',
                'exec_bars': execution_data['bar_count'],
                'execution_time_minutes': execution_data['execution_time_minutes'],
                'twap_price': execution_data['vwap_price'],
                'total_volume': execution_data['total_volume'],
                'num_phases': execution_data['num_phases'],
                'total_position_size': execution_data['total_position_size']
            }
            trade_records.append(trade_record)

        if trade_records:
            return pd.DataFrame(trade_records).sort_values('signal_bar')
        else:
            return pd.DataFrame()

    def _create_twap_summary(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for volume-weighted TWAP execution"""

        long_data = execution_results['long_execution_data']
        short_data = execution_results['short_execution_data']

        summary = {
            'total_signals': len(long_data) + len(short_data),
            'long_signals': len(long_data),
            'short_signals': len(short_data),
            'avg_execution_time_minutes': 0,
            'avg_execution_bars': 0,
            'min_execution_time': 0,
            'max_execution_time': 0,
            'total_phases': execution_results['total_natural_phases']
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

    def get_enhanced_trade_list(self, portfolio_result: Any = None) -> pd.DataFrame:
        """
        Create enhanced trade list with TWAP execution details

        This will show execBars and other TWAP metrics for each trade
        """

        if self.execution_metadata is None:
            return pd.DataFrame()

        # Create base trade metadata
        trade_metadata = self.twap_engine.create_trade_metadata(
            self.execution_metadata['long_twap_prices'],
            self.execution_metadata['short_twap_prices']
        )

        if trade_metadata.empty:
            return trade_metadata

        # Add additional computed columns
        trade_metadata['twap_vs_close'] = 0.0  # Would compare TWAP price to close price
        trade_metadata['execution_efficiency'] = trade_metadata['execution_time_minutes'] / trade_metadata['exec_bars']

        return trade_metadata

    def create_vectorbt_portfolio(self, df: pd.DataFrame, long_signals: pd.Series,
                                 short_signals: pd.Series = None, **kwargs):
        """
        Create VectorBT Pro portfolio with TWAP execution

        This method encapsulates the entire process and returns a VectorBT Portfolio object
        """

        try:
            # Import VectorBT Pro (this would fail gracefully if not available)
            import vectorbtpro as vbt

            # Execute TWAP calculation and prepare data
            results = self.execute_portfolio_with_twap(
                df, long_signals, short_signals, **kwargs
            )

            vectorbt_data = results['vectorbt_data']

            # Create VectorBT Pro portfolio
            portfolio = vbt.Portfolio.from_signals(**vectorbt_data)

            # Attach our metadata to the portfolio object for later use
            portfolio._twap_metadata = results['trade_metadata']
            portfolio._twap_summary = results['twap_summary']
            portfolio._execution_results = results['execution_results']

            return portfolio

        except ImportError:
            warnings.warn("VectorBT Pro not available. Returning prepared data instead.")
            return self.execute_portfolio_with_twap(df, long_signals, short_signals, **kwargs)

        except Exception as e:
            print(f"Error creating VectorBT Portfolio: {e}")
            return None

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get detailed execution statistics"""

        if self.execution_metadata is None:
            return {}

        long_data = self.execution_metadata['long_twap_prices']
        short_data = self.execution_metadata['short_twap_prices']

        stats = {
            'config': {
                'minimum_execution_minutes': self.config.minimum_execution_minutes,
                'max_bars_to_check': self.config.max_bars_to_check,
                'max_phases': self.config.max_phases
            },
            'execution_summary': self._create_twap_summary(self.execution_metadata),
            'signal_details': {
                'long_signals': [
                    {
                        'signal_bar': d['signal_bar'],
                        'exec_bars': d['bar_count'],
                        'execution_time': d['execution_time_minutes'],
                        'twap_price': d['twap_price'],
                        'phases': len(d['phase_prices'])
                    }
                    for d in long_data
                ],
                'short_signals': [
                    {
                        'signal_bar': d['signal_bar'],
                        'exec_bars': d['bar_count'],
                        'execution_time': d['execution_time_minutes'],
                        'twap_price': d['twap_price'],
                        'phases': len(d['phase_prices'])
                    }
                    for d in short_data
                ]
            }
        }

        return stats


def create_twap_adapter(config_path: str = None) -> VectorBTTWAPAdapter:
    """Factory function to create configured TWAP adapter"""
    config = TimeBasedTWAPConfig.from_yaml(config_path)
    return VectorBTTWAPAdapter(config)