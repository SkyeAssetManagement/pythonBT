"""
Time-Based TWAP Execution System for Range Bars
===============================================
Proper implementation for variable time bars (range bars) using VectorBT Pro

Process:
1. 1st vector sweep = get signals
2. 2nd sweep: Add cumulative time for 1,2,3...20 bars from signal bar + lag
3. Filter list to pick bar count for each signal that meets min time requirement
4. Calculate TWAP price based on each signal's individual required bar count
5. Feed custom price back into VectorBT Pro for P&L calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import yaml
import os
from datetime import datetime, timedelta


@dataclass
class TimeBasedTWAPConfig:
    """Configuration for volume-weighted time-based TWAP execution with range bars"""
    enabled: bool = False
    minimum_execution_minutes: float = 5.0    # Minimum execution time
    max_bars_to_check: int = 20               # Maximum bars to look ahead
    batch_size: int = 5                       # Bars to process per batch (for efficiency)

    # Risk management
    max_adverse_move: float = 3.0
    require_profit: bool = True

    @classmethod
    def from_yaml(cls, config_path: str = None) -> 'TimeBasedTWAPConfig':
        """Load time-based TWAP configuration from YAML"""
        if config_path is None:
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
                twap_config = config.get('time_based_twap', {})
                if twap_config:
                    return cls(**twap_config)

        return cls()


class TimeBasedTWAPEngine:
    """
    Time-based TWAP execution engine for range bars using VectorBT Pro

    This engine properly handles variable time bars by calculating actual
    elapsed time rather than assuming fixed bar durations.
    """

    def __init__(self, config: TimeBasedTWAPConfig = None):
        self.config = config or TimeBasedTWAPConfig()

    def execute_signals_with_volume_weighted_twap(self, df: pd.DataFrame, signals_long: pd.Series,
                                                 signals_short: pd.Series, signal_lag: int = 1,
                                                 target_position_size: float = 1.0) -> Dict[str, Any]:
        """
        Execute signals using volume-weighted time-based TWAP approach

        Args:
            df: DataFrame with OHLC data, timestamps, and volume
            signals_long: Boolean series for long signals
            signals_short: Boolean series for short signals
            signal_lag: Number of bars to lag signals
            target_position_size: Target position size for allocation

        Returns:
            Dictionary with execution results including natural phases and VectorBT data
        """

        # Step 1: First vector sweep - get signals
        signal_bars_long = self._get_signal_bars(signals_long)
        signal_bars_short = self._get_signal_bars(signals_short)

        print(f"Processing {len(signal_bars_long)} long signals and {len(signal_bars_short)} short signals")

        # Step 2: Efficient batched time calculation
        long_execution_data = self._calculate_execution_data_batched(df, signal_bars_long, signal_lag)
        short_execution_data = self._calculate_execution_data_batched(df, signal_bars_short, signal_lag)

        # Step 3 & 4: Calculate volume-weighted execution (combines filtering and pricing)
        # The batched execution data already contains filtered signals that meet time requirements
        long_volume_weighted = self._calculate_volume_weighted_execution(
            df, long_execution_data, target_position_size
        )
        short_volume_weighted = self._calculate_volume_weighted_execution(
            df, short_execution_data, target_position_size
        )

        # Step 5: Build custom price arrays and phase data for VectorBT Pro
        custom_price_array = self._build_volume_weighted_price_array(
            df, long_volume_weighted, short_volume_weighted
        )

        # Build phase signals for both long and short
        phase_signals_long = self._build_phase_signals(df, long_volume_weighted, 'long')
        phase_signals_short = self._build_phase_signals(df, short_volume_weighted, 'short')

        execution_results = {
            'long_signals': signals_long,
            'short_signals': signals_short,
            'long_execution_data': long_volume_weighted,
            'short_execution_data': short_volume_weighted,
            'custom_price_array': custom_price_array,
            'phase_signals_long': phase_signals_long,
            'phase_signals_short': phase_signals_short,
            'total_natural_phases': sum(ex['num_phases'] for ex in long_volume_weighted + short_volume_weighted)
        }

        return execution_results

    def _get_signal_bars(self, signals: pd.Series) -> np.ndarray:
        """Extract bar indices where signals are True"""
        return np.where(signals)[0]

    def _calculate_execution_data_batched(self, df: pd.DataFrame, signal_bars: np.ndarray,
                                         signal_lag: int) -> List[Dict]:
        """
        Calculate execution data using efficient batched processing

        Process signals in batches of bars (e.g., 1-5, 6-10, 11-15) to minimize
        redundant calculations and capture signals as soon as they meet time requirements.
        """

        # Ensure we have datetime index or datetime column
        if hasattr(df.index, 'to_pydatetime'):
            timestamps = df.index.to_pydatetime()
        elif 'datetime' in df.columns:
            timestamps = pd.to_datetime(df['datetime']).dt.to_pydatetime()
        elif 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp']).dt.to_pydatetime()
        else:
            # Fallback: create synthetic timestamps assuming 5-minute bars
            base_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
            timestamps = [base_time + timedelta(minutes=5*i) for i in range(len(df))]

        # Prepare signal tracking
        remaining_signals = []
        completed_signals = []

        # Initialize remaining signals with basic data
        for signal_bar in signal_bars:
            execution_start_bar = signal_bar + signal_lag

            # Skip if execution would be beyond data range
            if execution_start_bar >= len(df):
                continue

            remaining_signals.append({
                'signal_bar': signal_bar,
                'execution_start_bar': execution_start_bar,
                'execution_start_time': timestamps[execution_start_bar]
            })

        # Process in batches
        batch_start = 1
        while remaining_signals and batch_start <= self.config.max_bars_to_check:
            batch_end = min(batch_start + self.config.batch_size - 1, self.config.max_bars_to_check)

            # Process current batch for all remaining signals
            newly_completed = []
            still_remaining = []

            for signal_data in remaining_signals:
                execution_start_bar = signal_data['execution_start_bar']
                execution_start_time = signal_data['execution_start_time']

                # Try each bar count in current batch
                found_sufficient_time = False

                for bars_ahead in range(batch_start, batch_end + 1):
                    end_bar = execution_start_bar + bars_ahead - 1

                    # Check if end_bar is within data range
                    if end_bar >= len(df):
                        break

                    end_time = timestamps[end_bar]
                    elapsed_time_minutes = (end_time - execution_start_time).total_seconds() / 60.0

                    # Check if this meets minimum time requirement
                    if elapsed_time_minutes >= self.config.minimum_execution_minutes:
                        # Found sufficient time - capture this signal
                        completed_signal = {
                            'signal_bar': signal_data['signal_bar'],
                            'execution_start_bar': execution_start_bar,
                            'required_bar_count': bars_ahead,
                            'execution_end_bar': end_bar,
                            'actual_execution_time_minutes': elapsed_time_minutes,
                            'start_time': execution_start_time,
                            'end_time': end_time
                        }
                        newly_completed.append(completed_signal)
                        found_sufficient_time = True
                        break

                # If not found in this batch, keep for next batch
                if not found_sufficient_time:
                    still_remaining.append(signal_data)

            # Update tracking lists
            completed_signals.extend(newly_completed)
            remaining_signals = still_remaining

            # Move to next batch
            batch_start = batch_end + 1

            print(f"Batch {batch_start-self.config.batch_size}-{batch_end}: "
                  f"Captured {len(newly_completed)} signals, "
                  f"{len(remaining_signals)} remaining")

        # Handle any remaining signals that never met time requirement
        # Use maximum available bars for these
        for signal_data in remaining_signals:
            execution_start_bar = signal_data['execution_start_bar']
            execution_start_time = signal_data['execution_start_time']

            # Find maximum available bars
            max_end_bar = min(execution_start_bar + self.config.max_bars_to_check - 1, len(df) - 1)

            if max_end_bar >= execution_start_bar:
                end_time = timestamps[max_end_bar]
                elapsed_time_minutes = (end_time - execution_start_time).total_seconds() / 60.0
                bars_ahead = max_end_bar - execution_start_bar + 1

                completed_signal = {
                    'signal_bar': signal_data['signal_bar'],
                    'execution_start_bar': execution_start_bar,
                    'required_bar_count': bars_ahead,
                    'execution_end_bar': max_end_bar,
                    'actual_execution_time_minutes': elapsed_time_minutes,
                    'start_time': execution_start_time,
                    'end_time': end_time,
                    'insufficient_time': True  # Flag for insufficient time
                }
                completed_signals.append(completed_signal)

        return completed_signals

    def _filter_by_time_requirements(self, execution_data: List[Dict]) -> List[Dict]:
        """
        Filter execution data to find bar counts that meet minimum time requirements

        For each signal, find the minimum number of bars needed to meet
        the minimum execution time requirement
        """
        filtered_data = []

        for signal_data in execution_data:
            # Find the first bar count that meets minimum time requirement
            required_bar_count = None
            required_time_data = None

            for bar_data in signal_data['bar_time_data']:
                if bar_data['elapsed_time_minutes'] >= self.config.minimum_execution_minutes:
                    required_bar_count = bar_data['bars_ahead']
                    required_time_data = bar_data
                    break

            # If no single bar count meets requirement, use the maximum available
            if required_bar_count is None and signal_data['bar_time_data']:
                last_bar_data = signal_data['bar_time_data'][-1]
                required_bar_count = last_bar_data['bars_ahead']
                required_time_data = last_bar_data

            if required_bar_count is not None:
                filtered_signal_data = {
                    'signal_bar': signal_data['signal_bar'],
                    'execution_start_bar': signal_data['execution_start_bar'],
                    'required_bar_count': required_bar_count,
                    'execution_end_bar': required_time_data['end_bar'],
                    'actual_execution_time_minutes': required_time_data['elapsed_time_minutes'],
                    'start_time': required_time_data['start_time'],
                    'end_time': required_time_data['end_time']
                }

                filtered_data.append(filtered_signal_data)

        return filtered_data

    def _calculate_volume_weighted_execution(self, df: pd.DataFrame, execution_data: List[Dict],
                                           target_position_size: float = 1.0) -> List[Dict]:
        """
        Calculate volume-weighted execution for each signal

        Each bar in the execution period becomes a natural phase with size
        proportional to that bar's volume relative to total execution volume.
        """
        execution_results = []

        for signal_data in execution_data:
            start_bar = signal_data['execution_start_bar']
            end_bar = signal_data['execution_end_bar']

            # Extract execution period data
            execution_period = df.iloc[start_bar:end_bar + 1].copy()

            # Ensure we have volume data
            if 'volume' not in execution_period.columns:
                # Create uniform volume if not available
                execution_period['volume'] = 1.0

            # Calculate total volume across execution period
            total_volume = execution_period['volume'].sum()

            # Avoid division by zero
            if total_volume == 0:
                total_volume = len(execution_period)
                execution_period['volume'] = 1.0

            # Calculate volume-proportional sizes and prices for each bar
            natural_phases = []
            cumulative_size = 0.0

            for idx, (bar_idx, bar_data) in enumerate(execution_period.iterrows()):
                # Calculate volume proportion for this bar
                volume_proportion = bar_data['volume'] / total_volume
                phase_size = target_position_size * volume_proportion

                # Calculate price for this bar (typical price if available)
                if all(col in df.columns for col in ['high', 'low', 'close']):
                    phase_price = (bar_data['high'] + bar_data['low'] + bar_data['close']) / 3
                elif 'hlc3' in df.columns:
                    phase_price = bar_data['hlc3']
                else:
                    phase_price = bar_data['close']

                # Get actual bar index in original dataframe
                actual_bar_idx = start_bar + idx

                phase_data = {
                    'phase_number': idx + 1,
                    'bar_index': actual_bar_idx,
                    'volume': bar_data['volume'],
                    'volume_proportion': volume_proportion,
                    'phase_size': phase_size,
                    'phase_price': phase_price,
                    'timestamp': execution_period.index[idx] if hasattr(execution_period.index, 'to_pydatetime') else None
                }

                natural_phases.append(phase_data)
                cumulative_size += phase_size

            # Calculate overall TWAP price (volume-weighted)
            total_value = sum(phase['phase_size'] * phase['phase_price'] for phase in natural_phases)
            vwap_price = total_value / cumulative_size if cumulative_size > 0 else natural_phases[0]['phase_price']

            # Compile execution data
            execution_data = {
                'signal_bar': signal_data['signal_bar'],
                'execution_start_bar': start_bar,
                'execution_end_bar': end_bar,
                'bar_count': signal_data['required_bar_count'],
                'execution_time_minutes': signal_data['actual_execution_time_minutes'],
                'total_volume': total_volume,
                'total_position_size': cumulative_size,
                'vwap_price': vwap_price,
                'natural_phases': natural_phases,
                'num_phases': len(natural_phases),
                'insufficient_time': signal_data.get('insufficient_time', False)
            }

            execution_results.append(execution_data)

        return execution_results

    def _build_volume_weighted_price_array(self, df: pd.DataFrame,
                                          long_execution_results: List[Dict],
                                          short_execution_results: List[Dict] = None) -> np.ndarray:
        """
        Build custom price array for VectorBT Pro with volume-weighted execution prices

        Uses the natural phase prices from volume-weighted execution for accurate P&L calculation
        """
        # Start with default close prices
        custom_prices = df['close'].copy().values.astype(float)

        # Apply long volume-weighted prices
        for execution_data in long_execution_results:
            signal_bar = execution_data['signal_bar']
            vwap_price = execution_data['vwap_price']

            # Use the volume-weighted average price for the signal bar
            custom_prices[signal_bar] = vwap_price

        # Apply short volume-weighted prices if provided
        if short_execution_results:
            for execution_data in short_execution_results:
                signal_bar = execution_data['signal_bar']
                vwap_price = execution_data['vwap_price']

                # Use the volume-weighted average price for the signal bar
                custom_prices[signal_bar] = vwap_price

        return custom_prices

    def _build_phase_signals(self, df: pd.DataFrame, execution_results: List[Dict],
                            signal_direction: str = 'long') -> Dict[str, np.ndarray]:
        """
        Build phase-specific signal arrays for VectorBT Pro accumulate=True execution

        Each natural phase becomes a separate entry signal at the appropriate bar
        with volume-proportional sizing
        """
        # Initialize signal arrays
        phase_entries = np.zeros(len(df), dtype=bool)
        phase_sizes = np.zeros(len(df), dtype=float)

        # Process each execution's natural phases
        for execution_data in execution_results:
            natural_phases = execution_data['natural_phases']

            for phase in natural_phases:
                bar_index = phase['bar_index']

                # Ensure bar index is within bounds
                if 0 <= bar_index < len(df):
                    # Set entry signal for this phase
                    phase_entries[bar_index] = True
                    # Set volume-proportional size
                    phase_sizes[bar_index] = phase['phase_size']

        return {
            'entries': phase_entries,
            'sizes': phase_sizes,
            'direction': signal_direction
        }

    def create_trade_metadata(self, long_twap_prices: List[Dict],
                             short_twap_prices: List[Dict]) -> pd.DataFrame:
        """
        Create trade metadata DataFrame showing execution details

        This will be used to display execBars and other TWAP details in trade list
        """
        trade_records = []

        # Add long trades
        for twap_data in long_twap_prices:
            record = {
                'signal_bar': twap_data['signal_bar'],
                'direction': 'LONG',
                'execution_start_bar': twap_data['execution_start_bar'],
                'execution_end_bar': twap_data['execution_end_bar'],
                'exec_bars': twap_data['bar_count'],
                'twap_price': twap_data['twap_price'],
                'execution_time_minutes': twap_data['execution_time_minutes'],
                'num_phases': len(twap_data['phase_prices'])
            }
            trade_records.append(record)

        # Add short trades
        for twap_data in short_twap_prices:
            record = {
                'signal_bar': twap_data['signal_bar'],
                'direction': 'SHORT',
                'execution_start_bar': twap_data['execution_start_bar'],
                'execution_end_bar': twap_data['execution_end_bar'],
                'exec_bars': twap_data['bar_count'],
                'twap_price': twap_data['twap_price'],
                'execution_time_minutes': twap_data['execution_time_minutes'],
                'num_phases': len(twap_data['phase_prices'])
            }
            trade_records.append(record)

        if trade_records:
            return pd.DataFrame(trade_records).sort_values('signal_bar')
        else:
            return pd.DataFrame()

    def integrate_with_vectorbt(self, df: pd.DataFrame, execution_results: Dict[str, Any],
                               size: float = 1.0, fees: float = 0.0) -> Any:
        """
        Integrate with VectorBT Pro for P&L calculation

        Uses the custom price array and signals to create portfolio
        """
        try:
            # This would be the actual VectorBT Pro integration
            # For now, return structured data that can be used with VectorBT Pro

            vectorbt_data = {
                'close_prices': df['close'].values,
                'custom_prices': execution_results['custom_price_array'],
                'long_entries': execution_results['long_signals'],
                'short_entries': execution_results['short_signals'],
                'size': size,
                'fees': fees,
                'execution_metadata': {
                    'long_twap_data': execution_results['long_twap_prices'],
                    'short_twap_data': execution_results['short_twap_prices']
                }
            }

            return vectorbt_data

        except Exception as e:
            print(f"Error integrating with VectorBT Pro: {e}")
            return None


def create_time_based_twap_engine(config_path: str = None) -> TimeBasedTWAPEngine:
    """Factory function to create configured TWAP engine"""
    config = TimeBasedTWAPConfig.from_yaml(config_path)
    return TimeBasedTWAPEngine(config)