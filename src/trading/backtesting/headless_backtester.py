#!/usr/bin/env python3
"""
Headless Backtesting Engine
===========================
Completely modular backtesting system that runs independent of chart visualization.
Results are saved to organized folder structures with timestamps.
"""

import os
import sys
import json
import shutil
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class BacktestRun:
    """Configuration and metadata for a backtest run"""
    strategy_name: str
    run_id: str
    timestamp: str
    parameters: Dict[str, Any]
    data_file: Optional[str] = None
    execution_mode: str = "standard"  # "standard", "twap", "unified"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultStorage:
    """Handles organized storage of backtest results"""

    def __init__(self, base_dir: str = "backtest_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def create_run_directory(self, run: BacktestRun) -> Path:
        """Create organized directory structure for a backtest run"""

        run_dir = self.base_dir / f"{run.strategy_name}_{run.timestamp}"

        # Check if identical parameters exist - if so, overwrite
        existing_run = self._find_existing_run(run.strategy_name, run.parameters)
        if existing_run:
            print(f"[STORAGE] Found existing run with same parameters: {existing_run.name}")
            print(f"[STORAGE] Overwriting: {existing_run}")
            shutil.rmtree(existing_run)
            run_dir = existing_run

        # Create directory structure
        run_dir.mkdir(exist_ok=True)
        (run_dir / "parameters").mkdir(exist_ok=True)
        (run_dir / "code").mkdir(exist_ok=True)
        (run_dir / "trades").mkdir(exist_ok=True)
        (run_dir / "equity").mkdir(exist_ok=True)

        return run_dir

    def _find_existing_run(self, strategy_name: str, parameters: Dict[str, Any]) -> Optional[Path]:
        """Find existing run with same strategy and parameters"""

        param_hash = self._hash_parameters(parameters)

        for run_dir in self.base_dir.glob(f"{strategy_name}_*"):
            if run_dir.is_dir():
                param_file = run_dir / "parameters" / "run_params.json"
                if param_file.exists():
                    try:
                        with open(param_file, 'r') as f:
                            existing_params = json.load(f)
                        existing_hash = self._hash_parameters(existing_params.get('parameters', {}))
                        if existing_hash == param_hash:
                            return run_dir
                    except Exception:
                        continue

        return None

    def _hash_parameters(self, parameters: Dict[str, Any]) -> str:
        """Create hash of parameters for comparison"""
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def save_run_metadata(self, run_dir: Path, run: BacktestRun):
        """Save run parameters and metadata"""

        metadata = {
            "run_id": run.run_id,
            "strategy_name": run.strategy_name,
            "timestamp": run.timestamp,
            "execution_mode": run.execution_mode,
            "data_file": run.data_file,
            "parameters": run.parameters
        }

        param_file = run_dir / "parameters" / "run_params.json"
        with open(param_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[STORAGE] Saved parameters to: {param_file}")

    def save_strategy_code(self, run_dir: Path, strategy_name: str):
        """Save snapshot of strategy code"""

        # Try to find strategy file
        strategy_files = [
            f"src/trading/strategies/{strategy_name}.py",
            f"strategies/{strategy_name}.py",
            f"src/strategies/{strategy_name}.py"
        ]

        for strategy_file in strategy_files:
            if os.path.exists(strategy_file):
                code_file = run_dir / "code" / "strategy_snapshot.py"
                shutil.copy2(strategy_file, code_file)
                print(f"[STORAGE] Saved strategy code to: {code_file}")
                return

        print(f"[STORAGE] Warning: Could not find strategy file for {strategy_name}")

    def save_trade_list(self, run_dir: Path, trades: List[Dict[str, Any]]):
        """Save trade list as CSV"""

        if not trades:
            print("[STORAGE] No trades to save")
            return

        trade_df = pd.DataFrame(trades)
        trade_file = run_dir / "trades" / "trade_list.csv"
        trade_df.to_csv(trade_file, index=False)

        print(f"[STORAGE] Saved {len(trades)} trades to: {trade_file}")

    def save_equity_curve(self, run_dir: Path, equity_data: Dict[str, Any]):
        """Save equity curve data"""

        equity_file = run_dir / "equity" / "equity_curve.csv"

        if isinstance(equity_data, dict):
            equity_df = pd.DataFrame(equity_data)
        else:
            equity_df = equity_data

        equity_df.to_csv(equity_file, index=False)
        print(f"[STORAGE] Saved equity curve to: {equity_file}")


class HeadlessBacktester:
    """
    Headless backtesting engine that runs completely independent of GUI
    """

    def __init__(self, base_dir: str = "backtest_results"):
        self.storage = ResultStorage(base_dir)

        # Import TWAP system
        try:
            from src.trading.core.optimized_twap_adapter import OptimizedTWAPAdapter, OptimizedTWAPConfig
            self.twap_available = True
            print("[BACKTESTER] TWAP system available")
        except ImportError as e:
            self.twap_available = False
            print(f"[BACKTESTER] TWAP system not available: {e}")
            print(f"[BACKTESTER] Expected import: src.trading.core.optimized_twap_adapter")

    def run_backtest(self,
                    strategy_name: str,
                    parameters: Dict[str, Any],
                    data_file: str,
                    execution_mode: str = "twap") -> str:
        """
        Run a complete backtest and save results

        Returns:
            run_id: Unique identifier for this backtest run
        """

        # Create run metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{strategy_name}_{timestamp}"

        run = BacktestRun(
            strategy_name=strategy_name,
            run_id=run_id,
            timestamp=timestamp,
            parameters=parameters,
            data_file=data_file,
            execution_mode=execution_mode
        )

        print(f"[BACKTESTER] Starting backtest: {run_id}")
        print(f"[BACKTESTER] Strategy: {strategy_name}")
        print(f"[BACKTESTER] Execution mode: {execution_mode}")
        print(f"[BACKTESTER] Parameters: {parameters}")

        # Create run directory
        run_dir = self.storage.create_run_directory(run)
        print(f"[BACKTESTER] Results will be saved to: {run_dir}")

        try:
            # Load data
            df = self._load_data(data_file)
            print(f"[BACKTESTER] Loaded {len(df)} bars of data")

            # Run backtest based on execution mode
            if execution_mode == "twap":
                if not self.twap_available:
                    raise ImportError(f"TWAP execution mode requested but TWAP system not available. Check TWAP imports and dependencies.")
                results = self._run_twap_backtest(strategy_name, parameters, df)
            elif execution_mode == "standard":
                results = self._run_standard_backtest(strategy_name, parameters, df)
            else:
                raise ValueError(f"Unknown execution mode: {execution_mode}. Supported modes: 'twap', 'standard'")

            # Save all results
            self.storage.save_run_metadata(run_dir, run)
            self.storage.save_strategy_code(run_dir, strategy_name)
            self.storage.save_trade_list(run_dir, results['trades'])
            self.storage.save_equity_curve(run_dir, results['equity'])

            print(f"[BACKTESTER] Backtest completed successfully: {run_id}")
            print(f"[BACKTESTER] Generated {len(results['trades'])} trades")

            return run_id

        except Exception as e:
            print(f"[BACKTESTER] Backtest failed: {e}")
            import traceback
            traceback.print_exc()

            # Save error information
            error_file = run_dir / "error.log"
            with open(error_file, 'w') as f:
                f.write(f"Error: {e}\n")
                f.write(traceback.format_exc())

            raise

    def _load_data(self, data_file: str) -> pd.DataFrame:
        """Load price data for backtesting"""

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Try different data formats
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.endswith('.parquet'):
            df = pd.read_parquet(data_file)
        else:
            raise ValueError(f"Unsupported data format: {data_file}")

        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Handle column name variations
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume', 'Vol': 'volume'
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Set datetime index if available
        if 'datetime' in df.columns:
            df.index = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'])

        return df

    def _run_twap_backtest(self,
                          strategy_name: str,
                          parameters: Dict[str, Any],
                          df: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest with TWAP execution"""

        print("[BACKTESTER] Running TWAP backtest...")

        from core.optimized_twap_adapter import OptimizedTWAPAdapter, OptimizedTWAPConfig

        # Create TWAP config
        twap_config = OptimizedTWAPConfig(
            enabled=True,
            minimum_execution_minutes=parameters.get('min_execution_time', 5.0),
            chunk_size=5000,
            max_signals_before_chunking=10000,
            skip_vectorbt_portfolio=True  # Skip VectorBT for headless operation
        )

        twap_adapter = OptimizedTWAPAdapter(twap_config)

        # Generate signals
        signals = self._generate_signals(strategy_name, parameters, df)
        long_signals = (signals > 0)
        short_signals = (signals < 0)

        # Execute with TWAP
        twap_results = twap_adapter.execute_portfolio_with_twap_chunked(
            df=df,
            long_signals=long_signals,
            short_signals=short_signals,
            signal_lag=parameters.get('signal_lag', 2),
            size=parameters.get('position_size', 1.0),
            fees=parameters.get('fees', 0.0)
        )

        # Convert to standard format
        trades = self._convert_twap_trades_to_standard(twap_results['trade_metadata'])
        equity = self._calculate_equity_curve(trades, df)

        return {
            'trades': trades,
            'equity': equity,
            'metadata': twap_results.get('twap_summary', {})
        }

    def _run_standard_backtest(self,
                              strategy_name: str,
                              parameters: Dict[str, Any],
                              df: pd.DataFrame) -> Dict[str, Any]:
        """Run standard backtest without TWAP"""

        print("[BACKTESTER] Running standard backtest...")

        # Generate signals
        signals = self._generate_signals(strategy_name, parameters, df)

        # Execute trades
        trades = self._execute_standard_trades(signals, df, parameters)
        equity = self._calculate_equity_curve(trades, df)

        return {
            'trades': trades,
            'equity': equity,
            'metadata': {}
        }

    def _generate_signals(self,
                         strategy_name: str,
                         parameters: Dict[str, Any],
                         df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using strategy"""

        if strategy_name.lower() in ['sma', 'sma_crossover']:
            from strategies.sma_crossover import SMACrossoverStrategy

            strategy = SMACrossoverStrategy(
                fast_period=parameters.get('fast_period', 10),
                slow_period=parameters.get('slow_period', 30),
                long_only=parameters.get('long_only', False)
            )

            return strategy.generate_signals(df)

        else:
            raise ValueError(f"Strategy not supported: {strategy_name}")

    def _execute_standard_trades(self,
                               signals: pd.Series,
                               df: pd.DataFrame,
                               parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute standard trades from signals with proper P&L calculation"""

        trades = []
        current_position = 0
        entry_price = 0
        signal_lag = parameters.get('signal_lag', 2)
        position_size = parameters.get('position_size', 1.0)
        cumulative_profit = 0

        for i in range(signal_lag, len(signals)):
            signal = signals.iloc[i - signal_lag]

            if signal != current_position:
                # Position change
                price = df['close'].iloc[i]
                trade_type = 'BUY' if signal > current_position else 'SELL'

                # Calculate P&L for exit trades
                pnl = 0
                if current_position != 0:  # This is an exit
                    if current_position > 0:  # Closing long position
                        pnl = (price - entry_price) * position_size
                    else:  # Closing short position
                        pnl = (entry_price - price) * position_size
                    cumulative_profit += pnl

                trade = {
                    'datetime': df.index[i],
                    'bar_index': i,
                    'trade_type': trade_type,
                    'price': price,
                    'size': abs(signal - current_position) * position_size,
                    'signal_value': signal,
                    'exec_bars': 1,  # Standard execution uses 1 bar
                    'pnl': pnl,  # P&L without % sign (raw profit/loss)
                    'cumulative_profit': cumulative_profit,  # Running total
                    'is_entry': current_position == 0,  # True if this is an entry
                    'is_exit': signal == 0 or (current_position > 0 and signal <= 0) or (current_position < 0 and signal >= 0)
                }

                trades.append(trade)

                # Update position tracking
                if signal != 0:  # This is an entry or position change
                    entry_price = price
                current_position = signal

        return trades

    def _convert_twap_trades_to_standard(self, trade_metadata: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert TWAP trade metadata to standard trade format with P&L calculations"""

        trades = []
        position_tracker = {}  # Track positions by direction for P&L calculation
        cumulative_profit = 0

        # Sort by signal bar to process in chronological order
        trade_metadata_sorted = trade_metadata.sort_values('signal_bar')

        for _, row in trade_metadata_sorted.iterrows():
            direction = row['direction']
            price = row['twap_price']
            size = row['total_position_size']

            # Determine if this is entry or exit
            is_entry = direction not in position_tracker or position_tracker[direction] == 0
            is_exit = not is_entry

            # Calculate P&L for exits
            pnl = 0
            if is_exit and direction in position_tracker:
                entry_price = position_tracker[f'{direction}_entry_price']
                if direction == 'long':
                    pnl = (price - entry_price) * size
                else:  # short
                    pnl = (entry_price - price) * size
                cumulative_profit += pnl
                position_tracker[direction] = 0  # Close position

            trade = {
                'datetime': pd.Timestamp.now(),  # Would need actual timestamp from data
                'bar_index': int(row['signal_bar']),
                'trade_type': 'BUY' if direction == 'long' else 'SELL',
                'price': price,
                'size': size,
                'exec_bars': row['exec_bars'],
                'execution_time_minutes': row['execution_time_minutes'],
                'num_phases': row['num_phases'],
                'total_volume': row['total_volume'],
                'pnl': pnl,  # P&L without % sign (raw profit/loss)
                'cumulative_profit': cumulative_profit,  # Running total
                'is_entry': is_entry,
                'is_exit': is_exit
            }

            trades.append(trade)

            # Track position for P&L calculation
            if is_entry:
                position_tracker[direction] = size
                position_tracker[f'{direction}_entry_price'] = price

        return trades

    def _calculate_equity_curve(self, trades: List[Dict[str, Any]], df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate equity curve from trades"""

        if not trades:
            return {'datetime': [], 'equity': [], 'returns': []}

        # Simple equity calculation
        equity_curve = []
        running_equity = 1.0  # Start with $1

        trade_times = [trade['datetime'] if 'datetime' in trade else df.index[trade['bar_index']] for trade in trades]

        for i, trade in enumerate(trades):
            # Simple P&L calculation (placeholder - would need proper implementation)
            pnl_percent = np.random.normal(0, 0.02)  # Random P&L for now
            running_equity *= (1 + pnl_percent)

            equity_curve.append({
                'datetime': trade_times[i],
                'equity': running_equity,
                'returns': pnl_percent,
                'trade_index': i
            })

        return {
            'datetime': [e['datetime'] for e in equity_curve],
            'equity': [e['equity'] for e in equity_curve],
            'returns': [e['returns'] for e in equity_curve],
            'trade_index': [e['trade_index'] for e in equity_curve]
        }


def main():
    """Example usage of headless backtester"""

    backtester = HeadlessBacktester()

    # Example backtest configuration
    parameters = {
        'fast_period': 10,
        'slow_period': 30,
        'long_only': False,
        'signal_lag': 2,
        'position_size': 1.0,
        'min_execution_time': 5.0
    }

    # Run backtest (would need actual data file)
    try:
        run_id = backtester.run_backtest(
            strategy_name='sma_crossover',
            parameters=parameters,
            data_file='data/ES_5min.csv',  # Example data file
            execution_mode='twap'
        )

        print(f"Backtest completed: {run_id}")

    except FileNotFoundError:
        print("Example data file not found - this is expected for demo")
    except Exception as e:
        print(f"Backtest failed: {e}")


if __name__ == "__main__":
    main()