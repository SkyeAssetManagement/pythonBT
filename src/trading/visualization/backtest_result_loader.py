#!/usr/bin/env python3
"""
Backtest Result Loader for Chart Visualizer
============================================
Loads trade lists and results from headless backtest CSV files
for display in the chart visualizer
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from src.trading.data.trade_data import TradeData, TradeCollection


class BacktestResultLoader:
    """
    Loads backtest results from organized folder structure
    for display in chart visualizer
    """

    def __init__(self, results_dir: str = "backtest_results"):
        self.results_dir = Path(results_dir)

    def list_available_runs(self) -> List[Dict[str, Any]]:
        """List all available backtest runs"""

        runs = []

        if not self.results_dir.exists():
            return runs

        for run_dir in self.results_dir.iterdir():
            if run_dir.is_dir():
                try:
                    metadata = self._load_run_metadata(run_dir)
                    if metadata:
                        runs.append({
                            'run_id': metadata['run_id'],
                            'strategy_name': metadata['strategy_name'],
                            'timestamp': metadata['timestamp'],
                            'execution_mode': metadata.get('execution_mode', 'standard'),
                            'parameters': metadata['parameters'],
                            'folder_path': str(run_dir)
                        })
                except Exception as e:
                    print(f"[LOADER] Could not load metadata for {run_dir.name}: {e}")

        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        return runs

    def get_latest_run(self, strategy_name: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get the latest run for a strategy, optionally matching parameters"""

        runs = self.list_available_runs()

        # Filter by strategy name
        strategy_runs = [run for run in runs if run['strategy_name'] == strategy_name]

        if not strategy_runs:
            return None

        # If parameters specified, try to match them
        if parameters:
            for run in strategy_runs:
                if self._parameters_match(run['parameters'], parameters):
                    return run

        # Return latest run if no parameter match or no parameters specified
        return strategy_runs[0]

    def load_trade_list(self, run_id: str) -> Optional[TradeCollection]:
        """Load trade list from backtest results"""

        run_dir = self._find_run_directory(run_id)
        if not run_dir:
            print(f"[LOADER] Run not found: {run_id}")
            return None

        trade_file = run_dir / "trades" / "trade_list.csv"
        if not trade_file.exists():
            print(f"[LOADER] Trade file not found: {trade_file}")
            return None

        try:
            # Load CSV
            df = pd.read_csv(trade_file)

            if df.empty:
                print(f"[LOADER] No trades found in {trade_file}")
                return TradeCollection([])

            # Convert to TradeData objects
            trades = []
            for _, row in df.iterrows():
                trade = TradeData(
                    bar_index=int(row.get('bar_index', 0)),
                    trade_type=str(row.get('trade_type', 'BUY')),
                    price=float(row.get('price', 0.0)),
                    timestamp=pd.to_datetime(row.get('datetime', '1970-01-01')),
                    size=float(row.get('size', 1.0)),
                    strategy=run_id.split('_')[0]  # Extract strategy name from run_id
                )

                # Add TWAP metadata if available
                if 'exec_bars' in row:
                    trade.metadata = {
                        'exec_bars': int(row['exec_bars']),
                        'execution_time_minutes': float(row.get('execution_time_minutes', 0.0)),
                        'num_phases': int(row.get('num_phases', 1)),
                        'total_volume': float(row.get('total_volume', 0.0))
                    }

                trades.append(trade)

            print(f"[LOADER] Loaded {len(trades)} trades from {trade_file}")
            return TradeCollection(trades)

        except Exception as e:
            print(f"[LOADER] Error loading trades from {trade_file}: {e}")
            return None

    def load_equity_curve(self, run_id: str) -> Optional[pd.DataFrame]:
        """Load equity curve from backtest results"""

        run_dir = self._find_run_directory(run_id)
        if not run_dir:
            return None

        equity_file = run_dir / "equity" / "equity_curve.csv"
        if not equity_file.exists():
            print(f"[LOADER] Equity file not found: {equity_file}")
            return None

        try:
            df = pd.read_csv(equity_file)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])

            print(f"[LOADER] Loaded equity curve with {len(df)} points")
            return df

        except Exception as e:
            print(f"[LOADER] Error loading equity curve: {e}")
            return None

    def load_run_parameters(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load parameters for a specific run"""

        run_dir = self._find_run_directory(run_id)
        if not run_dir:
            return None

        return self._load_run_metadata(run_dir)

    def _find_run_directory(self, run_id: str) -> Optional[Path]:
        """Find directory for a specific run ID"""

        if not self.results_dir.exists():
            return None

        # Try exact match first
        exact_path = self.results_dir / run_id
        if exact_path.exists() and exact_path.is_dir():
            return exact_path

        # Try directories that start with the run_id
        for run_dir in self.results_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith(run_id):
                return run_dir

        # Try partial match
        for run_dir in self.results_dir.iterdir():
            if run_dir.is_dir() and run_id in run_dir.name:
                return run_dir

        return None

    def _load_run_metadata(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        """Load metadata for a run directory"""

        param_file = run_dir / "parameters" / "run_params.json"
        if not param_file.exists():
            return None

        try:
            with open(param_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[LOADER] Error loading metadata from {param_file}: {e}")
            return None

    def _parameters_match(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> bool:
        """Check if two parameter dictionaries match"""

        # Simple comparison - could be made more sophisticated
        try:
            return json.dumps(params1, sort_keys=True) == json.dumps(params2, sort_keys=True)
        except Exception:
            return False

    def get_run_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary of a backtest run"""

        run_dir = self._find_run_directory(run_id)
        if not run_dir:
            return None

        metadata = self._load_run_metadata(run_dir)
        if not metadata:
            return None

        # Load trade count
        trade_count = 0
        trade_file = run_dir / "trades" / "trade_list.csv"
        if trade_file.exists():
            try:
                df = pd.read_csv(trade_file)
                trade_count = len(df)
            except Exception:
                pass

        # Check for equity data
        has_equity = (run_dir / "equity" / "equity_curve.csv").exists()

        # Check for code snapshot
        has_code = (run_dir / "code" / "strategy_snapshot.py").exists()

        summary = {
            'run_id': run_id,
            'metadata': metadata,
            'trade_count': trade_count,
            'has_equity_curve': has_equity,
            'has_code_snapshot': has_code,
            'folder_path': str(run_dir)
        }

        return summary


class ChartVisualizerIntegration:
    """
    Integration layer between backtest results and chart visualizer
    """

    def __init__(self, results_dir: str = "backtest_results"):
        self.loader = BacktestResultLoader(results_dir)

    def get_trades_for_strategy(self,
                               strategy_name: str,
                               parameters: Dict[str, Any]) -> Tuple[Optional[TradeCollection], Dict[str, Any]]:
        """
        Get trades for chart visualizer display

        Returns:
            Tuple of (TradeCollection, metadata)
        """

        print(f"[INTEGRATION] Looking for trades: {strategy_name} with parameters {parameters}")

        # Try to find matching run
        run = self.loader.get_latest_run(strategy_name, parameters)

        if not run:
            print(f"[INTEGRATION] No matching run found for {strategy_name}")
            return None, {}

        print(f"[INTEGRATION] Found matching run: {run['run_id']}")

        # Load trades
        trades = self.loader.load_trade_list(run['run_id'])

        if not trades:
            print(f"[INTEGRATION] No trades loaded from {run['run_id']}")
            return None, {}

        # Prepare metadata for chart display
        metadata = {
            'run_id': run['run_id'],
            'execution_mode': run['execution_mode'],
            'timestamp': run['timestamp'],
            'parameters': run['parameters'],
            'trade_count': len(trades),
            'has_twap_data': any(hasattr(trade, 'metadata') and 'exec_bars' in (trade.metadata or {}) for trade in trades)
        }

        print(f"[INTEGRATION] Loaded {len(trades)} trades with TWAP data: {metadata['has_twap_data']}")

        return trades, metadata

    def trigger_headless_backtest(self,
                                 strategy_name: str,
                                 parameters: Dict[str, Any],
                                 data_file: str,
                                 execution_mode: str = "twap") -> str:
        """
        Trigger a new headless backtest and return run_id

        This allows the chart visualizer to request fresh backtests
        """

        print(f"[INTEGRATION] Triggering headless backtest: {strategy_name}")

        try:
            from src.trading.backtesting.headless_backtester import HeadlessBacktester

            backtester = HeadlessBacktester()
            run_id = backtester.run_backtest(
                strategy_name=strategy_name,
                parameters=parameters,
                data_file=data_file,
                execution_mode=execution_mode
            )

            print(f"[INTEGRATION] Headless backtest completed: {run_id}")
            return run_id

        except Exception as e:
            print(f"[INTEGRATION] Headless backtest failed: {e}")
            raise


# Example usage
def main():
    """Demo of backtest result loading"""

    loader = BacktestResultLoader()

    # List available runs
    runs = loader.list_available_runs()
    print(f"Found {len(runs)} backtest runs:")

    for run in runs:
        print(f"  {run['run_id']}: {run['strategy_name']} ({run['execution_mode']}) - {run['timestamp']}")

    # Load latest SMA run
    if runs:
        latest_run = runs[0]
        trades = loader.load_trade_list(latest_run['run_id'])
        if trades:
            print(f"Loaded {len(trades)} trades from latest run")


if __name__ == "__main__":
    main()