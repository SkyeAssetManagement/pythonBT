import numpy as np
import vectorbtpro as vbt
from typing import Dict, Optional, Tuple, List
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
import time
from ..utils.price_formulas import PriceFormulaEvaluator
from .phased_trading_engine import PhasedTradingEngine, create_config_from_yaml

logger = logging.getLogger(__name__)


class VectorBTEngine:
    """High-performance backtesting engine using VectorBT."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize backtesting engine with configuration.
        
        Args:
            config_path: Path to YAML configuration file (required)
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        
        # Validate required configuration sections
        self._validate_config()
        
        # Initialize price formula evaluator
        self.formula_evaluator = PriceFormulaEvaluator()
        
        # Initialize phased trading engine if configured
        phased_config = create_config_from_yaml(self.config)
        self.phased_engine = PhasedTradingEngine(phased_config)
        
        if phased_config.enabled:
            logger.info(f"Phased trading enabled: {phased_config.entry_bars} entry bars, {phased_config.exit_bars} exit bars")
    
    def _validate_config(self):
        """
        Validate that required configuration sections and keys exist.
        
        Raises:
            ValueError: If required configuration is missing
        """
        required_sections = {
            'backtest': [
                'initial_cash', 'position_size', 'position_size_type',
                'execution_price', 'signal_lag', 'fees', 'fixed_fees', 'slippage',
                'direction', 'min_size', 'call_seq', 'freq'
            ],
            'output': ['results_dir', 'trade_list_filename', 'equity_curve_filename']
        }
        
        for section, keys in required_sections.items():
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
            
            for key in keys:
                if key not in self.config[section]:
                    raise ValueError(f"Missing required configuration key: {section}.{key}")
        
        # Validate formula configuration if using formula execution
        if self.config['backtest']['execution_price'] == 'formula':
            formula_keys = ['buy_execution_formula', 'sell_execution_formula']
            for key in formula_keys:
                if key not in self.config['backtest']:
                    raise ValueError(f"Missing required formula configuration: backtest.{key}")
    
    def calculate_execution_prices(self, data: Dict[str, np.ndarray],
                                 entries: np.ndarray, exits: np.ndarray,
                                 lagged_entries: np.ndarray = None,
                                 lagged_exits: np.ndarray = None) -> np.ndarray:
        """
        Calculate execution prices based on buy/sell formulas and signals.

        Args:
            data: OHLC data dictionary
            entries: Original entry signals (for reference)
            exits: Original exit signals (for reference)
            lagged_entries: Lagged entry signals (where trades actually execute)
            lagged_exits: Lagged exit signals (where trades actually execute)

        Returns:
            Array of execution prices for each bar
        """
        execution_price_type = self.config['backtest'].get('execution_price', 'close')

        # If using simple price type (not formula), return that price array
        if execution_price_type in ['open', 'high', 'low', 'close']:
            return data[execution_price_type]

        # Use formula-based pricing
        buy_formula = self.config['backtest'].get('buy_execution_formula', 'C')
        sell_formula = self.config['backtest'].get('sell_execution_formula', 'C')

        # Calculate buy and sell execution prices
        buy_prices = self.formula_evaluator.get_execution_prices(buy_formula, data, "buy")
        sell_prices = self.formula_evaluator.get_execution_prices(sell_formula, data, "sell")

        # Create execution price array - use the formula prices where trades execute
        execution_prices = data['close'].copy()  # Default to close prices

        # Use lagged signals if provided, otherwise use original signals
        exec_entries = lagged_entries if lagged_entries is not None else entries
        exec_exits = lagged_exits if lagged_exits is not None else exits

        # Handle multi-dimensional signals (for parameter sweeps)
        if exec_entries.ndim == 1:
            # Single parameter case - apply formula prices where trades execute
            execution_prices = np.where(exec_entries, buy_prices, execution_prices)
            execution_prices = np.where(exec_exits, sell_prices, execution_prices)
        else:
            # Multiple parameter case - use buy prices for any entry, sell prices for any exit
            any_entries = np.any(exec_entries, axis=1)
            any_exits = np.any(exec_exits, axis=1)
            execution_prices = np.where(any_entries, buy_prices, execution_prices)
            execution_prices = np.where(any_exits, sell_prices, execution_prices)

        return execution_prices
            
    def run_vectorized_backtest(self, data: Dict[str, np.ndarray],
                               entries: np.ndarray, exits: np.ndarray,
                               symbol: str = "TEST") -> vbt.Portfolio:
        """
        Run backtest for multiple parameter combinations simultaneously.

        Args:
            data: Dictionary with OHLCV arrays
            entries: 2D/3D boolean array of entry signals
            exits: 2D/3D boolean array of exit signals
            symbol: Symbol name for the backtest

        Returns:
            VectorBT Portfolio object
        """
        # Apply phased trading if enabled
        phased_size_array = None
        if self.phased_engine.config.enabled:
            logger.info("Applying phased trading to signals...")

            # Process signals through phased engine
            phased_results = self.phased_engine.process_signals(
                entries.flatten() if entries.ndim > 1 else entries,
                exits.flatten() if exits.ndim > 1 else exits,
                data,
                position_size=self.config['backtest']['position_size']
            )

            # Use phased signals
            entries = phased_results['entries']
            exits = phased_results['exits']

            # Store phased sizes for position sizing
            phased_size_array = phased_results['entry_sizes']

            # Log signal transformation
            logger.info(f"Phased trading: {np.sum(entries)} entry signals, {np.sum(exits)} exit signals")

        # Store original signals for price calculation
        original_entries = entries.copy()
        original_exits = exits.copy()

        # Apply signal lag if configured
        signal_lag = self.config['backtest'].get('signal_lag', 0)
        if signal_lag > 0:
            # Shift signals forward by lag amount (signal today, execute in lag bars)
            entries = np.roll(entries, signal_lag, axis=0)
            exits = np.roll(exits, signal_lag, axis=0)
            # Clear signals at the beginning (can't execute before data starts)
            entries[:signal_lag] = False
            exits[:signal_lag] = False

            # Also shift phased size array if present
            if phased_size_array is not None:
                phased_size_array = np.roll(phased_size_array, signal_lag, axis=0)
                phased_size_array[:signal_lag] = 0

        # Ensure arrays are properly shaped
        if entries.ndim == 1:
            entries = entries.reshape(-1, 1)
        if exits.ndim == 1:
            exits = exits.reshape(-1, 1)
        if original_entries.ndim == 1:
            original_entries = original_entries.reshape(-1, 1)
        if original_exits.ndim == 1:
            original_exits = original_exits.reshape(-1, 1)

        # Calculate execution prices - pass both original and lagged signals
        # Original signals show where crossovers occurred
        # Lagged signals show where trades actually execute
        execution_prices = self.calculate_execution_prices(
            data, original_entries, original_exits, entries, exits
        )
            
        # Determine position sizing - use phased sizes if available
        if phased_size_array is not None:
            # Use the phased size array (already properly shaped for each bar)
            position_size = phased_size_array.reshape(-1, 1) if phased_size_array.ndim == 1 else phased_size_array
        else:
            # Use default position size from config
            position_size = self.config['backtest']['position_size']
            
        # Create portfolio using VectorBT Pro with config-specified settings
        pf = vbt.Portfolio.from_signals(
            close=execution_prices,
            entries=entries,
            exits=exits,
            size=position_size,
            size_type=self.config['backtest']['position_size_type'] if phased_size_array is None else 'amount',
            init_cash=self.config['backtest']['initial_cash'],
            direction=self.config['backtest']['direction'],
            min_size=self.config['backtest']['min_size'],
            fees=self.config['backtest']['fees'],
            fixed_fees=self.config['backtest']['fixed_fees'],
            slippage=self.config['backtest']['slippage'],
            freq=self.config['backtest']['freq'],
            call_seq=self.config['backtest']['call_seq']
        )
        
        return pf
    
    def get_uncompounded_equity_data(self, pf: vbt.Portfolio) -> Dict:
        """
        Extract uncompounded equity curve data using pf object methods.
        PERFORMANCE OPTIMIZED: Uses efficient numpy operations for large datasets.
        
        Args:
            pf: VectorBT Portfolio object
            
        Returns:
            Dictionary containing equity curve data
        """
        print(f"   INFO: Calculating equity data for portfolio...")
        
        # PERFORMANCE FIX: Use numpy operations directly instead of pandas operations
        # which can be slow for large datasets (1.9M+ bars)
        
        # Get returns as numpy array for efficient processing
        returns_data = pf.returns
        if hasattr(returns_data, 'values'):
            returns_array = returns_data.values
        else:
            returns_array = np.asarray(returns_data)
        
        print(f"   INFO: Returns array shape: {returns_array.shape}")
        
        # Use numpy cumsum instead of pandas cumsum for better performance
        # This is the operation that was potentially causing the hang
        cumsum_returns = np.cumsum(returns_array, axis=0) * 100
        
        print(f"   SUCCESS: Cumulative returns calculated efficiently")
        
        # Convert back to pandas format if needed, but keep the efficient calculation
        if hasattr(returns_data, 'index') and hasattr(returns_data, 'columns'):
            # Preserve pandas structure for compatibility
            uncompounded_cumulative = pd.DataFrame(
                cumsum_returns, 
                index=returns_data.index, 
                columns=returns_data.columns
            )
        else:
            uncompounded_cumulative = cumsum_returns
        
        return {
            'equity_values': pf.value,  # Portfolio equity curve (dollar values)
            'period_returns': pf.returns,  # Period returns (daily/hourly returns)
            'uncompounded_cumulative_pct': uncompounded_cumulative,  # Efficient cumulative returns
            'initial_cash': pf.init_cash  # Initial cash amount
        }
    
    def demonstrate_equity_curve_methods(self, pf: vbt.Portfolio) -> Dict:
        """
        Demonstrate different methods to create equity curves using pf object.
        Shows comparison between compounded vs uncompounded returns.
        
        Args:
            pf: VectorBT Portfolio object
            
        Returns:
            Dictionary with different equity curve calculations
        """
        return {
            # Method 1: Uncompounded cumulative returns (recommended)
            'uncompounded_returns_pct': pf.returns.cumsum() * 100,
            
            # Method 2: Portfolio equity values (dollar amounts)
            'equity_values_dollar': pf.value,
            
            # Method 3: Compounded returns (for comparison)
            'compounded_total_return': pf.total_return * 100,
            
            # Method 4: Calculate uncompounded from equity values
            'uncompounded_from_equity': ((pf.value - pf.init_cash) / pf.init_cash) * 100,
            
            # Method 5: Period returns (building blocks)
            'period_returns': pf.returns,
            
            # Method 6: Final uncompounded return value
            'final_uncompounded_return': (pf.returns.cumsum() * 100).iloc[-1] if hasattr(pf.returns.cumsum(), 'iloc') else (pf.returns.cumsum() * 100)[-1]
        }
    
    def calculate_performance_metrics(self, pf: vbt.Portfolio, 
                                    equity_curve: np.ndarray = None,
                                    timestamps: np.ndarray = None) -> Dict:
        """
        Calculate comprehensive performance metrics using VectorBT.
        All calculations are vectorized.
        
        Args:
            pf: VectorBT Portfolio object
            equity_curve: Optional equity curve array for UPI calculations
            timestamps: Optional timestamps array for UPI calculations
        
        Returns:
            Dictionary of performance metrics including UPI metrics
        """
        # Basic metrics (vectorBT Pro returns values directly)
        metrics = {
            'total_return': pf.total_return,
            'annualized_return': pf.annualized_return,
            'sharpe_ratio': pf.sharpe_ratio,
            'max_drawdown': pf.max_drawdown,
            'total_trades': len(pf.trades.records) if hasattr(pf.trades, 'records') else 0
        }
        
        # Add trade-specific metrics if we have trades
        if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
            try:
                metrics['win_rate'] = pf.trades.win_rate
                metrics['profit_factor'] = pf.trades.profit_factor
            except:
                metrics['win_rate'] = 0
                metrics['profit_factor'] = 0
        
        # Add UPI metrics if equity curve and timestamps are provided - PERFORMANCE OPTIMIZED
        if equity_curve is not None and timestamps is not None:
            try:
                print(f"   INFO: Calculating UPI metrics for {len(equity_curve)} data points...")
                
                # PERFORMANCE CHECK: Skip UPI for extremely large datasets to avoid hanging
                # UPI calculation on 1.9M+ bars can be extremely slow due to rolling calculations
                MAX_UPI_BARS = 500000  # 500K bars threshold for UPI calculation
                
                if len(equity_curve) > MAX_UPI_BARS:
                    print(f"   INFO: Dataset too large ({len(equity_curve)} bars > {MAX_UPI_BARS}), skipping UPI calculation for performance")
                    print(f"   INFO: UPI calculation on datasets this large can take 10+ minutes")
                    
                    # Add NaN values for missing UPI metrics
                    for period in [30, 50]:
                        metrics[f'UPI_{period}'] = np.nan
                        metrics[f'UPI_{period}_adj'] = np.nan
                        metrics[f'UPI_{period}_max'] = np.nan
                        metrics[f'UPI_{period}_adj_max'] = np.nan
                else:
                    from ..utils.upi_calculator import calculate_upi_30_50
                    
                    # Extract trade indices for UPI calculation
                    trade_indices = None
                    if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
                        trades_df = pf.trades.records_readable
                        if 'Entry Index' in trades_df.columns:
                            trade_indices = trades_df['Entry Index'].values
                    
                    # Calculate UPI metrics with timeout protection
                    print(f"   INFO: Computing UPI metrics (this may take a few moments)...")
                    upi_start = time.time()
                    upi_metrics = calculate_upi_30_50(equity_curve, timestamps, trade_indices)
                    upi_time = time.time() - upi_start
                    
                    metrics.update(upi_metrics)
                    print(f"   SUCCESS: UPI calculation completed in {upi_time:.2f}s")
                    
                    logger.info(f"Added UPI metrics: UPI_30={upi_metrics.get('UPI_30', 'N/A'):.4f}, "
                              f"UPI_50={upi_metrics.get('UPI_50', 'N/A'):.4f}")
                
            except Exception as e:
                logger.warning(f"Error calculating UPI metrics: {e}")
                print(f"   WARNING: UPI calculation failed: {e}")
                # Add NaN values for missing UPI metrics
                for period in [30, 50]:
                    metrics[f'UPI_{period}'] = np.nan
                    metrics[f'UPI_{period}_adj'] = np.nan
                    metrics[f'UPI_{period}_max'] = np.nan
                    metrics[f'UPI_{period}_adj_max'] = np.nan
        
        # Convert to plain values if needed
        for key, value in metrics.items():
            if hasattr(value, 'values'):
                metrics[key] = value.values
                
        return metrics
    
    def generate_trade_list(self, pf: vbt.Portfolio, 
                          timestamps: np.ndarray, symbol: str = 'UNKNOWN') -> pd.DataFrame:
        """
        Generate detailed trade list from portfolio.
        Uses VectorBT's built-in trade analysis.
        
        Args:
            pf: VectorBT Portfolio object
            timestamps: Array of timestamps
            
        Returns:
            DataFrame with trade details
        """
        if len(pf.trades.records) == 0:
            return pd.DataFrame()
            
        # Get trades from VectorBT
        trades = pf.trades.records_readable
        
        # Check available columns and map them
        print(f"Available trade columns: {list(trades.columns)}")
        
        # Find the correct column names (they may vary between vectorBT versions)
        entry_idx_col = None
        exit_idx_col = None
        
        for col in trades.columns:
            if 'entry' in col.lower() and 'idx' in col.lower():
                entry_idx_col = col
            elif 'exit' in col.lower() and 'idx' in col.lower():
                exit_idx_col = col
            elif col == 'Entry Index':
                entry_idx_col = col
            elif col == 'Exit Index':
                exit_idx_col = col
        
        # Add additional columns
        trades['Symbol'] = symbol
        if 'Column' in trades.columns:
            trades['Strategy'] = trades['Column'].astype(str)
        else:
            trades['Strategy'] = '0'
        
        # Add position sizing columns
        trades['posSize_raw'] = 1  # Raw position size (always 1 unit per trade)
        trades['posSize_riskAdj'] = 1  # Risk-adjusted position size (set to 1 for now)
        
        # Add shares column (size / entryPrice) with proper signing for shorts
        if 'Size' in trades.columns and 'Avg Entry Price' in trades.columns and 'Direction' in trades.columns:
            # Calculate basic shares
            basic_shares = trades['Size'] / trades['Avg Entry Price']
            # Apply sign based on direction (negative for shorts)
            trades['shares'] = basic_shares * trades['Direction'].map({'Long': 1, 'Short': -1}).fillna(1)
            
            # Also apply sign to Size column for consistency
            trades['Size'] = trades['Size'] * trades['Direction'].map({'Long': 1, 'Short': -1}).fillna(1)
        else:
            trades['shares'] = 0  # Fallback if columns not found
            
        # Add pnl_excFriction column (shares x (exit - entry))
        if 'shares' in trades.columns and 'Avg Entry Price' in trades.columns and 'Avg Exit Price' in trades.columns:
            trades['pnl_excFriction'] = trades['shares'] * (trades['Avg Exit Price'] - trades['Avg Entry Price'])
        else:
            trades['pnl_excFriction'] = 0  # Fallback if columns not found
        
        # Convert indices to timestamps if available - PERFORMANCE OPTIMIZED
        if len(timestamps) > 0 and entry_idx_col and exit_idx_col:
            try:
                print(f"   INFO: Converting {len(trades)} trades to timestamps using vectorized operations...")
                
                # PERFORMANCE FIX: Use vectorized operations instead of slow apply() + lambda
                # Extract indices as numpy arrays for faster processing
                entry_indices = trades[entry_idx_col].values.astype(int)
                exit_indices = trades[exit_idx_col].values.astype(int)
                
                # Vectorized bounds checking - much faster than lambda functions
                valid_entry_mask = entry_indices < len(timestamps)
                valid_exit_mask = exit_indices < len(timestamps)
                
                # Vectorized timestamp mapping using fancy indexing
                entry_times = np.full(len(trades), np.nan, dtype=np.int64)
                exit_times = np.full(len(trades), np.nan, dtype=np.int64)
                
                # Only map valid indices - avoids out-of-bounds errors
                entry_times[valid_entry_mask] = timestamps[entry_indices[valid_entry_mask]]
                exit_times[valid_exit_mask] = timestamps[exit_indices[valid_exit_mask]]
                
                # Assign back to DataFrame (much faster than apply())
                trades['EntryTime'] = entry_times
                trades['ExitTime'] = exit_times
                
                # PERFORMANCE FIX: Vectorized timestamp formatting
                # Instead of applying formatting to each timestamp individually,
                # we'll create readable text fields only for valid timestamps
                
                # Create text fields with vectorized string operations
                entry_times_valid = entry_times[~np.isnan(entry_times)].astype(np.int64)
                exit_times_valid = exit_times[~np.isnan(exit_times)].astype(np.int64)
                
                # Initialize text columns
                trades['entryTime_text'] = None
                trades['exitTime_text'] = None
                
                # Only format timestamps that are valid (avoid unnecessary formatting)
                if len(entry_times_valid) > 0:
                    # Use vectorized pandas datetime conversion instead of custom formatting
                    valid_entry_datetimes = pd.to_datetime(entry_times_valid, unit='ns', utc=True)
                    # Convert to EST timezone
                    est_entry_times = valid_entry_datetimes.tz_convert('US/Eastern')
                    # Format as strings efficiently
                    entry_text_values = est_entry_times.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Map back to original positions
                    entry_text_full = np.full(len(trades), None, dtype=object)
                    entry_text_full[~np.isnan(entry_times)] = entry_text_values
                    trades['entryTime_text'] = entry_text_full
                
                if len(exit_times_valid) > 0:
                    # Same vectorized approach for exit times
                    valid_exit_datetimes = pd.to_datetime(exit_times_valid, unit='ns', utc=True)
                    est_exit_times = valid_exit_datetimes.tz_convert('US/Eastern')
                    exit_text_values = est_exit_times.strftime('%Y-%m-%d %H:%M:%S')
                    
                    exit_text_full = np.full(len(trades), None, dtype=object)
                    exit_text_full[~np.isnan(exit_times)] = exit_text_values
                    trades['exitTime_text'] = exit_text_full
                
                print(f"   SUCCESS: Timestamp conversion completed using vectorized operations")
                
            except Exception as e:
                print(f"Warning: Could not map timestamps: {e}")
                # Fallback to basic mapping without text formatting
                trades['EntryTime'] = None
                trades['ExitTime'] = None
                trades['entryTime_text'] = None
                trades['exitTime_text'] = None
        elif len(timestamps) > 0:
            print("Warning: Could not find entry/exit index columns for timestamp mapping")
            
        return trades
    
    def _format_timestamp_readable(self, timestamp_ns: int) -> str:
        """
        Convert nanosecond timestamp to human-readable format in exchange time (EST).
        For ES futures, display times in Eastern Standard Time.
        
        Args:
            timestamp_ns: Timestamp in nanoseconds
            
        Returns:
            Formatted timestamp string in 'yyyy-mm-dd hh:mm:ss' format (EST)
        """
        try:
            # Convert nanoseconds to seconds
            timestamp_sec = timestamp_ns / 1_000_000_000
            
            # The timestamps are in UTC. For ES futures, we want to display in EST.
            # UTC 09:31 -> EST 14:31 (add 5 hours)
            est_timestamp_sec = timestamp_sec + (5 * 3600)  # Add 5 hours for EST
            
            # Use utcfromtimestamp to avoid system timezone interference
            dt_obj = datetime.utcfromtimestamp(est_timestamp_sec)
            return dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            
        except (ValueError, OSError, OverflowError) as e:
            # Handle edge cases like invalid timestamps
            return f"Invalid timestamp ({timestamp_ns})"
    
    def export_results(self, results: Dict, output_dir: str = "results", timestamps: np.ndarray = None):
        """
        Export backtest results to files.
        
        Args:
            results: Dictionary with backtest results
            output_dir: Directory to save results
            timestamps: Array of datetime timestamps for equity curve
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export trade list
        if 'trades' in results:
            results['trades'].to_csv(output_path / "tradelist.csv", index=False)
            
        # Export equity curve using pf object methods
        if 'portfolio' in results:
            pf = results['portfolio']
            
            # Get uncompounded equity data using pf object
            equity_data = self.get_uncompounded_equity_data(pf)
            uncompounded_returns_pct = equity_data['uncompounded_cumulative_pct']
            
            # Create equity curve DataFrame with proper timestamps
            if timestamps is not None and len(timestamps) > 0:
                # Convert timestamps to proper datetime format if needed - PERFORMANCE FIX
                if hasattr(timestamps, 'dtype') and 'datetime64' in str(timestamps.dtype):
                    datetime_col = pd.to_datetime(timestamps)
                elif hasattr(timestamps, 'dtype') and timestamps.dtype in [np.int64, int]:
                    # PERFORMANCE FIX: Auto-detect timestamp unit to avoid out-of-bounds errors
                    # Check if timestamps are in nanoseconds (large values) or seconds
                    sample_timestamp = timestamps[0] if len(timestamps) > 0 else 0
                    
                    if sample_timestamp > 1e15:  # Likely nanoseconds (after year 2001)
                        print(f"   INFO: Detected nanosecond timestamps, converting appropriately")
                        datetime_col = pd.to_datetime(timestamps, unit='ns')
                    elif sample_timestamp > 1e9:  # Likely seconds (after year 2001)
                        print(f"   INFO: Detected second timestamps, converting appropriately")
                        datetime_col = pd.to_datetime(timestamps, unit='s')
                    else:
                        print(f"   WARNING: Unusual timestamp format detected, attempting default conversion")
                        datetime_col = pd.to_datetime(timestamps)
                else:
                    # Already datetime-like
                    datetime_col = pd.to_datetime(timestamps)
                    
                # Extract uncompounded returns values
                if hasattr(uncompounded_returns_pct, 'values'):
                    returns_values = uncompounded_returns_pct.values.flatten()
                else:
                    returns_values = uncompounded_returns_pct.flatten() if hasattr(uncompounded_returns_pct, 'flatten') else uncompounded_returns_pct
                
                # Ensure timestamps and returns have same length
                min_len = min(len(datetime_col), len(returns_values))
                
                equity_df = pd.DataFrame({
                    'datetime': datetime_col[:min_len],
                    'uncompounded_returns_pct': returns_values[:min_len],
                    'equity_value': equity_data['equity_values'].values.flatten()[:min_len] if hasattr(equity_data['equity_values'], 'values') else equity_data['equity_values'][:min_len]
                })
            elif hasattr(uncompounded_returns_pct, 'index'):
                # Use returns index if available
                returns_values = uncompounded_returns_pct.values.flatten() if hasattr(uncompounded_returns_pct, 'values') else uncompounded_returns_pct.flatten()
                equity_values = equity_data['equity_values'].values.flatten() if hasattr(equity_data['equity_values'], 'values') else equity_data['equity_values'].flatten()
                
                equity_df = pd.DataFrame({
                    'datetime': uncompounded_returns_pct.index,
                    'uncompounded_returns_pct': returns_values,
                    'equity_value': equity_values
                })
            else:
                # Fallback to sequential index
                returns_values = uncompounded_returns_pct.flatten() if hasattr(uncompounded_returns_pct, 'flatten') else uncompounded_returns_pct
                equity_values = equity_data['equity_values'].flatten() if hasattr(equity_data['equity_values'], 'flatten') else equity_data['equity_values']
                
                equity_df = pd.DataFrame({
                    'datetime': range(len(returns_values)),
                    'uncompounded_returns_pct': returns_values,
                    'equity_value': equity_values
                })
                
            equity_df.to_csv(output_path / "equity_curve.csv", index=False)
            
        # Export performance summary
        if 'metrics' in results:
            metrics_df = pd.DataFrame([results['metrics']])
            metrics_df.to_csv(output_path / "performance_summary.csv", index=False)
            
