"""
VectorBT Engine V2 - Enhanced to support separate phased trades
"""

import numpy as np
import vectorbtpro as vbt
from typing import Dict, Optional
import yaml
from pathlib import Path
import pandas as pd
import logging
from ..utils.price_formulas import PriceFormulaEvaluator
from .phased_trading_engine_v2 import PhasedTradingEngineV2, create_config_from_yaml

logger = logging.getLogger(__name__)


class VectorBTEngineV2:
    """
    Enhanced VectorBT engine that forces separate trades for phased trading.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._validate_config()
        self.formula_evaluator = PriceFormulaEvaluator()
        
        # Initialize V2 phased trading engine
        self.phased_engine = None
        if self.config['backtest'].get('phased_trading_enabled', False):
            phased_config = create_config_from_yaml(self.config)
            self.phased_engine = PhasedTradingEngineV2(phased_config)
            logger.info(f"Phased Trading V2 enabled: {phased_config.entry_bars} entry bars, "
                       f"{phased_config.exit_bars} exit bars, force_separate={phased_config.force_separate_trades}")
    
    def _validate_config(self):
        """Validate configuration."""
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
    
    def run_phased_backtest(self, data: Dict[str, np.ndarray],
                           entries: np.ndarray, exits: np.ndarray,
                           symbol: str = "TEST") -> vbt.Portfolio:
        """
        Run backtest with enhanced phased trading that creates separate trades.
        
        Args:
            data: OHLCV data dictionary
            entries: Entry signals
            exits: Exit signals
            symbol: Symbol name
            
        Returns:
            VectorBT Portfolio with separate trades for each phase
        """
        # Apply signal lag if configured
        signal_lag = self.config['backtest'].get('signal_lag', 0)
        if signal_lag > 0:
            entries = np.roll(entries, signal_lag, axis=0)
            exits = np.roll(exits, signal_lag, axis=0)
            entries[:signal_lag] = False
            exits[:signal_lag] = False
        
        # Ensure proper shape
        if entries.ndim == 1:
            entries = entries.reshape(-1, 1)
        if exits.ndim == 1:
            exits = exits.reshape(-1, 1)
        
        # Apply phased trading if enabled
        if self.phased_engine is not None:
            print(f"\n[PHASED TRADING V2] Processing signals...")
            print(f"  Original signals: {np.sum(entries)} entries, {np.sum(exits)} exits")
            
            # Process each column separately
            n_cols = entries.shape[1]
            phased_entries_list = []
            phased_exits_list = []
            phased_sizes_list = []
            
            for col in range(n_cols):
                col_entries = entries[:, col].flatten()
                col_exits = exits[:, col].flatten()
                
                # Apply advanced phasing
                phased_result = self.phased_engine.apply_advanced_phasing(
                    col_entries, col_exits, data,
                    position_size=self.config['backtest']['position_size']
                )
                
                phased_entries_list.append(phased_result['entries'])
                phased_exits_list.append(phased_result['exits'])
                phased_sizes_list.append(phased_result['sizes'])
            
            # Stack results
            if n_cols > 1:
                phased_entries = np.column_stack(phased_entries_list)
                phased_exits = np.column_stack(phased_exits_list)
                phased_sizes = np.column_stack(phased_sizes_list)
            else:
                phased_entries = phased_entries_list[0].reshape(-1, 1)
                phased_exits = phased_exits_list[0].reshape(-1, 1)
                phased_sizes = phased_sizes_list[0].reshape(-1, 1)
            
            print(f"  Phased signals: {np.sum(phased_entries)} entries, {np.sum(phased_exits)} exits")
            
            # Validate signals
            if np.sum(phased_entries) != np.sum(phased_exits):
                print(f"  WARNING: Entry/exit mismatch - attempting to balance...")
                # Ensure we have matching entries and exits
                n_entries = np.sum(phased_entries)
                n_exits = np.sum(phased_exits)
                if n_entries > n_exits:
                    # Add exit at the end for unmatched entries
                    phased_exits[-1] = True
                elif n_exits > n_entries:
                    # Remove extra exits
                    exit_indices = np.where(phased_exits)[0]
                    extra_exits = n_exits - n_entries
                    for i in range(extra_exits):
                        phased_exits[exit_indices[-(i+1)]] = False
            
            # Use phased signals with individual sizes
            entries_final = phased_entries
            exits_final = phased_exits
            sizes_array = phased_sizes
            
            print(f"  Creating portfolio with {np.sum(entries_final)} separate trades...")
            
            # Create portfolio with size array to force separate trades
            pf = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries_final,
                exits=exits_final,
                size=sizes_array,  # Use the size array directly
                size_type='value',  # Force value sizing
                init_cash=self.config['backtest']['initial_cash'],
                direction=self.config['backtest']['direction'],
                min_size=self.config['backtest']['min_size'],
                fees=self.config['backtest']['fees'],
                fixed_fees=self.config['backtest']['fixed_fees'],
                slippage=self.config['backtest']['slippage'],
                freq=self.config['backtest']['freq'],
                call_seq=self.config['backtest']['call_seq'],
                accumulate=False,  # Don't accumulate positions
                conflict_mode='exit'  # Exit takes priority in conflicts
            )
            
        else:
            # Standard backtest without phasing
            execution_prices = self._calculate_execution_prices(data, entries, exits)
            
            pf = vbt.Portfolio.from_signals(
                close=execution_prices,
                entries=entries,
                exits=exits,
                size=self.config['backtest']['position_size'],
                size_type=self.config['backtest']['position_size_type'],
                init_cash=self.config['backtest']['initial_cash'],
                direction=self.config['backtest']['direction'],
                min_size=self.config['backtest']['min_size'],
                fees=self.config['backtest']['fees'],
                fixed_fees=self.config['backtest']['fixed_fees'],
                slippage=self.config['backtest']['slippage'],
                freq=self.config['backtest']['freq'],
                call_seq=self.config['backtest']['call_seq']
            )
        
        # Log trade information
        if hasattr(pf.trades, 'records') and len(pf.trades.records) > 0:
            trades_df = pf.trades.records_readable
            print(f"\n[RESULTS] Created {len(trades_df)} trades")
            
            # Show first few trades for verification
            if len(trades_df) > 0:
                print("\nFirst 5 trades:")
                print(trades_df[['Entry Index', 'Exit Index', 'Size', 'PnL']].head())
        
        return pf
    
    def _calculate_execution_prices(self, data: Dict[str, np.ndarray], 
                                   entries: np.ndarray, exits: np.ndarray) -> np.ndarray:
        """Calculate execution prices."""
        execution_price_type = self.config['backtest'].get('execution_price', 'close')
        
        if execution_price_type in ['open', 'high', 'low', 'close']:
            return data[execution_price_type]
        
        # Formula-based pricing
        buy_formula = self.config['backtest'].get('buy_execution_formula', 'C')
        sell_formula = self.config['backtest'].get('sell_execution_formula', 'C')
        
        buy_prices = self.formula_evaluator.get_execution_prices(buy_formula, data, "buy")
        sell_prices = self.formula_evaluator.get_execution_prices(sell_formula, data, "sell")
        
        execution_prices = data['close'].copy()
        
        if entries.ndim == 1:
            execution_prices = np.where(entries, buy_prices, execution_prices)
            execution_prices = np.where(exits, sell_prices, execution_prices)
        else:
            any_entries = np.any(entries, axis=1)
            any_exits = np.any(exits, axis=1)
            execution_prices = np.where(any_entries, buy_prices, execution_prices)
            execution_prices = np.where(any_exits, sell_prices, execution_prices)
        
        return execution_prices
    
    def generate_trade_list(self, pf: vbt.Portfolio, 
                          timestamps: np.ndarray, symbol: str = 'UNKNOWN') -> pd.DataFrame:
        """Generate detailed trade list."""
        if len(pf.trades.records) == 0:
            return pd.DataFrame()
        
        trades = pf.trades.records_readable
        
        # Add metadata
        trades['Symbol'] = symbol
        trades['Strategy'] = trades.get('Column', '0').astype(str)
        
        # Add position sizing info
        trades['posSize_raw'] = 1
        trades['posSize_riskAdj'] = 1
        
        # Calculate shares
        if 'Size' in trades.columns and 'Avg Entry Price' in trades.columns:
            trades['shares'] = trades['Size'] / trades['Avg Entry Price']
            if 'Direction' in trades.columns:
                direction_map = {'Long': 1, 'Short': -1}
                trades['shares'] = trades['shares'] * trades['Direction'].map(direction_map).fillna(1)
                trades['Size'] = trades['Size'] * trades['Direction'].map(direction_map).fillna(1)
        
        # Add P&L columns
        if 'shares' in trades.columns and 'Avg Entry Price' in trades.columns and 'Avg Exit Price' in trades.columns:
            trades['pnl_excFriction'] = trades['shares'] * (trades['Avg Exit Price'] - trades['Avg Entry Price'])
        
        # Convert indices to timestamps if available
        if len(timestamps) > 0:
            entry_idx_col = 'Entry Index' if 'Entry Index' in trades.columns else None
            exit_idx_col = 'Exit Index' if 'Exit Index' in trades.columns else None
            
            if entry_idx_col and exit_idx_col:
                entry_indices = trades[entry_idx_col].values.astype(int)
                exit_indices = trades[exit_idx_col].values.astype(int)
                
                # Map to timestamps
                valid_entry_mask = entry_indices < len(timestamps)
                valid_exit_mask = exit_indices < len(timestamps)
                
                entry_times = np.full(len(trades), np.nan, dtype=np.int64)
                exit_times = np.full(len(trades), np.nan, dtype=np.int64)
                
                entry_times[valid_entry_mask] = timestamps[entry_indices[valid_entry_mask]]
                exit_times[valid_exit_mask] = timestamps[exit_indices[valid_exit_mask]]
                
                trades['EntryTime'] = entry_times
                trades['ExitTime'] = exit_times
        
        return trades
    
    def export_results(self, pf: vbt.Portfolio, output_dir: str = "results", 
                      timestamps: np.ndarray = None, symbol: str = "TEST"):
        """Export backtest results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate and export trade list
        trades_df = self.generate_trade_list(pf, timestamps if timestamps is not None else np.array([]), symbol)
        if not trades_df.empty:
            trades_df.to_csv(output_path / self.config['output']['trade_list_filename'], index=False)
            print(f"\nTrade list exported to: {output_path / self.config['output']['trade_list_filename']}")
            print(f"Total trades: {len(trades_df)}")
        
        # Export equity curve
        equity_df = pd.DataFrame({
            'datetime': range(len(pf.value)),
            'equity_value': pf.value.values.flatten()
        })
        equity_df.to_csv(output_path / self.config['output']['equity_curve_filename'], index=False)
        
        # Export performance summary
        metrics = {
            'total_return': float(pf.total_return),
            'sharpe_ratio': float(pf.sharpe_ratio) if hasattr(pf, 'sharpe_ratio') else 0,
            'max_drawdown': float(pf.max_drawdown) if hasattr(pf, 'max_drawdown') else 0,
            'total_trades': len(trades_df)
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_path / "performance_summary.csv", index=False)