#!/usr/bin/env python3
"""
VectorBT Integration - Load trades from VectorBT portfolio output
===============================================================

Convert VectorBT Portfolio trades to our standard TradeData format.
Supports gradual entry/exit trades and parallel processing results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

try:
    import vectorbtpro as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    vbt = None

from trade_data import TradeData, TradeCollection

logger = logging.getLogger(__name__)

class VBTTradeLoader:
    """Load trades from VectorBT Portfolio objects"""
    
    def __init__(self):
        """Initialize VBT trade loader"""
        if not VBT_AVAILABLE:
            raise ImportError("VectorBT Pro is required for VBT integration. Install with: pip install vectorbtpro")
    
    def load_vbt_trades(self, portfolio: 'vbt.Portfolio', 
                       price_data: Optional[Dict[str, np.ndarray]] = None) -> TradeCollection:
        """
        Convert VectorBT trades to our format
        
        Args:
            portfolio: VectorBT Portfolio object with trades
            price_data: Optional price data dict with 'datetime' array for timestamps
            
        Returns:
            TradeCollection with parsed trades
        """
        logger.info("Loading trades from VectorBT Portfolio")
        
        # Get trades records
        trades_df = portfolio.trades.records_readable
        
        if trades_df.empty:
            logger.warning("No trades found in VectorBT Portfolio")
            return TradeCollection([])
        
        logger.info(f"Found {len(trades_df)} trade records")
        
        # Handle gradual entry/exit (multiple fractional positions)
        trades = self._parse_vbt_trades(trades_df, price_data)
        
        logger.info(f"Parsed {len(trades)} individual trades")
        return TradeCollection(trades)
    
    def _parse_vbt_trades(self, trades_df: pd.DataFrame, 
                         price_data: Optional[Dict[str, np.ndarray]] = None) -> List[TradeData]:
        """Parse VectorBT trades DataFrame into TradeData objects"""
        trades = []
        
        for idx, trade_row in trades_df.iterrows():
            try:
                # Parse entry trade
                entry_trade = self._parse_vbt_entry(trade_row, idx * 2, price_data)
                trades.append(entry_trade)
                
                # Parse exit trade
                exit_trade = self._parse_vbt_exit(trade_row, idx * 2 + 1, price_data)
                trades.append(exit_trade)
                
            except Exception as e:
                logger.warning(f"Failed to parse VBT trade at index {idx}: {e}")
                continue
        
        return trades
    
    def _parse_vbt_entry(self, trade_row: pd.Series, trade_id: int,
                        price_data: Optional[Dict[str, np.ndarray]] = None) -> TradeData:
        """Parse VBT trade row into entry TradeData"""
        # Get entry data
        entry_idx = int(trade_row['Entry Idx'])
        entry_price = float(trade_row['Entry Price'])
        size = float(trade_row['Size'])
        
        # Determine trade type (BUY for positive size, SHORT for negative)
        trade_type = 'BUY' if size > 0 else 'SHORT'
        size = abs(size)  # Always store positive size
        
        # Get timestamp
        timestamp = self._get_timestamp(entry_idx, price_data)
        
        # Get column info (for multi-column portfolios)
        column = trade_row.get('Column', 0)
        strategy = f"Col_{column}" if column != 0 else None
        
        return TradeData(
            trade_id=trade_id,
            timestamp=timestamp,
            bar_index=entry_idx,
            trade_type=trade_type,
            price=entry_price,
            size=size,
            pnl=None,  # P&L is calculated on exit
            strategy=strategy,
            symbol=None  # Would need to be passed separately
        )
    
    def _parse_vbt_exit(self, trade_row: pd.Series, trade_id: int,
                       price_data: Optional[Dict[str, np.ndarray]] = None) -> TradeData:
        """Parse VBT trade row into exit TradeData"""
        # Get exit data
        exit_idx = int(trade_row['Exit Idx'])
        exit_price = float(trade_row['Exit Price'])
        size = float(trade_row['Size'])
        pnl = float(trade_row['PnL'])
        
        # Determine trade type (SELL for positive original size, COVER for negative)
        trade_type = 'SELL' if size > 0 else 'COVER'
        size = abs(size)  # Always store positive size
        
        # Get timestamp
        timestamp = self._get_timestamp(exit_idx, price_data)
        
        # Get column info (for multi-column portfolios)
        column = trade_row.get('Column', 0)
        strategy = f"Col_{column}" if column != 0 else None
        
        return TradeData(
            trade_id=trade_id,
            timestamp=timestamp,
            bar_index=exit_idx,
            trade_type=trade_type,
            price=exit_price,
            size=size,
            pnl=pnl,
            strategy=strategy,
            symbol=None  # Would need to be passed separately
        )
    
    def _get_timestamp(self, bar_index: int, 
                      price_data: Optional[Dict[str, np.ndarray]] = None) -> pd.Timestamp:
        """Get timestamp for bar index"""
        if price_data and 'datetime' in price_data:
            try:
                # Convert numpy datetime64 to pandas Timestamp
                dt = price_data['datetime'][bar_index]
                return pd.Timestamp(dt)
            except (IndexError, KeyError):
                pass
        
        # Fallback: create synthetic timestamp
        base_time = pd.Timestamp('2024-01-01 09:30:00')
        return base_time + pd.Timedelta(minutes=bar_index * 5)
    
    def load_from_strategy_results(self, strategy_results: Dict, 
                                 price_data: Dict[str, np.ndarray]) -> TradeCollection:
        """
        Load trades from strategy results dictionary
        
        Args:
            strategy_results: Dictionary with 'portfolio' key containing VBT Portfolio
            price_data: Price data dictionary with datetime array
            
        Returns:
            TradeCollection with trades
        """
        if 'portfolio' not in strategy_results:
            raise ValueError("Strategy results must contain 'portfolio' key")
        
        portfolio = strategy_results['portfolio']
        return self.load_vbt_trades(portfolio, price_data)
    
    def load_from_parallel_results(self, parallel_results: List[Dict], 
                                 price_data: Dict[str, np.ndarray]) -> TradeCollection:
        """
        Load trades from parallel processing results
        
        Args:
            parallel_results: List of strategy result dictionaries
            price_data: Price data dictionary
            
        Returns:
            Combined TradeCollection with all trades
        """
        all_trades = []
        
        for i, result in enumerate(parallel_results):
            try:
                trade_collection = self.load_from_strategy_results(result, price_data)
                
                # Add strategy identifier for parallel results
                for trade in trade_collection.trades:
                    if trade.strategy is None:
                        trade.strategy = f"Strategy_{i}"
                    else:
                        trade.strategy = f"Strategy_{i}_{trade.strategy}"
                
                all_trades.extend(trade_collection.trades)
                
            except Exception as e:
                logger.warning(f"Failed to load trades from parallel result {i}: {e}")
                continue
        
        logger.info(f"Loaded {len(all_trades)} total trades from {len(parallel_results)} parallel results")
        return TradeCollection(all_trades)

def create_sample_vbt_portfolio():
    """Create sample VectorBT portfolio for testing"""
    if not VBT_AVAILABLE:
        raise ImportError("VectorBT Pro is required for testing")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    close = 4000 + np.cumsum(np.random.randn(1000) * 0.1)
    
    # Create simple buy/sell signals
    entries = np.zeros(1000, dtype=bool)
    exits = np.zeros(1000, dtype=bool)
    
    # Add some random entry/exit signals
    entry_indices = np.random.choice(range(100, 900), size=10, replace=False)
    for entry_idx in entry_indices:
        entries[entry_idx] = True
        # Add exit 10-50 bars later
        exit_idx = min(entry_idx + np.random.randint(10, 50), 999)
        exits[exit_idx] = True
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        freq='5min'
    )
    
    return portfolio, {'datetime': dates.values, 'close': close}

if __name__ == "__main__":
    # Test VBT integration
    if not VBT_AVAILABLE:
        print("VectorBT Pro not available. Install with: pip install vectorbtpro")
        exit(1)
    
    print("Testing VBT Trade Loader...")
    
    try:
        # Create sample portfolio
        portfolio, price_data = create_sample_vbt_portfolio()
        
        print(f"Created sample portfolio with {len(portfolio.trades.records)} trade records")
        
        # Load trades
        loader = VBTTradeLoader()
        trades = loader.load_vbt_trades(portfolio, price_data)
        
        print(f"Loaded {len(trades)} individual trades")
        
        # Show statistics
        stats = trades.get_statistics()
        print(f"Trade statistics: {stats}")
        
        # Test range queries
        range_trades = trades.get_trades_in_range(100, 200)
        print(f"Trades in range 100-200: {len(range_trades)}")
        
        # Show first few trades
        print("\nFirst 5 trades:")
        for i, trade in enumerate(trades[:5]):
            print(f"  {i+1}. Bar {trade.bar_index}: {trade.trade_type} {trade.size} @ ${trade.price:.2f}")
        
        print("VBT integration test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()