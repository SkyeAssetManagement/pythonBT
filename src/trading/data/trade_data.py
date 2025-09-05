#!/usr/bin/env python3
"""
Trade Data Structures - Core trade data format compatible with VectorBT output
=============================================================================

Standard trade data format for visualization with efficient lookups and filtering.
Designed for high performance with large datasets (100,000+ trades).
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import bisect

@dataclass
class TradeData:
    """Standard trade data format compatible with VectorBT output"""
    trade_id: int
    timestamp: pd.Timestamp
    bar_index: int
    trade_type: str  # 'BUY', 'SELL', 'SHORT', 'COVER'
    price: float
    size: float
    pnl: Optional[float] = None
    strategy: Optional[str] = None
    symbol: Optional[str] = None
    
    def __post_init__(self):
        """Validate trade data after initialization"""
        valid_types = {'BUY', 'SELL', 'SHORT', 'COVER'}
        if self.trade_type not in valid_types:
            raise ValueError(f"Invalid trade_type: {self.trade_type}. Must be one of {valid_types}")
        
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got: {self.price}")
            
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got: {self.size}")
    
    @property
    def is_entry(self) -> bool:
        """Check if this is an entry trade (BUY or SHORT)"""
        return self.trade_type in {'BUY', 'SHORT'}
    
    @property
    def is_exit(self) -> bool:
        """Check if this is an exit trade (SELL or COVER)"""
        return self.trade_type in {'SELL', 'COVER'}
    
    @property
    def is_long(self) -> bool:
        """Check if this is a long side trade (BUY or SELL)"""
        return self.trade_type in {'BUY', 'SELL'}
    
    @property
    def is_short(self) -> bool:
        """Check if this is a short side trade (SHORT or COVER)"""
        return self.trade_type in {'SHORT', 'COVER'}

class TradeCollection:
    """
    Efficient container for all trades with fast lookups
    
    Optimized for:
    - Fast range queries by bar index
    - Fast time-based queries
    - Efficient viewport filtering
    - Memory-efficient storage for large datasets
    """
    
    def __init__(self, trades: List[TradeData]):
        """
        Initialize trade collection with efficient indexes
        
        Args:
            trades: List of TradeData objects
        """
        self.trades = sorted(trades, key=lambda t: t.bar_index)
        
        # Create efficient lookup structures
        self._build_indexes()
        
        # Statistics
        self.total_trades = len(self.trades)
        self.date_range = self._calculate_date_range()
        self.price_range = self._calculate_price_range()
    
    def _build_indexes(self):
        """Build efficient lookup indexes"""
        # Bar index lookup (sorted for binary search)
        self.by_bar: Dict[int, List[TradeData]] = {}
        self.bar_indexes = []
        
        # Time-based lookup
        self.by_time: Dict[pd.Timestamp, List[TradeData]] = {}
        
        # Build indexes
        for trade in self.trades:
            # Bar index lookup
            if trade.bar_index not in self.by_bar:
                self.by_bar[trade.bar_index] = []
                self.bar_indexes.append(trade.bar_index)
            self.by_bar[trade.bar_index].append(trade)
            
            # Time-based lookup
            if trade.timestamp not in self.by_time:
                self.by_time[trade.timestamp] = []
            self.by_time[trade.timestamp].append(trade)
        
        # Sort bar indexes for binary search
        self.bar_indexes.sort()
    
    def _calculate_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Calculate min/max dates in collection"""
        if not self.trades:
            return None, None
        return min(t.timestamp for t in self.trades), max(t.timestamp for t in self.trades)
    
    def _calculate_price_range(self) -> Tuple[float, float]:
        """Calculate min/max prices in collection"""
        if not self.trades:
            return None, None
        return min(t.price for t in self.trades), max(t.price for t in self.trades)
    
    def get_trades_in_range(self, start_bar: int, end_bar: int) -> List[TradeData]:
        """
        Get all trades within bar index range (efficient viewport filtering)
        
        Args:
            start_bar: Starting bar index (inclusive)
            end_bar: Ending bar index (inclusive)
            
        Returns:
            List of trades in range, sorted by bar index
        """
        # Use binary search to find range efficiently
        start_idx = bisect.bisect_left(self.bar_indexes, start_bar)
        end_idx = bisect.bisect_right(self.bar_indexes, end_bar)
        
        # Collect trades in range
        trades_in_range = []
        for bar_idx in self.bar_indexes[start_idx:end_idx]:
            trades_in_range.extend(self.by_bar[bar_idx])
        
        return sorted(trades_in_range, key=lambda t: t.bar_index)
    
    def get_trades_at_bar(self, bar_index: int) -> List[TradeData]:
        """Get all trades at specific bar index"""
        return self.by_bar.get(bar_index, [])
    
    def get_trades_by_type(self, trade_type: str) -> List[TradeData]:
        """Filter trades by type (BUY, SELL, SHORT, COVER)"""
        return [trade for trade in self.trades if trade.trade_type == trade_type]
    
    def get_entry_trades(self) -> List[TradeData]:
        """Get all entry trades (BUY and SHORT)"""
        return [trade for trade in self.trades if trade.is_entry]
    
    def get_exit_trades(self) -> List[TradeData]:
        """Get all exit trades (SELL and COVER)"""
        return [trade for trade in self.trades if trade.is_exit]
    
    def get_first_visible_trade(self, start_bar: int, end_bar: int) -> Optional[TradeData]:
        """
        Get the first trade visible in the given range
        Used for auto-syncing trade list with chart viewport
        
        Args:
            start_bar: Starting bar index
            end_bar: Ending bar index
            
        Returns:
            First trade in range or None if no trades
        """
        trades_in_range = self.get_trades_in_range(start_bar, end_bar)
        return trades_in_range[0] if trades_in_range else None
    
    def get_statistics(self) -> Dict:
        """Get collection statistics for debugging/display"""
        if not self.trades:
            return {}
        
        return {
            'total_trades': self.total_trades,
            'date_range': self.date_range,
            'price_range': self.price_range,
            'trade_types': {
                'BUY': len(self.get_trades_by_type('BUY')),
                'SELL': len(self.get_trades_by_type('SELL')),
                'SHORT': len(self.get_trades_by_type('SHORT')),
                'COVER': len(self.get_trades_by_type('COVER'))
            },
            'unique_bars': len(self.bar_indexes),
            'avg_trades_per_bar': self.total_trades / len(self.bar_indexes) if self.bar_indexes else 0
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis/export"""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp,
                'bar_index': trade.bar_index,
                'trade_type': trade.trade_type,
                'price': trade.price,
                'size': trade.size,
                'pnl': trade.pnl,
                'strategy': trade.strategy,
                'symbol': trade.symbol
            })
        
        return pd.DataFrame(data)
    
    def __len__(self) -> int:
        """Return number of trades"""
        return len(self.trades)
    
    def __getitem__(self, index: int) -> TradeData:
        """Get trade by index"""
        return self.trades[index]
    
    def __iter__(self):
        """Iterate over trades"""
        return iter(self.trades)

def create_sample_trades(n_trades: int = 1000, 
                        start_bar: int = 0, 
                        end_bar: int = 10000,
                        chart_timestamps: Optional[pd.Series] = None,
                        bar_data: Optional[Dict[str, np.ndarray]] = None) -> TradeCollection:
    """
    Create sample trades for testing
    
    Args:
        n_trades: Number of trades to create
        start_bar: Starting bar index
        end_bar: Ending bar index
        chart_timestamps: Optional pandas Series of chart timestamps to use for realistic dates
        bar_data: Optional dict with 'open', 'high', 'low', 'close' arrays for realistic pricing
        
    Returns:
        TradeCollection with sample data
    """
    import random
    
    trades = []
    
    # Use chart timestamps if provided, otherwise default to 2024
    if chart_timestamps is not None and len(chart_timestamps) > end_bar:
        base_time = None  # Will use actual chart timestamps
    else:
        base_time = pd.Timestamp('2024-01-01 09:30:00')
    
    for i in range(n_trades):
        # Random bar within range
        bar_idx = random.randint(start_bar, end_bar)
        
        # Use actual chart timestamp if available, otherwise calculate from base_time
        if base_time is None:
            timestamp = chart_timestamps[bar_idx]
        else:
            # Random timestamp (5-minute bars)
            timestamp = base_time + pd.Timedelta(minutes=bar_idx * 5)
        
        # Random trade type
        trade_type = random.choice(['BUY', 'SELL', 'SHORT', 'COVER'])
        
        # Generate realistic price within the bar's OHLC range
        if bar_data is not None and bar_idx < len(bar_data['high']):
            bar_high = bar_data['high'][bar_idx]
            bar_low = bar_data['low'][bar_idx]
            bar_open = bar_data['open'][bar_idx]
            bar_close = bar_data['close'][bar_idx]
            
            # Generate price within the bar's actual range
            # Use 80% of the range to ensure it's clearly within the bar
            price_range = bar_high - bar_low
            min_price = bar_low + (price_range * 0.1)
            max_price = bar_high - (price_range * 0.1)
            price = random.uniform(min_price, max_price)
        else:
            # Fallback to default pricing
            price = 4000 + random.uniform(-200, 200)
        
        # Random size
        size = random.randint(1, 10)
        
        # Random P&L for exits
        pnl = random.uniform(-500, 500) if trade_type in ['SELL', 'COVER'] else None
        
        trade = TradeData(
            trade_id=i,
            timestamp=timestamp,
            bar_index=bar_idx,
            trade_type=trade_type,
            price=price,
            size=size,
            pnl=pnl,
            strategy='Test',
            symbol='ES'
        )
        
        trades.append(trade)
    
    return TradeCollection(trades)

if __name__ == "__main__":
    # Test the data structures
    print("Testing TradeData and TradeCollection...")
    
    # Create sample trades
    trades = create_sample_trades(100, 0, 1000)
    print(f"Created {len(trades)} sample trades")
    
    # Test statistics
    stats = trades.get_statistics()
    print(f"Statistics: {stats}")
    
    # Test range queries
    range_trades = trades.get_trades_in_range(100, 200)
    print(f"Trades in range 100-200: {len(range_trades)}")
    
    # Test first visible trade
    first_trade = trades.get_first_visible_trade(100, 200)
    if first_trade:
        print(f"First visible trade: Bar {first_trade.bar_index}, Type: {first_trade.trade_type}")
    
    print("Trade data structures test completed!")