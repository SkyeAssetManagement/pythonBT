#!/usr/bin/env python3
"""
Trade Data Classes
==================
Simple trade data structures for visualization
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

class TradeData:
    """Single trade data point"""
    
    def __init__(self, bar_index: int, price: float, trade_type: str = 'Buy', 
                 timestamp: Optional[datetime] = None, **kwargs):
        self.bar_index = bar_index
        self.price = price
        self.trade_type = trade_type  # 'Buy' or 'Sell'
        self.timestamp = timestamp
        self.extra_data = kwargs  # Store any additional fields
        
    def __str__(self):
        return f"{self.trade_type} @ {self.price:.2f} (bar {self.bar_index})"
    
    def __repr__(self):
        return f"TradeData({self.trade_type}, {self.price}, {self.bar_index})"

class TradeCollection:
    """Collection of trades"""
    
    def __init__(self, trades: Optional[List[TradeData]] = None):
        self.trades = trades if trades is not None else []
        
    def add_trade(self, trade: TradeData):
        """Add a trade to the collection"""
        self.trades.append(trade)
        
    def __len__(self):
        return len(self.trades)
    
    def __iter__(self):
        return iter(self.trades)
    
    def __getitem__(self, index):
        return self.trades[index]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trade statistics"""
        stats = {
            'total_trades': len(self.trades),
            'trade_types': {}
        }
        
        # Count trade types
        for trade in self.trades:
            trade_type = trade.trade_type.upper()
            if trade_type not in stats['trade_types']:
                stats['trade_types'][trade_type] = 0
            stats['trade_types'][trade_type] += 1
        
        # Ensure all expected types exist
        for trade_type in ['BUY', 'SELL', 'SHORT', 'COVER']:
            if trade_type not in stats['trade_types']:
                stats['trade_types'][trade_type] = 0
        
        return stats
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, bar_index_col: str = 'bar_index',
                      price_col: str = 'price', type_col: str = 'type'):
        """Create from DataFrame"""
        trades = []
        for _, row in df.iterrows():
            trade = TradeData(
                bar_index=int(row[bar_index_col]) if bar_index_col in row else 0,
                price=float(row[price_col]) if price_col in row else 0.0,
                trade_type=row[type_col] if type_col in row else 'Buy'
            )
            trades.append(trade)
        return cls(trades)
    
    @classmethod
    def create_sample_trades(cls, num_trades: int = 20, num_bars: int = 500):
        """Create sample trades for testing"""
        import numpy as np
        np.random.seed(42)
        
        trades = []
        indices = np.random.choice(range(50, num_bars - 50), num_trades, replace=False)
        indices.sort()
        
        for i, idx in enumerate(indices):
            trades.append(TradeData(
                bar_index=idx,
                price=4000 + np.random.randn() * 10,  # Random price around 4000
                trade_type='Buy' if i % 2 == 0 else 'Sell'
            ))
        
        return cls(trades)

# Standalone function for compatibility with trade_panel imports
def create_sample_trades(num_trades: int = 100, start_bar: int = 0, end_bar: int = 1000, 
                        timestamps: Optional[List] = None, bar_data: Optional[Dict] = None):
    """Create sample trades with optional timestamp and price coordination"""
    import numpy as np
    
    if end_bar <= start_bar:
        end_bar = start_bar + 100
    
    # Generate random bar indices
    num_trades = min(num_trades, (end_bar - start_bar) // 2)
    trade_bars = np.random.choice(range(start_bar, end_bar), num_trades, replace=False)
    trade_bars.sort()
    
    trades = []
    for i, bar_idx in enumerate(trade_bars):
        # Determine price based on bar data if available
        if bar_data and 'close' in bar_data and bar_idx < len(bar_data['close']):
            base_price = bar_data['close'][bar_idx]
            price = base_price + np.random.randn() * 2
        else:
            price = 4000 + np.random.randn() * 10
        
        # Get timestamp if available
        timestamp = None
        if timestamps and bar_idx < len(timestamps):
            timestamp = timestamps[bar_idx]
        
        trade = TradeData(
            bar_index=bar_idx,
            price=price,
            trade_type='Buy' if i % 2 == 0 else 'Sell',
            timestamp=timestamp
        )
        trades.append(trade)
    
    collection = TradeCollection(trades)
    
    # Add date_range attribute for compatibility
    if timestamps and len(timestamps) > 0:
        collection.date_range = (timestamps[0], timestamps[-1])
    else:
        from datetime import datetime, timedelta
        now = datetime.now()
        collection.date_range = (now - timedelta(days=30), now)
    
    return collection