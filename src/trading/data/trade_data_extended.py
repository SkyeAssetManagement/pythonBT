"""
Extended Trade Collection Methods - Additional functionality
===========================================================
These methods extend the base TradeCollection class
"""

from typing import List, Dict, Tuple, Optional
from .trade_data import TradeData, TradeCollection


def get_entry_exit_pairs(collection: TradeCollection) -> List[Tuple[TradeData, TradeData]]:
    """
    Pair entry and exit trades
    
    Returns:
        List of (entry, exit) trade pairs
    """
    pairs = []
    pending_entries = []
    
    for trade in collection.trades:
        if trade.is_entry:
            pending_entries.append(trade)
        elif trade.is_exit and pending_entries:
            # Match with the first pending entry of compatible type
            for i, entry in enumerate(pending_entries):
                if (entry.trade_type == 'BUY' and trade.trade_type == 'SELL') or \
                   (entry.trade_type == 'SHORT' and trade.trade_type == 'COVER'):
                    pairs.append((entry, trade))
                    pending_entries.pop(i)
                    break
    
    return pairs


def calculate_statistics(collection: TradeCollection) -> Dict:
    """
    Calculate comprehensive trade statistics
    
    Returns:
        Dictionary with statistics
    """
    if not collection.trades:
        return {
            'total_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'avg_pnl': 0
        }
    
    # Calculate P&L statistics
    total_pnl = 0
    wins = 0
    losses = 0
    
    for trade in collection.trades:
        if trade.pnl is not None:
            total_pnl += trade.pnl
            if trade.pnl > 0:
                wins += 1
            elif trade.pnl < 0:
                losses += 1
    
    total_with_pnl = wins + losses
    win_rate = (wins / total_with_pnl * 100) if total_with_pnl > 0 else 0
    avg_pnl = total_pnl / total_with_pnl if total_with_pnl > 0 else 0
    
    return {
        'total_trades': collection.total_trades,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'wins': wins,
        'losses': losses
    }


def filter_by_type(collection: TradeCollection, trade_type: str) -> List[TradeData]:
    """
    Filter trades by type
    
    Args:
        trade_type: 'BUY', 'SELL', 'SHORT', or 'COVER'
        
    Returns:
        List of trades of specified type
    """
    return [trade for trade in collection.trades if trade.trade_type == trade_type]


def filter_by_strategy(collection: TradeCollection, strategy: str) -> List[TradeData]:
    """
    Filter trades by strategy
    
    Args:
        strategy: Strategy name to filter by
        
    Returns:
        List of trades with specified strategy
    """
    return [trade for trade in collection.trades if trade.strategy == strategy]


# Monkey-patch the methods onto TradeCollection
TradeCollection.get_entry_exit_pairs = lambda self: get_entry_exit_pairs(self)
TradeCollection.calculate_statistics = lambda self: calculate_statistics(self)
TradeCollection.filter_by_type = lambda self, trade_type: filter_by_type(self, trade_type)
TradeCollection.filter_by_strategy = lambda self, strategy: filter_by_strategy(self, strategy)