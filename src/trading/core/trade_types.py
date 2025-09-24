"""
Enhanced trade data structures for unified trading system
Compatible with existing trade_panel.py but with additional fields
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class TradeRecord:
    """
    Complete trade record with all execution details
    Compatible with existing trade_panel.py display
    """
    # Core fields (required)
    trade_id: int
    bar_index: int  # Execution bar for compatibility
    trade_type: str  # BUY, SELL, SHORT, COVER
    price: float  # Execution price

    # Extended execution details
    signal_bar: int = None  # Bar where signal was generated
    execution_bar: int = None  # Bar where trade was executed
    lag: int = 0  # Bars between signal and execution
    signal_price: float = None  # Price at signal bar
    execution_formula: str = None  # Formula used for execution price

    # P&L tracking
    pnl_points: float = None  # P&L in price points
    pnl_percent: float = None  # P&L as percentage
    cumulative_pnl_percent: float = None  # Running total P&L %

    # Position sizing
    size: float = 1.0  # Position size (shares/contracts)
    value: float = None  # Position value (size * price)

    # Friction costs
    fees: float = 0.0  # Total fees paid
    slippage_cost: float = 0.0  # Slippage cost in dollars

    # Timestamps
    timestamp: pd.Timestamp = None  # Execution timestamp
    signal_timestamp: pd.Timestamp = None  # Signal generation timestamp

    # Strategy metadata
    strategy: str = None  # Strategy name
    strategy_params: Dict = field(default_factory=dict)  # Strategy parameters

    # Additional metadata
    metadata: Dict = field(default_factory=dict)  # Any additional data

    def __post_init__(self):
        """Post-initialization validation and calculation"""
        # Set execution_bar to bar_index if not specified
        if self.execution_bar is None:
            self.execution_bar = self.bar_index

        # Calculate lag if both bars are specified
        if self.signal_bar is not None and self.execution_bar is not None:
            self.lag = self.execution_bar - self.signal_bar

        # Calculate value if size and price are available
        if self.value is None and self.size and self.price:
            self.value = self.size * self.price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'trade_id': self.trade_id,
            'bar_index': self.bar_index,
            'trade_type': self.trade_type,
            'price': self.price,
            'signal_bar': self.signal_bar,
            'execution_bar': self.execution_bar,
            'lag': self.lag,
            'signal_price': self.signal_price,
            'execution_formula': self.execution_formula,
            'pnl_points': self.pnl_points,
            'pnl_percent': self.pnl_percent,
            'cumulative_pnl_percent': self.cumulative_pnl_percent,
            'size': self.size,
            'value': self.value,
            'fees': self.fees,
            'slippage_cost': self.slippage_cost,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'signal_timestamp': self.signal_timestamp.isoformat() if self.signal_timestamp else None,
            'strategy': self.strategy,
            'strategy_params': self.strategy_params,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """Create from dictionary"""
        # Handle timestamp conversion
        if 'timestamp' in data and data['timestamp']:
            if isinstance(data['timestamp'], str):
                data['timestamp'] = pd.Timestamp(data['timestamp'])

        if 'signal_timestamp' in data and data['signal_timestamp']:
            if isinstance(data['signal_timestamp'], str):
                data['signal_timestamp'] = pd.Timestamp(data['signal_timestamp'])

        return cls(**data)

    def to_legacy_trade_data(self):
        """
        Convert to legacy TradeData format for compatibility
        Returns a compatible object for existing trade_panel.py
        """
        # Import here to avoid circular dependency
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.trade_data import TradeData

        legacy_trade = TradeData(
            bar_index=self.bar_index,
            trade_type=self.trade_type,
            price=self.price,
            trade_id=self.trade_id,
            timestamp=self.timestamp,
            pnl=self.pnl_points,  # Legacy uses points not percent
            strategy=self.strategy
        )

        # IMPORTANT: Copy the pnl_percent attribute for proper display
        if self.pnl_percent is not None:
            legacy_trade.pnl_percent = self.pnl_percent

        # Also copy cumulative P&L if available
        if self.cumulative_pnl_percent is not None:
            legacy_trade.cumulative_pnl_percent = self.cumulative_pnl_percent

        # Copy other important fields
        if hasattr(self, 'fees') and self.fees is not None:
            legacy_trade.fees = self.fees
        if hasattr(self, 'signal_lag') and self.signal_lag is not None:
            legacy_trade.lag = self.signal_lag

        return legacy_trade

    def format_pnl_display(self) -> str:
        """Format P&L for display (as percentage)"""
        if self.pnl_percent is not None:
            sign = '+' if self.pnl_percent >= 0 else ''
            return f"{sign}{self.pnl_percent:.2f}%"
        return ""

    def format_cumulative_pnl(self) -> str:
        """Format cumulative P&L for display"""
        if self.cumulative_pnl_percent is not None:
            sign = '+' if self.cumulative_pnl_percent >= 0 else ''
            return f"{sign}{self.cumulative_pnl_percent:.2f}%"
        return ""

    def is_entry(self) -> bool:
        """Check if this is an entry trade"""
        return self.trade_type in ['BUY', 'SHORT']

    def is_exit(self) -> bool:
        """Check if this is an exit trade"""
        return self.trade_type in ['SELL', 'COVER']

    def is_long(self) -> bool:
        """Check if this is a long-side trade"""
        return self.trade_type in ['BUY', 'SELL']

    def is_short(self) -> bool:
        """Check if this is a short-side trade"""
        return self.trade_type in ['SHORT', 'COVER']


class TradeRecordCollection:
    """
    Collection of TradeRecords with analysis capabilities
    Compatible with existing TradeCollection interface
    """

    def __init__(self, trades: List[TradeRecord] = None):
        self.trades = trades or []
        self._update_cumulative_pnl()

    def _update_cumulative_pnl(self):
        """Update cumulative P&L for all trades using simple summation"""
        cumulative = 0.0
        for trade in self.trades:
            if trade.pnl_percent is not None:
                # Simple sum of percentages
                cumulative += trade.pnl_percent
                trade.cumulative_pnl_percent = cumulative

    def add_trade(self, trade: TradeRecord):
        """Add a trade and update cumulative P&L"""
        self.trades.append(trade)
        self._update_cumulative_pnl()

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl_percent': 0.0,
                'avg_pnl_percent': 0.0,
                'max_win_percent': 0.0,
                'max_loss_percent': 0.0,
                'sharpe_ratio': 0.0
            }

        # Get trades with P&L
        trades_with_pnl = [t for t in self.trades if t.pnl_percent is not None]

        if not trades_with_pnl:
            return {
                'total_trades': len(self.trades),
                'closed_trades': 0,
                'win_rate': 0.0,
                'total_pnl_percent': 0.0,
                'avg_pnl_percent': 0.0,
                'max_win_percent': 0.0,
                'max_loss_percent': 0.0,
                'sharpe_ratio': 0.0
            }

        pnl_values = [t.pnl_percent for t in trades_with_pnl]
        wins = [p for p in pnl_values if p > 0]
        losses = [p for p in pnl_values if p < 0]

        # Calculate Sharpe ratio (simplified)
        sharpe = 0.0
        if len(pnl_values) > 1:
            avg_return = np.mean(pnl_values)
            std_return = np.std(pnl_values)
            if std_return > 0:
                sharpe = avg_return / std_return * np.sqrt(252)  # Annualized

        return {
            'total_trades': len(self.trades),
            'closed_trades': len(trades_with_pnl),
            'open_trades': len(self.trades) - len(trades_with_pnl),
            'win_rate': len(wins) / len(trades_with_pnl) * 100 if trades_with_pnl else 0,
            'wins': len(wins),
            'losses': len(losses),
            'total_pnl_percent': sum(pnl_values),
            'avg_pnl_percent': np.mean(pnl_values) if pnl_values else 0,
            'max_win_percent': max(wins) if wins else 0,
            'max_loss_percent': min(losses) if losses else 0,
            'sharpe_ratio': sharpe,
            'avg_lag': np.mean([t.lag for t in self.trades if t.lag is not None]) if self.trades else 0
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        data = [t.to_dict() for t in self.trades]
        df = pd.DataFrame(data)

        # Ensure proper column order
        column_order = [
            'trade_id', 'timestamp', 'trade_type', 'price',
            'size', 'value', 'pnl_percent', 'cumulative_pnl_percent',
            'signal_bar', 'execution_bar', 'lag', 'strategy'
        ]

        # Only include columns that exist
        columns = [c for c in column_order if c in df.columns]
        return df[columns]

    def to_legacy_collection(self):
        """Convert to legacy TradeCollection for compatibility"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.trade_data import TradeCollection

        legacy_trades = [t.to_legacy_trade_data() for t in self.trades]
        return TradeCollection(legacy_trades)

    def filter_by_date_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Filter trades by date range"""
        filtered = [
            t for t in self.trades
            if t.timestamp and start_date <= t.timestamp <= end_date
        ]
        return TradeRecordCollection(filtered)

    def filter_by_strategy(self, strategy_name: str):
        """Filter trades by strategy"""
        filtered = [
            t for t in self.trades
            if t.strategy == strategy_name
        ]
        return TradeRecordCollection(filtered)

    def get_summary_stats(self) -> str:
        """Get formatted summary statistics"""
        metrics = self.get_metrics()

        summary = f"""
Trade Summary:
--------------
Total Trades: {metrics['total_trades']}
Closed Trades: {metrics.get('closed_trades', 0)}
Win Rate: {metrics['win_rate']:.1f}%
Total P&L: {metrics['total_pnl_percent']:.2f}%
Avg P&L: {metrics['avg_pnl_percent']:.2f}%
Max Win: {metrics['max_win_percent']:.2f}%
Max Loss: {metrics['max_loss_percent']:.2f}%
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
Avg Execution Lag: {metrics.get('avg_lag', 0):.1f} bars
"""
        return summary

    def __len__(self):
        return len(self.trades)

    def __getitem__(self, index):
        return self.trades[index]

    def __iter__(self):
        return iter(self.trades)