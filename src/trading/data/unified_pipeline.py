#!/usr/bin/env python3
"""
Unified Data Pipeline - Bridges OMtree and ABtoPython data formats
==================================================================
Provides adapters and converters for seamless data flow between systems
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

# Feature flag protection
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from feature_flags import feature_flag, get_feature_flags

# Import both data formats
try:
    from .trade_data import TradeData, TradeCollection
except ImportError:
    from trade_data import TradeData, TradeCollection

logger = logging.getLogger(__name__)


@dataclass
class UnifiedDataFormat:
    """
    Unified data format that both systems can use
    Bridges OMtree DataFrame format with ABtoPython TradeData
    """
    # Common fields
    timestamp: pd.Timestamp
    symbol: str
    price: float
    
    # OMtree specific
    features: Optional[Dict[str, float]] = None
    target: Optional[float] = None
    prediction: Optional[float] = None
    
    # ABtoPython specific
    trade_type: Optional[str] = None
    size: Optional[float] = None
    pnl: Optional[float] = None
    bar_index: Optional[int] = None
    
    # Metadata
    source_system: str = 'unknown'
    

class DataPipelineAdapter:
    """
    Adapter to convert between different data formats
    Ensures compatibility between OMtree and ABtoPython
    """
    
    def __init__(self):
        self.flags = get_feature_flags()
        self.conversion_stats = {
            'omtree_to_unified': 0,
            'unified_to_omtree': 0,
            'trades_to_unified': 0,
            'unified_to_trades': 0
        }
    
    @feature_flag('unified_data_pipeline')
    def omtree_to_unified(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[UnifiedDataFormat]:
        """
        Convert OMtree DataFrame to unified format
        
        Args:
            df: OMtree DataFrame with features and targets
            config: Configuration with column mappings
            
        Returns:
            List of UnifiedDataFormat objects
        """
        logger.info(f"Converting OMtree data to unified format: {len(df)} rows")
        
        unified_data = []
        
        # Extract configuration
        date_col = config.get('date_column', 'Date')
        time_col = config.get('time_column', 'Time')
        target_col = config.get('target_column', 'Ret_fwd6hr')
        feature_cols = config.get('feature_columns', [])
        
        for idx, row in df.iterrows():
            # Combine date and time
            if time_col in df.columns:
                timestamp = pd.Timestamp(f"{row[date_col]} {row[time_col]}")
            else:
                timestamp = pd.Timestamp(row[date_col])
            
            # Extract features
            features = {}
            for col in feature_cols:
                if col in df.columns:
                    features[col] = row[col]
            
            # Create unified format
            unified = UnifiedDataFormat(
                timestamp=timestamp,
                symbol=config.get('symbol', 'UNKNOWN'),
                price=row.get('Close', 0.0),
                features=features,
                target=row.get(target_col),
                prediction=row.get('Prediction'),
                source_system='omtree'
            )
            
            unified_data.append(unified)
        
        self.conversion_stats['omtree_to_unified'] += len(unified_data)
        return unified_data
    
    @feature_flag('unified_data_pipeline')
    def unified_to_omtree(self, unified_data: List[UnifiedDataFormat]) -> pd.DataFrame:
        """
        Convert unified format back to OMtree DataFrame
        
        Args:
            unified_data: List of UnifiedDataFormat objects
            
        Returns:
            DataFrame suitable for OMtree processing
        """
        logger.info(f"Converting unified data to OMtree format: {len(unified_data)} items")
        
        records = []
        for item in unified_data:
            record = {
                'Date': item.timestamp.date(),
                'Time': item.timestamp.time(),
                'Symbol': item.symbol,
                'Close': item.price
            }
            
            # Add features
            if item.features:
                record.update(item.features)
            
            # Add target and prediction
            if item.target is not None:
                record['Target'] = item.target
            if item.prediction is not None:
                record['Prediction'] = item.prediction
            
            records.append(record)
        
        df = pd.DataFrame(records)
        self.conversion_stats['unified_to_omtree'] += len(df)
        return df
    
    @feature_flag('unified_data_pipeline')
    def trades_to_unified(self, trades: Union[TradeCollection, List[TradeData]]) -> List[UnifiedDataFormat]:
        """
        Convert ABtoPython trades to unified format
        
        Args:
            trades: TradeCollection or list of TradeData
            
        Returns:
            List of UnifiedDataFormat objects
        """
        if isinstance(trades, TradeCollection):
            trade_list = trades.trades
        else:
            trade_list = trades
        
        logger.info(f"Converting trades to unified format: {len(trade_list)} trades")
        
        unified_data = []
        for trade in trade_list:
            unified = UnifiedDataFormat(
                timestamp=trade.timestamp,
                symbol=trade.symbol or 'UNKNOWN',
                price=trade.price,
                trade_type=trade.trade_type,
                size=trade.size,
                pnl=trade.pnl,
                bar_index=trade.bar_index,
                source_system='abtopython'
            )
            unified_data.append(unified)
        
        self.conversion_stats['trades_to_unified'] += len(unified_data)
        return unified_data
    
    @feature_flag('unified_data_pipeline')
    def unified_to_trades(self, unified_data: List[UnifiedDataFormat]) -> TradeCollection:
        """
        Convert unified format to ABtoPython trades
        
        Args:
            unified_data: List of UnifiedDataFormat objects
            
        Returns:
            TradeCollection with converted trades
        """
        logger.info(f"Converting unified data to trades: {len(unified_data)} items")
        
        trades = []
        for idx, item in enumerate(unified_data):
            # Only convert items that have trade information
            if item.trade_type:
                trade = TradeData(
                    trade_id=idx,
                    timestamp=item.timestamp,
                    bar_index=item.bar_index or idx,
                    trade_type=item.trade_type,
                    price=item.price,
                    size=item.size or 1.0,
                    pnl=item.pnl,
                    symbol=item.symbol
                )
                trades.append(trade)
        
        self.conversion_stats['unified_to_trades'] += len(trades)
        return TradeCollection(trades)
    
    def merge_data_sources(self, 
                          omtree_df: Optional[pd.DataFrame] = None,
                          trades: Optional[TradeCollection] = None,
                          config: Optional[Dict[str, Any]] = None) -> List[UnifiedDataFormat]:
        """
        Merge data from both sources into unified format
        
        Args:
            omtree_df: OMtree DataFrame
            trades: ABtoPython trades
            config: Configuration for conversion
            
        Returns:
            Merged list of UnifiedDataFormat objects
        """
        unified_data = []
        
        # Convert OMtree data if provided
        if omtree_df is not None and not omtree_df.empty:
            if self.flags.is_enabled('unified_data_pipeline'):
                omtree_unified = self.omtree_to_unified(omtree_df, config or {})
                unified_data.extend(omtree_unified)
            else:
                logger.warning("Unified data pipeline feature flag disabled")
        
        # Convert trades if provided
        if trades is not None and len(trades) > 0:
            if self.flags.is_enabled('unified_data_pipeline'):
                trades_unified = self.trades_to_unified(trades)
                unified_data.extend(trades_unified)
            else:
                logger.warning("Unified data pipeline feature flag disabled")
        
        # Sort by timestamp
        unified_data.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Merged {len(unified_data)} items from data sources")
        return unified_data
    
    def get_statistics(self) -> Dict[str, int]:
        """Get conversion statistics"""
        return self.conversion_stats.copy()
    

class DataValidator:
    """
    Validates data integrity during conversions
    Ensures no data loss or corruption
    """
    
    @staticmethod
    def validate_omtree_df(df: pd.DataFrame, required_cols: List[str]) -> bool:
        """Validate OMtree DataFrame has required columns"""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        return True
    
    @staticmethod
    def validate_trades(trades: TradeCollection) -> bool:
        """Validate trade collection integrity"""
        if len(trades) == 0:
            logger.warning("Empty trade collection")
            return True  # Empty is valid
        
        # Check for required fields
        for trade in trades:
            if trade.price <= 0:
                logger.error(f"Invalid price in trade {trade.trade_id}: {trade.price}")
                return False
            if trade.size <= 0:
                logger.error(f"Invalid size in trade {trade.trade_id}: {trade.size}")
                return False
        
        return True
    
    @staticmethod
    def validate_unified_data(data: List[UnifiedDataFormat]) -> bool:
        """Validate unified data format"""
        if not data:
            logger.warning("Empty unified data")
            return True
        
        # Check timestamps are in order
        timestamps = [item.timestamp for item in data]
        if timestamps != sorted(timestamps):
            logger.warning("Unified data not sorted by timestamp")
        
        return True


# Global adapter instance
_adapter = None

def get_data_adapter() -> DataPipelineAdapter:
    """Get global data pipeline adapter"""
    global _adapter
    if _adapter is None:
        _adapter = DataPipelineAdapter()
    return _adapter


class OMtreeAdapter:
    """
    Adapter specifically for OMtree system integration
    Handles OMtree-specific data transformations
    """
    
    def __init__(self, adapter: DataPipelineAdapter):
        self.adapter = adapter
        self.flags = get_feature_flags()
    
    def prepare_data_for_model(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare data for OMtree model training/prediction
        """
        if not self.flags.is_enabled('unified_data_pipeline'):
            return df  # Return unchanged if feature disabled
        
        # Convert to unified format
        unified = self.adapter.omtree_to_unified(df, config)
        
        # Apply any transformations in unified space
        # (future enhancement point)
        
        # Convert back to OMtree format
        return self.adapter.unified_to_omtree(unified)
    
    def process_predictions(self, df: pd.DataFrame, predictions: np.ndarray, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Process model predictions and add to DataFrame
        """
        df['Prediction'] = predictions
        return df


class ABtoPythonAdapter:
    """
    Adapter specifically for ABtoPython system integration
    Handles trade-specific data transformations
    """
    
    def __init__(self, adapter: DataPipelineAdapter):
        self.adapter = adapter
        self.flags = get_feature_flags()
    
    def import_from_vectorbt(self, vbt_data: Any) -> TradeCollection:
        """
        Import trades from VectorBT format
        """
        # Implementation would go here
        # For now, return empty collection
        return TradeCollection([])
    
    def export_to_visualization(self, trades: TradeCollection) -> Dict[str, Any]:
        """
        Export trades for visualization in PyQtGraph
        """
        if not self.flags.is_enabled('unified_data_pipeline'):
            # Direct export without conversion
            return {'trades': trades.trades}
        
        # Convert to unified format
        unified = self.adapter.trades_to_unified(trades)
        
        # Prepare for visualization
        viz_data = {
            'timestamps': [item.timestamp for item in unified],
            'prices': [item.price for item in unified],
            'types': [item.trade_type for item in unified],
            'sizes': [item.size for item in unified]
        }
        
        return viz_data


if __name__ == "__main__":
    # Test the unified pipeline
    print("Testing Unified Data Pipeline...")
    
    # Enable feature flag for testing
    flags = get_feature_flags()
    flags.enable('unified_data_pipeline')
    
    # Create adapter
    adapter = get_data_adapter()
    
    # Test OMtree to unified conversion
    test_df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Time': ['09:00:00', '10:00:00'],
        'Close': [100.0, 101.0],
        'PIR_0-1hr': [0.5, -0.3],
        'Ret_fwd6hr': [0.8, -0.2]
    })
    
    config = {
        'date_column': 'Date',
        'time_column': 'Time',
        'target_column': 'Ret_fwd6hr',
        'feature_columns': ['PIR_0-1hr'],
        'symbol': 'TEST'
    }
    
    unified = adapter.omtree_to_unified(test_df, config)
    print(f"Converted {len(unified)} OMtree rows to unified format")
    
    # Convert back
    df_back = adapter.unified_to_omtree(unified)
    print(f"Converted back to DataFrame with shape: {df_back.shape}")
    
    # Test trade conversion
    test_trade = TradeData(
        trade_id=1,
        timestamp=pd.Timestamp('2024-01-01 09:00:00'),
        bar_index=100,
        trade_type='BUY',
        price=100.0,
        size=10.0,
        symbol='TEST'
    )
    
    collection = TradeCollection([test_trade])
    trade_unified = adapter.trades_to_unified(collection)
    print(f"Converted {len(trade_unified)} trades to unified format")
    
    # Test system-specific adapters
    omtree_adapter = OMtreeAdapter(adapter)
    processed_df = omtree_adapter.prepare_data_for_model(test_df, config)
    print(f"OMtree adapter processed DataFrame with shape: {processed_df.shape}")
    
    abtopython_adapter = ABtoPythonAdapter(adapter)
    viz_data = abtopython_adapter.export_to_visualization(collection)
    print(f"ABtoPython adapter exported {len(viz_data['timestamps'])} timestamps for visualization")
    
    # Get statistics
    stats = adapter.get_statistics()
    print(f"Conversion statistics: {stats}")
    
    print("Unified Data Pipeline test complete!")