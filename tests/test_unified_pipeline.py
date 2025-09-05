#!/usr/bin/env python3
"""
Unit tests for the unified data pipeline
Tests conversion between OMtree and ABtoPython formats
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.feature_flags import get_feature_flags
from src.trading.data.unified_pipeline import (
    UnifiedDataFormat,
    DataPipelineAdapter,
    DataValidator,
    OMtreeAdapter,
    ABtoPythonAdapter,
    get_data_adapter
)
from src.trading.data.trade_data import TradeData, TradeCollection


class TestUnifiedDataFormat:
    """Test UnifiedDataFormat dataclass"""
    
    def test_create_unified_format(self):
        """Test creating UnifiedDataFormat instance"""
        unified = UnifiedDataFormat(
            timestamp=pd.Timestamp('2024-01-01 09:00:00'),
            symbol='TEST',
            price=100.0,
            features={'feature1': 0.5},
            target=0.8,
            prediction=0.7,
            trade_type='BUY',
            size=10.0,
            pnl=50.0,
            bar_index=1,
            source_system='test'
        )
        
        assert unified.timestamp == pd.Timestamp('2024-01-01 09:00:00')
        assert unified.symbol == 'TEST'
        assert unified.price == 100.0
        assert unified.features['feature1'] == 0.5
        assert unified.source_system == 'test'
    
    def test_optional_fields(self):
        """Test that optional fields default to None"""
        unified = UnifiedDataFormat(
            timestamp=pd.Timestamp.now(),
            symbol='TEST',
            price=100.0
        )
        
        assert unified.features is None
        assert unified.target is None
        assert unified.prediction is None
        assert unified.trade_type is None
        assert unified.size is None
        assert unified.pnl is None
        assert unified.bar_index is None


class TestDataPipelineAdapter:
    """Test DataPipelineAdapter conversions"""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance with feature flags enabled"""
        flags = get_feature_flags()
        flags.enable('unified_data_pipeline')
        return DataPipelineAdapter()
    
    @pytest.fixture
    def sample_omtree_df(self):
        """Create sample OMtree DataFrame"""
        return pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Time': ['09:00:00', '10:00:00', '11:00:00'],
            'Close': [100.0, 101.0, 102.0],
            'PIR_0-1hr': [0.5, -0.3, 0.2],
            'PIR_1-3hr': [0.8, -0.1, 0.4],
            'Ret_fwd6hr': [0.8, -0.2, 0.5],
            'Prediction': [0.7, -0.1, 0.4]
        })
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration"""
        return {
            'date_column': 'Date',
            'time_column': 'Time',
            'target_column': 'Ret_fwd6hr',
            'feature_columns': ['PIR_0-1hr', 'PIR_1-3hr'],
            'symbol': 'TEST'
        }
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trade collection"""
        trades = [
            TradeData(
                trade_id=1,
                timestamp=pd.Timestamp('2024-01-01 09:00:00'),
                bar_index=100,
                trade_type='BUY',
                price=100.0,
                size=10.0,
                pnl=50.0,
                symbol='TEST'
            ),
            TradeData(
                trade_id=2,
                timestamp=pd.Timestamp('2024-01-01 10:00:00'),
                bar_index=101,
                trade_type='SELL',
                price=101.0,
                size=10.0,
                pnl=10.0,
                symbol='TEST'
            )
        ]
        return TradeCollection(trades)
    
    def test_omtree_to_unified(self, adapter, sample_omtree_df, sample_config):
        """Test converting OMtree DataFrame to unified format"""
        unified = adapter.omtree_to_unified(sample_omtree_df, sample_config)
        
        assert len(unified) == 3
        assert all(isinstance(item, UnifiedDataFormat) for item in unified)
        
        # Check first item
        first = unified[0]
        assert first.symbol == 'TEST'
        assert first.price == 100.0
        assert first.features['PIR_0-1hr'] == 0.5
        assert first.features['PIR_1-3hr'] == 0.8
        assert first.target == 0.8
        assert first.prediction == 0.7
        assert first.source_system == 'omtree'
    
    def test_unified_to_omtree(self, adapter, sample_omtree_df, sample_config):
        """Test converting unified format back to OMtree DataFrame"""
        unified = adapter.omtree_to_unified(sample_omtree_df, sample_config)
        df_back = adapter.unified_to_omtree(unified)
        
        assert isinstance(df_back, pd.DataFrame)
        assert len(df_back) == 3
        assert 'Date' in df_back.columns
        assert 'Time' in df_back.columns
        assert 'Close' in df_back.columns
        assert 'PIR_0-1hr' in df_back.columns
        assert df_back['Close'].iloc[0] == 100.0
    
    def test_trades_to_unified(self, adapter, sample_trades):
        """Test converting trades to unified format"""
        unified = adapter.trades_to_unified(sample_trades)
        
        assert len(unified) == 2
        assert all(isinstance(item, UnifiedDataFormat) for item in unified)
        
        # Check first trade
        first = unified[0]
        assert first.symbol == 'TEST'
        assert first.price == 100.0
        assert first.trade_type == 'BUY'
        assert first.size == 10.0
        assert first.pnl == 50.0
        assert first.bar_index == 100
        assert first.source_system == 'abtopython'
    
    def test_unified_to_trades(self, adapter, sample_trades):
        """Test converting unified format to trades"""
        unified = adapter.trades_to_unified(sample_trades)
        trades_back = adapter.unified_to_trades(unified)
        
        assert isinstance(trades_back, TradeCollection)
        assert len(trades_back) == 2
        assert trades_back.trades[0].price == 100.0
        assert trades_back.trades[0].trade_type == 'BUY'
    
    def test_merge_data_sources(self, adapter, sample_omtree_df, sample_config, sample_trades):
        """Test merging data from both sources"""
        unified = adapter.merge_data_sources(
            omtree_df=sample_omtree_df,
            trades=sample_trades,
            config=sample_config
        )
        
        # Should have data from both sources
        assert len(unified) == 5  # 3 from OMtree + 2 from trades
        
        # Check sorting by timestamp
        timestamps = [item.timestamp for item in unified]
        assert timestamps == sorted(timestamps)
    
    def test_get_statistics(self, adapter, sample_omtree_df, sample_config):
        """Test getting conversion statistics"""
        adapter.omtree_to_unified(sample_omtree_df, sample_config)
        stats = adapter.get_statistics()
        
        assert 'omtree_to_unified' in stats
        assert stats['omtree_to_unified'] == 3


class TestDataValidator:
    """Test DataValidator validation methods"""
    
    def test_validate_omtree_df_valid(self):
        """Test validating valid OMtree DataFrame"""
        df = pd.DataFrame({
            'Date': ['2024-01-01'],
            'Time': ['09:00:00'],
            'Close': [100.0],
            'PIR_0-1hr': [0.5]
        })
        
        required_cols = ['Date', 'Time', 'Close']
        assert DataValidator.validate_omtree_df(df, required_cols) is True
    
    def test_validate_omtree_df_missing_columns(self):
        """Test validating OMtree DataFrame with missing columns"""
        df = pd.DataFrame({
            'Date': ['2024-01-01'],
            'Close': [100.0]
        })
        
        required_cols = ['Date', 'Time', 'Close']
        assert DataValidator.validate_omtree_df(df, required_cols) is False
    
    def test_validate_trades_valid(self):
        """Test validating valid trade collection"""
        trades = [
            TradeData(
                trade_id=1,
                timestamp=pd.Timestamp.now(),
                bar_index=1,
                trade_type='BUY',
                price=100.0,
                size=10.0
            )
        ]
        collection = TradeCollection(trades)
        
        assert DataValidator.validate_trades(collection) is True
    
    def test_validate_trades_invalid_price(self):
        """Test validating trades with invalid price"""
        # Create trade with invalid price should raise ValueError at creation
        with pytest.raises(ValueError, match="Price must be positive"):
            TradeData(
                trade_id=1,
                timestamp=pd.Timestamp.now(),
                bar_index=1,
                trade_type='BUY',
                price=-100.0,  # Invalid negative price
                size=10.0
            )
    
    def test_validate_unified_data(self):
        """Test validating unified data format"""
        data = [
            UnifiedDataFormat(
                timestamp=pd.Timestamp('2024-01-01 09:00:00'),
                symbol='TEST',
                price=100.0
            ),
            UnifiedDataFormat(
                timestamp=pd.Timestamp('2024-01-01 10:00:00'),
                symbol='TEST',
                price=101.0
            )
        ]
        
        assert DataValidator.validate_unified_data(data) is True


class TestOMtreeAdapter:
    """Test OMtree specific adapter"""
    
    @pytest.fixture
    def omtree_adapter(self):
        """Create OMtree adapter instance"""
        flags = get_feature_flags()
        flags.enable('unified_data_pipeline')
        base_adapter = DataPipelineAdapter()
        return OMtreeAdapter(base_adapter)
    
    def test_prepare_data_for_model(self, omtree_adapter):
        """Test preparing data for OMtree model"""
        df = pd.DataFrame({
            'Date': ['2024-01-01'],
            'Time': ['09:00:00'],
            'Close': [100.0],
            'PIR_0-1hr': [0.5],
            'Ret_fwd6hr': [0.8]
        })
        
        config = {
            'date_column': 'Date',
            'time_column': 'Time',
            'target_column': 'Ret_fwd6hr',
            'feature_columns': ['PIR_0-1hr'],
            'symbol': 'TEST'
        }
        
        processed_df = omtree_adapter.prepare_data_for_model(df, config)
        
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == 1
        assert 'Close' in processed_df.columns
    
    def test_process_predictions(self, omtree_adapter):
        """Test processing model predictions"""
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Close': [100.0, 101.0]
        })
        
        predictions = np.array([0.5, -0.3])
        config = {}
        
        result_df = omtree_adapter.process_predictions(df, predictions, config)
        
        assert 'Prediction' in result_df.columns
        assert result_df['Prediction'].iloc[0] == 0.5
        assert result_df['Prediction'].iloc[1] == -0.3


class TestABtoPythonAdapter:
    """Test ABtoPython specific adapter"""
    
    @pytest.fixture
    def ab_adapter(self):
        """Create ABtoPython adapter instance"""
        flags = get_feature_flags()
        flags.enable('unified_data_pipeline')
        base_adapter = DataPipelineAdapter()
        return ABtoPythonAdapter(base_adapter)
    
    def test_export_to_visualization(self, ab_adapter):
        """Test exporting trades for visualization"""
        trades = [
            TradeData(
                trade_id=1,
                timestamp=pd.Timestamp('2024-01-01 09:00:00'),
                bar_index=1,
                trade_type='BUY',
                price=100.0,
                size=10.0
            ),
            TradeData(
                trade_id=2,
                timestamp=pd.Timestamp('2024-01-01 10:00:00'),
                bar_index=2,
                trade_type='SELL',
                price=101.0,
                size=10.0
            )
        ]
        collection = TradeCollection(trades)
        
        viz_data = ab_adapter.export_to_visualization(collection)
        
        assert 'timestamps' in viz_data
        assert 'prices' in viz_data
        assert 'types' in viz_data
        assert 'sizes' in viz_data
        assert len(viz_data['timestamps']) == 2
        assert viz_data['prices'][0] == 100.0
        assert viz_data['types'][0] == 'BUY'


class TestGlobalAdapter:
    """Test global adapter singleton"""
    
    def test_get_data_adapter_singleton(self):
        """Test that get_data_adapter returns singleton"""
        adapter1 = get_data_adapter()
        adapter2 = get_data_adapter()
        
        assert adapter1 is adapter2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])