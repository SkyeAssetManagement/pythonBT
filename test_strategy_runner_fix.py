#!/usr/bin/env python3
"""
Test script to verify strategy runner fix for TradeCollection error
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/trading')
sys.path.insert(0, 'src/trading/visualization')

import pandas as pd
import numpy as np
from core.strategy_runner_adapter import StrategyRunnerAdapter
from core.trade_types import TradeRecordCollection
from data.trade_data import TradeCollection

def test_adapter_conversion():
    """Test that adapter correctly converts TradeRecordCollection to TradeCollection"""
    print("\n=== Testing Strategy Runner Adapter Conversion ===")

    # Create adapter
    adapter = StrategyRunnerAdapter()

    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    prices = 4000 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        'DateTime': dates,
        'Open': prices + np.random.randn(100) * 1,
        'High': prices + np.abs(np.random.randn(100) * 2),
        'Low': prices - np.abs(np.random.randn(100) * 2),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    })

    # Test parameters
    params = {
        'fast_period': 10,
        'slow_period': 20,
        'long_only': True
    }

    # Test with unified engine if enabled
    if adapter.use_unified_engine:
        print(f"Unified engine is ENABLED")

        # Run strategy
        trades = adapter.run_strategy('sma_crossover', params, df)

        print(f"Result type: {type(trades)}")
        print(f"Is TradeRecordCollection: {isinstance(trades, TradeRecordCollection)}")
        print(f"Is TradeCollection: {isinstance(trades, TradeCollection)}")

        # If it's TradeRecordCollection, convert it
        if isinstance(trades, TradeRecordCollection):
            print("Converting TradeRecordCollection to legacy format...")
            legacy_trades = trades.to_legacy_collection()
            print(f"Converted type: {type(legacy_trades)}")
            print(f"Is TradeCollection after conversion: {isinstance(legacy_trades, TradeCollection)}")

            # Verify we can emit this
            try:
                # Simulate signal emission
                from PyQt5.QtCore import pyqtSignal, QObject

                class TestEmitter(QObject):
                    trades_generated = pyqtSignal(TradeCollection)

                    def test_emit(self, trades):
                        self.trades_generated.emit(trades)
                        print("SUCCESS: Signal emission works!")

                # Test emission
                from PyQt5.QtWidgets import QApplication
                app = QApplication([])
                emitter = TestEmitter()
                emitter.test_emit(legacy_trades)

            except Exception as e:
                print(f"ERROR during emission test: {e}")

    else:
        print("Unified engine is DISABLED - using legacy path")

        # Run strategy
        trades = adapter.run_strategy('sma_crossover', params, df)
        print(f"Result type: {type(trades)}")
        print(f"Is TradeCollection: {isinstance(trades, TradeCollection)}")

    print("\n=== Test Complete ===")

def test_import_fix():
    """Test that the import in strategy_runner.py will work"""
    print("\n=== Testing Import Fix ===")

    # Test the import that strategy_runner uses
    try:
        from core.trade_types import TradeRecordCollection
        print("SUCCESS: Can import TradeRecordCollection")

        # Test isinstance check
        from core.strategy_runner_adapter import StrategyRunnerAdapter
        adapter = StrategyRunnerAdapter()

        # Create sample data
        df = pd.DataFrame({
            'DateTime': pd.date_range(start='2024-01-01', periods=50, freq='5min'),
            'Close': 4000 + np.cumsum(np.random.randn(50) * 2),
            'Open': 4000 + np.cumsum(np.random.randn(50) * 2),
            'High': 4100 + np.cumsum(np.random.randn(50) * 2),
            'Low': 3900 + np.cumsum(np.random.randn(50) * 2)
        })

        result = adapter.run_strategy('sma_crossover', {'fast_period': 5, 'slow_period': 10, 'long_only': True}, df)

        if isinstance(result, TradeRecordCollection):
            print("Adapter returned TradeRecordCollection - conversion needed")
            legacy = result.to_legacy_collection()
            print(f"Conversion successful: {isinstance(legacy, TradeCollection)}")
        else:
            print(f"Adapter returned {type(result)} - no conversion needed")

    except ImportError as e:
        print(f"IMPORT ERROR: {e}")

    print("\n=== Import Test Complete ===")

if __name__ == "__main__":
    test_import_fix()
    test_adapter_conversion()