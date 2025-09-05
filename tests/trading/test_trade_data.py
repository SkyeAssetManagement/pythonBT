#!/usr/bin/env python3
"""
Characterization Tests for TradeData and TradeCollection
=========================================================
100% coverage tests for trade data structures before refactoring.
Following safety-first principles: NO refactoring without tests.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from trading.data.trade_data import TradeData, TradeCollection
import trading.data.trade_data_extended  # Import extensions


class TestTradeData(unittest.TestCase):
    """Test TradeData class with 100% coverage"""
    
    def test_valid_trade_creation(self):
        """Test creating valid trades"""
        # Test BUY trade
        trade = TradeData(
            trade_id=1,
            timestamp=pd.Timestamp('2024-01-01 10:00:00'),
            bar_index=100,
            trade_type='BUY',
            price=100.50,
            size=10.0,
            pnl=None,
            strategy='test_strategy',
            symbol='AAPL'
        )
        self.assertEqual(trade.trade_id, 1)
        self.assertEqual(trade.trade_type, 'BUY')
        self.assertEqual(trade.price, 100.50)
        self.assertEqual(trade.size, 10.0)
        
    def test_invalid_trade_type(self):
        """Test that invalid trade types raise ValueError"""
        with self.assertRaises(ValueError) as context:
            TradeData(
                trade_id=1,
                timestamp=pd.Timestamp('2024-01-01'),
                bar_index=100,
                trade_type='INVALID',
                price=100.0,
                size=10.0
            )
        self.assertIn('Invalid trade_type', str(context.exception))
        
    def test_invalid_price(self):
        """Test that negative/zero prices raise ValueError"""
        with self.assertRaises(ValueError) as context:
            TradeData(
                trade_id=1,
                timestamp=pd.Timestamp('2024-01-01'),
                bar_index=100,
                trade_type='BUY',
                price=0,
                size=10.0
            )
        self.assertIn('Price must be positive', str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            TradeData(
                trade_id=1,
                timestamp=pd.Timestamp('2024-01-01'),
                bar_index=100,
                trade_type='BUY',
                price=-10.0,
                size=10.0
            )
        self.assertIn('Price must be positive', str(context.exception))
        
    def test_invalid_size(self):
        """Test that negative/zero sizes raise ValueError"""
        with self.assertRaises(ValueError) as context:
            TradeData(
                trade_id=1,
                timestamp=pd.Timestamp('2024-01-01'),
                bar_index=100,
                trade_type='BUY',
                price=100.0,
                size=0
            )
        self.assertIn('Size must be positive', str(context.exception))
        
    def test_trade_properties(self):
        """Test all trade property methods"""
        # Test BUY trade properties
        buy_trade = TradeData(
            trade_id=1,
            timestamp=pd.Timestamp('2024-01-01'),
            bar_index=100,
            trade_type='BUY',
            price=100.0,
            size=10.0
        )
        self.assertTrue(buy_trade.is_entry)
        self.assertFalse(buy_trade.is_exit)
        self.assertTrue(buy_trade.is_long)
        self.assertFalse(buy_trade.is_short)
        
        # Test SELL trade properties
        sell_trade = TradeData(
            trade_id=2,
            timestamp=pd.Timestamp('2024-01-01'),
            bar_index=101,
            trade_type='SELL',
            price=105.0,
            size=10.0
        )
        self.assertFalse(sell_trade.is_entry)
        self.assertTrue(sell_trade.is_exit)
        self.assertTrue(sell_trade.is_long)
        self.assertFalse(sell_trade.is_short)
        
        # Test SHORT trade properties
        short_trade = TradeData(
            trade_id=3,
            timestamp=pd.Timestamp('2024-01-01'),
            bar_index=102,
            trade_type='SHORT',
            price=110.0,
            size=5.0
        )
        self.assertTrue(short_trade.is_entry)
        self.assertFalse(short_trade.is_exit)
        self.assertFalse(short_trade.is_long)
        self.assertTrue(short_trade.is_short)
        
        # Test COVER trade properties
        cover_trade = TradeData(
            trade_id=4,
            timestamp=pd.Timestamp('2024-01-01'),
            bar_index=103,
            trade_type='COVER',
            price=108.0,
            size=5.0
        )
        self.assertFalse(cover_trade.is_entry)
        self.assertTrue(cover_trade.is_exit)
        self.assertFalse(cover_trade.is_long)
        self.assertTrue(cover_trade.is_short)


class TestTradeCollection(unittest.TestCase):
    """Test TradeCollection class with 100% coverage"""
    
    def setUp(self):
        """Create sample trades for testing"""
        self.trades = [
            TradeData(1, pd.Timestamp('2024-01-01 09:00:00'), 0, 'BUY', 100.0, 10.0),
            TradeData(2, pd.Timestamp('2024-01-01 10:00:00'), 10, 'SELL', 105.0, 10.0, pnl=50.0),
            TradeData(3, pd.Timestamp('2024-01-01 11:00:00'), 20, 'SHORT', 110.0, 5.0),
            TradeData(4, pd.Timestamp('2024-01-01 12:00:00'), 30, 'COVER', 108.0, 5.0, pnl=10.0),
            TradeData(5, pd.Timestamp('2024-01-01 13:00:00'), 40, 'BUY', 107.0, 15.0),
        ]
        
    def test_collection_creation(self):
        """Test creating a trade collection"""
        collection = TradeCollection(self.trades)
        self.assertEqual(collection.total_trades, 5)
        self.assertEqual(len(collection.trades), 5)
        
    def test_empty_collection(self):
        """Test creating an empty collection"""
        collection = TradeCollection([])
        self.assertEqual(collection.total_trades, 0)
        self.assertEqual(collection.date_range, (None, None))
        self.assertEqual(collection.price_range, (None, None))
        
    def test_collection_sorting(self):
        """Test that trades are sorted by bar_index"""
        # Create unsorted trades
        unsorted_trades = [
            TradeData(1, pd.Timestamp('2024-01-01'), 30, 'BUY', 100.0, 10.0),
            TradeData(2, pd.Timestamp('2024-01-01'), 10, 'SELL', 105.0, 10.0),
            TradeData(3, pd.Timestamp('2024-01-01'), 20, 'SHORT', 110.0, 5.0),
        ]
        collection = TradeCollection(unsorted_trades)
        
        # Check trades are sorted
        bar_indexes = [t.bar_index for t in collection.trades]
        self.assertEqual(bar_indexes, [10, 20, 30])
        
    def test_get_trades_in_range(self):
        """Test getting trades within a bar range"""
        collection = TradeCollection(self.trades)
        
        # Get trades in range [10, 30]
        trades_in_range = collection.get_trades_in_range(10, 30)
        self.assertEqual(len(trades_in_range), 3)
        self.assertEqual(trades_in_range[0].bar_index, 10)
        self.assertEqual(trades_in_range[-1].bar_index, 30)
        
        # Test empty range
        empty_range = collection.get_trades_in_range(100, 200)
        self.assertEqual(len(empty_range), 0)
        
    def test_get_entry_exit_pairs(self):
        """Test pairing entry and exit trades"""
        collection = TradeCollection(self.trades)
        pairs = collection.get_entry_exit_pairs()
        
        # Should have 2 complete pairs
        self.assertEqual(len(pairs), 2)
        
        # First pair: BUY -> SELL
        self.assertEqual(pairs[0][0].trade_type, 'BUY')
        self.assertEqual(pairs[0][1].trade_type, 'SELL')
        
        # Second pair: SHORT -> COVER
        self.assertEqual(pairs[1][0].trade_type, 'SHORT')
        self.assertEqual(pairs[1][1].trade_type, 'COVER')
        
    def test_calculate_statistics(self):
        """Test statistics calculation"""
        collection = TradeCollection(self.trades)
        stats = collection.calculate_statistics()
        
        self.assertIn('total_trades', stats)
        self.assertIn('total_pnl', stats)
        self.assertIn('win_rate', stats)
        self.assertIn('avg_pnl', stats)
        self.assertEqual(stats['total_trades'], 5)
        
    def test_filter_by_type(self):
        """Test filtering trades by type"""
        collection = TradeCollection(self.trades)
        
        # Filter BUY trades
        buy_trades = collection.filter_by_type('BUY')
        self.assertEqual(len(buy_trades), 2)
        for trade in buy_trades:
            self.assertEqual(trade.trade_type, 'BUY')
            
    def test_filter_by_strategy(self):
        """Test filtering by strategy"""
        # Create trades with strategies
        trades_with_strategy = [
            TradeData(1, pd.Timestamp('2024-01-01'), 0, 'BUY', 100.0, 10.0, strategy='strat1'),
            TradeData(2, pd.Timestamp('2024-01-01'), 10, 'SELL', 105.0, 10.0, strategy='strat1'),
            TradeData(3, pd.Timestamp('2024-01-01'), 20, 'BUY', 110.0, 5.0, strategy='strat2'),
        ]
        collection = TradeCollection(trades_with_strategy)
        
        strat1_trades = collection.filter_by_strategy('strat1')
        self.assertEqual(len(strat1_trades), 2)
        
    def test_date_range_calculation(self):
        """Test date range calculation"""
        collection = TradeCollection(self.trades)
        date_range = collection.date_range
        
        self.assertIsNotNone(date_range)
        self.assertEqual(date_range[0], pd.Timestamp('2024-01-01 09:00:00'))
        self.assertEqual(date_range[1], pd.Timestamp('2024-01-01 13:00:00'))
        
    def test_price_range_calculation(self):
        """Test price range calculation"""
        collection = TradeCollection(self.trades)
        price_range = collection.price_range
        
        self.assertIsNotNone(price_range)
        self.assertEqual(price_range[0], 100.0)  # min price
        self.assertEqual(price_range[1], 110.0)  # max price


class TestTradeCollectionPerformance(unittest.TestCase):
    """Test performance with large datasets"""
    
    def test_large_dataset_performance(self):
        """Test with 10,000 trades"""
        import time
        
        # Create 10,000 trades
        large_trades = []
        base_time = pd.Timestamp('2024-01-01')
        
        for i in range(10000):
            trade = TradeData(
                trade_id=i,
                timestamp=base_time + pd.Timedelta(minutes=i),
                bar_index=i,
                trade_type='BUY' if i % 2 == 0 else 'SELL',
                price=100.0 + np.random.randn(),
                size=10.0
            )
            large_trades.append(trade)
        
        # Measure creation time
        start = time.time()
        collection = TradeCollection(large_trades)
        creation_time = time.time() - start
        
        # Should create in under 1 second
        self.assertLess(creation_time, 1.0)
        
        # Test range query performance
        start = time.time()
        trades_in_range = collection.get_trades_in_range(5000, 6000)
        query_time = time.time() - start
        
        # Should query in under 0.1 seconds
        self.assertLess(query_time, 0.1)
        self.assertEqual(len(trades_in_range), 1001)


if __name__ == '__main__':
    # Run tests with coverage
    import coverage
    
    cov = coverage.Coverage()
    cov.start()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTradeData))
    suite.addTests(loader.loadTestsFromTestCase(TestTradeCollection))
    suite.addTests(loader.loadTestsFromTestCase(TestTradeCollectionPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Stop coverage and report
    cov.stop()
    cov.save()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    # Coverage report
    print("\n" + "="*60)
    print("COVERAGE REPORT")
    print("="*60)
    try:
        cov.report(include=['*/trading/data/trade_data.py'])
    except:
        print("Run from project root to see coverage report")