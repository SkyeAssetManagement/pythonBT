#!/usr/bin/env python3
"""
CSV Trade Loader
================
Loads trade data from CSV files for visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import logging

from trade_data import TradeData, TradeCollection

logger = logging.getLogger(__name__)

class CSVTradeLoader:
    """Loads trades from CSV files"""
    
    def __init__(self):
        self.last_loaded_file = None
        self.last_loaded_trades = None
        
    def load_csv_trades(self, csv_path: str) -> Optional[TradeCollection]:
        """Alias for load_trades to match expected interface"""
        return self.load_trades(csv_path)
    
    def load_trades(self, csv_path: str) -> Optional[TradeCollection]:
        """
        Load trades from CSV file
        
        Expected CSV format:
        - bar_index or index: Bar number
        - price or trade_price: Trade execution price
        - type or trade_type: Buy/Sell
        - timestamp (optional): Trade timestamp
        """
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            if df.empty:
                logger.warning(f"CSV file is empty: {csv_path}")
                return TradeCollection([])
            
            trades = []
            
            # Detect column names
            bar_col = None
            price_col = None
            type_col = None
            time_col = None
            
            # Find bar index column
            for col in ['bar_index', 'index', 'bar', 'Bar']:
                if col in df.columns:
                    bar_col = col
                    break
            
            # Find price column  
            for col in ['price', 'trade_price', 'Price', 'execution_price']:
                if col in df.columns:
                    price_col = col
                    break
                    
            # Find type column
            for col in ['type', 'trade_type', 'Type', 'side', 'Side']:
                if col in df.columns:
                    type_col = col
                    break
                    
            # Find timestamp column
            for col in ['timestamp', 'time', 'datetime', 'DateTime']:
                if col in df.columns:
                    time_col = col
                    break
            
            if not bar_col or not price_col:
                # Try to infer from numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    bar_col = numeric_cols[0]
                    price_col = numeric_cols[1]
                    logger.info(f"Inferred columns: bar={bar_col}, price={price_col}")
                else:
                    logger.error(f"Could not find required columns in CSV: {csv_path}")
                    return None
            
            # Create trades
            for idx, row in df.iterrows():
                try:
                    bar_index = int(row[bar_col]) if bar_col else idx
                    price = float(row[price_col]) if price_col else 0.0
                    trade_type = str(row[type_col]) if type_col and type_col in row else 'Buy'
                    
                    # Normalize trade type
                    if trade_type.lower() in ['buy', 'long', 'b', '1']:
                        trade_type = 'Buy'
                    elif trade_type.lower() in ['sell', 'short', 's', '-1', '0']:
                        trade_type = 'Sell'
                    
                    # Parse timestamp if available
                    timestamp = None
                    if time_col and time_col in row:
                        try:
                            timestamp = pd.to_datetime(row[time_col])
                        except:
                            pass
                    
                    trade = TradeData(
                        bar_index=bar_index,
                        price=price,
                        trade_type=trade_type,
                        timestamp=timestamp
                    )
                    trades.append(trade)
                    
                except Exception as e:
                    logger.warning(f"Error parsing trade row {idx}: {e}")
                    continue
            
            logger.info(f"Loaded {len(trades)} trades from {csv_path}")
            
            collection = TradeCollection(trades)
            self.last_loaded_file = csv_path
            self.last_loaded_trades = collection
            
            return collection
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            return None
    
    def get_sample_trades(self, num_bars: int = 500) -> TradeCollection:
        """Generate sample trades for testing"""
        return TradeCollection.create_sample_trades(
            num_trades=20,
            num_bars=num_bars
        )
    
    @staticmethod
    def find_trade_files(directory: str = '.') -> List[str]:
        """Find potential trade CSV files in directory"""
        trade_files = []
        path = Path(directory)
        
        # Look for files with 'trade' in the name
        for pattern in ['*trade*.csv', '*trades*.csv', '*Trade*.csv']:
            trade_files.extend(path.glob(pattern))
        
        # Also check common locations
        common_paths = [
            'trades.csv',
            'data/trades.csv', 
            'results/trades.csv',
            'output/trades.csv'
        ]
        
        for csv_path in common_paths:
            full_path = path / csv_path
            if full_path.exists():
                trade_files.append(full_path)
        
        return [str(f) for f in trade_files]