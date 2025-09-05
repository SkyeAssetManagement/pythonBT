#!/usr/bin/env python3
"""
CSV Trade Loader - Flexible CSV import for trade data
====================================================

Supports multiple CSV formats with intelligent column mapping and data parsing.
Compatible with various trading platforms and export formats.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from trade_data import TradeData, TradeCollection

logger = logging.getLogger(__name__)

class CSVTradeLoader:
    """Flexible CSV trade loader with intelligent format detection"""
    
    # Column name mappings (case-insensitive)
    DATETIME_COLUMNS = ['datetime', 'timestamp', 'time', 'date_time', 'date', 'entry_time', 'exit_time']
    PRICE_COLUMNS = ['price', 'entry_price', 'exit_price', 'fill_price', 'avg_price']
    DIRECTION_COLUMNS = ['direction', 'type', 'trade_type', 'side', 'action', 'signal']
    SIZE_COLUMNS = ['size', 'quantity', 'qty', 'shares', 'contracts', 'volume', 'amount']
    PNL_COLUMNS = ['pnl', 'profit', 'profit_loss', 'pl', 'return', 'gain_loss']
    SYMBOL_COLUMNS = ['symbol', 'instrument', 'ticker', 'contract', 'security']
    STRATEGY_COLUMNS = ['strategy', 'system', 'method', 'algo', 'model']
    
    # Direction mappings (flexible input -> standard output)
    DIRECTION_MAPPING = {
        # Buy variations
        'buy': 'BUY', 'b': 'BUY', 'long': 'BUY', 'l': 'BUY',
        'enter long': 'BUY', 'open long': 'BUY', '1': 'BUY',
        
        # Sell variations  
        'sell': 'SELL', 's': 'SELL', 'exit': 'SELL', 'close': 'SELL',
        'exit long': 'SELL', 'close long': 'SELL', '-1': 'SELL',
        
        # Short variations
        'short': 'SHORT', 'sh': 'SHORT', 'sell short': 'SHORT',
        'enter short': 'SHORT', 'open short': 'SHORT', '-2': 'SHORT',
        
        # Cover variations
        'cover': 'COVER', 'c': 'COVER', 'buy cover': 'COVER',
        'exit short': 'COVER', 'close short': 'COVER', '2': 'COVER'
    }
    
    def __init__(self):
        """Initialize CSV loader"""
        self.detected_format = None
        self.column_mapping = {}
        
    def load_csv_trades(self, file_path: Union[str, Path]) -> TradeCollection:
        """
        Load trades from CSV file with automatic format detection
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            TradeCollection with parsed trades
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns missing or data invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Loading CSV trades from: {file_path}")
        
        try:
            # Read CSV with flexible parameters
            df = pd.read_csv(file_path, 
                           parse_dates=False,  # We'll handle datetime parsing
                           dtype=str,  # Read as strings first for cleaning
                           encoding='utf-8')
            
            logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
            
        except Exception as e:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    df = pd.read_csv(file_path, 
                                   parse_dates=False,
                                   dtype=str,
                                   encoding=encoding)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not read CSV file with any encoding: {e}")
        
        # Clean and detect format
        df = self._clean_dataframe(df)
        self._detect_format(df)
        
        # Parse trades
        trades = self._parse_trades(df)
        
        logger.info(f"Successfully parsed {len(trades)} trades")
        return TradeCollection(trades)
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe - remove empty rows, strip whitespace, etc."""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Strip whitespace from string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        # Remove rows where critical fields are empty/null
        df = df.replace('', np.nan)
        
        return df
    
    def _detect_format(self, df: pd.DataFrame):
        """Detect CSV format and create column mapping"""
        self.column_mapping = {}
        columns = [col.lower().strip() for col in df.columns]
        
        logger.info(f"Detecting format for columns: {columns}")
        
        # Find datetime column
        datetime_col = self._find_column(columns, self.DATETIME_COLUMNS)
        if not datetime_col:
            raise ValueError(f"No datetime column found. Expected one of: {self.DATETIME_COLUMNS}")
        
        # Find price column
        price_col = self._find_column(columns, self.PRICE_COLUMNS)
        if not price_col:
            raise ValueError(f"No price column found. Expected one of: {self.PRICE_COLUMNS}")
        
        # Find direction column
        direction_col = self._find_column(columns, self.DIRECTION_COLUMNS)
        if not direction_col:
            raise ValueError(f"No direction column found. Expected one of: {self.DIRECTION_COLUMNS}")
        
        # Required columns
        self.column_mapping = {
            'datetime': datetime_col,
            'price': price_col,
            'direction': direction_col
        }
        
        # Optional columns
        optional_mappings = {
            'size': self._find_column(columns, self.SIZE_COLUMNS),
            'pnl': self._find_column(columns, self.PNL_COLUMNS),
            'symbol': self._find_column(columns, self.SYMBOL_COLUMNS),
            'strategy': self._find_column(columns, self.STRATEGY_COLUMNS)
        }
        
        # Add found optional columns
        for key, col in optional_mappings.items():
            if col:
                self.column_mapping[key] = col
        
        logger.info(f"Detected column mapping: {self.column_mapping}")
    
    def _find_column(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        """Find matching column name (case-insensitive)"""
        for candidate in candidates:
            for col in columns:
                if candidate.lower() in col.lower() or col.lower() in candidate.lower():
                    return col
        return None
    
    def _parse_trades(self, df: pd.DataFrame) -> List[TradeData]:
        """Parse dataframe into TradeData objects"""
        trades = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                trade = self._parse_trade_row(row, idx)
                if trade:
                    trades.append(trade)
            except Exception as e:
                errors.append(f"Row {idx}: {e}")
                continue
        
        if errors:
            logger.warning(f"Encountered {len(errors)} parsing errors:")
            for error in errors[:5]:  # Show first 5 errors
                logger.warning(f"  {error}")
            if len(errors) > 5:
                logger.warning(f"  ... and {len(errors) - 5} more errors")
        
        if not trades:
            raise ValueError("No valid trades could be parsed from CSV")
        
        return trades
    
    def _parse_trade_row(self, row: pd.Series, row_idx: int) -> Optional[TradeData]:
        """Parse single row into TradeData"""
        # Get original column names for row access
        orig_columns = {v: k for k, v in self.column_mapping.items()}
        
        # Parse datetime
        datetime_val = row[self.column_mapping['datetime']]
        if pd.isna(datetime_val) or str(datetime_val).strip() == '':
            return None  # Skip rows with no datetime
        
        timestamp = self._parse_datetime(datetime_val)
        
        # Parse price
        price_val = row[self.column_mapping['price']]
        price = self._parse_float(price_val, 'price')
        
        # Parse direction
        direction_val = row[self.column_mapping['direction']]
        trade_type = self._parse_direction(direction_val)
        
        # Parse optional fields
        size = 1.0  # Default size
        if 'size' in self.column_mapping:
            size_val = row[self.column_mapping['size']]
            if not pd.isna(size_val) and str(size_val).strip():
                size = self._parse_float(size_val, 'size')
        
        pnl = None
        if 'pnl' in self.column_mapping:
            pnl_val = row[self.column_mapping['pnl']]
            if not pd.isna(pnl_val) and str(pnl_val).strip():
                pnl = self._parse_float(pnl_val, 'pnl', allow_negative=True)
        
        symbol = None
        if 'symbol' in self.column_mapping:
            symbol_val = row[self.column_mapping['symbol']]
            if not pd.isna(symbol_val) and str(symbol_val).strip():
                symbol = str(symbol_val).strip()
        
        strategy = None
        if 'strategy' in self.column_mapping:
            strategy_val = row[self.column_mapping['strategy']]
            if not pd.isna(strategy_val) and str(strategy_val).strip():
                strategy = str(strategy_val).strip()
        
        # Create TradeData (bar_index will be set later when we have price data context)
        trade = TradeData(
            trade_id=row_idx,
            timestamp=timestamp,
            bar_index=0,  # Placeholder - will be set by caller with price data context
            trade_type=trade_type,
            price=price,
            size=size,
            pnl=pnl,
            strategy=strategy,
            symbol=symbol
        )
        
        return trade
    
    def _parse_datetime(self, value: str) -> pd.Timestamp:
        """Parse datetime with multiple format support"""
        if pd.isna(value):
            raise ValueError("Datetime value is null")
        
        value = str(value).strip()
        if not value:
            raise ValueError("Datetime value is empty")
        
        # Try common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',    # 2024-01-15 09:30:00
            '%m/%d/%Y %H:%M:%S',    # 01/15/2024 09:30:00
            '%d/%m/%Y %H:%M:%S',    # 15/01/2024 09:30:00
            '%Y-%m-%d %H:%M',       # 2024-01-15 09:30
            '%m/%d/%Y %H:%M',       # 01/15/2024 09:30
            '%Y-%m-%d',             # 2024-01-15
            '%m/%d/%Y',             # 01/15/2024
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(value, format=fmt)
            except:
                continue
        
        # Try pandas auto-parsing as last resort
        try:
            return pd.to_datetime(value)
        except:
            raise ValueError(f"Could not parse datetime: {value}")
    
    def _parse_float(self, value: str, field_name: str, allow_negative: bool = False) -> float:
        """Parse float value with error handling"""
        if pd.isna(value):
            raise ValueError(f"{field_name} value is null")
        
        # Clean value
        value = str(value).strip().replace(',', '').replace('$', '')
        
        if not value:
            raise ValueError(f"{field_name} value is empty")
        
        try:
            result = float(value)
            if not allow_negative and result <= 0:
                raise ValueError(f"{field_name} must be positive, got: {result}")
            return result
        except ValueError as e:
            if "could not convert" in str(e):
                raise ValueError(f"Could not parse {field_name}: {value}")
            raise
    
    def _parse_direction(self, value: str) -> str:
        """Parse direction with flexible mapping"""
        if pd.isna(value):
            raise ValueError("Direction value is null")
        
        value = str(value).strip().lower()
        if not value:
            raise ValueError("Direction value is empty")
        
        # Direct lookup
        if value in self.DIRECTION_MAPPING:
            return self.DIRECTION_MAPPING[value]
        
        # Partial matching for compound terms
        for key, mapped in self.DIRECTION_MAPPING.items():
            if key in value or value in key:
                return mapped
        
        raise ValueError(f"Unknown direction: {value}")

def create_sample_csv(file_path: Union[str, Path], format_type: str = 'minimal', n_rows: int = 100):
    """
    Create sample CSV files for testing
    
    Args:
        file_path: Output file path
        format_type: 'minimal' or 'extended'
        n_rows: Number of rows to create
    """
    import random
    from datetime import datetime, timedelta
    
    # Generate sample data
    data = []
    base_time = datetime(2024, 1, 15, 9, 30, 0)
    
    for i in range(n_rows):
        # Random time
        time_offset = timedelta(minutes=random.randint(0, 1440))  # Random time within day
        timestamp = base_time + time_offset
        
        # Random trade data
        price = 4000 + random.uniform(-200, 200)
        direction = random.choice(['BUY', 'SELL', 'SHORT', 'COVER'])
        
        row = {
            'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'price': round(price, 2),
            'direction': direction
        }
        
        if format_type == 'extended':
            size = random.randint(1, 10)
            pnl = round(random.uniform(-500, 500), 2) if direction in ['SELL', 'COVER'] else 0
            
            row.update({
                'size': size,
                'symbol': 'ES',
                'pnl': pnl,
                'strategy': random.choice(['TimeWindow', 'Momentum', 'Reversal'])
            })
        
        data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Created sample CSV: {file_path} with {n_rows} rows ({format_type} format)")

if __name__ == "__main__":
    # Test the CSV loader
    print("Testing CSV Trade Loader...")
    
    # Create sample CSV files
    create_sample_csv('sample_trades_minimal.csv', 'minimal', 50)
    create_sample_csv('sample_trades_extended.csv', 'extended', 50)
    
    # Test loading
    loader = CSVTradeLoader()
    
    try:
        # Test minimal format
        trades_minimal = loader.load_csv_trades('sample_trades_minimal.csv')
        print(f"Loaded {len(trades_minimal)} trades from minimal CSV")
        print(f"Statistics: {trades_minimal.get_statistics()}")
        
        # Test extended format
        trades_extended = loader.load_csv_trades('sample_trades_extended.csv')
        print(f"Loaded {len(trades_extended)} trades from extended CSV")
        print(f"Statistics: {trades_extended.get_statistics()}")
        
        print("CSV loader test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up test files
    Path('sample_trades_minimal.csv').unlink(missing_ok=True)
    Path('sample_trades_extended.csv').unlink(missing_ok=True)