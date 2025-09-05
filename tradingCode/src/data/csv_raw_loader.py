"""
CSV Raw Data Loader for new dataRaw structure.
Handles organized data by frequency/symbol/adjustment type.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class CSVRawLoader:
    """Load data from the new dataRaw CSV structure."""
    
    def __init__(self, data_raw_root: str = None):
        """Initialize with path to dataRaw folder."""
        if data_raw_root is None:
            # Default to relative path from tradingCode
            self.data_root = Path(__file__).parent.parent.parent.parent / "dataRaw"
        else:
            self.data_root = Path(data_raw_root)
        
        logger.info(f"Initialized CSV Raw Loader: {self.data_root}")
    
    def discover_symbols(self, frequency: str = "1m") -> List[str]:
        """Discover all available symbols for a given frequency."""
        freq_dir = self.data_root / frequency
        if not freq_dir.exists():
            logger.warning(f"Frequency directory not found: {freq_dir}")
            return []
        
        symbols = []
        for symbol_dir in freq_dir.iterdir():
            if symbol_dir.is_dir():
                symbols.append(symbol_dir.name)
        return sorted(symbols)
    
    def load_symbol_data(self, symbol: str, frequency: str = "1m", 
                        adjustment: str = "diffAdjusted") -> Optional[Dict[str, np.ndarray]]:
        """
        Load symbol data as numpy arrays for vectorBT.
        
        Args:
            symbol: Symbol to load (e.g., 'GC', 'AD')
            frequency: Data frequency ('1m', '5m', etc.)
            adjustment: 'Current' or 'diffAdjusted'
            
        Returns:
            Dictionary with numpy arrays: {datetime, open, high, low, close, volume}
        """
        # Construct file path
        if adjustment == "Current":
            file_path = self.data_root / frequency / symbol / "Current" / f"{symbol}-NONE-{frequency}-EST-NoPad.csv"
        elif adjustment == "diffAdjusted":
            file_path = self.data_root / frequency / symbol / "diffAdjusted" / f"{symbol}-DIFF-{frequency}-EST-NoPad.csv"
        else:
            # Fallback for other adjustment types
            file_path = self.data_root / frequency / symbol / adjustment / f"{symbol}-DIFF-{frequency}-EST-NoPad.csv"
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            return None
        
        try:
            logger.info(f"Loading {symbol} {frequency} data from {file_path}")
            
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Handle different column formats
            if 'Date' in df.columns and 'Time' in df.columns:
                # Combine Date and Time columns
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            elif 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
            else:
                logger.error(f"No DateTime or Date/Time columns found in {file_path}")
                return None
            
            # Expected columns after DateTime creation
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in {file_path}: {missing_cols}")
                logger.info(f"Available columns: {list(df.columns)}")
                return None
            
            # Sort by datetime to ensure proper order
            df = df.sort_values('DateTime').reset_index(drop=True)
            
            # Convert to numpy arrays for vectorBT compatibility
            data_arrays = {
                'datetime': df['DateTime'].astype('datetime64[ns]').values.astype('datetime64[ns]').view('int64'),
                'datetime_ns': df['DateTime'].astype('datetime64[ns]').values,  # Keep ns version for timestamps
                'open': df['Open'].astype(np.float64).values,
                'high': df['High'].astype(np.float64).values,
                'low': df['Low'].astype(np.float64).values,
                'close': df['Close'].astype(np.float64).values,
                'volume': df['Volume'].astype(np.float64).values
            }
            
            logger.info(f"Successfully loaded {len(data_arrays['close'])} bars for {symbol}")
            logger.info(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
            
            return data_arrays
            
        except Exception as e:
            logger.error(f"Error loading {symbol} data from {file_path}: {e}")
            return None
    
    def load_symbol_data_filtered(self, symbol: str, frequency: str = "1m", 
                                 adjustment: str = "diffAdjusted",
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Load symbol data with date filtering.
        
        Args:
            symbol: Symbol to load
            frequency: Data frequency  
            adjustment: 'Current' or 'diffAdjusted'
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Filtered data arrays
        """
        # Load full data first
        data = self.load_symbol_data(symbol, frequency, adjustment)
        if data is None:
            return None
        
        # Apply date filtering if specified
        if start_date or end_date:
            df_temp = pd.DataFrame({
                'DateTime': data['datetime_ns'],
                'open': data['open'],
                'high': data['high'], 
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume']
            })
            
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df_temp = df_temp[df_temp['DateTime'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df_temp = df_temp[df_temp['DateTime'] <= end_dt]
            
            # Convert back to arrays
            data = {
                'datetime': df_temp['DateTime'].astype('datetime64[ns]').values.astype('datetime64[ns]').view('int64'),
                'datetime_ns': df_temp['DateTime'].values,
                'open': df_temp['open'].values,
                'high': df_temp['high'].values,
                'low': df_temp['low'].values,
                'close': df_temp['close'].values,
                'volume': df_temp['volume'].values
            }
            
            logger.info(f"Filtered to {len(data['close'])} bars")
        
        return data
    
    def check_data_availability(self, symbol: str, frequency: str = "1m") -> Dict[str, bool]:
        """Check what data types are available for a symbol."""
        availability = {
            'Current': False,
            'diffAdjusted': False
        }
        
        symbol_dir = self.data_root / frequency / symbol
        if not symbol_dir.exists():
            return availability
        
        # Check Current data
        current_file = symbol_dir / "Current" / f"{symbol}-NONE-{frequency}-EST-NoPad.csv"
        availability['Current'] = current_file.exists()
        
        # Check diffAdjusted data  
        diff_file = symbol_dir / "diffAdjusted" / f"{symbol}-DIFF-{frequency}-EST-NoPad.csv"
        availability['diffAdjusted'] = diff_file.exists()
        
        return availability