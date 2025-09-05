"""
Lightning-fast parquet data retrieval system for backtesting.
Optimized for maximum speed with array processing.
"""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Union
import time
import logging

logger = logging.getLogger(__name__)

class ParquetLoader:
    """Ultra-fast parquet data retrieval system."""
    
    def __init__(self, parquet_root: str):
        self.parquet_root = Path(parquet_root)
        self.cache = {}  # In-memory cache for backtesting
        
        logger.info(f"Initialized loader: {self.parquet_root}")
    
    def discover_symbols(self) -> List[str]:
        """Discover all available symbols."""
        symbols = []
        for symbol_dir in self.parquet_root.iterdir():
            if symbol_dir.is_dir():
                symbols.append(symbol_dir.name)
        return sorted(symbols)
    
    def load_symbol_data(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, np.ndarray]]:
        """
        Load symbol data as numpy arrays for vectorBT.
        
        Args:
            symbol: Symbol to load
            use_cache: Whether to use in-memory cache
            
        Returns:
            Dictionary with numpy arrays: {datetime, open, high, low, close, volume}
        """
        cache_key = f"{symbol}_full"
        
        if use_cache and cache_key in self.cache:
            logger.debug(f"Using cached data for {symbol}")
            return self.cache[cache_key]
        
        try:
            start_time = time.time()
            
            # Find parquet file
            symbol_dir = self.parquet_root / symbol
            parquet_files = list(symbol_dir.glob("*.parquet"))
            
            if not parquet_files:
                logger.warning(f"No parquet files found for {symbol}")
                return None
            
            # Use the first parquet file found
            parquet_file = parquet_files[0]
            
            # Load using Polars for speed
            df = pl.read_parquet(parquet_file)
            
            # Convert to numpy arrays
            datetime_array = df['datetime'].to_numpy().astype('datetime64[ns]')
            
            data = {
                'datetime': datetime_array.astype('datetime64[s]').astype(np.int64),  # Convert to Unix seconds
                'datetime_ns': datetime_array,  # Keep original for display
                'open': df['open'].to_numpy().astype(np.float64),
                'high': df['high'].to_numpy().astype(np.float64),
                'low': df['low'].to_numpy().astype(np.float64),
                'close': df['close'].to_numpy().astype(np.float64),
                'volume': df['volume'].to_numpy().astype(np.float64)
            }
            
            # Cache the data
            if use_cache:
                self.cache[cache_key] = data
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {symbol}: {len(data['close'])} bars in {load_time:.3f}s")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
            return None
    
    def load_date_range(self, symbol: str, start_date: Union[str, datetime], 
                       end_date: Union[str, datetime]) -> Optional[Dict[str, np.ndarray]]:
        """
        Load symbol data for specific date range.
        
        Args:
            symbol: Symbol to load
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            
        Returns:
            Dictionary with numpy arrays for date range
        """
        try:
            # Load full data
            data = self.load_symbol_data(symbol, use_cache=True)
            if data is None:
                return None
            
            # Convert dates to datetime if needed
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Filter by date range using vectorized operations
            datetime_array = data['datetime']
            mask = (datetime_array >= np.datetime64(start_date)) & (datetime_array <= np.datetime64(end_date))
            
            # Apply mask to all arrays
            filtered_data = {}
            for key, array in data.items():
                filtered_data[key] = array[mask]
            
            logger.info(f"Filtered {symbol}: {len(filtered_data['close'])} bars in date range")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Failed to filter {symbol} by date: {e}")
            return None
    
    def get_latest_bars(self, symbol: str, n_bars: int = 1000) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the latest N bars for a symbol.
        
        Args:
            symbol: Symbol to load
            n_bars: Number of latest bars to return
            
        Returns:
            Dictionary with numpy arrays for latest bars
        """
        try:
            data = self.load_symbol_data(symbol, use_cache=True)
            if data is None:
                return None
            
            # Get latest N bars using array slicing
            latest_data = {}
            for key, array in data.items():
                latest_data[key] = array[-n_bars:] if len(array) > n_bars else array
            
            logger.info(f"Retrieved latest {len(latest_data['close'])} bars for {symbol}")
            return latest_data
            
        except Exception as e:
            logger.error(f"Failed to get latest bars for {symbol}: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get basic information about a symbol.
        
        Returns:
            Dictionary with symbol metadata
        """
        try:
            data = self.load_symbol_data(symbol, use_cache=True)
            if data is None:
                return None
            
            datetime_array = data['datetime']
            close_array = data['close']
            
            info = {
                'symbol': symbol,
                'total_bars': len(close_array),
                'start_date': str(datetime_array[0]),
                'end_date': str(datetime_array[-1]),
                'first_price': float(close_array[0]),
                'last_price': float(close_array[-1]),
                'min_price': float(np.min(close_array)),
                'max_price': float(np.max(close_array)),
                'total_volume': float(np.sum(data['volume']))
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        return {
            'cached_symbols': list(self.cache.keys()),
            'cache_size': len(self.cache)
        }


def main():
    """Test the parquet loader."""
    # Path relative to the main ABtoPython directory
    parquet_root = Path(__file__).parent.parent.parent.parent / "parquet_data"
    
    loader = ParquetLoader(str(parquet_root))
    
    print("Discovering symbols...")
    symbols = loader.discover_symbols()
    print(f"Found symbols: {symbols}")
    
    if symbols:
        symbol = symbols[0]
        print(f"\nTesting with symbol: {symbol}")
        
        # Get symbol info
        info = loader.get_symbol_info(symbol)
        if info:
            print(f"Symbol info: {info}")
        
        # Load latest 100 bars
        latest_data = loader.get_latest_bars(symbol, 100)
        if latest_data:
            print(f"Latest data shape: {len(latest_data['close'])} bars")
            print(f"Date range: {latest_data['datetime'][0]} to {latest_data['datetime'][-1]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()