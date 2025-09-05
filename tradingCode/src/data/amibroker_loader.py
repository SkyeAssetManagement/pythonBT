import numpy as np
import amipy
from typing import Dict, Optional
from pathlib import Path
import yaml
from numba import jit


class AmiBrokerArrayLoader:
    """Load AmiBroker data directly into numpy arrays for vectorized processing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize loader with configuration."""
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                'data': {
                    'amibroker_path': r"D:\oneModelProduction\2.AmiBrokerImportRawData\OneModel5MinDataABDB"
                }
            }
        self.db_path = self.config['data']['amibroker_path']
        
    def load_as_arrays(self, symbol: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Load symbol data directly into numpy arrays.
        
        Returns dict of numpy arrays:
        {
            'datetime': np.array([...]),  # int64 timestamps
            'open': np.array([...], dtype=np.float64),
            'high': np.array([...], dtype=np.float64),
            'low': np.array([...], dtype=np.float64),
            'close': np.array([...], dtype=np.float64),
            'volume': np.array([...], dtype=np.float64)
        }
        """
        # Connect to AmiBroker database
        db = amipy.Database(self.db_path)
        
        # Get symbol data
        ticker = db.get_stock(symbol)
        if ticker is None:
            raise ValueError(f"Symbol {symbol} not found in database")
        
        # Get quotes - amipy returns a list of Quote objects
        quotes = ticker.get_quotes(start_date, end_date)
        
        # Pre-allocate arrays for maximum efficiency
        n_bars = len(quotes)
        data = {
            'datetime': np.empty(n_bars, dtype=np.int64),
            'open': np.empty(n_bars, dtype=np.float64),
            'high': np.empty(n_bars, dtype=np.float64),
            'low': np.empty(n_bars, dtype=np.float64),
            'close': np.empty(n_bars, dtype=np.float64),
            'volume': np.empty(n_bars, dtype=np.float64)
        }
        
        # Vectorized data extraction
        for i, quote in enumerate(quotes):
            data['datetime'][i] = int(quote.datetime.timestamp())
            data['open'][i] = quote.open
            data['high'][i] = quote.high
            data['low'][i] = quote.low
            data['close'][i] = quote.close
            data['volume'][i] = quote.volume
            
        db.close()
        
        return data
    
    def validate_arrays(self, data: Dict[str, np.ndarray]) -> bool:
        """Validate data arrays using vectorized checks."""
        # Check array lengths
        lengths = [len(arr) for arr in data.values()]
        if len(set(lengths)) != 1:
            return False
            
        # Vectorized validation checks
        if np.any(data['high'] < data['low']):
            return False
            
        if np.any(data['close'] <= 0) or np.any(np.isnan(data['close'])):
            return False
            
        if np.any(data['volume'] < 0):
            return False
            
        return True
    
    @staticmethod
    @jit(nopython=True)
    def resample_ohlc(timestamps: np.ndarray, open_prices: np.ndarray,
                     high_prices: np.ndarray, low_prices: np.ndarray,
                     close_prices: np.ndarray, volumes: np.ndarray,
                     target_frequency: int) -> tuple:
        """
        Numba-optimized OHLC resampling.
        
        Args:
            timestamps: Unix timestamps
            open_prices, high_prices, low_prices, close_prices: Price arrays
            volumes: Volume array
            target_frequency: Target frequency in seconds (e.g., 3600 for 1H)
            
        Returns:
            Resampled arrays tuple
        """
        # Group bars by target frequency
        period_starts = timestamps // target_frequency * target_frequency
        unique_periods = np.unique(period_starts)
        
        n_periods = len(unique_periods)
        
        # Pre-allocate result arrays
        new_timestamps = np.empty(n_periods, dtype=np.int64)
        new_open = np.empty(n_periods, dtype=np.float64)
        new_high = np.empty(n_periods, dtype=np.float64)
        new_low = np.empty(n_periods, dtype=np.float64)
        new_close = np.empty(n_periods, dtype=np.float64)
        new_volume = np.empty(n_periods, dtype=np.float64)
        
        # Process each period
        for i, period in enumerate(unique_periods):
            mask = period_starts == period
            
            new_timestamps[i] = period
            new_open[i] = open_prices[mask][0]  # First open
            new_high[i] = np.max(high_prices[mask])
            new_low[i] = np.min(low_prices[mask])
            new_close[i] = close_prices[mask][-1]  # Last close
            new_volume[i] = np.sum(volumes[mask])
            
        return new_timestamps, new_open, new_high, new_low, new_close, new_volume