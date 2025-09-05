"""
Parquet Converter - Automatically converts CSV data to Parquet format
for lightning-fast backtesting performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ParquetConverter:
    """Convert CSV raw data to optimized Parquet format."""
    
    def __init__(self, data_raw_root: str = None, parquet_root: str = None):
        """Initialize converter with data paths."""
        if data_raw_root is None:
            self.data_raw_root = Path(__file__).parent.parent.parent.parent / "dataRaw"
        else:
            self.data_raw_root = Path(data_raw_root)
        
        if parquet_root is None:
            # Mirror the dataRaw structure in parquetData
            self.parquet_root = Path(__file__).parent.parent.parent.parent / "parquetData"
        else:
            self.parquet_root = Path(parquet_root)
        
        # Ensure parquet root exists
        self.parquet_root.mkdir(exist_ok=True)
        
        logger.info(f"Parquet Converter initialized:")
        logger.info(f"  CSV Raw Root: {self.data_raw_root}")
        logger.info(f"  Parquet Root: {self.parquet_root}")
    
    def get_parquet_path(self, symbol: str, frequency: str = "1m", adjustment: str = "diffAdjusted") -> Path:
        """Get the parquet file path for a given symbol/frequency/adjustment."""
        if adjustment == "Current":
            filename = f"{symbol}-NONE-{frequency}-EST-NoPad.parquet"
        else:
            filename = f"{symbol}-DIFF-{frequency}-EST-NoPad.parquet"
        
        return self.parquet_root / frequency / symbol / adjustment / filename
    
    def get_csv_path(self, symbol: str, frequency: str = "1m", adjustment: str = "diffAdjusted") -> Path:
        """Get the CSV file path for a given symbol/frequency/adjustment."""
        if adjustment == "Current":
            filename = f"{symbol}-NONE-{frequency}-EST-NoPad.csv"
        else:
            filename = f"{symbol}-DIFF-{frequency}-EST-NoPad.csv"
        
        return self.data_raw_root / frequency / symbol / adjustment / filename
    
    def parquet_exists(self, symbol: str, frequency: str = "1m", adjustment: str = "diffAdjusted") -> bool:
        """Check if parquet file exists and is newer than CSV."""
        parquet_path = self.get_parquet_path(symbol, frequency, adjustment)
        csv_path = self.get_csv_path(symbol, frequency, adjustment)
        
        if not parquet_path.exists():
            return False
        
        if not csv_path.exists():
            # CSV doesn't exist, so parquet is valid
            return True
        
        # Check if parquet is newer than CSV
        parquet_mtime = parquet_path.stat().st_mtime
        csv_mtime = csv_path.stat().st_mtime
        
        return parquet_mtime >= csv_mtime
    
    def convert_csv_to_parquet(self, symbol: str, frequency: str = "1m", adjustment: str = "diffAdjusted") -> bool:
        """Convert CSV file to optimized Parquet format."""
        csv_path = self.get_csv_path(symbol, frequency, adjustment)
        parquet_path = self.get_parquet_path(symbol, frequency, adjustment)
        
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            return False
        
        try:
            logger.info(f"Converting {symbol} {frequency} {adjustment} CSV to Parquet...")
            
            # Ensure parquet directory exists
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load CSV data
            df = pd.read_csv(csv_path)
            
            # Handle different column formats
            if 'Date' in df.columns and 'Time' in df.columns:
                # Combine Date and Time columns
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                # Drop original columns to save space
                df = df.drop(['Date', 'Time'], axis=1)
            elif 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
            else:
                logger.error(f"No DateTime or Date/Time columns found in {csv_path}")
                return False
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in {csv_path}: {missing_cols}")
                return False
            
            # Sort by datetime to ensure proper order
            df = df.sort_values('DateTime').reset_index(drop=True)
            
            # Optimize data types for storage and speed
            df['Open'] = df['Open'].astype(np.float32)
            df['High'] = df['High'].astype(np.float32)
            df['Low'] = df['Low'].astype(np.float32)
            df['Close'] = df['Close'].astype(np.float32)
            df['Volume'] = df['Volume'].astype(np.int32)
            
            # Add additional useful columns
            df['datetime_int'] = df['DateTime'].astype('datetime64[ns]').values.view('int64')
            
            # Keep only essential columns in optimized order
            columns_to_keep = ['DateTime', 'datetime_int', 'Open', 'High', 'Low', 'Close', 'Volume']
            # Add any extra columns that exist
            extra_cols = [col for col in df.columns if col not in columns_to_keep and col not in ['Date', 'Time']]
            final_columns = columns_to_keep + extra_cols
            
            df_final = df[final_columns].copy()
            
            # Write to parquet with optimal compression
            df_final.to_parquet(
                parquet_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            logger.info(f"Successfully converted {len(df_final)} bars to parquet: {parquet_path}")
            logger.info(f"Date range: {df_final['DateTime'].min()} to {df_final['DateTime'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting {symbol} to parquet: {e}")
            if parquet_path.exists():
                parquet_path.unlink()  # Remove partial file
            return False
    
    def load_parquet_data(self, symbol: str, frequency: str = "1m", 
                         adjustment: str = "diffAdjusted") -> Optional[Dict[str, np.ndarray]]:
        """Load data from parquet file as numpy arrays."""
        parquet_path = self.get_parquet_path(symbol, frequency, adjustment)
        
        if not parquet_path.exists():
            return None
        
        try:
            logger.info(f"Loading {symbol} {frequency} data from parquet: {parquet_path}")
            
            # Load parquet file
            df = pd.read_parquet(parquet_path)
            
            # Convert to numpy arrays for vectorBT compatibility
            data_arrays = {
                'datetime': df['datetime_int'].values if 'datetime_int' in df.columns else df['DateTime'].astype('datetime64[ns]').values.view('int64'),
                'datetime_ns': df['DateTime'].values,
                'open': df['Open'].astype(np.float64).values,
                'high': df['High'].astype(np.float64).values,
                'low': df['Low'].astype(np.float64).values,
                'close': df['Close'].astype(np.float64).values,
                'volume': df['Volume'].astype(np.float64).values
            }
            
            logger.info(f"Successfully loaded {len(data_arrays['close'])} bars from parquet")
            
            return data_arrays
            
        except Exception as e:
            logger.error(f"Error loading parquet data for {symbol}: {e}")
            return None
    
    def load_or_convert(self, symbol: str, frequency: str = "1m", 
                       adjustment: str = "diffAdjusted") -> Optional[Dict[str, np.ndarray]]:
        """Load parquet data, converting from CSV if necessary."""
        
        # Check if parquet exists and is up to date
        if self.parquet_exists(symbol, frequency, adjustment):
            logger.info(f"Using existing parquet for {symbol} {frequency} {adjustment}")
            return self.load_parquet_data(symbol, frequency, adjustment)
        
        # Need to convert CSV to parquet first
        logger.info(f"Parquet not found or outdated for {symbol} {frequency} {adjustment}")
        
        if self.convert_csv_to_parquet(symbol, frequency, adjustment):
            return self.load_parquet_data(symbol, frequency, adjustment)
        else:
            logger.warning(f"Failed to convert {symbol} to parquet, falling back to CSV")
            return None
    
    def filter_data_by_date(self, data: Dict[str, np.ndarray], 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Filter data arrays by date range."""
        if not start_date and not end_date:
            return data
        
        # Create temporary DataFrame for filtering
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
        filtered_data = {
            'datetime': df_temp['DateTime'].astype('datetime64[ns]').values.view('int64'),
            'datetime_ns': df_temp['DateTime'].values,
            'open': df_temp['open'].values,
            'high': df_temp['high'].values,
            'low': df_temp['low'].values,
            'close': df_temp['close'].values,
            'volume': df_temp['volume'].values
        }
        
        logger.info(f"Filtered to {len(filtered_data['close'])} bars")
        return filtered_data