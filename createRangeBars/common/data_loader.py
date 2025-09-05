"""
Ultra-Fast Data Loader for Tick Data Processing

This module provides optimized I/O operations for converting large CSV tick files
to Parquet format and loading data with maximum performance. Designed to handle
37GB+ files efficiently using chunked processing and memory optimization.
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Union, Dict, Any, Iterator, List
import os
from datetime import datetime, timedelta
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from performance import measure_time, measure_io, benchmark_func

class TickDataLoader:
    """
    High-performance tick data loader with optimized I/O operations.
    
    Features:
    - Chunked processing for memory efficiency
    - Optimized data types for reduced memory usage
    - Fast Parquet conversion with compression
    - Parallel processing capabilities
    - Memory-mapped file support
    - Progress tracking and performance monitoring
    """
    
    # Optimized data types for tick data columns
    TICK_DTYPES = {
        'Date': 'str',           # Will convert to datetime
        'Time': 'str',           # Will convert to datetime  
        'Open': 'float32',       # Price data - float32 sufficient for most tick data
        'High': 'float32',
        'Low': 'float32', 
        'Close': 'float32',
        'Volume': 'int32',       # Volume can be large but int32 usually sufficient
        'Up Ticks': 'int16',     # Tick counts are typically small
        'Down Ticks': 'int16',
        'Same Ticks': 'int16'
    }
    
    def __init__(self, chunk_size: int = 10_000_000, 
                 n_cores: Optional[int] = None,
                 memory_limit_gb: float = 8.0):
        """
        Initialize the tick data loader.
        
        Args:
            chunk_size: Number of rows to process per chunk
            n_cores: Number of CPU cores to use (None = auto-detect)
            memory_limit_gb: Memory limit in GB for processing
        """
        self.chunk_size = chunk_size
        self.n_cores = n_cores or max(1, mp.cpu_count() - 1)
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        
        print(f"TickDataLoader initialized:")
        print(f"   Chunk size: {chunk_size:,} rows")
        print(f"   CPU cores: {self.n_cores}")
        print(f"   Memory limit: {memory_limit_gb:.1f} GB")
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame data types for memory efficiency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized data types
        """
        with measure_time("optimize_dtypes", rows=len(df)):
            # Convert date/time columns to datetime
            if 'Date' in df.columns and 'Time' in df.columns:
                # Combine date and time into single datetime column
                datetime_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
                df['datetime'] = pd.to_datetime(datetime_str, format='%Y/%m/%d %H:%M:%S.%f')
                
                # Drop original date/time columns to save memory
                df = df.drop(['Date', 'Time'], axis=1)
            
            # Apply optimized numeric types
            for col in df.columns:
                if col == 'datetime':
                    continue
                    
                if col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df[col].astype('float32')
                elif col == 'Volume':
                    df[col] = df[col].astype('int32')
                elif col in ['Up Ticks', 'Down Ticks', 'Same Ticks']:
                    df[col] = df[col].astype('int16')
            
            return df
    
    def csv_to_parquet_chunked(self, 
                              csv_path: Union[str, Path],
                              parquet_path: Optional[Union[str, Path]] = None,
                              compression: str = 'snappy') -> str:
        """
        Convert large CSV file to Parquet using chunked processing.
        
        Args:
            csv_path: Path to input CSV file
            parquet_path: Path for output Parquet file (auto-generated if None)
            compression: Compression algorithm ('snappy', 'lz4', 'gzip', 'brotli')
            
        Returns:
            Path to created Parquet file
        """
        csv_path = Path(csv_path)
        if parquet_path is None:
            parquet_path = csv_path.with_suffix('.parquet')
        else:
            parquet_path = Path(parquet_path)
        
        print(f"Converting CSV to Parquet: {csv_path.name}")
        print(f"Output: {parquet_path}")
        print(f"🗜️  Compression: {compression}")
        
        with measure_io("csv_to_parquet_conversion", str(csv_path), "read"):
            # Get file size for progress tracking
            file_size_mb = csv_path.stat().st_size / (1024 * 1024)
            print(f"Input file size: {file_size_mb:,.1f} MB")
            
            # Initialize Parquet writer
            parquet_writer = None
            schema = None
            total_rows = 0
            chunk_num = 0
            
            try:
                # Process file in chunks
                for chunk_df in pd.read_csv(csv_path, 
                                          dtype=self.TICK_DTYPES,
                                          chunksize=self.chunk_size):
                    
                    chunk_num += 1
                    current_rows = len(chunk_df)
                    total_rows += current_rows
                    
                    print(f"Processing chunk {chunk_num}: {current_rows:,} rows "
                          f"(Total: {total_rows:,})")
                    
                    # Optimize data types
                    chunk_df = self.optimize_dtypes(chunk_df)
                    
                    # Convert to PyArrow table for efficient storage
                    table = pa.Table.from_pandas(chunk_df)
                    
                    # Initialize schema on first chunk
                    if schema is None:
                        schema = table.schema
                        parquet_writer = pq.ParquetWriter(
                            parquet_path, 
                            schema,
                            compression=compression,
                            use_dictionary=True,  # Enable dictionary encoding
                            row_group_size=self.chunk_size,
                            data_page_size=2*1024*1024  # 2MB pages for better compression
                        )
                    
                    # Write chunk to parquet
                    parquet_writer.write_table(table)
                    
                    # Force garbage collection to manage memory
                    del chunk_df, table
                    gc.collect()
                    
                    # Memory usage check
                    import psutil
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 85:
                        print(f"⚠️  High memory usage: {memory_percent:.1f}%")
                        gc.collect()
            
            finally:
                if parquet_writer:
                    parquet_writer.close()
            
            # Get output file stats
            output_size_mb = parquet_path.stat().st_size / (1024 * 1024)
            compression_ratio = output_size_mb / file_size_mb
            
            print(f"Conversion complete!")
            print(f"Output size: {output_size_mb:,.1f} MB")
            print(f"Compression ratio: {compression_ratio:.3f}")
            print(f"Total rows processed: {total_rows:,}")
        
        return str(parquet_path)
    
    def load_parquet_fast(self, 
                         parquet_path: Union[str, Path],
                         columns: Optional[List[str]] = None,
                         date_range: Optional[tuple] = None,
                         use_threads: bool = True) -> pd.DataFrame:
        """
        Load Parquet file with maximum speed optimizations.
        
        Args:
            parquet_path: Path to Parquet file
            columns: Specific columns to load (None = all)
            date_range: Tuple of (start_date, end_date) for filtering
            use_threads: Enable multi-threaded reading
            
        Returns:
            DataFrame with loaded data
        """
        parquet_path = Path(parquet_path)
        
        with measure_io("load_parquet", str(parquet_path), "read"):
            print(f"Loading Parquet file: {parquet_path.name}")
            
            # Configure PyArrow for maximum speed
            filters = None
            if date_range:
                start_date, end_date = date_range
                filters = [('datetime', '>=', start_date), ('datetime', '<=', end_date)]
                print(f"📅 Date filter: {start_date} to {end_date}")
            
            # Load with optimizations
            table = pq.read_table(
                parquet_path,
                columns=columns,
                filters=filters,
                use_threads=use_threads,
                pre_buffer=True,  # Pre-buffer data for speed
                memory_map=True   # Memory-map for large files
            )
            
            # Convert to pandas with zero-copy where possible
            df = table.to_pandas(
                use_threads=use_threads,
                split_blocks=True,     # Split into separate blocks for better performance
                self_destruct=True     # Destroy Arrow table to save memory
            )
            
            print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            return df
    
    def get_parquet_info(self, parquet_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive information about a Parquet file.
        
        Args:
            parquet_path: Path to Parquet file
            
        Returns:
            Dictionary with file information
        """
        parquet_path = Path(parquet_path)
        
        with measure_time("get_parquet_info"):
            # Read metadata without loading data
            parquet_file = pq.ParquetFile(parquet_path)
            
            info = {
                'file_path': str(parquet_path),
                'file_size_mb': parquet_path.stat().st_size / (1024 * 1024),
                'num_rows': parquet_file.metadata.num_rows,
                'num_columns': parquet_file.metadata.num_columns,
                'num_row_groups': parquet_file.metadata.num_row_groups,
                'compression': parquet_file.metadata.row_group(0).column(0).compression,
                'schema': parquet_file.schema_arrow,
                'columns': parquet_file.schema_arrow.names,
                'created_by': parquet_file.metadata.created_by
            }
            
            # Get date range if datetime column exists
            if 'datetime' in info['columns']:
                # Read just the datetime column from first and last row groups
                first_rg = parquet_file.read_row_group(0, columns=['datetime'])
                last_rg = parquet_file.read_row_group(
                    parquet_file.metadata.num_row_groups - 1, 
                    columns=['datetime']
                )
                
                first_df = first_rg.to_pandas()
                last_df = last_rg.to_pandas()
                
                info['date_range'] = {
                    'start': first_df['datetime'].min(),
                    'end': last_df['datetime'].max()
                }
            
            return info
    
    def validate_tick_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate tick data quality and provide statistics.
        
        Args:
            df: DataFrame with tick data
            
        Returns:
            Dictionary with validation results
        """
        with measure_time("validate_tick_data", rows=len(df)):
            print("Validating tick data quality...")
            
            validation = {
                'total_rows': len(df),
                'date_range': {
                    'start': df['datetime'].min(),
                    'end': df['datetime'].max()
                },
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'price_statistics': {},
                'volume_statistics': {},
                'data_quality_score': 0.0
            }
            
            # Price validation
            price_cols = ['Open', 'High', 'Low', 'Close']
            if all(col in df.columns for col in price_cols):
                validation['price_statistics'] = {
                    'min_price': float(df[price_cols].min().min()),
                    'max_price': float(df[price_cols].max().max()),
                    'zero_prices': (df[price_cols] == 0).sum().sum(),
                    'negative_prices': (df[price_cols] < 0).sum().sum(),
                    'ohlc_inconsistencies': ((df['High'] < df['Low']) | 
                                           (df['High'] < df['Open']) |
                                           (df['High'] < df['Close']) |
                                           (df['Low'] > df['Open']) |
                                           (df['Low'] > df['Close'])).sum()
                }
            
            # Volume validation
            if 'Volume' in df.columns:
                validation['volume_statistics'] = {
                    'total_volume': int(df['Volume'].sum()),
                    'zero_volume_ticks': (df['Volume'] == 0).sum(),
                    'negative_volume': (df['Volume'] < 0).sum(),
                    'avg_volume_per_tick': float(df['Volume'].mean())
                }
            
            # Calculate data quality score (0-100)
            quality_score = 100.0
            
            # Deduct for missing values
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            quality_score -= missing_ratio * 50
            
            # Deduct for duplicates
            duplicate_ratio = validation['duplicate_rows'] / len(df)
            quality_score -= duplicate_ratio * 30
            
            # Deduct for price inconsistencies
            if validation['price_statistics']:
                inconsistency_ratio = validation['price_statistics']['ohlc_inconsistencies'] / len(df)
                quality_score -= inconsistency_ratio * 20
            
            validation['data_quality_score'] = max(0, quality_score)
            
            print(f"Data Quality Score: {validation['data_quality_score']:.1f}/100")
            
            return validation

# Convenience functions for common operations
def convert_csv_to_parquet(csv_path: str, 
                          chunk_size: int = 10_000_000,
                          compression: str = 'snappy') -> str:
    """Quick conversion function"""
    loader = TickDataLoader(chunk_size=chunk_size)
    return loader.csv_to_parquet_chunked(csv_path, compression=compression)

def load_tick_data(parquet_path: str, 
                   date_range: Optional[tuple] = None,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Quick loading function"""
    loader = TickDataLoader()
    return loader.load_parquet_fast(parquet_path, columns=columns, date_range=date_range)

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    loader = TickDataLoader(chunk_size=1_000_000)
    
    # Example: Convert CSV to Parquet
    csv_file = r"C:\Users\skyeAM\SkyeAM Dropbox\SAMresearch\ABtoPython\dataRaw\tick\ES-DIFF-Tick1-21toT.tick"
    
    if os.path.exists(csv_file):
        print("🧪 Testing CSV to Parquet conversion...")
        parquet_file = loader.csv_to_parquet_chunked(csv_file)
        
        print("🧪 Testing Parquet info...")
        info = loader.get_parquet_info(parquet_file)
        for key, value in info.items():
            if key != 'schema':  # Schema is verbose
                print(f"   {key}: {value}")
        
        print("🧪 Testing fast load (first 1M rows)...")
        df_sample = loader.load_parquet_fast(parquet_file, 
                                           columns=['datetime', 'Close', 'Volume'])
        print(f"Sample loaded: {len(df_sample):,} rows")
        
        from performance import print_summary
        print_summary()
    else:
        print(f"Test file not found: {csv_file}")