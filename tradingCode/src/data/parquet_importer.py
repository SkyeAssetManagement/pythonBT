"""
Lightning-fast CSV to Parquet import system with incremental loading.
Uses array processing for maximum performance.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import hashlib
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ParquetImporter:
    """Fast CSV to Parquet importer with incremental loading."""
    
    def __init__(self, data_root: str, parquet_root: str):
        self.data_root = Path(data_root)
        self.parquet_root = Path(parquet_root)
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        
        # Metadata tracking
        self.metadata_file = self.parquet_root / "import_metadata.parquet"
        self.load_metadata()
    
    def load_metadata(self):
        """Load import metadata to track file changes."""
        try:
            self.metadata = pl.read_parquet(self.metadata_file).to_pandas()
        except FileNotFoundError:
            self.metadata = pd.DataFrame(columns=[
                'symbol', 'source_file', 'file_hash', 'last_modified', 
                'rows_imported', 'import_timestamp'
            ])
    
    def save_metadata(self):
        """Save metadata using Polars for speed."""
        pl.from_pandas(self.metadata).write_parquet(self.metadata_file)
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for change detection."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def needs_import(self, symbol: str, csv_file: Path) -> bool:
        """Check if file needs to be imported/updated."""
        if not csv_file.exists():
            return False
            
        current_hash = self.get_file_hash(csv_file)
        current_modified = csv_file.stat().st_mtime
        
        # Check if we have this file in metadata
        existing = self.metadata[
            (self.metadata['symbol'] == symbol) & 
            (self.metadata['source_file'] == str(csv_file))
        ]
        
        if existing.empty:
            return True
            
        last_hash = existing.iloc[0]['file_hash']
        return current_hash != last_hash
    
    def import_csv_to_parquet(self, symbol: str, csv_file: Path) -> bool:
        """
        Import CSV to Parquet using vectorized operations.
        
        Args:
            symbol: Symbol name
            csv_file: Path to CSV file
            
        Returns:
            True if import successful
        """
        try:
            start_time = time.time()
            
            # Use Polars for lightning-fast CSV reading
            df = pl.read_csv(
                csv_file,
                try_parse_dates=True,
                infer_schema_length=10000
            )
            
            # Convert datetime if needed
            if 'DateTime' in df.columns:
                df = df.with_columns([
                    pl.col('DateTime').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias('datetime')
                ]).drop('DateTime')
            elif 'Date' in df.columns and 'Time' in df.columns:
                df = df.with_columns([
                    pl.concat_str([pl.col('Date'), pl.col('Time')], separator=' ')
                    .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
                    .alias('datetime')
                ]).drop(['Date', 'Time'])
            
            # Standardize column names
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename({old_col: new_col})
            
            # Create symbol directory
            symbol_dir = self.parquet_root / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Save to parquet
            parquet_file = symbol_dir / f"{symbol}_5min_data.parquet"
            df.write_parquet(parquet_file)
            
            # Update metadata
            file_hash = self.get_file_hash(csv_file)
            file_modified = csv_file.stat().st_mtime
            
            # Remove existing entry
            self.metadata = self.metadata[
                ~((self.metadata['symbol'] == symbol) & 
                  (self.metadata['source_file'] == str(csv_file)))
            ]
            
            # Add new entry
            new_entry = pd.DataFrame({
                'symbol': [symbol],
                'source_file': [str(csv_file)],
                'file_hash': [file_hash],
                'last_modified': [file_modified],
                'rows_imported': [len(df)],
                'import_timestamp': [datetime.now()]
            })
            
            self.metadata = pd.concat([self.metadata, new_entry], ignore_index=True)
            self.save_metadata()
            
            import_time = time.time() - start_time
            logger.info(f"Imported {symbol}: {len(df)} rows in {import_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to import {symbol}: {e}")
            return False
    
    def import_all_symbols(self) -> Dict[str, bool]:
        """
        Import all symbols from dataRaw directory.
        
        Returns:
            Dictionary of symbol: success status
        """
        results = {}
        
        # Find all CSV files in dataRaw
        for symbol_dir in self.data_root.iterdir():
            if not symbol_dir.is_dir():
                continue
                
            symbol = symbol_dir.name
            
            # Look for CSV files in Current subdirectory
            current_dir = symbol_dir / "Current"
            if current_dir.exists():
                for csv_file in current_dir.glob("*.csv"):
                    if self.needs_import(symbol, csv_file):
                        logger.info(f"Importing {symbol} from {csv_file}")
                        results[symbol] = self.import_csv_to_parquet(symbol, csv_file)
                    else:
                        logger.info(f"Skipping {symbol} - no changes detected")
                        results[symbol] = True
        
        return results
    
    def get_import_summary(self) -> pd.DataFrame:
        """Get summary of all imports."""
        return self.metadata.copy()


def main():
    """Main import function."""
    # Paths relative to the main ABtoPython directory
    data_root = Path(__file__).parent.parent.parent.parent / "dataRaw"
    parquet_root = Path(__file__).parent.parent.parent.parent / "parquet_data"
    
    importer = ParquetImporter(str(data_root), str(parquet_root))
    
    print("Starting CSV to Parquet import...")
    results = importer.import_all_symbols()
    
    print(f"\nImport Results:")
    for symbol, success in results.items():
        status = "[OK]" if success else "[X]"
        print(f"{status} {symbol}")
    
    print(f"\nImport Summary:")
    summary = importer.get_import_summary()
    print(summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()