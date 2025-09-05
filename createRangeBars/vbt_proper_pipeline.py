#!/usr/bin/env python3
"""
VectorBT Pro: Proper Pipeline for Tick to Daily with ATR
=========================================================

Uses VectorBT Pro's Data object and resampling capabilities properly:
1. Load tick data in blocks into VBT Data objects
2. Use VBT's resample to convert to daily
3. Calculate ATR on daily data
4. Export with lagged ATR
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging
import gc
import time

# Configuration
ATR_PERIOD = 30
ATR_MULTIPLIER = 0.1
BLOCK_SIZE = 50_000_000  # Process 50M ticks at a time

# Paths
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_CSV = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
DAILY_PARQUET = Path("../parquetData/ES-DIFF-daily-with-atr.parquet")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def process_ticks_to_daily_vbt():
    """Process tick data to daily using VBT properly."""
    
    logger.info("=" * 80)
    logger.info("VECTORBT PRO: PROPER TICK TO DAILY PIPELINE")
    logger.info("=" * 80)
    
    # First, let's read all tick data efficiently
    logger.info("Reading tick data...")
    parquet_file = pq.ParquetFile(TICK_FILE)
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Total ticks: {total_rows:,}")
    
    # Read data using row groups to avoid corruption issues
    logger.info("Loading tick data using row groups...")
    all_data = []
    
    for rg_idx in range(parquet_file.num_row_groups):
        if rg_idx % 50 == 0:
            logger.info(f"  Reading row group {rg_idx}/{parquet_file.num_row_groups}")
        
        table = parquet_file.read_row_group(rg_idx)
        df = table.to_pandas()
        all_data.append(df)
        
    logger.info("Combining all row groups...")
    tick_df = pd.concat(all_data, ignore_index=True)
    del all_data
    gc.collect()
    
    # Parse datetime
    logger.info("Parsing datetime...")
    tick_df['datetime'] = pd.to_datetime(
        tick_df['Date'].astype(str) + ' ' + tick_df['Time'].astype(str),
        format='%Y/%m/%d %H:%M:%S.%f',
        errors='coerce'
    )
    tick_df = tick_df[tick_df['datetime'].notna()]
    tick_df = tick_df.set_index('datetime')
    tick_df = tick_df.sort_index()
    
    logger.info(f"Loaded {len(tick_df):,} valid ticks")
    logger.info(f"Date range: {tick_df.index[0]} to {tick_df.index[-1]}")
    
    # Create VBT Data object
    logger.info("Creating VectorBT Data object...")
    tick_data = vbt.Data.from_data(
        data=dict(
            open=tick_df['Close'],
            high=tick_df['Close'], 
            low=tick_df['Close'],
            close=tick_df['Close'],
            volume=tick_df['Volume']
        ),
        tz_convert=None,  # Keep timezone as is
        missing_index='drop'
    )
    
    # Resample to daily using VBT
    logger.info("Resampling to daily frequency using VBT...")
    daily_data = tick_data.resample('1D')
    
    # Extract daily OHLCV
    daily_df = pd.DataFrame({
        'date': daily_data.index,
        'open': daily_data.open,
        'high': daily_data.high,
        'low': daily_data.low, 
        'close': daily_data.close,
        'volume': daily_data.volume
    })
    
    # Remove any days with NaN values (non-trading days)
    daily_df = daily_df.dropna()
    
    logger.info(f"Created {len(daily_df)} daily bars")
    logger.info(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    
    # Clean up tick data
    del tick_df, tick_data
    gc.collect()
    
    return daily_df

def calculate_atr_and_export(daily_df):
    """Calculate ATR and export with lag."""
    
    logger.info("\n" + "=" * 80)
    logger.info(f"CALCULATING ATR-{ATR_PERIOD}")
    logger.info("=" * 80)
    
    # Calculate ATR using VBT
    logger.info("Calculating ATR with VectorBT Pro...")
    atr_indicator = vbt.ATR.run(
        high=daily_df['high'].values,
        low=daily_df['low'].values,
        close=daily_df['close'].values,
        window=ATR_PERIOD,
        ewm=True  # Use Wilder's smoothing
    )
    
    # Add ATR to dataframe
    daily_df['atr'] = atr_indicator.atr.values
    
    # Create lagged ATR (previous day's ATR for today's bars)
    daily_df['atr_lagged'] = daily_df['atr'].shift(1)
    
    # For first day, use its own ATR
    if pd.isna(daily_df['atr_lagged'].iloc[0]):
        daily_df.loc[daily_df.index[0], 'atr_lagged'] = daily_df['atr'].iloc[0]
    
    # Calculate range size
    daily_df['range_size'] = daily_df['atr_lagged'] * ATR_MULTIPLIER
    
    # Statistics
    logger.info("\nATR Statistics:")
    logger.info(f"  Current ATR: {daily_df['atr'].min():.2f} to {daily_df['atr'].max():.2f} (mean: {daily_df['atr'].mean():.2f})")
    logger.info(f"  Lagged ATR:  {daily_df['atr_lagged'].min():.2f} to {daily_df['atr_lagged'].max():.2f} (mean: {daily_df['atr_lagged'].mean():.2f})") 
    logger.info(f"  Range size:  {daily_df['range_size'].min():.2f} to {daily_df['range_size'].max():.2f} (mean: {daily_df['range_size'].mean():.2f})")
    
    # Show how ATR evolves
    logger.info("\nATR Evolution Sample:")
    sample_indices = np.linspace(0, len(daily_df)-1, min(10, len(daily_df)), dtype=int)
    for idx in sample_indices:
        row = daily_df.iloc[idx]
        logger.info(f"  {row['date'].date()}: ATR={row['atr']:.2f}, Lagged={row['atr_lagged']:.2f}, Range={row['range_size']:.2f}")
    
    # Save to CSV
    daily_df.to_csv(DAILY_CSV, index=False)
    logger.info(f"\nSaved to CSV: {DAILY_CSV}")
    
    # Save to Parquet
    daily_df.to_parquet(DAILY_PARQUET, index=False)
    logger.info(f"Saved to Parquet: {DAILY_PARQUET}")
    
    # Show first and last 5 rows
    logger.info("\nFirst 5 days:")
    logger.info("Date, Open, High, Low, Close, Volume, ATR, ATR_Lagged, Range_Size")
    for _, row in daily_df.head(5).iterrows():
        logger.info(f"{row['date'].date()}, {row['open']:.2f}, {row['high']:.2f}, {row['low']:.2f}, "
                   f"{row['close']:.2f}, {row['volume']:.0f}, {row['atr']:.2f}, "
                   f"{row['atr_lagged']:.2f}, {row['range_size']:.2f}")
    
    logger.info("\nLast 5 days:")
    for _, row in daily_df.tail(5).iterrows():
        logger.info(f"{row['date'].date()}, {row['open']:.2f}, {row['high']:.2f}, {row['low']:.2f}, "
                   f"{row['close']:.2f}, {row['volume']:.0f}, {row['atr']:.2f}, "
                   f"{row['atr_lagged']:.2f}, {row['range_size']:.2f}")
    
    return daily_df

def main():
    """Main pipeline execution."""
    
    logger.info("=" * 100)
    logger.info("VECTORBT PRO: COMPLETE PIPELINE")
    logger.info("=" * 100)
    logger.info(f"Configuration:")
    logger.info(f"  ATR Period: {ATR_PERIOD}")
    logger.info(f"  ATR Multiplier: {ATR_MULTIPLIER}")
    logger.info("  Method: Load all ticks -> VBT Data -> Resample to Daily -> Calculate ATR")
    logger.info("=" * 100)
    
    start_time = time.time()
    
    try:
        # Check if daily data already exists
        if DAILY_PARQUET.exists():
            logger.info(f"Loading existing daily data from {DAILY_PARQUET}")
            daily_df = pd.read_parquet(DAILY_PARQUET)
            logger.info(f"Loaded {len(daily_df)} days with ATR")
            logger.info(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        else:
            # Step 1: Convert ticks to daily using VBT
            daily_df = process_ticks_to_daily_vbt()
            
            # Step 2: Calculate ATR and export
            daily_df = calculate_atr_and_export(daily_df)
        
        elapsed = (time.time() - start_time) / 60
        
        logger.info("\n" + "=" * 100)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 100)
        logger.info(f"Processing time: {elapsed:.1f} minutes")
        logger.info(f"Created {len(daily_df)} daily bars with lagged ATR")
        logger.info(f"Files created:")
        logger.info(f"  {DAILY_CSV}")
        logger.info(f"  {DAILY_PARQUET}")
        logger.info("\nDaily ATR values are ready for range bar creation")
        logger.info("Each day will use the previous day's ATR (no peeking)")
        logger.info("=" * 100)
        
        return daily_df
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()