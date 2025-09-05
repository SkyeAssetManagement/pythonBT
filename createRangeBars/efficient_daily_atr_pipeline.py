#!/usr/bin/env python3
"""
Efficient Daily ATR Pipeline
=============================

Efficient pipeline that:
1. Aggregates tick data to daily OHLCV in batches
2. Uses VectorBT Pro to calculate ATR on daily data
3. Lags ATR by 1 day (no peeking)
4. Exports daily data with ATR to CSV
5. Creates range bars using lagged daily ATR
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
import pyarrow.parquet as pq
from pathlib import Path
import time
import logging
import gc

# Configuration
ATR_PERIOD = 30
ATR_MULTIPLIER = 0.1

# Paths
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_FILE = Path("../parquetData/ES-DIFF-daily-with-atr.parquet")
DAILY_CSV = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
OUTPUT_DIR = Path("../parquetData/range")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_daily_from_ticks():
    """Create daily OHLCV from tick data efficiently."""
    
    logger.info("Creating daily OHLCV from tick data...")
    
    parquet_file = pq.ParquetFile(TICK_FILE)
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Processing {total_rows:,} ticks...")
    
    daily_data = {}
    
    # Process in row groups
    for rg_idx in range(parquet_file.num_row_groups):
        if rg_idx % 50 == 0:
            logger.info(f"  Processing row group {rg_idx}/{parquet_file.num_row_groups}")
        
        # Read row group
        table = parquet_file.read_row_group(rg_idx)
        df = table.to_pandas()
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format='%Y/%m/%d %H:%M:%S.%f',
            errors='coerce'
        )
        df = df[df['datetime'].notna()]
        
        if len(df) == 0:
            continue
        
        # Get date
        df['date'] = df['datetime'].dt.date
        
        # Group by date and aggregate
        for date, group in df.groupby('date'):
            if date not in daily_data:
                daily_data[date] = {
                    'open': group.iloc[0]['Close'],
                    'high': group['Close'].max(),
                    'low': group['Close'].min(),
                    'close': group.iloc[-1]['Close'],
                    'volume': len(group),
                    'first_time': group.iloc[0]['datetime'],
                    'last_time': group.iloc[-1]['datetime']
                }
            else:
                # Update existing day
                daily_data[date]['high'] = max(daily_data[date]['high'], group['Close'].max())
                daily_data[date]['low'] = min(daily_data[date]['low'], group['Close'].min())
                daily_data[date]['close'] = group.iloc[-1]['Close']
                daily_data[date]['volume'] += len(group)
                daily_data[date]['last_time'] = group.iloc[-1]['datetime']
        
        del df, table
        gc.collect()
    
    # Convert to DataFrame
    daily_df = pd.DataFrame.from_dict(daily_data, orient='index')
    daily_df.index.name = 'date'
    daily_df = daily_df.reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date')
    
    logger.info(f"Created {len(daily_df)} daily bars")
    logger.info(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    
    return daily_df

def calculate_atr_with_vbt(daily_df):
    """Calculate ATR using VectorBT Pro."""
    
    logger.info(f"Calculating ATR-{ATR_PERIOD} using VectorBT Pro...")
    
    # Calculate ATR using VBT
    atr_indicator = vbt.ATR.run(
        high=daily_df['high'].values,
        low=daily_df['low'].values,
        close=daily_df['close'].values,
        window=ATR_PERIOD,
        ewm=True  # Wilder's smoothing
    )
    
    # Add ATR to dataframe
    daily_df['atr'] = atr_indicator.atr.values
    
    # Lag ATR by 1 day (use previous day's ATR for today's bars)
    daily_df['atr_lagged'] = daily_df['atr'].shift(1)
    
    # Fill first day with first ATR value
    if pd.isna(daily_df['atr_lagged'].iloc[0]):
        daily_df.loc[0, 'atr_lagged'] = daily_df['atr'].iloc[0]
    
    # Calculate range size
    daily_df['range_size'] = daily_df['atr_lagged'] * ATR_MULTIPLIER
    
    logger.info(f"ATR Statistics:")
    logger.info(f"  Current ATR range: {daily_df['atr'].min():.2f} to {daily_df['atr'].max():.2f}")
    logger.info(f"  Lagged ATR range: {daily_df['atr_lagged'].min():.2f} to {daily_df['atr_lagged'].max():.2f}")
    logger.info(f"  Range size: {daily_df['range_size'].min():.2f} to {daily_df['range_size'].max():.2f}")
    
    return daily_df

def save_daily_data(daily_df):
    """Save daily data to parquet and CSV."""
    
    # Save to parquet
    daily_df.to_parquet(DAILY_FILE, index=False)
    logger.info(f"Saved daily data to: {DAILY_FILE}")
    
    # Save to CSV for verification
    daily_df.to_csv(DAILY_CSV, index=False)
    logger.info(f"Saved daily data to: {DAILY_CSV}")
    
    # Print sample
    logger.info("\nSample of daily data with ATR:")
    logger.info("Date, Open, High, Low, Close, Volume, ATR, ATR_Lagged, Range_Size")
    for idx, row in daily_df.head(10).iterrows():
        logger.info(f"{row['date'].date()}, {row['open']:.2f}, {row['high']:.2f}, {row['low']:.2f}, "
                   f"{row['close']:.2f}, {row['volume']:.0f}, {row['atr']:.2f}, "
                   f"{row['atr_lagged']:.2f}, {row['range_size']:.2f}")

def main():
    """Main pipeline execution."""
    
    logger.info("=" * 80)
    logger.info("EFFICIENT DAILY ATR PIPELINE")
    logger.info("=" * 80)
    logger.info(f"ATR Period: {ATR_PERIOD}")
    logger.info(f"ATR Multiplier: {ATR_MULTIPLIER}")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Check if daily data exists
        if DAILY_FILE.exists():
            logger.info(f"Loading existing daily data from {DAILY_FILE}")
            daily_df = pd.read_parquet(DAILY_FILE)
            logger.info(f"Loaded {len(daily_df)} days")
        else:
            # Step 1: Create daily OHLCV from ticks
            daily_df = create_daily_from_ticks()
            
            # Step 2: Calculate ATR with VBT
            daily_df = calculate_atr_with_vbt(daily_df)
            
            # Step 3: Save daily data
            save_daily_data(daily_df)
        
        elapsed = (time.time() - start_time) / 60
        
        logger.info("\n" + "=" * 80)
        logger.info("DAILY DATA WITH ATR COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Processing time: {elapsed:.1f} minutes")
        logger.info(f"Daily data with lagged ATR saved to:")
        logger.info(f"  Parquet: {DAILY_FILE}")
        logger.info(f"  CSV: {DAILY_CSV}")
        logger.info("\nNext step: Run range bar creation using the daily ATR values")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()