#!/usr/bin/env python3
"""
VectorBT Pro: Tick to Daily Pipeline
=====================================

Uses VectorBT Pro to efficiently:
1. Read tick data from parquet in blocks
2. Convert each block to daily bars using VBT's resample
3. Calculate ATR-30 on the daily bars
4. Lag ATR by 1 day (no peeking)
5. Export to CSV for verification
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging
import gc

# Configuration
ATR_PERIOD = 30
ATR_MULTIPLIER = 0.1
BLOCK_SIZE = 10_000_000  # Process 10M ticks at a time

# Paths
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_CSV = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
DAILY_PARQUET = Path("../parquetData/ES-DIFF-daily-with-atr.parquet")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def process_ticks_to_daily():
    """Process tick data to daily bars using VectorBT Pro in blocks."""
    
    logger.info("=" * 80)
    logger.info("VectorBT Pro: Tick to Daily Conversion with ATR")
    logger.info("=" * 80)
    
    # Open parquet file
    parquet_file = pq.ParquetFile(TICK_FILE)
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Total ticks: {total_rows:,}")
    logger.info(f"Processing in blocks of {BLOCK_SIZE:,}")
    
    # Process in blocks
    all_daily_bars = []
    rows_processed = 0
    
    # Calculate number of blocks
    num_blocks = (total_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_idx in range(num_blocks):
        start_row = block_idx * BLOCK_SIZE
        end_row = min((block_idx + 1) * BLOCK_SIZE, total_rows)
        
        logger.info(f"\nBlock {block_idx + 1}/{num_blocks}: Processing rows {start_row:,} to {end_row:,}")
        
        # Read block of ticks using pyarrow for efficient slicing
        parquet_file = pq.ParquetFile(TICK_FILE)
        
        # Calculate which row groups to read
        row_group_size = 1048576  # Standard row group size
        start_rg = start_row // row_group_size
        end_rg = min((end_row - 1) // row_group_size + 1, parquet_file.num_row_groups)
        
        # Read the row groups
        tables = []
        for rg_idx in range(start_rg, end_rg):
            tables.append(parquet_file.read_row_group(rg_idx))
        
        if not tables:
            continue
            
        combined_table = pa.concat_tables(tables)
        tick_block = combined_table.to_pandas()
        
        # Slice to exact range if needed
        if start_row > 0 or end_row < len(tick_block):
            start_offset = start_row - (start_rg * row_group_size)
            end_offset = start_offset + (end_row - start_row)
            tick_block = tick_block.iloc[start_offset:end_offset]
        
        # Parse datetime
        tick_block['datetime'] = pd.to_datetime(
            tick_block['Date'].astype(str) + ' ' + tick_block['Time'].astype(str),
            format='%Y/%m/%d %H:%M:%S.%f',
            errors='coerce'
        )
        tick_block = tick_block[tick_block['datetime'].notna()]
        
        if len(tick_block) == 0:
            continue
        
        # Set datetime as index for VBT
        tick_block = tick_block.set_index('datetime')
        tick_block = tick_block.sort_index()
        
        logger.info(f"  Block date range: {tick_block.index[0]} to {tick_block.index[-1]}")
        
        # Create VBT Data object from this block
        tick_data = vbt.Data.from_data(
            data=dict(
                open=tick_block['Close'],
                high=tick_block['Close'],
                low=tick_block['Close'],
                close=tick_block['Close'],
                volume=tick_block['Volume']
            ),
            tz_convert=None,  # Keep original timezone
            missing_index='drop',
            freq=None  # Irregular frequency
        )
        
        # Resample to daily bars using VBT
        daily_data = tick_data.resample('1D')
        
        # Convert to DataFrame
        daily_df = pd.DataFrame({
            'date': daily_data.index,
            'open': daily_data.open,
            'high': daily_data.high,
            'low': daily_data.low,
            'close': daily_data.close,
            'volume': daily_data.volume
        })
        
        # Remove any days with no data
        daily_df = daily_df.dropna()
        
        if len(daily_df) > 0:
            all_daily_bars.append(daily_df)
            logger.info(f"  Created {len(daily_df)} daily bars from this block")
        
        rows_processed += len(tick_block)
        logger.info(f"  Total processed: {rows_processed:,}/{total_rows:,} ({100*rows_processed/total_rows:.1f}%)")
        
        # Clean up
        del tick_block, tick_data, daily_data, daily_df
        gc.collect()
    
    # Combine all daily bars
    logger.info("\nCombining all daily bars...")
    combined_daily = pd.concat(all_daily_bars, ignore_index=True)
    
    # Group by date (in case same date appears in multiple blocks)
    combined_daily = combined_daily.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    combined_daily = combined_daily.sort_values('date')
    
    logger.info(f"Total daily bars: {len(combined_daily)}")
    logger.info(f"Date range: {combined_daily['date'].min()} to {combined_daily['date'].max()}")
    
    return combined_daily

def calculate_atr_and_lag(daily_df):
    """Calculate ATR using VectorBT Pro and lag by 1 day."""
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Calculating ATR-{ATR_PERIOD} with VectorBT Pro")
    logger.info("=" * 80)
    
    # Calculate ATR using VBT
    atr_indicator = vbt.ATR.run(
        high=daily_df['high'].values,
        low=daily_df['low'].values,
        close=daily_df['close'].values,
        window=ATR_PERIOD,
        ewm=True  # Use Wilder's smoothing
    )
    
    # Add ATR to dataframe
    daily_df['atr'] = atr_indicator.atr.values
    
    # Lag ATR by 1 day (use previous day's ATR for today's bars - no peeking!)
    daily_df['atr_lagged'] = daily_df['atr'].shift(1)
    
    # For the first day, use its own ATR (no previous day available)
    if pd.isna(daily_df['atr_lagged'].iloc[0]) and len(daily_df) > 0:
        daily_df.loc[daily_df.index[0], 'atr_lagged'] = daily_df['atr'].iloc[0]
    
    # Calculate range size (lagged ATR × multiplier)
    daily_df['range_size'] = daily_df['atr_lagged'] * ATR_MULTIPLIER
    
    # Statistics
    logger.info(f"ATR Statistics:")
    logger.info(f"  Current ATR: {daily_df['atr'].min():.2f} to {daily_df['atr'].max():.2f} (mean: {daily_df['atr'].mean():.2f})")
    logger.info(f"  Lagged ATR:  {daily_df['atr_lagged'].min():.2f} to {daily_df['atr_lagged'].max():.2f} (mean: {daily_df['atr_lagged'].mean():.2f})")
    logger.info(f"  Range size:  {daily_df['range_size'].min():.2f} to {daily_df['range_size'].max():.2f} (mean: {daily_df['range_size'].mean():.2f})")
    
    # Show how ATR changes over time
    logger.info("\nSample ATR progression:")
    sample_indices = [0, len(daily_df)//4, len(daily_df)//2, 3*len(daily_df)//4, -1]
    for idx in sample_indices:
        if 0 <= idx < len(daily_df) or idx == -1:
            row = daily_df.iloc[idx]
            logger.info(f"  {row['date']}: ATR={row['atr']:.2f}, Lagged={row['atr_lagged']:.2f}, Range={row['range_size']:.2f}")
    
    return daily_df

def save_results(daily_df):
    """Save daily data with ATR to CSV and Parquet."""
    
    logger.info("\n" + "=" * 80)
    logger.info("Saving Results")
    logger.info("=" * 80)
    
    # Save to CSV for easy verification
    daily_df.to_csv(DAILY_CSV, index=False)
    logger.info(f"Saved to CSV: {DAILY_CSV}")
    
    # Save to Parquet for efficient loading
    daily_df.to_parquet(DAILY_PARQUET, index=False)
    logger.info(f"Saved to Parquet: {DAILY_PARQUET}")
    
    # Print first 10 rows
    logger.info("\nFirst 10 days of data:")
    logger.info("Date, Open, High, Low, Close, Volume, ATR, ATR_Lagged, Range_Size")
    for _, row in daily_df.head(10).iterrows():
        logger.info(f"{row['date'].date()}, {row['open']:.2f}, {row['high']:.2f}, {row['low']:.2f}, "
                   f"{row['close']:.2f}, {row['volume']:.0f}, {row['atr']:.2f}, "
                   f"{row['atr_lagged']:.2f}, {row['range_size']:.2f}")

def main():
    """Main pipeline execution."""
    
    logger.info("=" * 100)
    logger.info("VECTORBT PRO PIPELINE: TICK TO DAILY WITH ATR")
    logger.info("=" * 100)
    logger.info(f"Configuration:")
    logger.info(f"  ATR Period: {ATR_PERIOD}")
    logger.info(f"  ATR Multiplier: {ATR_MULTIPLIER}")
    logger.info(f"  Block Size: {BLOCK_SIZE:,} ticks")
    logger.info("=" * 100)
    
    try:
        # Step 1: Convert ticks to daily bars using VBT
        daily_df = process_ticks_to_daily()
        
        # Step 2: Calculate ATR and lag it
        daily_df = calculate_atr_and_lag(daily_df)
        
        # Step 3: Save results
        save_results(daily_df)
        
        logger.info("\n" + "=" * 100)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 100)
        logger.info(f"Successfully created {len(daily_df)} daily bars with lagged ATR")
        logger.info(f"Files created:")
        logger.info(f"  {DAILY_CSV}")
        logger.info(f"  {DAILY_PARQUET}")
        logger.info("\nNext step: Use this daily ATR data to create range bars")
        logger.info("=" * 100)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()