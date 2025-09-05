#!/usr/bin/env python3
"""
Manual ATR Calculation Pipeline
================================

Creates daily bars and calculates ATR manually (without VBT issues).
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import logging
import gc
import time

# Configuration
ATR_PERIOD = 30
ATR_MULTIPLIER = 0.1

# Paths  
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_CSV = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
DAILY_PARQUET = Path("../parquetData/ES-DIFF-daily-with-atr.parquet")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_daily_bars():
    """Create daily OHLCV bars from tick data."""
    
    logger.info("Creating daily bars from tick data...")
    
    parquet_file = pq.ParquetFile(TICK_FILE)
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Processing {total_rows:,} ticks...")
    
    daily_data = {}
    
    # Process row groups
    for rg_idx in range(parquet_file.num_row_groups):
        if rg_idx % 50 == 0:
            logger.info(f"  Row group {rg_idx}/{parquet_file.num_row_groups}")
        
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
        
        df['date'] = df['datetime'].dt.date
        
        # Aggregate to daily
        for date, group in df.groupby('date'):
            if date not in daily_data:
                daily_data[date] = {
                    'open': group.iloc[0]['Close'],
                    'high': group['Close'].max(),
                    'low': group['Close'].min(),
                    'close': group.iloc[-1]['Close'],
                    'volume': len(group)
                }
            else:
                # Update existing
                daily_data[date]['high'] = max(daily_data[date]['high'], group['Close'].max())
                daily_data[date]['low'] = min(daily_data[date]['low'], group['Close'].min())
                daily_data[date]['close'] = group.iloc[-1]['Close']
                daily_data[date]['volume'] += len(group)
        
        del df, table
        gc.collect()
    
    # Convert to DataFrame
    daily_df = pd.DataFrame.from_dict(daily_data, orient='index')
    daily_df.index.name = 'date'
    daily_df = daily_df.reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Created {len(daily_df)} daily bars")
    logger.info(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    
    return daily_df

def calculate_atr_manual(daily_df, period=30):
    """Calculate ATR manually using Wilder's method."""
    
    logger.info(f"\nCalculating ATR-{period} manually...")
    
    # Calculate True Range
    high = daily_df['high'].values
    low = daily_df['low'].values
    close = daily_df['close'].values
    
    # TR = max of:
    # 1. High - Low
    # 2. |High - Previous Close|
    # 3. |Low - Previous Close|
    
    tr = np.zeros(len(daily_df))
    
    # First day: just high - low
    tr[0] = high[0] - low[0]
    
    # Subsequent days
    for i in range(1, len(daily_df)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calculate ATR using Wilder's smoothing
    atr = np.zeros(len(daily_df))
    
    # First ATR is average of first 'period' TR values
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
        
        # Subsequent ATR values use Wilder's smoothing
        for i in range(period, len(daily_df)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    else:
        # Not enough data, use simple moving average
        for i in range(len(daily_df)):
            window_start = max(0, i - period + 1)
            atr[i] = np.mean(tr[window_start:i+1])
    
    return atr

def add_atr_to_daily(daily_df):
    """Add ATR to daily bars."""
    
    # Calculate ATR manually
    atr_values = calculate_atr_manual(daily_df, ATR_PERIOD)
    
    # Add to dataframe
    daily_df['atr'] = atr_values
    
    # Lag ATR by 1 day (use previous day's ATR for today)
    daily_df['atr_lagged'] = daily_df['atr'].shift(1)
    
    # For first day, use its own ATR
    if pd.isna(daily_df['atr_lagged'].iloc[0]):
        daily_df.loc[0, 'atr_lagged'] = daily_df['atr'].iloc[0]
    
    # Calculate range size
    daily_df['range_size'] = daily_df['atr_lagged'] * ATR_MULTIPLIER
    
    # Statistics
    logger.info(f"ATR Statistics:")
    logger.info(f"  Current ATR: {daily_df['atr'].min():.2f} to {daily_df['atr'].max():.2f}")
    logger.info(f"  Lagged ATR:  {daily_df['atr_lagged'].min():.2f} to {daily_df['atr_lagged'].max():.2f}")
    logger.info(f"  Range size:  {daily_df['range_size'].min():.2f} to {daily_df['range_size'].max():.2f}")
    
    return daily_df

def save_daily_data(daily_df):
    """Save daily data with ATR to files."""
    
    # Save to CSV
    daily_df.to_csv(DAILY_CSV, index=False)
    logger.info(f"\nSaved to CSV: {DAILY_CSV}")
    
    # Save to Parquet
    daily_df.to_parquet(DAILY_PARQUET, index=False)
    logger.info(f"Saved to Parquet: {DAILY_PARQUET}")
    
    # Print sample
    logger.info("\nSample daily data with ATR:")
    logger.info("Date, Open, High, Low, Close, Volume, ATR, ATR_Lagged, Range_Size")
    
    # Show first 5, middle 5, and last 5
    samples = []
    if len(daily_df) > 15:
        samples = list(range(5)) + list(range(len(daily_df)//2 - 2, len(daily_df)//2 + 3)) + list(range(-5, 0))
    else:
        samples = range(len(daily_df))
    
    for idx in samples:
        if 0 <= idx < len(daily_df) or idx < 0:
            row = daily_df.iloc[idx]
            logger.info(f"{row['date'].date()}, {row['open']:.2f}, {row['high']:.2f}, {row['low']:.2f}, "
                       f"{row['close']:.2f}, {row['volume']:.0f}, {row['atr']:.2f}, "
                       f"{row['atr_lagged']:.2f}, {row['range_size']:.2f}")

def main():
    """Main execution."""
    
    logger.info("=" * 80)
    logger.info("MANUAL ATR CALCULATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"ATR Period: {ATR_PERIOD}")
    logger.info(f"ATR Multiplier: {ATR_MULTIPLIER}")
    logger.info("Using manual ATR calculation (Wilder's method)")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Check if daily data exists
        if DAILY_PARQUET.exists():
            logger.info(f"Loading existing daily data from {DAILY_PARQUET}")
            daily_df = pd.read_parquet(DAILY_PARQUET)
            logger.info(f"Loaded {len(daily_df)} days with ATR")
            logger.info(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        else:
            # Step 1: Create daily bars
            daily_df = create_daily_bars()
            
            # Step 2: Add ATR manually
            daily_df = add_atr_to_daily(daily_df)
            
            # Step 3: Save results
            save_daily_data(daily_df)
        
        elapsed = (time.time() - start_time) / 60
        
        logger.info("\n" + "=" * 80)
        logger.info("DAILY ATR PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Processing time: {elapsed:.1f} minutes")
        logger.info(f"Created {len(daily_df)} daily bars with lagged ATR")
        logger.info("\nNext: Run create_range_bars_from_daily_atr.py to build range bars")
        logger.info("=" * 80)
        
        return daily_df
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()