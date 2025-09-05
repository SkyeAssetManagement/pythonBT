#!/usr/bin/env python3
"""
VectorBT Pro Pipeline: Tick to Daily with ATR Calculation
==========================================================

This pipeline:
1. Uses VectorBT Pro to convert tick data to daily OHLCV
2. Calculates ATR-30 using VBT's built-in indicators
3. Lags ATR by 1 day to avoid peeking
4. Exports daily data with ATR to CSV for verification
5. Creates range bars using the lagged ATR values

Each day's range bars use the previous day's ATR value.
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parameters
ATR_PERIOD = 30  # ATR period
ATR_MULTIPLIER = 0.1  # Multiplier for ATR to get range size

# Paths
TICK_FILE = Path("../parquetData/tick/ES/diffAdjusted/ES-DIFF-tick-EST.parquet")
DAILY_FILE = Path("../parquetData/ES-DIFF-daily-with-atr.parquet")
DAILY_CSV = Path("../parquetData/ES-DIFF-daily-with-atr.csv")
OUTPUT_DIR = Path("../parquetData/range")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: CREATE DAILY DATA WITH ATR USING VECTORBT PRO
# ============================================================================

def create_daily_data_with_atr():
    """
    Create daily OHLCV data with ATR using VectorBT Pro.
    
    Returns:
        pd.DataFrame: Daily data with OHLCV, ATR, and lagged ATR
    """
    
    logger.info("=" * 80)
    logger.info("STEP 1: CREATE DAILY DATA WITH ATR USING VECTORBT PRO")
    logger.info("=" * 80)
    
    # Check if daily data already exists
    if DAILY_FILE.exists():
        logger.info(f"Loading existing daily data from {DAILY_FILE}")
        daily_df = pd.read_parquet(DAILY_FILE)
        logger.info(f"Loaded {len(daily_df)} days of data")
        logger.info(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        logger.info(f"ATR range: {daily_df['atr_lagged'].min():.2f} to {daily_df['atr_lagged'].max():.2f}")
        return daily_df
    
    logger.info("Creating daily data from tick file...")
    logger.info(f"Reading: {TICK_FILE}")
    
    # Read tick data
    tick_df = pd.read_parquet(TICK_FILE)
    logger.info(f"Loaded {len(tick_df):,} ticks")
    
    # Parse datetime
    tick_df['datetime'] = pd.to_datetime(
        tick_df['Date'].astype(str) + ' ' + tick_df['Time'].astype(str),
        format='%Y/%m/%d %H:%M:%S.%f',
        errors='coerce'
    )
    tick_df = tick_df[tick_df['datetime'].notna()]
    
    # Create VBT Data object from tick data
    logger.info("Creating VectorBT Data object...")
    tick_data = vbt.Data.from_data(
        data=dict(
            open=tick_df['Close'].values,
            high=tick_df['Close'].values,
            low=tick_df['Close'].values,
            close=tick_df['Close'].values,
            volume=tick_df['Volume'].values
        ),
        tz_convert='UTC',
        missing_index='drop',
        freq='1min'  # Tick frequency
    )
    
    # Resample to daily using VBT
    logger.info("Resampling to daily frequency...")
    daily_data = tick_data.resample('1D')
    
    # Get the resampled data
    daily_df = pd.DataFrame({
        'date': daily_data.index,
        'open': daily_data.open,
        'high': daily_data.high,
        'low': daily_data.low,
        'close': daily_data.close,
        'volume': daily_data.volume
    })
    
    # Calculate ATR using VBT
    logger.info(f"Calculating ATR-{ATR_PERIOD}...")
    atr_indicator = vbt.ATR.run(
        high=daily_df['high'].values,
        low=daily_df['low'].values,
        close=daily_df['close'].values,
        window=ATR_PERIOD,
        ewm=True  # Use exponential weighted (Wilder's method)
    )
    
    # Get ATR values
    daily_df['atr'] = atr_indicator.atr
    
    # Create lagged ATR (use previous day's ATR for today's bars)
    daily_df['atr_lagged'] = daily_df['atr'].shift(1)
    
    # Fill first day with first ATR value
    if pd.isna(daily_df['atr_lagged'].iloc[0]) and not pd.isna(daily_df['atr'].iloc[0]):
        daily_df.loc[0, 'atr_lagged'] = daily_df['atr'].iloc[0]
    
    # Calculate range size for each day
    daily_df['range_size'] = daily_df['atr_lagged'] * ATR_MULTIPLIER
    
    logger.info(f"Created daily data: {len(daily_df)} days")
    logger.info(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    logger.info(f"ATR range: {daily_df['atr'].min():.2f} to {daily_df['atr'].max():.2f}")
    logger.info(f"Lagged ATR range: {daily_df['atr_lagged'].min():.2f} to {daily_df['atr_lagged'].max():.2f}")
    logger.info(f"Range size: {daily_df['range_size'].min():.2f} to {daily_df['range_size'].max():.2f}")
    
    # Save to parquet
    daily_df.to_parquet(DAILY_FILE, index=False)
    logger.info(f"Saved to: {DAILY_FILE}")
    
    # Also save to CSV for verification
    daily_df.to_csv(DAILY_CSV, index=False)
    logger.info(f"Saved to: {DAILY_CSV}")
    
    # Show sample
    logger.info("\nSample of daily data with ATR:")
    sample = daily_df[['date', 'open', 'high', 'low', 'close', 'volume', 'atr', 'atr_lagged', 'range_size']].head(10)
    for idx, row in sample.iterrows():
        logger.info(f"  {row['date']}: OHLC=[{row['open']:.2f}, {row['high']:.2f}, {row['low']:.2f}, {row['close']:.2f}] "
                   f"ATR={row['atr']:.2f} Lagged={row['atr_lagged']:.2f} Range={row['range_size']:.2f}")
    
    return daily_df

# ============================================================================
# STEP 2: CREATE RANGE BARS USING DAILY ATR VALUES  
# ============================================================================

def create_range_bars_with_daily_atr(daily_df):
    """
    Create range bars using daily ATR values.
    Each day uses the previous day's ATR (lagged).
    
    Args:
        daily_df: DataFrame with daily data and lagged ATR
    """
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CREATE RANGE BARS WITH DAILY ATR")
    logger.info("=" * 80)
    
    # Create output directory
    output_subdir = OUTPUT_DIR / f"ATR{ATR_PERIOD}x{ATR_MULTIPLIER}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Output paths
    diff_path = output_subdir / f"ES-DIFF-range-ATR{ATR_PERIOD}x{ATR_MULTIPLIER}-amibroker.parquet"
    none_path = output_subdir / f"ES-NONE-range-ATR{ATR_PERIOD}x{ATR_MULTIPLIER}-amibroker.parquet"
    
    logger.info(f"Output: {diff_path.name}")
    
    # Read tick data
    logger.info("Loading tick data...")
    tick_df = pd.read_parquet(TICK_FILE)
    
    # Parse datetime
    tick_df['datetime'] = pd.to_datetime(
        tick_df['Date'].astype(str) + ' ' + tick_df['Time'].astype(str),
        format='%Y/%m/%d %H:%M:%S.%f',
        errors='coerce'
    )
    tick_df = tick_df[tick_df['datetime'].notna()]
    tick_df['date'] = tick_df['datetime'].dt.date
    
    logger.info(f"Loaded {len(tick_df):,} ticks")
    
    # Process each day's ticks with that day's range size
    all_bars = []
    
    for idx, daily_row in daily_df.iterrows():
        current_date = pd.to_datetime(daily_row['date']).date()
        range_size = daily_row['range_size']
        atr_value = daily_row['atr_lagged']
        
        # Skip if no range size
        if pd.isna(range_size) or range_size <= 0:
            continue
        
        # Get ticks for this day
        day_ticks = tick_df[tick_df['date'] == current_date].copy()
        
        if len(day_ticks) == 0:
            continue
        
        # Create range bars for this day
        bars = create_range_bars_for_day(day_ticks, range_size, atr_value)
        
        if len(bars) > 0:
            all_bars.append(bars)
            
        if idx % 100 == 0:
            logger.info(f"  Processed {idx}/{len(daily_df)} days, created {sum(len(b) for b in all_bars)} bars so far")
    
    # Combine all bars
    if all_bars:
        final_df = pd.concat(all_bars, ignore_index=True)
        
        # Calculate gaps
        final_df['gap'] = 0.0
        if len(final_df) > 1:
            gaps = final_df['open'].iloc[1:].values - final_df['close'].iloc[:-1].values
            final_df.loc[1:, 'gap'] = gaps
        
        # Save DIFF bars
        final_df.to_parquet(diff_path, compression='snappy')
        logger.info(f"Saved {len(final_df):,} DIFF bars to {diff_path}")
        
        # Create and save NONE bars
        none_df = final_df.copy()
        none_df[['open', 'high', 'low', 'close']] -= 50
        none_df.to_parquet(none_path, compression='snappy')
        logger.info(f"Saved {len(none_df):,} NONE bars to {none_path}")
        
        # Show ATR distribution
        atr_stats = final_df['AUX1'].describe()
        logger.info("\nATR Distribution in Range Bars:")
        logger.info(f"  Min: {atr_stats['min']:.2f}")
        logger.info(f"  25%: {atr_stats['25%']:.2f}")
        logger.info(f"  50%: {atr_stats['50%']:.2f}")
        logger.info(f"  75%: {atr_stats['75%']:.2f}")
        logger.info(f"  Max: {atr_stats['max']:.2f}")
        logger.info(f"  Unique values: {final_df['AUX1'].nunique()}")
        
        return final_df
    
    return pd.DataFrame()

def create_range_bars_for_day(day_ticks, range_size, atr_value):
    """
    Create range bars for a single day's ticks.
    
    Args:
        day_ticks: DataFrame with one day's tick data
        range_size: Range size for this day (ATR * multiplier)
        atr_value: ATR value used (for AUX1 field)
    
    Returns:
        pd.DataFrame: Range bars for this day
    """
    
    bars = []
    
    if len(day_ticks) == 0:
        return pd.DataFrame()
    
    # Sort by datetime
    day_ticks = day_ticks.sort_values('datetime')
    
    # Initialize first bar
    bar_start_idx = 0
    bar_high = day_ticks.iloc[0]['Close']
    bar_low = day_ticks.iloc[0]['Close']
    bar_start_time = day_ticks.iloc[0]['datetime']
    
    min_duration = pd.Timedelta(minutes=1)
    
    for i in range(1, len(day_ticks)):
        current_price = day_ticks.iloc[i]['Close']
        current_time = day_ticks.iloc[i]['datetime']
        
        # Update high/low
        bar_high = max(bar_high, current_price)
        bar_low = min(bar_low, current_price)
        
        # Check if bar completes
        if (bar_high - bar_low) >= range_size and (current_time - bar_start_time) >= min_duration:
            # Create bar
            bar_data = day_ticks.iloc[bar_start_idx:i+1]
            
            bars.append({
                'timestamp': bar_data.iloc[-1]['datetime'],
                'open': float(bar_data.iloc[0]['Close']),
                'high': float(bar_data['Close'].max()),
                'low': float(bar_data['Close'].min()),
                'close': float(bar_data.iloc[-1]['Close']),
                'volume': int(bar_data['Volume'].sum()),
                'ticks': len(bar_data),
                'AUX1': float(atr_value),  # ATR value used
                'AUX2': float(ATR_MULTIPLIER)  # Multiplier
            })
            
            # Reset for next bar
            bar_start_idx = i + 1
            if bar_start_idx < len(day_ticks):
                bar_high = day_ticks.iloc[bar_start_idx]['Close']
                bar_low = day_ticks.iloc[bar_start_idx]['Close']
                bar_start_time = day_ticks.iloc[bar_start_idx]['datetime']
    
    return pd.DataFrame(bars)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution"""
    
    logger.info("=" * 100)
    logger.info("VECTORBT PRO PIPELINE: DAILY ATR WITH RANGE BARS")
    logger.info("=" * 100)
    logger.info(f"ATR Period: {ATR_PERIOD}")
    logger.info(f"ATR Multiplier: {ATR_MULTIPLIER}")
    logger.info("Each day uses previous day's ATR (lagged to avoid peeking)")
    logger.info("=" * 100)
    
    start_time = time.time()
    
    try:
        # Step 1: Create daily data with ATR
        daily_df = create_daily_data_with_atr()
        
        # Step 2: Create range bars using daily ATR
        range_bars = create_range_bars_with_daily_atr(daily_df)
        
        elapsed = (time.time() - start_time) / 60
        
        logger.info("\n" + "=" * 100)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 100)
        logger.info(f"Total processing time: {elapsed:.1f} minutes")
        logger.info(f"Created {len(range_bars):,} range bars")
        logger.info(f"Daily data exported to: {DAILY_CSV}")
        logger.info("=" * 100)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()