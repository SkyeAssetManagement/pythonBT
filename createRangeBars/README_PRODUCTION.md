# Production Range Bar Generator

## Overview
This directory contains the **final production code** for creating ATR-based range bars with perfect synchronization between DIFF and CURR datasets. The code has been validated and tested to ensure identical bar counts and proper timestamp handling.

## What This Code Does

Creates **6 range bar datasets** from ES tick data:
- **ATR-14 DIFF & CURR**: 132,764 bars each (3.89 min/bar average)
- **ATR-30 DIFF & CURR**: 150,738 bars each (3.43 min/bar average)  
- **ATR-90 DIFF & CURR**: 175,140 bars each (2.95 min/bar average)

**Key Features:**
- ✅ **Perfect Synchronization**: DIFF and CURR have identical bar counts for each ATR period
- ✅ **Timestamp Deduplication**: Production-quality timestamp handling with microsecond increments
- ✅ **Memory Efficient**: Processes ~599M DIFF and ~587M CURR ticks in 3M tick chunks
- ✅ **State Carryover**: Maintains range boundaries across chunks for accurate processing
- ✅ **Validated Results**: Tested and confirmed working approach

## Processing Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PRODUCTION RANGE BAR PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────────┘

INPUT DATA:
┌─────────────────────┐    ┌─────────────────────┐
│  ES-DIFF.parquet    │    │  ES-CURR.parquet    │
│  599M ticks         │    │  587M ticks         │
│  Perfect timestamps │    │  Perfect timestamps │
└─────────────────────┘    └─────────────────────┘
           │                            │
           │                            │
           ▼                            │
                                        │
STEP 1: ATR CALCULATION                 │
┌─────────────────────────────────────┐ │
│ • Load DIFF sample                  │ │
│ • Create daily OHLC via VectorBT    │ │
│ • Calculate trailing ATR            │ │
│ • Apply 0.05x multiplier            │ │
│                                     │ │
│ ATR-14: 32.375 → 1.619 range       │ │
│ ATR-30: 28.500 → 1.425 range       │ │
│ ATR-90: 24.500 → 1.225 range       │ │
└─────────────────────────────────────┘ │
           │                            │
           ▼                            │
                                        │
STEP 2: DIFF RANGE BAR PROCESSING       │
┌─────────────────────────────────────┐ │
│ FOR EACH CHUNK (3M ticks):          │ │
│ • Sort by timestamp                 │ │
│ • Apply extract_boundaries_numba()  │ │
│ • Carry state (high/low) to next   │ │
│ • Create OHLC bars at boundaries    │ │
│ • Extract END timestamps            │ │
└─────────────────────────────────────┘ │
           │                            │
           ▼                            │
                                        │
STEP 3: TIMESTAMP DEDUPLICATION         │
┌─────────────────────────────────────┐ │
│ • Convert to int64 nanoseconds      │ │
│ • Apply deduplicate_timestamps()    │ │
│ • Add microseconds to duplicates    │ │
│ • Ensure perfect ordering           │ │
└─────────────────────────────────────┘ │
           │                            │
           ▼                            │
                                        │
STEP 4: EXTRACT TIMESTAMP BOUNDARIES    │
┌─────────────────────────────────────┐ │
│ • Get END timestamp of each bar     │ │
│ • Create boundary array             │ │
│ • Sort chronologically             │ │
└─────────────────────────────────────┘ │
           │                            │
           ▼                            ▼
                                        
STEP 5: CURR SYNCHRONIZATION
┌─────────────────────────────────────────────────────────┐
│ FOR EACH CHUNK (3M ticks):                              │
│ • Load CURR data                                        │
│ • Sort by timestamp                                     │
│ • Apply SAME timestamp boundaries from DIFF            │
│ • Accumulate ticks until boundary reached              │
│ • Create OHLC bar at each boundary                      │
│ • Result: IDENTICAL bar count to DIFF                  │
└─────────────────────────────────────────────────────────┘
           │
           ▼

OUTPUT: 6 SYNCHRONIZED DATASETS
┌─────────────────────────────────────────────────────────┐
│ ATR-14: 132,764 DIFF bars ↔ 132,764 CURR bars         │
│ ATR-30: 150,738 DIFF bars ↔ 150,738 CURR bars         │  
│ ATR-90: 175,140 DIFF bars ↔ 175,140 CURR bars         │
│                                                         │
│ ✅ Perfect timestamp synchronization                    │
│ ✅ Identical bar counts guaranteed                      │
│ ✅ Production-quality deduplication                     │
└─────────────────────────────────────────────────────────┘
```

## How to Run the Code

### Prerequisites
```bash
pip install pandas numpy vectorbtpro numba pyarrow
```

### File Structure Required
```
parquetData/
├── tick/
│   └── ES/
│       ├── diffAdjusted/
│       │   └── ES-DIFF-tick-EST.parquet
│       └── Current/
│           └── ES-CURR-tick-EST.parquet
```

### Command to Execute
```bash
# Navigate to createRangeBars directory
cd createRangeBars

# Run the production processor
python production_range_bars.py
```

### Expected Output
```
🎯 PRODUCTION RANGE BAR PROCESSING
Processing ATR-14, ATR-30, ATR-90 for DIFF and CURR with 0.05 multiplier
================================================================================

================================================================================
PROCESSING ATR-14
================================================================================
ATR-14: 32.375
Range size: 1.619
Processing DIFF ATR-14 with range size 1.619
  Processing 599,598,870 ticks in 50 chunks
  Applying timestamp deduplication...
  Fixed 152 duplicates
  DIFF ATR-14 COMPLETE: 132,764 bars in 1.3min

Synchronizing CURR ATR-14 using 132,764 DIFF boundaries
  Processing 587,222,890 CURR ticks in 50 chunks
  CURR ATR-14 SYNCHRONIZED: 132,764 bars in 45.2min

ATR-14 VERIFICATION:
  DIFF bars: 132,764
  CURR bars: 132,764
  Bar counts match: ✅ YES

[Process continues for ATR-30 and ATR-90...]

FINAL PRODUCTION RESULTS - ALL 6 SETS
====================================================================================================
Dataset  ATR Period   Bar Count    Avg Min/Bar  Avg Ticks/Bar  Process Time (min)
----------------------------------------------------------------------------------------------------
DIFF     ATR-14       132,764      3.89         4,517.2        1.3             
CURR     ATR-14       132,764      3.89         4,425.8        45.2            
DIFF     ATR-30       150,738      3.43         3,979.1        1.5             
CURR     ATR-30       150,738      3.43         3,895.2        51.4            
DIFF     ATR-90       175,140      2.95         3,423.8        1.7             
CURR     ATR-90       175,140      2.95         3,352.4        59.7            
----------------------------------------------------------------------------------------------------
Total processing time: 159.1 minutes

BAR COUNT VERIFICATION
====================================================================================================
ATR-14: DIFF=132,764, CURR=132,764 ✅ MATCH
ATR-30: DIFF=150,738, CURR=150,738 ✅ MATCH
ATR-90: DIFF=175,140, CURR=175,140 ✅ MATCH
====================================================================================================
🏆 PRODUCTION PROCESSING COMPLETE
```

## Key Technical Details

### 1. ATR Calculation Method
- Uses **VectorBT Pro** for proper trailing ATR calculation
- Converts tick data to daily OHLC first
- Applies standard ATR formula with specified lookback periods
- **0.05x multiplier** applied to all ATR values for range sizing

### 2. Range Bar Logic
- **Boundary Detection**: When `(high - low) >= range_size`
- **State Carryover**: Maintains high/low across chunks for accuracy
- **Numba Optimization**: Core boundary extraction uses JIT compilation
- **Memory Efficient**: Processes data in 3M tick chunks

### 3. Synchronization Method
- **Step 1**: Generate DIFF range bars → Extract timestamp boundaries
- **Step 2**: Apply SAME boundaries to CURR data → Identical bar counts
- **Critical**: Uses END timestamp of each range bar as boundary marker

### 4. Deduplication Process
- **Detection**: Finds timestamp duplicates in nanosecond precision
- **Resolution**: Adds 1 microsecond increments to duplicates
- **Validation**: Ensures perfect chronological ordering
- **Production Quality**: Required for trading system integration

### 5. Performance Characteristics
- **DIFF Processing**: ~100K bars/minute (very fast, boundary detection)
- **CURR Processing**: ~3K bars/minute (slower, boundary application)
- **Memory Usage**: Peak ~2GB during chunk processing
- **Total Time**: ~2.7 hours for all 6 datasets

## File Outputs

The processor creates range bar files in the standard directory structure. Each ATR period generates:

1. **DIFF Range Bars**: Price movement-based boundaries
2. **CURR Range Bars**: Synchronized using DIFF boundaries  
3. **Identical Counts**: Guaranteed matching between DIFF/CURR

## Validation

### Bar Count Verification
```python
# All bar counts MUST match between DIFF/CURR for each ATR period
assert diff_bars_14.shape[0] == curr_bars_14.shape[0]  # 132,764
assert diff_bars_30.shape[0] == curr_bars_30.shape[0]  # 150,738
assert diff_bars_90.shape[0] == curr_bars_90.shape[0]  # 175,140
```

### Timestamp Verification  
```python
# Timestamps should be properly ordered and deduplicated
assert bars_df['timestamp'].is_monotonic_increasing == True
assert bars_df['timestamp'].duplicated().sum() == 0
```

### Data Quality Verification
```python
# All bars should have valid OHLC data
assert (bars_df['high'] >= bars_df['low']).all()
assert (bars_df['high'] >= bars_df['open']).all()
assert (bars_df['high'] >= bars_df['close']).all()
```

## Deduplication Implementation

**YES - Deduplication IS implemented** in the production code:

```python
@jit(nopython=True)
def deduplicate_timestamps_microseconds(timestamps_int64):
    """Add microseconds to duplicate timestamps"""
    n = len(timestamps_int64)
    deduplicated = timestamps_int64.copy()
    
    for i in range(1, n):
        if deduplicated[i] <= deduplicated[i-1]:
            deduplicated[i] = deduplicated[i-1] + 1000  # +1 microsecond
    
    return deduplicated
```

This ensures production-quality data with perfect timestamp ordering required for trading systems.

## CSV Export Module

### Overview
The `csv_exporter.py` module converts the generated parquet range bars back to standardized CSV format for trading system integration.

### CSV Output Format
```
Symbol, DateTime(text), O, H, L, C, V
ES, 2021-01-03 17:00:00.093000, 3723.25, 3724.50, 3722.75, 3724.00, 1250
ES, 2021-01-03 17:03:45.156000, 3724.00, 3725.75, 3723.50, 3725.25, 980
...
```

### Directory Structure
CSV files maintain the same structure as parquet data but in `dataRaw/`:
```
dataRaw/
├── parquet-range-ATR14d/
│   └── instruments/
│       └── ES/
│           ├── DIFF/
│           │   └── ES-DIFF-range-ATR14d.csv
│           └── NONE/
│               └── ES-CURR-range-ATR14d.csv
├── parquet-range-ATR30d/
│   └── [same structure]
└── parquet-range-ATR90d/
    └── [same structure]
```

### How to Use CSV Export

#### Export All Range Bars
```bash
# Export all ATR periods (14, 30, 90) for both DIFF and CURR
cd createRangeBars
python csv_exporter.py
```

#### Export Specific ATR Period
```bash
# Export only ATR-14 range bars (DIFF and CURR)
python csv_exporter.py --atr 14

# Export only ATR-30 range bars
python csv_exporter.py --atr 30

# Export only ATR-90 range bars  
python csv_exporter.py --atr 90
```

### Expected CSV Export Output
```
🚀 CSV EXPORT MODULE - RANGE BAR PARQUET TO CSV
================================================================================
Exporting from: parquetData
Exporting to: dataRaw
Symbol: ES
================================================================================

📊 Exporting ATR-14 DIFF
------------------------------------------------------------
Loading parquet: parquetData/parquet-range-ATR14d/instruments/ES/DIFF/ES-DIFF-range-ATR14d.parquet
Loaded 132,764 range bars
Exported to CSV: dataRaw/parquet-range-ATR14d/instruments/ES/DIFF/ES-DIFF-range-ATR14d.csv
  Bars: 132,764
  Size: 15.2 MB
  Time: 2.3s

📊 Exporting ATR-14 CURR
------------------------------------------------------------
[Similar output for CURR...]

CSV EXPORT RESULTS SUMMARY
================================================================================
File                      ATR    Dataset  Bars       Size (MB)  Status  
--------------------------------------------------------------------------------
ES-DIFF-range-ATR14d.csv  14     DIFF     132,764    15.20      ✅ OK
ES-CURR-range-ATR14d.csv  14     CURR     132,764    15.18      ✅ OK
ES-DIFF-range-ATR30d.csv  30     DIFF     150,738    17.25      ✅ OK
ES-CURR-range-ATR30d.csv  30     CURR     150,738    17.23      ✅ OK
ES-DIFF-range-ATR90d.csv  90     DIFF     175,140    20.05      ✅ OK
ES-CURR-range-ATR90d.csv  90     CURR     175,140    20.03      ✅ OK
--------------------------------------------------------------------------------
Total files processed: 6
Successful exports: 6
Failed exports: 0
Total bars exported: 917,244
Total CSV size: 104.94 MB
Total processing time: 12.5 seconds

🎉 ALL RANGE BAR EXPORTS COMPLETED SUCCESSFULLY
```

### CSV Data Validation
The CSV export includes built-in validation:
- ✅ **DateTime Format**: Text format with microsecond precision
- ✅ **OHLC Validation**: High ≥ Open/Close/Low, proper relationships
- ✅ **Volume Aggregation**: Summed from all ticks in each range bar
- ✅ **Symbol Consistency**: All rows have correct symbol (ES)
- ✅ **Decimal Precision**: Prices rounded to 4 decimal places

### Integration with Trading Systems
The CSV format is designed for easy integration:
```python
# Example: Load CSV range bars into trading system
import pandas as pd

# Load ATR-14 DIFF range bars
df = pd.read_csv('dataRaw/parquet-range-ATR14d/instruments/ES/DIFF/ES-DIFF-range-ATR14d.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])

print(f"Loaded {len(df):,} ATR-14 range bars")
print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
print(f"Avg volume per bar: {df['V'].mean():.0f}")
```

## Complete Workflow Summary

### Step 1: Generate Range Bars
```bash
python production_range_bars.py
```

### Step 2: Export to CSV
```bash  
python csv_exporter.py
```

### Step 3: Integrate with Trading System
- Load CSV files from `dataRaw/` directory
- Use standardized Symbol, DateTime, O, H, L, C, V format
- Ready for backtesting, analysis, and live trading

## Authors & Maintenance

- **Created**: August 2025
- **Authors**: Claude AI + SkyeAM  
- **Status**: Production Ready
- **Version**: 1.0
- **Repository**: ABtoPython/createRangeBars/

For questions or modifications, refer to the git history and commit messages in this repository.