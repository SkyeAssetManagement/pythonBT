# Ultra-Fast Range Bar Conversion Guide

## Overview
This guide provides hyper-speed ETL processing for converting tick data to range bars using advanced array and parallel processing techniques in Python. Designed to handle massive datasets (37GB+ tick files) with maximum efficiency.

## Data Structure
**Input Format**: CSV tick data with columns:
- Date, Time, Open, High, Low, Close, Volume, Up Ticks, Down Ticks, Same Ticks

**Output Format**: Parquet files optimized for fast I/O and compression

## Three Range Bar Methods (All Optimized for ~5 Minutes Per Bar)

### 1. Raw Price Range Bars
- **Fixed price range: 0.22 points**
- New bar created when price moves 0.22 points from previous bar's close
- **Target: 204 bars/day (~5.0 min/bar)**
- Ultra-fast using NumPy vectorized operations

### 2. Percentage Range Bars  
- **Dynamic range: 0.0051% of current price**
- Adapts to price level changes (≈0.22 points at ES 4300 level)
- **Target: 204 bars/day (~5.0 min/bar)**
- Range = Current Price × 0.0051% / 100

### 3. ATR Range Bars (Daily H-L Approach)
- Range based on Average True Range with optimized multipliers:
  - **ATR-14d: 0.0088x multiplier**
  - **ATR-30d: 0.0074x multiplier** 
  - **ATR-90d: 0.0063x multiplier**
- **Target: 204 bars/day (~5.0 min/bar)**
- Most adaptive method, adjusting to market volatility

## Performance Optimization Strategies

### 1. Memory Management
- **Chunked Processing**: Process data in memory-efficient chunks (10M-50M rows)
- **dtype Optimization**: Use float32 instead of float64 where precision allows
- **Memory Mapping**: Use memory-mapped files for large datasets

### 2. Vectorized Operations
- **NumPy Arrays**: Convert to NumPy arrays for vectorized operations
- **Pandas Categorical**: Use categorical data types for repeated values
- **Avoid Python Loops**: Use NumPy array operations instead of Python loops

### 3. JIT Compilation
- **Numba**: JIT compile critical range bar logic functions
- **Cython**: Pre-compile performance-critical sections
- **Parallel Execution**: Leverage multiple CPU cores

### 4. I/O Optimization
- **PyArrow**: Fast parquet reading/writing
- **Compression**: Use Snappy or LZ4 compression for balance of speed/size
- **Column Store**: Leverage columnar format benefits
- **Partitioning**: Partition large datasets by date/symbol

### 5. Parallel Processing
- **Multiprocessing**: Process different date ranges in parallel
- **Threading**: Use for I/O bound operations
- **Chunked Parallel**: Combine chunking with parallel processing

## Performance Benchmarks

### Target Metrics
- **CSV to Parquet Conversion**: < 2 minutes for 37GB file
- **Parquet Load Time**: < 10 seconds for full dataset
- **Range Bar Generation**: < 30 seconds per method
- **Memory Usage**: < 8GB peak RAM usage

### Measurement Points
1. Raw CSV read time
2. Data type conversion time  
3. Parquet write time
4. Parquet read time
5. Range bar calculation time
6. Final parquet write time
7. Total end-to-end time

## Implementation Architecture

```
createRangeBars/
├── README.md                    # This guide
├── main.py                      # Main orchestration pipeline
├── common/
│   ├── performance.py           # Benchmarking utilities
│   └── data_loader.py           # Optimized CSV/Parquet I/O
├── raw_price/
│   └── range_bars_raw.py        # Raw price range bars
├── percentage/
│   └── range_bars_percent.py    # Percentage range bars  
├── atr/
│   └── range_bars_atr.py        # ATR range bars with daily H-L
└── output/                      # Generated outputs
    ├── parquet/                 # Converted parquet files
    ├── raw_price/               # Raw price range bar results
    ├── percentage/              # Percentage range bar results
    ├── atr/                     # ATR range bar results
    ├── analysis/                # Analysis reports
    └── performance/             # Performance logs
```

## Usage Examples

### Command Line Usage
```bash
# Process complete pipeline with optimized 5-minute bar settings
python main.py "path/to/ES-DIFF-Tick1-21toT.tick"

# Custom configurations
python main.py "path/to/ES-DIFF-Tick1-21toT.tick" \
    --output "my_output" \
    --raw-ranges 0.22 \
    --percentages 0.0051 \
    --cores 8
```

### Programmatic Usage
```python
from createRangeBars.main import RangeBarProcessor

# Initialize processor
processor = RangeBarProcessor(output_base_dir="output", n_cores=8)

# Complete processing pipeline with optimized 5-minute parameters
results = processor.process_all_range_types(
    csv_path='ES-DIFF-Tick1-21toT.tick',
    raw_range_sizes=[0.22],
    percentages=[0.0051],
    atr_configs=[
        {'lookback_days': 14, 'multiplier': 0.0088, 'name': 'atr_14d_5min'},
        {'lookback_days': 30, 'multiplier': 0.0074, 'name': 'atr_30d_5min'},
        {'lookback_days': 90, 'multiplier': 0.0063, 'name': 'atr_90d_5min'}
    ]
)
```

### Individual Range Bar Types (Optimized Parameters)
```python
# Just raw price range bars (5-minute optimized)
from createRangeBars.raw_price.range_bars_raw import create_raw_range_bars
range_bars = create_raw_range_bars(tick_data, range_size=0.22)

# Just percentage range bars (5-minute optimized)
from createRangeBars.percentage.range_bars_percent import create_percentage_range_bars
range_bars = create_percentage_range_bars(tick_data, percentage=0.0051)

# Just ATR range bars (5-minute optimized)
from createRangeBars.atr.range_bars_atr import create_atr_range_bars  
range_bars = create_atr_range_bars(tick_data, atr_lookback_days=14, atr_multiplier=0.0088)
```

## Performance Tips

### 1. Hardware Optimization
- **SSD Storage**: Use NVMe SSD for data files
- **RAM**: 32GB+ recommended for large datasets
- **CPU**: Multi-core CPU for parallel processing

### 2. Python Environment
- **Python 3.9+**: Latest Python version for performance improvements
- **NumPy**: Latest version with optimized BLAS libraries
- **Pandas 2.0+**: PyArrow backend for improved performance

### 3. System Tuning
- **Memory Overcommit**: Adjust vm.overcommit_memory for large datasets
- **File System**: Use appropriate file system (ext4/NTFS) with large block sizes
- **Swap**: Disable swap or use fast NVMe swap

## Expected Performance Results

For 37GB ES tick data file (approximately 400M+ rows):

| Operation | Time | Memory |
|-----------|------|--------|
| CSV → Parquet | 90s | 4GB |
| Parquet Load | 8s | 6GB |
| Raw Range Bars | 15s | 3GB |
| Percentage Range Bars | 18s | 3GB |
| ATR Range Bars | 25s | 4GB |
| **Total End-to-End** | **3-4 minutes** | **8GB peak** |

## Troubleshooting

### Memory Issues
- Reduce chunk size in data_loader.py
- Use float32 instead of float64
- Enable memory mapping for large files

### Performance Issues  
- Profile with cProfile to identify bottlenecks
- Ensure NumPy uses optimized BLAS (Intel MKL/OpenBLAS)
- Check I/O wait times with iostat

### Data Quality Issues
- Validate tick data for missing timestamps
- Handle price gaps and splits appropriately
- Verify range bar logic with small test datasets