# Modular AmiBroker Trading System

A clean, modular trading system with vectorized backtesting using VectorBT (open source).

## [TARGET] Key Features

- **[PUZZLE] Modular Strategy Architecture**: Plug-and-play strategy system
- **[FAST] Lightning-Fast Data Loading**: Parquet-based caching with Polars
- **[NUMBERS] Array Processing**: All operations use numpy/vectorBT arrays (NO LOOPS)
- **[CHART] VectorBT Integration**: Professional-grade backtesting engine
- **[TOOLS] Parameter Sweeps**: Test hundreds of parameter combinations simultaneously
- **[GROWTH] Multiple Data Sources**: AmiBroker DB -> CSV -> Parquet pipeline

## [LAUNCH] Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Import your data:**
```bash
python data_manager.py import
```

3. **Run a strategy:**
```bash
python main.py AD simpleSMA
```

## [FOLDER] Project Structure

```
amibroker-trading-system/
|--── src/
|   |--── data/              # Data loading (AmiBroker, Parquet)
|   |--── backtest/          # VectorBT backtesting engine
|   `--── utils/             # Utility functions
|--── strategies/            # 🔥 Modular strategy system
|   |--── base_strategy.py   # Abstract base class
|   `--── simpleSMA.py       # SMA crossover strategies
|--── main.py               # Main execution script
|--── data_manager.py       # Data import/management utility
`--── config.yaml          # Configuration
```

## [BUILD] Data Management

### Import CSV to Parquet
```bash
python data_manager.py import      # Import all CSV files
python data_manager.py list       # List available symbols
python data_manager.py info AD    # Show symbol information
python data_manager.py test AD    # Test symbol loading
```

### Data Flow
1. **Raw CSV** files in `../dataRaw/SYMBOL/Current/`
2. **Parquet** cache in `../parquet_data/SYMBOL/`
3. **Fast loading** with automatic fallback to AmiBroker

## [PUZZLE] Creating Strategies

Create a new strategy by inheriting from `BaseStrategy`:

```python
# strategies/my_strategy.py
from .base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("MyStrategy")
    
    def get_parameter_combinations(self):
        # Return list of parameter dictionaries
        return [{'param1': 10, 'param2': 20}]
    
    def _generate_signals_for_params(self, data, params):
        # Generate entry/exit signals for given parameters
        # Return (entries, exits) as boolean numpy arrays
        pass
```

Then run with:
```bash
python main.py SYMBOL my_strategy
```

## [CHART] Example Strategies

### SimpleSMA (Single Test)
- SMA 100/20 crossover
- Single parameter combination

### SimpleSMAParameterSweep 
- Tests hundreds of SMA combinations
- Fast periods: 5-50 (steps of 5)
- Slow periods: 50-200 (steps of 10)
- Fully vectorized processing

## [FAST] Performance

- **Data Loading**: Sub-second with Parquet cache
- **Signal Generation**: Vectorized operations only
- **Backtesting**: 100+ parameter combinations simultaneously
- **No Python Loops**: Pure array processing

## [TOOLS] Configuration

Edit `config.yaml`:
```yaml
data:
  amibroker_path: "path/to/amibroker/db"
  
backtest:
  initial_cash: 100000
  commission: 0.001
  slippage: 0.0005
  
output:
  results_dir: "results"
```

## [GROWTH] Output

Generated files in `results/`:
- `tradelist.csv`: Detailed trade records
- `equity_curve.csv`: Portfolio equity over time
- `performance_summary.csv`: Key metrics

## [TEST] Testing

```bash
python test_system.py    # Test the modular system
```

## [IDEA] Design Philosophy

1. **Modular**: Strategies are completely separate from core engine
2. **Fast**: Vectorized operations with Polars/VectorBT
3. **Clean**: No hardcoded strategies, no loops, minimal dependencies
4. **Scalable**: Easy to add new strategies without touching core code