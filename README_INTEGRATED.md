# SAM - Integrated Backtesting System

## Overview

SAM (Integrated Backtesting System) combines multiple powerful trading tools into a unified platform:

- **Range Bar Creation**: Convert tick data to range bars using ATR-based dynamic ranges
- **OMtree ML Models**: Machine learning decision trees for trading signal generation
- **VectorBT Pro Backtesting**: High-performance vectorized backtesting engine
- **Advanced Visualization**: Interactive charting dashboard with trade visualization
- **Data Management**: Parquet data handling and quick viewing tools

## Quick Start

### Windows
```bash
launch_integrated_system.bat
```

### Linux/Mac
```bash
chmod +x launch_integrated_system.sh
./launch_integrated_system.sh
```

### Python Direct
```bash
python integrated_trading_launcher.py
```

## Features

### 1. Data Preparation (Range Bars)
- Convert tick data to range bars
- ATR-based dynamic range calculation
- Multiple timeframe support
- Parallel processing for speed
- Export to CSV and Parquet formats

### 2. ML Backtesting (OMtree)
- Decision tree models
- Walk-forward analysis
- Feature selection and importance
- Performance metrics and statistics
- Regression and classification modes

### 3. VectorBT Pro Backtesting
- Vectorized backtesting for speed
- Strategy optimization
- Portfolio analysis
- Risk metrics calculation
- Multiple strategy support

### 4. Data Visualization
- Interactive candlestick charts
- Trade marker visualization
- Technical indicators overlay
- Crosshair and zoom controls
- Real-time updates

### 5. Quick Chart Viewer
- Browse parquet data files
- Quick data validation
- Basic charting capabilities
- Data statistics display

### 6. Unified System
- All features in one interface
- Workflow automation
- Custom pipeline creation
- Advanced configuration options

## Directory Structure

```
PythonBT/
├── integrated_trading_launcher.py  # Main launcher GUI
├── OMtree_gui.py                  # ML trading models GUI
├── unified_gui.py                 # Unified system GUI
├── createRangeBars/              # Range bar creation modules
│   ├── main.py                   # Range bar GUI
│   └── ...                       # Processing modules
├── tradingCode/                  # Trading visualization
│   ├── main.py                   # Dashboard main
│   └── ...                       # Dashboard components
├── src/                          # Core modules
│   ├── trading/                  # Trading components
│   ├── backtest/                 # Backtesting engines
│   └── ...                       # Other modules
├── data/                         # Data files
├── dataRaw/                      # Raw range bar data
├── parquetData/                  # Parquet format data
└── results/                      # Output results
```

## Requirements

- Python 3.8 or higher
- tkinter (usually included with Python)
- pandas
- numpy
- PyQt5 (for advanced dashboard)
- vectorbt (for backtesting)
- Additional requirements in requirements.txt

## Installation

1. Clone or download the repository
2. Install requirements:
```bash
pip install -r requirements.txt
```
3. Launch the integrated system using one of the methods above

## Usage Guide

### Creating Range Bars

1. Launch the system and select "Data Preparation"
2. Browse for your tick data file (CSV or Parquet)
3. Set ATR parameters:
   - Period (default: 30)
   - Multipliers (0.05, 0.10, 0.20)
4. Click "Start Processing"
5. Results will be saved to the output directory

### Running ML Backtests

1. Select "ML Backtesting" from the launcher
2. Configure your model parameters
3. Select data file and date range
4. Run walk-forward analysis
5. View performance metrics and results

### Using VectorBT

1. Select "VectorBT Pro" from the launcher
2. Choose strategy and parameters
3. Load historical data
4. Run optimization or single backtest
5. Analyze results and metrics

### Viewing Charts

1. Select "Data Visualization" for full dashboard
2. Or use "Quick Chart" for simple viewing
3. Load data file
4. Navigate using keyboard shortcuts:
   - Arrow keys: Pan
   - +/-: Zoom
   - Space: Reset view

## Keyboard Shortcuts

### Dashboard Controls
- `←/→`: Pan left/right
- `↑/↓`: Zoom in/out
- `Space`: Reset view
- `T`: Toggle trade markers
- `I`: Toggle indicators
- `G`: Toggle grid

## Configuration

Configuration files are located in:
- `config/`: General configuration
- `model_settings/`: ML model settings
- `data_settings/`: Data configuration

## Troubleshooting

### Python Not Found
- Ensure Python 3.8+ is installed
- Add Python to system PATH

### Module Import Errors
- Install all requirements: `pip install -r requirements.txt`
- Check Python version compatibility

### GUI Not Launching
- Verify tkinter is installed
- On Linux: `sudo apt-get install python3-tk`

### Performance Issues
- Reduce data size for testing
- Enable parallel processing
- Check available RAM

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages in the console
3. Verify all dependencies are installed

## License

This is a proprietary trading system. All rights reserved.

## Version History

- v2.0: Integrated system with unified launcher
- v1.5: Added VectorBT integration
- v1.0: Initial release with separate components