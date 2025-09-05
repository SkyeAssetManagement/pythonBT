# OMtree Trading System

A machine learning-based trading strategy system using ensemble decision trees for directional market prediction.

## Quick Start

### 1. Installation (Windows)
```cmd
python setup.py
```

### 2. Install Dependencies
```cmd
pip install -r requirements.txt
```

### 3. Generate Sample Data
```cmd
cd data
python sample_data_generator.py
cd ..
```

### 4. Run the System

**Option A: GUI Interface**
```cmd
python OMtree_gui.py
```

**Option B: Command Line**
```cmd
python OMtree_walkforward.py
```

## Documentation

- **[HOW-TO-GUIDE.md](HOW-TO-GUIDE.md)** - Complete usage instructions
- **[CODE-DOCUMENTATION.md](.claude/CODE-DOCUMENTATION.md)** - Technical documentation

## Project Structure

```
PythonBT/
├── src/                    # Core modules
├── data/                   # Data files
├── OMtree_gui.py          # Main GUI
├── OMtree_config.ini      # Configuration
├── requirements.txt       # Dependencies
├── setup.py              # Setup script
└── HOW-TO-GUIDE.md       # User guide
```

## Configuration

1. Copy `OMtree_config_example.ini` to `OMtree_config.ini`
2. Update the `csv_file` path to your data
3. Adjust model parameters as needed

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Features

- **Walk-Forward Validation**: Robust backtesting methodology
- **Ensemble Decision Trees**: Multiple tree models for predictions
- **GUI Interface**: User-friendly analysis interface
- **Performance Metrics**: Comprehensive trading statistics
- **Feature Selection**: Dynamic feature importance analysis

## Support

For detailed instructions, see:
- [HOW-TO-GUIDE.md](HOW-TO-GUIDE.md)
- [Code Documentation](.claude/CODE-DOCUMENTATION.md)