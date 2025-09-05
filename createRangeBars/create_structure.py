"""
Create organized folder structure for parquet data
"""

import os
from pathlib import Path

def create_parquet_structure():
    """Create the organized parquet data structure"""
    
    base_dir = Path("parquetData")
    
    # Data types
    data_types = [
        "parquet-1mn",           # Original tick data converted to 1-minute bars
        "parquet-range-raw",     # Raw price range bars
        "parquet-range-perc",    # Percentage range bars
        "parquet-range-ATR14d",  # ATR 14-day range bars
        "parquet-range-ATR30d",  # ATR 30-day range bars
        "parquet-range-ATR90d"   # ATR 90-day range bars
    ]
    
    # Instruments and symbols
    instruments = ["ES"]
    symbols = ["NONE", "DIFF"]
    
    print("Creating parquet data structure...")
    print(f"Base directory: {base_dir}")
    
    created_folders = []
    
    # Create base directory
    base_dir.mkdir(exist_ok=True)
    created_folders.append(str(base_dir))
    
    # Create each data type folder with instrument/symbol subfolders
    for data_type in data_types:
        data_type_path = base_dir / data_type
        data_type_path.mkdir(exist_ok=True)
        created_folders.append(str(data_type_path))
        
        # Create instruments folder
        instruments_path = data_type_path / "instruments" 
        instruments_path.mkdir(exist_ok=True)
        created_folders.append(str(instruments_path))
        
        # Create instrument subfolders
        for instrument in instruments:
            instrument_path = instruments_path / instrument
            instrument_path.mkdir(exist_ok=True)
            created_folders.append(str(instrument_path))
            
            # Create symbol subfolders
            for symbol in symbols:
                symbol_path = instrument_path / symbol
                symbol_path.mkdir(exist_ok=True)
                created_folders.append(str(symbol_path))
                
                print(f"Created: {symbol_path}")
    
    # Create main README
    readme_content = """# Parquet Data Structure

## Directory Organization

This directory contains organized parquet files for different data types:

### Data Types:
- **parquet-1mn**: Base tick data (for 1-minute aggregation)
- **parquet-range-raw**: Raw price range bars (2.10 points)
- **parquet-range-perc**: Percentage range bars (0.049%)
- **parquet-range-ATR14d**: ATR 14-day range bars (0.084x multiplier)
- **parquet-range-ATR30d**: ATR 30-day range bars (0.070x multiplier) 
- **parquet-range-ATR90d**: ATR 90-day range bars (0.060x multiplier)

### Structure:
```
parquetData/
├── parquet-1mn/
├── parquet-range-raw/
├── parquet-range-perc/  
├── parquet-range-ATR14d/
├── parquet-range-ATR30d/
└── parquet-range-ATR90d/
    └── instruments/
        └── ES/
            ├── NONE/     # No adjustments
            └── DIFF/     # Spread/difference data
```

### Performance:
- All range bar types target ~5 minutes per bar (204 bars/day)
- Compression: 20,785:1 tick-to-bar ratio
- File sizes: ~2.9 MB per range bar file
"""
    
    with open(base_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"\nStructure created successfully!")
    print(f"Created {len(created_folders)} folders")
    
    # Show directory tree
    print(f"\nDirectory structure:")
    print("parquetData/")
    for data_type in data_types:
        print(f"├── {data_type}/")
        print(f"│   └── instruments/")
        print(f"│       └── ES/")
        print(f"│           ├── NONE/")
        print(f"│           └── DIFF/")
    
    return base_dir

if __name__ == "__main__":
    create_parquet_structure()