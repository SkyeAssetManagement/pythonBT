"""
Structured Data Organizer for Range Bar Parquet Files

Creates organized folder structure:
parquetData/
├── parquet-1mn/
├── parquet-range-raw/
├── parquet-range-perc/  
├── parquet-range-ATR14d/
├── parquet-range-ATR30d/
└── parquet-range-ATR90d/
    └── instruments/
        └── ES/
            ├── NONE/
            └── DIFF/
"""

import os
from pathlib import Path
import shutil
from typing import Dict, List

class StructuredDataOrganizer:
    """
    Organizes parquet data files into structured hierarchy matching existing parquetData structure
    """
    
    def __init__(self, base_output_dir: str = "parquetData"):
        self.base_dir = Path(base_output_dir)
        
        # Define the folder structure
        self.data_types = [
            "parquet-1mn",           # Original tick data converted to 1-minute bars
            "parquet-range-raw",     # Raw price range bars
            "parquet-range-perc",    # Percentage range bars
            "parquet-range-ATR14d",  # ATR 14-day range bars
            "parquet-range-ATR30d",  # ATR 30-day range bars
            "parquet-range-ATR90d"   # ATR 90-day range bars
        ]
        
        # Instrument and symbol structure
        self.instruments = ["ES"]  # Can be expanded for other instruments
        self.symbols = ["NONE", "DIFF"]  # NONE = no adjustment, DIFF = difference/spread
        
        print("📁 Structured Data Organizer initialized")
        print(f"   Base directory: {self.base_dir}")
        print(f"   Data types: {len(self.data_types)}")
        print(f"   Instruments: {self.instruments}")
        print(f"   Symbols: {self.symbols}")
    
    def create_folder_structure(self):
        """Create the complete folder structure"""
        
        print("\n🏗️ Creating folder structure...")
        
        created_folders = []
        
        # Create base directory
        self.base_dir.mkdir(exist_ok=True)
        created_folders.append(str(self.base_dir))
        
        # Create each data type folder with instrument/symbol subfolders
        for data_type in self.data_types:
            data_type_path = self.base_dir / data_type
            data_type_path.mkdir(exist_ok=True)
            created_folders.append(str(data_type_path))
            
            # Create instruments folder
            instruments_path = data_type_path / "instruments" 
            instruments_path.mkdir(exist_ok=True)
            created_folders.append(str(instruments_path))
            
            # Create instrument subfolders
            for instrument in self.instruments:
                instrument_path = instruments_path / instrument
                instrument_path.mkdir(exist_ok=True)
                created_folders.append(str(instrument_path))
                
                # Create symbol subfolders
                for symbol in self.symbols:
                    symbol_path = instrument_path / symbol
                    symbol_path.mkdir(exist_ok=True)
                    created_folders.append(str(symbol_path))
        
        print(f"✅ Created {len(created_folders)} folders")
        
        return created_folders
    
    def get_output_path(self, data_type: str, instrument: str = "ES", symbol: str = "DIFF") -> Path:
        """
        Get the output path for a specific data type, instrument, and symbol
        
        Args:
            data_type: Type of data (parquet-1mn, parquet-range-raw, etc.)
            instrument: Instrument name (ES, NQ, etc.)
            symbol: Symbol type (NONE, DIFF)
            
        Returns:
            Path object for the target directory
        """
        
        if data_type not in self.data_types:
            raise ValueError(f"Invalid data_type: {data_type}. Must be one of {self.data_types}")
        
        return self.base_dir / data_type / "instruments" / instrument / symbol
    
    def organize_range_bar_files(self, source_files: Dict[str, str], 
                                instrument: str = "ES", symbol: str = "DIFF"):
        """
        Organize range bar files into the structured hierarchy
        
        Args:
            source_files: Dictionary mapping data type to source file path
            instrument: Instrument name
            symbol: Symbol type
        """
        
        print(f"\n📋 Organizing range bar files for {instrument}-{symbol}...")
        
        # Mapping from source keys to folder names
        type_mapping = {
            "raw_price": "parquet-range-raw",
            "percentage": "parquet-range-perc", 
            "atr_14d": "parquet-range-ATR14d",
            "atr_30d": "parquet-range-ATR30d",
            "atr_90d": "parquet-range-ATR90d"
        }
        
        organized_files = {}
        
        for source_key, source_file in source_files.items():
            if source_key in type_mapping:
                data_type = type_mapping[source_key]
                target_dir = self.get_output_path(data_type, instrument, symbol)
                
                # Create filename based on parameters
                source_path = Path(source_file)
                target_filename = f"{instrument}-{symbol}-range-bars.parquet"
                target_path = target_dir / target_filename
                
                # Copy file to organized location
                if source_path.exists():
                    shutil.copy2(source_file, target_path)
                    organized_files[data_type] = str(target_path)
                    print(f"   ✅ {source_key} → {target_path}")
                else:
                    print(f"   ❌ Source file not found: {source_file}")
            else:
                print(f"   ⚠️  Unknown source key: {source_key}")
        
        return organized_files
    
    def organize_base_parquet(self, source_parquet: str, 
                            instrument: str = "ES", symbol: str = "DIFF"):
        """
        Organize the base tick data parquet file
        
        Args:
            source_parquet: Path to source parquet file
            instrument: Instrument name  
            symbol: Symbol type
        """
        
        print(f"\n📦 Organizing base parquet for {instrument}-{symbol}...")
        
        # Place base tick data in parquet-1mn folder (can be used for 1-minute aggregation)
        target_dir = self.get_output_path("parquet-1mn", instrument, symbol)
        target_filename = f"{instrument}-{symbol}-tick-data.parquet"
        target_path = target_dir / target_filename
        
        source_path = Path(source_parquet)
        if source_path.exists():
            shutil.copy2(source_parquet, target_path)
            print(f"   ✅ Base data → {target_path}")
            return str(target_path)
        else:
            print(f"   ❌ Source parquet not found: {source_parquet}")
            return None
    
    def generate_directory_tree(self) -> str:
        """Generate a visual directory tree"""
        
        tree_lines = [f"📁 {self.base_dir}/"]
        
        for data_type in self.data_types:
            tree_lines.append(f"├── {data_type}/")
            tree_lines.append(f"│   └── instruments/")
            
            for i, instrument in enumerate(self.instruments):
                is_last_instrument = (i == len(self.instruments) - 1)
                prefix = "│       └──" if is_last_instrument else "│       ├──"
                tree_lines.append(f"{prefix} {instrument}/")
                
                for j, symbol in enumerate(self.symbols):
                    is_last_symbol = (j == len(self.symbols) - 1)
                    if is_last_instrument:
                        symbol_prefix = "            └──" if is_last_symbol else "            ├──"
                    else:
                        symbol_prefix = "│           └──" if is_last_symbol else "│           ├──"
                    tree_lines.append(f"{symbol_prefix} {symbol}/")
        
        return "\n".join(tree_lines)
    
    def create_readme_files(self):
        """Create README files explaining the structure"""
        
        # Main README
        main_readme = self.base_dir / "README.md"
        with open(main_readme, 'w') as f:
            f.write("""# Parquet Data Structure

## Directory Organization

This directory contains organized parquet files for different data types and aggregation methods:

### Data Types:
- **parquet-1mn**: Base tick data (can be used for 1-minute aggregation)
- **parquet-range-raw**: Raw price range bars (2.10 points)
- **parquet-range-perc**: Percentage range bars (0.049%)
- **parquet-range-ATR14d**: ATR 14-day range bars (0.084x multiplier)
- **parquet-range-ATR30d**: ATR 30-day range bars (0.070x multiplier) 
- **parquet-range-ATR90d**: ATR 90-day range bars (0.060x multiplier)

### Structure:
```
parquetData/
├── [data-type]/
│   └── instruments/
│       └── [instrument]/
│           ├── NONE/     # No adjustments
│           └── DIFF/     # Spread/difference data
```

### Instruments:
- **ES**: E-mini S&P 500 futures

### Symbol Types:
- **NONE**: Raw price data without adjustments
- **DIFF**: Spread or difference calculations

## Performance:
- All range bar types target ~5 minutes per bar (204 bars/day)
- Massive compression: 20,785:1 tick-to-bar ratio
- File sizes: ~2.9 MB per range bar file
""")
        
        # Create README for each data type
        descriptions = {
            "parquet-1mn": "Base tick data in parquet format, optimized for 1-minute aggregation",
            "parquet-range-raw": "Fixed 2.10-point range bars (5-minute average)",
            "parquet-range-perc": "0.049% percentage range bars (price-adaptive, 5-minute average)", 
            "parquet-range-ATR14d": "ATR-based range bars with 14-day lookback (0.084x multiplier)",
            "parquet-range-ATR30d": "ATR-based range bars with 30-day lookback (0.070x multiplier)",
            "parquet-range-ATR90d": "ATR-based range bars with 90-day lookback (0.060x multiplier)"
        }
        
        for data_type, description in descriptions.items():
            readme_path = self.base_dir / data_type / "README.md"
            with open(readme_path, 'w') as f:
                f.write(f"# {data_type.title().replace('-', ' ')}\n\n")
                f.write(f"{description}\n\n")
                f.write("## Contents:\n")
                f.write("- instruments/ES/NONE/: Raw ES data\n")
                f.write("- instruments/ES/DIFF/: ES spread data\n")
        
        print("📝 README files created")

def create_organized_structure():
    """Create the complete organized structure"""
    
    organizer = StructuredDataOrganizer()
    
    print("🚀 Creating organized parquet data structure...")
    
    # Create folder structure
    folders = organizer.create_folder_structure()
    
    # Create README files
    organizer.create_readme_files()
    
    # Display directory tree
    print(f"\n📂 Directory structure created:")
    print(organizer.generate_directory_tree())
    
    print(f"\n✅ Structured data organization complete!")
    print(f"   📁 Created {len(folders)} folders")
    print(f"   📝 Created README files")
    print(f"   🎯 Ready for range bar data organization")
    
    return organizer

if __name__ == "__main__":
    organizer = create_organized_structure()