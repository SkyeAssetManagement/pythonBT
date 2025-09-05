"""
Data management utility for importing and managing symbol data.
Usage: python data_manager.py [command] [options]
"""

import argparse
import sys
from pathlib import Path
from src.data.parquet_importer import ParquetImporter
from src.data.parquet_loader import ParquetLoader


def import_data():
    """Import all CSV data from dataRaw to parquet format."""
    print("Starting data import...")
    
    # Paths relative to the main ABtoPython directory
    data_root = Path(__file__).parent.parent / "dataRaw"
    parquet_root = Path(__file__).parent.parent / "parquet_data"
    
    if not data_root.exists():
        print(f"[X] dataRaw directory not found: {data_root}")
        return False
    
    importer = ParquetImporter(str(data_root), str(parquet_root))
    results = importer.import_all_symbols()
    
    print(f"\n[CHART] Import Results:")
    success_count = 0
    for symbol, success in results.items():
        status = "[OK]" if success else "[X]"
        print(f"  {status} {symbol}")
        if success:
            success_count += 1
    
    print(f"\n[OK] Successfully imported {success_count}/{len(results)} symbols")
    return success_count > 0


def list_symbols():
    """List all available symbols."""
    parquet_root = Path(__file__).parent.parent / "parquet_data"
    
    if not parquet_root.exists():
        print("[X] No parquet data found. Run 'import' first.")
        return
    
    loader = ParquetLoader(str(parquet_root))
    symbols = loader.discover_symbols()
    
    if not symbols:
        print("[X] No symbols found in parquet data.")
        return
    
    print(f"[GROWTH] Available symbols ({len(symbols)}):")
    for symbol in symbols:
        info = loader.get_symbol_info(symbol)
        if info:
            print(f"  {symbol}: {info['total_bars']} bars ({info['start_date']} to {info['end_date']})")
        else:
            print(f"  {symbol}: [X] Error loading info")


def symbol_info(symbol: str):
    """Show detailed information about a symbol."""
    parquet_root = Path(__file__).parent.parent / "parquet_data"
    
    if not parquet_root.exists():
        print("[X] No parquet data found. Run 'import' first.")
        return
    
    loader = ParquetLoader(str(parquet_root))
    info = loader.get_symbol_info(symbol)
    
    if not info:
        print(f"[X] Symbol '{symbol}' not found or error loading data.")
        return
    
    print(f"[CHART] Symbol Information: {symbol}")
    print(f"  Total bars: {info['total_bars']:,}")
    print(f"  Date range: {info['start_date']} to {info['end_date']}")
    print(f"  Price range: ${info['min_price']:.2f} - ${info['max_price']:.2f}")
    print(f"  First price: ${info['first_price']:.2f}")
    print(f"  Last price: ${info['last_price']:.2f}")
    print(f"  Total volume: {info['total_volume']:,.0f}")


def test_symbol(symbol: str):
    """Test loading a symbol and show sample data."""
    parquet_root = Path(__file__).parent.parent / "parquet_data"
    
    if not parquet_root.exists():
        print("[X] No parquet data found. Run 'import' first.")
        return
    
    loader = ParquetLoader(str(parquet_root))
    data = loader.get_latest_bars(symbol, 10)
    
    if not data:
        print(f"[X] Failed to load data for '{symbol}'")
        return
    
    print(f"[GROWTH] Sample data for {symbol} (last 10 bars):")
    print(f"{'Date':<20} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<10}")
    print("-" * 70)
    
    for i in range(len(data['datetime'])):
        dt = str(data['datetime'][i])[:19]  # Remove microseconds
        print(f"{dt:<20} {data['open'][i]:<8.2f} {data['high'][i]:<8.2f} "
              f"{data['low'][i]:<8.2f} {data['close'][i]:<8.2f} {data['volume'][i]:<10.0f}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Data management utility for the trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_manager.py import              # Import all CSV data to Parquet
  python data_manager.py list               # List all available symbols
  python data_manager.py info AD            # Show info for AD symbol
  python data_manager.py test AD            # Test loading AD symbol data
        """
    )
    
    parser.add_argument('command', 
                       choices=['import', 'list', 'info', 'test'],
                       help='Command to execute')
    
    parser.add_argument('symbol', 
                       nargs='?',
                       help='Symbol name (required for info/test commands)')
    
    args = parser.parse_args()
    
    if args.command == 'import':
        import_data()
        
    elif args.command == 'list':
        list_symbols()
        
    elif args.command == 'info':
        if not args.symbol:
            print("[X] Symbol name required for 'info' command")
            sys.exit(1)
        symbol_info(args.symbol)
        
    elif args.command == 'test':
        if not args.symbol:
            print("[X] Symbol name required for 'test' command")
            sys.exit(1)
        test_symbol(args.symbol)


if __name__ == "__main__":
    main()