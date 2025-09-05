import pandas as pd
from date_parser import FlexibleDateParser

def test_file(filename):
    """Test date parsing on a specific file"""
    print(f"\n{'='*60}")
    print(f"Testing: {filename}")
    print('='*60)
    
    try:
        # Load the file
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # Auto-detect date columns
        date_columns = FlexibleDateParser.get_date_columns(df)
        print(f"\nDetected date columns:")
        print(f"  Date column: {date_columns.get('date_column')}")
        print(f"  Time column: {date_columns.get('time_column')}")
        print(f"  DateTime column: {date_columns.get('datetime_column')}")
        
        # Parse dates
        parsed_dates = FlexibleDateParser.parse_dates(df)
        print(f"\nParsed {len(parsed_dates)} dates successfully")
        print(f"Date range: {parsed_dates.min()} to {parsed_dates.max()}")
        
        # Show first few dates
        print(f"\nFirst 5 parsed dates:")
        for i in range(min(5, len(parsed_dates))):
            original_cols = []
            if date_columns.get('datetime_column'):
                original_cols.append(df[date_columns['datetime_column']].iloc[i])
            else:
                if date_columns.get('date_column'):
                    original_cols.append(df[date_columns['date_column']].iloc[i])
                if date_columns.get('time_column'):
                    original_cols.append(df[date_columns['time_column']].iloc[i])
            
            original_str = ' '.join(str(col) for col in original_cols)
            print(f"  {original_str} -> {parsed_dates.iloc[i]}")
        
        # Test with different dayfirst settings
        print(f"\nTesting dayfirst settings:")
        for dayfirst in [True, False, None]:
            try:
                parsed = FlexibleDateParser.parse_dates(
                    df,
                    date_column=date_columns.get('date_column'),
                    time_column=date_columns.get('time_column'),
                    datetime_column=date_columns.get('datetime_column'),
                    dayfirst=dayfirst
                )
                print(f"  dayfirst={dayfirst}: Success - {parsed.iloc[0]}")
            except Exception as e:
                print(f"  dayfirst={dayfirst}: Failed - {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test both data files"""
    print("Testing Flexible Date Parser")
    print("="*60)
    
    # Test DTSmlDATA7x7.csv (separate Date and Time columns)
    test1 = test_file("DTSmlDATA7x7.csv")
    
    # Test DTSnnData.csv (combined Date/Time column)
    test2 = test_file("DTSnnData.csv")
    
    print(f"\n{'='*60}")
    print("Test Summary:")
    print(f"  DTSmlDATA7x7.csv: {'PASSED' if test1 else 'FAILED'}")
    print(f"  DTSnnData.csv: {'PASSED' if test2 else 'FAILED'}")
    print('='*60)

if __name__ == "__main__":
    main()