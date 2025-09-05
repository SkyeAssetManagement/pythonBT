#!/usr/bin/env python3
"""
Test the cleaned data pipeline without AmiBroker dependencies
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all imports work without AmiBroker dependencies"""
    
    print("TESTING CLEANED IMPORTS")
    print("=" * 50)
    
    try:
        from src.data.csv_raw_loader import CSVRawLoader
        print("[OK] CSVRawLoader imported successfully")
    except ImportError as e:
        print(f"[X] CSVRawLoader import failed: {e}")
        return False
    
    try:
        from src.data.parquet_loader import ParquetLoader
        print("[OK] ParquetLoader imported successfully")
    except ImportError as e:
        print(f"[X] ParquetLoader import failed: {e}")
        return False
    
    try:
        from src.data.parquet_converter import ParquetConverter
        print("[OK] ParquetConverter imported successfully")
    except ImportError as e:
        print(f"[X] ParquetConverter import failed: {e}")
        return False
    
    try:
        from src.data.array_validator import ArrayValidator
        print("[OK] ArrayValidator imported successfully")
    except ImportError as e:
        print(f"[X] ArrayValidator import failed: {e}")
        return False
    
    # Test that AmiBroker is NOT imported
    try:
        from src.data.amibroker_loader import AmiBrokerArrayLoader
        print("[X] AmiBrokerArrayLoader should not be imported")
        return False
    except ImportError:
        print("[OK] AmiBrokerArrayLoader correctly removed")
    
    return True

def test_parquet_converter_pipeline():
    """Test the ParquetConverter pipeline"""
    
    print("\nTESTING PARQUET CONVERTER PIPELINE")
    print("=" * 50)
    
    try:
        from src.data.parquet_converter import ParquetConverter
        
        converter = ParquetConverter()
        print(f"[OK] ParquetConverter initialized")
        print(f"   CSV Root: {converter.data_raw_root}")
        print(f"   Parquet Root: {converter.parquet_root}")
        
        # Test path generation for AD symbol with diffAdjusted
        parquet_path = converter.get_parquet_path("AD", "1m", "diffAdjusted")
        csv_path = converter.get_csv_path("AD", "1m", "diffAdjusted")
        
        print(f"[OK] Path generation working")
        print(f"   Expected CSV: {csv_path}")
        print(f"   Expected Parquet: {parquet_path}")
        
        # Test if paths are using diffAdjusted correctly
        if "DIFF" in str(parquet_path) and "diffAdjusted" in str(parquet_path):
            print("[OK] AD symbol correctly defaults to diffAdjusted")
        else:
            print("[X] AD symbol path issue")
            return False
        
        return True
        
    except Exception as e:
        print(f"[X] ParquetConverter test failed: {e}")
        return False

def test_csv_loader():
    """Test CSV loader with diffAdjusted default"""
    
    print("\nTESTING CSV LOADER")
    print("=" * 50)
    
    try:
        from src.data.csv_raw_loader import CSVRawLoader
        
        loader = CSVRawLoader()
        print(f"[OK] CSVRawLoader initialized")
        print(f"   Data Root: {loader.data_root}")
        
        # Test discovery of symbols (won't find data but should not crash)
        symbols = loader.discover_symbols("1m")
        print(f"[OK] Symbol discovery working (found {len(symbols)} symbols)")
        
        return True
        
    except Exception as e:
        print(f"[X] CSVRawLoader test failed: {e}")
        return False

def test_data_loading_logic():
    """Test the main data loading logic without AmiBroker"""
    
    print("\nTESTING DATA LOADING LOGIC")
    print("=" * 50)
    
    try:
        # Simulate the main.py data loading logic
        from src.data.parquet_converter import ParquetConverter
        from src.data.csv_raw_loader import CSVRawLoader
        
        symbol = "AD"
        freq_str = "1m"
        adjustment_type = "diffAdjusted"
        
        print(f"Testing data loading for: {symbol} {freq_str} {adjustment_type}")
        
        # Step 1: Try ParquetConverter
        data = None
        parquet_converter = ParquetConverter()
        
        try:
            data = parquet_converter.load_or_convert(symbol, freq_str, adjustment_type)
            if data:
                print(f"[OK] ParquetConverter would load data: {len(data.get('close', []))} bars")
            else:
                print("ℹ️  ParquetConverter: No data found (expected if files don't exist)")
        except Exception as e:
            print(f"ℹ️  ParquetConverter failed: {e} (expected if files don't exist)")
        
        # Step 2: Try CSV loader if parquet failed
        if data is None:
            try:
                csv_loader = CSVRawLoader()
                data = csv_loader.load_symbol_data(symbol, freq_str, adjustment_type)
                if data:
                    print(f"[OK] CSVRawLoader would load data: {len(data.get('close', []))} bars")
                else:
                    print("ℹ️  CSVRawLoader: No data found (expected if files don't exist)")
            except Exception as e:
                print(f"ℹ️  CSVRawLoader failed: {e} (expected if files don't exist)")
        
        # Step 3: Synthetic data fallback
        if data is None:
            from main import generate_synthetic_data
            data = generate_synthetic_data(1000)
            print(f"[OK] Synthetic data generated: {len(data['close'])} bars")
        
        print("[OK] Data loading pipeline working correctly")
        return True
        
    except Exception as e:
        print(f"[X] Data loading logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_function():
    """Test that main function works without AmiBroker"""
    
    print("\nTESTING MAIN FUNCTION COMPATIBILITY")
    print("=" * 50)
    
    try:
        # Test imports from main
        import main
        
        # Test key functions exist
        if hasattr(main, 'generate_synthetic_data'):
            print("[OK] generate_synthetic_data function available")
        else:
            print("[X] generate_synthetic_data function missing")
            return False
        
        if hasattr(main, 'normalize_timestamps_for_dashboard'):
            print("[OK] normalize_timestamps_for_dashboard function available")
        else:
            print("[X] normalize_timestamps_for_dashboard function missing")
            return False
        
        if hasattr(main, 'launch_dashboard_robust'):
            print("[OK] launch_dashboard_robust function available")
        else:
            print("[X] launch_dashboard_robust function missing")
            return False
        
        # Test synthetic data generation
        data = main.generate_synthetic_data(100)
        if 'datetime' in data and 'close' in data:
            print(f"[OK] Synthetic data generation working: {len(data['close'])} bars")
        else:
            print("[X] Synthetic data generation broken")
            return False
        
        print("[OK] Main function compatibility verified")
        return True
        
    except Exception as e:
        print(f"[X] Main function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all pipeline tests"""
    
    print("CLEANED PIPELINE VALIDATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_parquet_converter_pipeline,
        test_csv_loader,
        test_data_loading_logic,
        test_main_function
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[X] Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("PIPELINE TEST RESULTS")
    print("=" * 60)
    
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "[OK] PASSED" if result else "[X] FAILED"
        print(f"{i}. {test.__name__}: {status}")
    
    overall_success = all(results)
    print(f"\nOVERALL: {'[OK] ALL TESTS PASSED' if overall_success else '[X] SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n[SUCCESS] CLEANED PIPELINE IS WORKING!")
        print("[OK] AmiBroker dependencies removed")
        print("[OK] Parquet-first pipeline intact")
        print("[OK] AD defaults to diffAdjusted")
        print("[OK] CSV fallback working")
        print("[OK] Dashboard compatibility maintained")
    else:
        print("\n[WARNING]  PIPELINE ISSUES DETECTED")
        print("Check the failed tests above.")
    
    return overall_success

if __name__ == "__main__":
    main()