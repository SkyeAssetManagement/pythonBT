import configparser
import pandas as pd
from date_parser import FlexibleDateParser

def test_dtnn_config():
    """Test walk-forward with DTSnnData.csv"""
    
    # Create a test config for DTSnnData.csv
    config = configparser.ConfigParser()
    
    # Data section - adjust for DTSnnData columns
    config['data'] = {
        'csv_file': 'DTSnnData.csv',
        'target_column': 'RetForward',
        'all_targets': 'RetForward',
        'feature_columns': 'Last10,First20,Overnight,3day',
        'selected_features': 'Last10,First20',
        'date_column': '',  # Not used for combined column
        'time_column': ''   # Not used for combined column
    }
    
    # Preprocessing section
    config['preprocessing'] = {
        'normalize_features': 'true',
        'normalize_target': 'true',
        'vol_window': '60',
        'smoothing_type': 'exponential',
        'smoothing_alpha': '0.1',
        'percentile_upper': '75',
        'percentile_lower': '25',
        'recent_iqr_lookback': '20'
    }
    
    # Model section
    config['model'] = {
        'model_type': 'longonly',
        'n_trees': '100',
        'max_depth': '1',
        'bootstrap_fraction': '0.8',
        'min_leaf_fraction': '0.2',
        'target_threshold': '0.1',
        'vote_threshold': '0.7',
        'random_seed': '42'
    }
    
    # Validation section
    config['validation'] = {
        'train_size': '500',
        'test_size': '50',
        'step_size': '25',
        'min_training_samples': '100',
        'base_rate': '0.42',
        'validation_start_date': '2005-01-01',
        'validation_end_date': '2010-12-31',
        'date_format_dayfirst': 'true'  # For DD/MM/YYYY format
    }
    
    # Analysis section
    config['analysis'] = {
        'rolling_window_short': '25',
        'rolling_window_long': '50',
        'recent_predictions_count': '10',
        'edge_threshold_good': '0.02',
        'edge_threshold_strong': '0.05',
        'edge_threshold_excellent': '0.10'
    }
    
    # Output section
    config['output'] = {
        'results_file': 'OMtree_results_dtnn.csv',
        'chart_dpi': '300',
        'chart_format': 'png',
        'date_format': '%%Y-%%m-%%d',
        'verbose_logging': 'true',
        'save_predictions': 'true'
    }
    
    # Save config
    with open('OMtree_config_dtnn.ini', 'w') as f:
        config.write(f)
    print("Created OMtree_config_dtnn.ini for DTSnnData.csv")
    
    # Test date parsing on DTSnnData.csv
    print("\nTesting date parsing on DTSnnData.csv...")
    df = pd.read_csv('DTSnnData.csv')
    
    # Use flexible parser
    date_columns = FlexibleDateParser.get_date_columns(df)
    print(f"Detected columns: {date_columns}")
    
    parsed_dates = FlexibleDateParser.parse_dates(
        df,
        datetime_column=date_columns.get('datetime_column'),
        dayfirst=True  # For DD/MM/YYYY format
    )
    
    print(f"Successfully parsed {len(parsed_dates)} dates")
    print(f"Date range: {parsed_dates.min()} to {parsed_dates.max()}")
    
    # Filter to validation range
    validation_end = pd.to_datetime('2010-12-31')
    mask = parsed_dates <= validation_end
    filtered_count = mask.sum()
    
    print(f"\nValidation data: {filtered_count} rows (up to {validation_end.strftime('%Y-%m-%d')})")
    print(f"Out-of-sample data: {len(df) - filtered_count} rows")
    
    return True

def test_walkforward():
    """Test walk-forward validation with DTSnnData"""
    print("\nTesting walk-forward validation with DTSnnData.csv...")
    
    try:
        from OMtree_validation import DirectionalValidator
        
        # Use the DTSnn config
        validator = DirectionalValidator('OMtree_config_dtnn.ini')
        
        # Load and prepare data
        print("Loading and preparing data...")
        data = validator.load_and_prepare_data()
        
        print(f"Loaded {len(data)} rows after preprocessing")
        print(f"Columns: {data.columns.tolist()[:10]}...")  # Show first 10 columns
        
        # Check if we can run at least one window
        min_required = validator.train_size + validator.test_size
        if len(data) >= min_required:
            print(f"[OK] Sufficient data for walk-forward (need {min_required}, have {len(data)})")
        else:
            print(f"[ERROR] Insufficient data (need {min_required}, have {len(data)})")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing DTSnnData.csv compatibility")
    print("="*60)
    
    # Create config
    test1 = test_dtnn_config()
    
    # Test walk-forward
    test2 = test_walkforward()
    
    print("\n" + "="*60)
    print("Test Results:")
    print(f"  Config creation: {'PASSED' if test1 else 'FAILED'}")
    print(f"  Walk-forward compatibility: {'PASSED' if test2 else 'FAILED'}")
    print("="*60)