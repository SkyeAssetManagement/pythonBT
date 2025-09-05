"""
Test PermuteAlpha direction setting (shortonly vs longonly)
"""

import configparser
import tempfile
import os

print("="*60)
print("PERMUTE DIRECTION TEST")
print("="*60)

# Test parameters
ticker = "ES"
target = "Ret_fwd3hr"
hour = "10"
direction = "shortonly"  # Testing shortonly specifically
features = ['Ret_0-1hr', 'Ret_1-2hr', 'Ret_2-4hr', 'Ret_4-8hr']

print(f"\nTest Configuration:")
print(f"  Ticker: {ticker}")
print(f"  Target: {target}")
print(f"  Hour: {hour}")
print(f"  Direction: {direction}")
print(f"  Features: {features}")

# Create config (simulating what PermuteAlpha does)
config = configparser.ConfigParser()
config.read('OMtree_config.ini')

print(f"\nOriginal model_type in config: {config['model']['model_type']}")

# Create temp config
temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
temp_path = temp_config.name

# Update for this combination (same as PermuteAlpha)
config['data']['target_column'] = target
config['data']['selected_features'] = ','.join(features)
config['model']['model_type'] = direction  # Should be 'shortonly'
config['data']['ticker_filter'] = ticker
config['data']['hour_filter'] = hour

print(f"Setting model_type to: {direction}")

# Write config
config.write(temp_config)
temp_config.close()

print(f"\nCreated temp config: {temp_path}")

# Read back the config to verify
verify_config = configparser.ConfigParser()
verify_config.read(temp_path)

print(f"\nVerification:")
print(f"  model_type in temp config: {verify_config['model']['model_type']}")
print(f"  target_column: {verify_config['data']['target_column']}")
print(f"  ticker_filter: {verify_config['data']['ticker_filter']}")
print(f"  hour_filter: {verify_config['data']['hour_filter']}")

# Now create a validator and check what model type it uses
from OMtree_validation import DirectionalValidator

try:
    print(f"\nCreating validator with temp config...")
    validator = DirectionalValidator(temp_path)
    
    print(f"  Validator model_type: {validator.model_type}")
    
    # Check the config object in validator
    if hasattr(validator, 'config'):
        print(f"  Validator config['model']['model_type']: {validator.config['model']['model_type']}")
    
    # Load and prepare data to see preprocessing messages
    print(f"\nLoading and preparing data...")
    data = validator.load_and_prepare_data()
    
    if data is not None:
        print(f"  Data loaded: {len(data)} rows")
        
        # Check if the model created would be correct type
        from OMtree_model import DirectionalTreeEnsemble
        
        print(f"\nCreating model with temp config...")
        model = DirectionalTreeEnsemble(config_path=temp_path, verbose=False)
        
        print(f"  Model model_type: {model.model_type}")
        
        if model.model_type == direction:
            print(f"\n[SUCCESS] Model correctly created as {direction}")
        else:
            print(f"\n[ERROR] Model created as {model.model_type} instead of {direction}")
    
except Exception as e:
    print(f"\n[ERROR] Test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # Clean up temp file
    if os.path.exists(temp_path):
        os.unlink(temp_path)
        print(f"\nCleaned up temp config")

print("="*60)