#!/usr/bin/env python3

import configparser
from model_directional import DirectionalTreeEnsemble

def test_directional_model():
    print("Testing Directional Model Implementation")
    print("=" * 50)
    
    # Test with longonly first
    print("\n1. Testing LONGONLY model...")
    try:
        model = DirectionalTreeEnsemble(verbose=True)
        info = model.get_model_info()
        print(f"Model type: {info['model_type']}")
        print(f"Target threshold: {info['target_threshold']}")
        print(f"Effective threshold: {info['effective_threshold']}")
        print(f"Direction: {info['direction_name']}")
        print(f"Signal: {info['signal_name']}")
        print("[OK] LONGONLY model creation successful")
    except Exception as e:
        print(f"[FAIL] LONGONLY model failed: {e}")
        return False
    
    # Test with shortonly
    print("\n2. Testing SHORTONLY model...")
    # Temporarily change config to shortonly
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    original_model_type = config['model']['model_type']
    config['model']['model_type'] = 'shortonly'
    with open('config_longonly.ini', 'w') as f:
        config.write(f)
    
    try:
        model = DirectionalTreeEnsemble(verbose=True)
        info = model.get_model_info()
        print(f"Model type: {info['model_type']}")
        print(f"Target threshold: {info['target_threshold']}")
        print(f"Effective threshold: {info['effective_threshold']}")
        print(f"Direction: {info['direction_name']}")
        print(f"Signal: {info['signal_name']}")
        print("[OK] SHORTONLY model creation successful")
    except Exception as e:
        print(f"[FAIL] SHORTONLY model failed: {e}")
        return False
    finally:
        # Restore original config
        config['model']['model_type'] = original_model_type
        with open('config_longonly.ini', 'w') as f:
            config.write(f)
    
    print("\n3. Testing parameter validation...")
    # Test invalid model type
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    config['model']['model_type'] = 'invalid'
    with open('config_longonly.ini', 'w') as f:
        config.write(f)
    
    try:
        model = DirectionalTreeEnsemble(verbose=False)
        print("[FAIL] Invalid model type should have failed")
        return False
    except ValueError as e:
        print(f"[OK] Invalid model type correctly rejected: {e}")
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False
    finally:
        # Restore original config
        config['model']['model_type'] = original_model_type
        with open('config_longonly.ini', 'w') as f:
            config.write(f)
    
    print("\n[SUCCESS] All directional model tests passed!")
    return True

if __name__ == "__main__":
    success = test_directional_model()
    if success:
        print("\nDirectional model implementation is working correctly!")
    else:
        print("\nTests failed - check implementation")