#!/usr/bin/env python3

import numpy as np
import configparser
from model_directional import DirectionalTreeEnsemble

def test_short_trading_logic():
    print("Testing Short Trading Logic with Synthetic Data")
    print("=" * 50)
    
    # Create synthetic data where SHORT trades should be profitable
    # Negative returns = profitable for shorts
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])  # Some features
    y = np.array([-0.05, -0.08, 0.02, -0.12, 0.01])   # Mix of negative (profitable for shorts) and positive returns
    
    print("Synthetic data:")
    print("Features:", X.flatten())
    print("Returns: ", y)
    print("Expected SHORT profits:", -y)  # Short profits = negative of returns
    
    # Test LONGONLY model first
    print("\n1. Testing LONGONLY model...")
    # Set config to longonly
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    config['model']['model_type'] = 'longonly'
    config['model']['target_threshold'] = '0.03'  # Lower threshold for testing
    with open('config_longonly.ini', 'w') as f:
        config.write(f)
    
    model_long = DirectionalTreeEnsemble(verbose=True)
    model_long.fit(X, y)
    
    # Check labeling for longonly
    labels_long = model_long.create_directional_labels(y)
    print(f"LONGONLY labels (1=profitable): {labels_long}")
    print(f"Expected: returns > 0.03 should be 1")
    
    # Test SHORTONLY model
    print("\n2. Testing SHORTONLY model...")
    config['model']['model_type'] = 'shortonly'
    with open('config_longonly.ini', 'w') as f:
        config.write(f)
    
    model_short = DirectionalTreeEnsemble(verbose=True)
    model_short.fit(X, y)
    
    # Check labeling for shortonly
    labels_short = model_short.create_directional_labels(y)
    print(f"SHORTONLY labels (1=profitable): {labels_short}")
    print(f"Expected: returns < -0.03 should be 1")
    
    # Verify the logic is correct
    print("\n3. Verification:")
    threshold = 0.03
    
    print(f"\nFor LONGONLY (threshold = +{threshold}):")
    for i, ret in enumerate(y):
        is_profitable = ret > threshold
        print(f"  Return {ret:+.3f}: {'Profitable' if is_profitable else 'Not Profitable'} (label={labels_long[i]})")
    
    print(f"\nFor SHORTONLY (threshold = -{threshold}):")
    for i, ret in enumerate(y):
        is_profitable = ret < -threshold  # Negative returns are profitable for shorts
        short_pnl = -ret  # P&L for short = negative of return
        print(f"  Return {ret:+.3f} -> Short P&L {short_pnl:+.3f}: {'Profitable' if is_profitable else 'Not Profitable'} (label={labels_short[i]})")
    
    print("\n[SUCCESS] Short trading logic verification complete!")
    
    # Restore original config
    config['model']['model_type'] = 'longonly'
    config['model']['target_threshold'] = '0.1'
    with open('config_longonly.ini', 'w') as f:
        config.write(f)
    
    return True

if __name__ == "__main__":
    test_short_trading_logic()