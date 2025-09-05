#!/usr/bin/env python3

import configparser

def demo_short_trading():
    print("DIRECTIONAL TRADING SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    print("This codebase now supports both LONG and SHORT trading!")
    print()
    
    print("KEY CHANGES MADE:")
    print("1. Renamed 'up_threshold' -> 'target_threshold'")
    print("2. Added 'model_type' setting: 'longonly' or 'shortonly'")
    print("3. For SHORT trades, target_threshold is multiplied by -1")
    print("4. SHORT trades profit from NEGATIVE returns")
    print()
    
    # Read current config
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    
    print("CURRENT CONFIGURATION:")
    print(f"  model_type: {config['model']['model_type']}")
    print(f"  target_threshold: {config['model']['target_threshold']}")
    print(f"  features: {config['data']['selected_features']}")
    print()
    
    print("HOW TO USE:")
    print()
    
    print("FOR LONG-ONLY TRADING:")
    print("  model_type = longonly")
    print("  target_threshold = 0.1  # Look for returns > +10%")
    print("  -> Model learns to predict POSITIVE returns")
    print("  -> Trades when expecting UP moves")
    print("  -> P&L = actual return (positive = profit)")
    print()
    
    print("FOR SHORT-ONLY TRADING:")
    print("  model_type = shortonly") 
    print("  target_threshold = 0.1   # Look for returns < -10%")
    print("  -> Model learns to predict NEGATIVE returns")
    print("  -> Trades when expecting DOWN moves")
    print("  -> P&L = -actual return (negative return = profit)")
    print()
    
    print("EXAMPLE SCENARIOS:")
    print()
    print("Scenario 1: Market goes UP +5%")
    print("  LONG trade P&L: +5% (profit)")
    print("  SHORT trade P&L: -5% (loss)")
    print()
    print("Scenario 2: Market goes DOWN -8%")
    print("  LONG trade P&L: -8% (loss)")
    print("  SHORT trade P&L: +8% (profit)")
    print()
    
    print("FILES TO USE:")
    print("  main_directional.py       - Run validation")
    print("  walkforward_directional.py - Full analysis")
    print("  model_directional.py      - Core model")
    print("  validation_directional.py - Validation framework")
    print()
    
    print("BACKWARD COMPATIBILITY:")
    print("  walkforward_complete.py   - Updated to use directional model")
    print("  All experiment scripts work with new system")
    print()
    
    print("TO SWITCH TO SHORT TRADING:")
    print("1. Edit config_longonly.ini:")
    print("   model_type = shortonly")
    print("2. Run any of the validation scripts")
    print("3. Model will automatically learn short patterns")

if __name__ == "__main__":
    demo_short_trading()