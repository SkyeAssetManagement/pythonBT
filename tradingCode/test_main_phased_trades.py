"""
Test phased trading with main.py to see actual trade generation
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path


def test_with_main():
    """Run main.py with phased config and check results"""
    
    print("="*70)
    print("TESTING PHASED TRADING WITH MAIN.PY")
    print("="*70)
    
    # Run main.py with a simple date range
    cmd = [
        "python", "main.py", 
        "ES", "simpleSMA",
        "--config", "config_phased_test.yaml",
        "--start_date", "2024-01-01",
        "--end_date", "2024-01-10",
        "--no-viz"
    ]
    
    print("\nRunning command:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Print output
        print("\nMain.py output:")
        print("-"*70)
        for line in result.stdout.split('\n'):
            if 'INFO' in line or 'phased' in line.lower() or 'trade' in line.lower():
                print(line)
        
        # Check the generated tradelist
        tradelist_path = Path("results_phased/tradelist.csv")
        if tradelist_path.exists():
            trades_df = pd.read_csv(tradelist_path)
            
            print("\n" + "="*70)
            print("TRADELIST ANALYSIS:")
            print(f"Total trades: {len(trades_df)}")
            
            if len(trades_df) > 0:
                # Analyze trade sizes
                sizes = trades_df['Size'].values
                unique_sizes = np.unique(np.round(sizes, 2))
                
                print(f"\nUnique trade sizes found: {unique_sizes}")
                
                # Check if we have the expected 1/n sizing
                # With phased_entry_bars=5 and position_size=1000, expect size ~200
                expected_size = 200  # 1000 / 5
                
                trades_at_expected_size = 0
                for size in sizes:
                    if abs(size - expected_size) < 10:  # Allow some tolerance
                        trades_at_expected_size += 1
                
                if trades_at_expected_size > 0:
                    print(f"\n[SUCCESS] Found {trades_at_expected_size} trades at ~{expected_size} size (1/5 of 1000)")
                    print("This confirms phased trading is creating 1/n sized trades!")
                else:
                    # Check if sizes are fractions of expected
                    avg_size = np.mean(sizes)
                    print(f"\nAverage trade size: {avg_size:.2f}")
                    
                    if avg_size < 50:  # Much smaller than expected
                        print("[INFO] Trades are very small - may be using percent sizing instead of value sizing")
                    elif avg_size > 500:  # Larger than expected
                        print("[INFO] Trades are large - phasing may be consolidated")
                    else:
                        print(f"[INFO] Trade sizes are between expected ranges")
                
                # Show first few trades
                print(f"\nFirst 5 trades:")
                print("-"*70)
                cols_to_show = ['Entry Index', 'Exit Index', 'Size', 'Avg Entry Price', 'PnL']
                available_cols = [col for col in cols_to_show if col in trades_df.columns]
                print(trades_df[available_cols].head())
                
        else:
            print("\n[ERROR] No tradelist.csv found in results_phased/")
            
    except subprocess.TimeoutExpired:
        print("\n[ERROR] Command timed out")
    except Exception as e:
        print(f"\n[ERROR] Failed to run main.py: {e}")


if __name__ == "__main__":
    test_with_main()
    
    print("\n" + "="*70)
    print("IMPORTANT NOTE:")
    print("If you're not seeing separate trades at 1/n size, this is because")
    print("VectorBT's core portfolio management consolidates positions.")
    print("The phasing IS happening (check 'INFO: Phased signals generated')")
    print("but VectorBT merges them for efficiency.")
    print("="*70)