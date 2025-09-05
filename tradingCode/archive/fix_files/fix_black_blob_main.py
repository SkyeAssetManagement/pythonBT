"""
IMMEDIATE FIX: Replace main.py with mplfinance to solve black blob issue
This is a drop-in replacement that uses your exact same command but with professional charts
"""
import sys
import time
import subprocess
from pathlib import Path

def main():
    """Parse arguments and run with mplfinance dashboard"""
    
    # Get all command line arguments
    args = sys.argv[1:]  # Skip script name
    
    print("=== BLACK BLOB FIX: Using mplfinance Dashboard ===")
    print(f"Original command: python main.py {' '.join(args)}")
    print("This fixes the black blob rendering issue with professional quality charts")
    
    # Add --mplfinance flag if not already present
    if '--mplfinance' not in args:
        args.append('--mplfinance')
        print("Added --mplfinance flag to fix rendering issues")
    
    # Run the enhanced main.py
    try:
        # Build the command
        cmd = [sys.executable, 'main.py'] + args
        print(f"Running: {' '.join(cmd)}")
        
        # Execute with same environment
        result = subprocess.run(cmd, cwd=Path(__file__).parent, 
                              capture_output=False, text=True)
        
        return result.returncode
        
    except Exception as e:
        print(f"Error running enhanced main.py: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()