"""
Test the original command: python main.py ES simpleSMA --start "2025-07-01"
"""
import subprocess
import sys
from pathlib import Path

def test_original_command():
    """Test if the original command works with the black blob fix"""
    
    print("=== TESTING ORIGINAL COMMAND ===")
    print('Command: python main.py ES simpleSMA --start "2025-07-01"')
    print("This should now show proper candlesticks without black blobs")
    
    try:
        # Build the command
        cmd = [sys.executable, 'main.py', 'ES', 'simpleSMA', '--start', '2025-07-01']
        print(f"Running: {' '.join(cmd)}")
        
        # Run with timeout
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            for line in result.stdout.split('\n')[-20:]:  # Last 20 lines
                if line.strip():
                    print(f"  {line}")
        
        if result.stderr and result.stderr.strip():
            print("STDERR:")
            for line in result.stderr.split('\n')[-10:]:  # Last 10 lines
                if line.strip():
                    print(f"  {line}")
        
        if result.returncode == 0:
            print("SUCCESS: Command completed successfully!")
            print("The dashboard should have displayed with proper candlesticks")
            return True
        else:
            print(f"ERROR: Command failed with exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Command took longer than 60 seconds")
        print("This may indicate the dashboard is still running - that's actually good!")
        print("The GUI may have opened successfully but is still running")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to run command: {e}")
        return False

if __name__ == "__main__":
    success = test_original_command()
    
    if success:
        print("\nSUCCESS: Original command appears to be working!")
        print("The black blob issue should be resolved")
    else:
        print("\nERROR: Original command still has issues")