#!/usr/bin/env python
"""
OMtree Trading System - Setup Script
Automated installation and configuration
"""

import os
import sys
import subprocess
import platform

def print_header():
    print("=" * 60)
    print("OMtree Trading System - Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERROR: Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"OK - Python {version.major}.{version.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("OK - All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directory structure...")
    directories = [
        "data",
        "output",
        "logs",
        "configs",
        ".claude"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"OK - Created {directory}/")
        else:
            print(f"  {directory}/ already exists")
    return True

def generate_sample_data():
    """Generate sample data files"""
    print("\nGenerating sample data...")
    try:
        # Check if sample data already exists
        if os.path.exists("data/sample_trading_data.csv"):
            response = input("Sample data already exists. Regenerate? (y/n): ")
            if response.lower() != 'y':
                print("  Keeping existing sample data")
                return True
        
        # Generate new sample data
        os.chdir("data")
        subprocess.check_call([sys.executable, "sample_data_generator.py"])
        os.chdir("..")
        print("OK - Sample data generated")
        return True
    except Exception as e:
        print(f"Warning: Could not generate sample data: {e}")
        print("  You can generate it manually later")
        return True

def update_config():
    """Update configuration file with correct paths"""
    print("\nUpdating configuration...")
    
    # Get current working directory
    cwd = os.getcwd().replace('\\', '/')
    
    # Read current config
    config_file = "OMtree_config.ini"
    if not os.path.exists(config_file):
        print(f"ERROR: {config_file} not found")
        return False
    
    # Create example config with local paths
    example_config = f"""# Example configuration with local paths
# Copy this to OMtree_config.ini and adjust as needed

[data]
csv_file = {cwd}/data/sample_trading_data.csv
target_column = Ret_fwd6hr
all_targets = Ret_fwd1hr,Ret_fwd3hr,Ret_fwd6hr,Ret_fwd12hr,Ret_fwd1d
feature_columns = PIR_64-128hr,PIR_32-64hr,PIR_16-32hr,PIR_8-16hr,PIR_4-8hr,PIR_2-4hr,PIR_1-2hr,PIR_0-1hr
selected_features = PIR_64-128hr,PIR_32-64hr,PIR_16-32hr,PIR_8-16hr,PIR_4-8hr,PIR_2-4hr,PIR_1-2hr,PIR_0-1hr
date_column = Date
time_column = Time
hour_filter = 10
ticker_filter = NQ
"""
    
    # Save example config
    with open("configs/example_config.ini", "w") as f:
        f.write(example_config)
    
    print(f"OK - Created example config at configs/example_config.ini")
    print(f"  Update OMtree_config.ini with your data path:")
    print(f"  csv_file = {cwd}/data/your_data.csv")
    return True

def test_import():
    """Test if main modules can be imported"""
    print("\nTesting module imports...")
    try:
        import numpy
        import pandas
        import sklearn
        import matplotlib
        from PIL import Image
        print("OK - All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import module: {e}")
        return False

def create_shortcuts():
    """Create convenient shortcuts/batch files"""
    print("\nCreating shortcuts...")
    
    if platform.system() == "Windows":
        # Windows batch files
        with open("run_gui.bat", "w") as f:
            f.write("@echo off\n")
            f.write("python OMtree_gui.py\n")
            f.write("pause\n")
        
        with open("run_walkforward.bat", "w") as f:
            f.write("@echo off\n")
            f.write("python OMtree_walkforward.py\n")
            f.write("pause\n")
        
        print("OK - Created run_gui.bat and run_walkforward.bat")
    else:
        # Unix shell scripts
        with open("run_gui.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("python3 OMtree_gui.py\n")
        
        with open("run_walkforward.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("python3 OMtree_walkforward.py\n")
        
        os.chmod("run_gui.sh", 0o755)
        os.chmod("run_walkforward.sh", 0o755)
        
        print("OK - Created run_gui.sh and run_walkforward.sh")
    
    return True

def main():
    """Main setup routine"""
    print_header()
    
    # Track setup status
    success = True
    
    # Run setup steps
    if not check_python_version():
        success = False
    
    if success and not install_requirements():
        success = False
    
    if not create_directories():
        success = False
    
    if not generate_sample_data():
        pass  # Non-critical, continue
    
    if not update_config():
        pass  # Non-critical, continue
    
    if success and not test_import():
        success = False
    
    if not create_shortcuts():
        pass  # Non-critical, continue
    
    # Print summary
    print("\n" + "=" * 60)
    if success:
        print("OK - Setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and update OMtree_config.ini with your data path")
        print("2. Run the GUI: python OMtree_gui.py")
        print("3. Or run walk-forward: python OMtree_walkforward.py")
        print("\nFor detailed instructions, see HOW-TO-GUIDE.md")
    else:
        print("WARNING: Setup completed with warnings")
        print("\nPlease address any errors above before running the system")
        print("You may need to manually install some packages")
    print("=" * 60)

if __name__ == "__main__":
    main()