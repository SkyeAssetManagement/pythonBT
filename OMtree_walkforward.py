#!/usr/bin/env python
"""
Walk-forward validation runner
This script runs the walk-forward validation from the correct location
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the actual walkforward script
from src import OMtree_walkforward

# The script runs on import (it's not wrapped in if __name__ == '__main__')