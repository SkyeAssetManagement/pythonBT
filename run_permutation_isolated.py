"""
Isolated permutation runner
This script runs a single permutation in complete isolation
Called by parallel processing to avoid file conflicts
"""

import sys
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import configparser
import subprocess
import json


def main():
    """Run a single permutation in isolation"""
    
    if len(sys.argv) < 2:
        print("Usage: run_permutation_isolated.py <config_json>")
        sys.exit(1)
    
    # Parse arguments
    config_json = sys.argv[1]
    config_data = json.loads(config_json)
    
    ticker = config_data['ticker']
    target = config_data['target']
    hour = config_data['hour']
    direction = config_data['direction']
    features = config_data['features']
    base_config_path = config_data['base_config']
    output_dir = config_data['output_dir']
    perm_id = config_data['perm_id']
    
    # Create isolated working directory
    work_dir = tempfile.mkdtemp(prefix=f"perm_{perm_id}_")
    original_dir = os.getcwd()
    
    try:
        # Copy necessary files to work directory
        shutil.copy(base_config_path, os.path.join(work_dir, 'config.ini'))
        
        # Copy data files if they exist
        data_files = ['data.csv', 'OMtree_data.csv']
        for data_file in data_files:
            if os.path.exists(data_file):
                shutil.copy(data_file, work_dir)
        
        # Change to work directory
        os.chdir(work_dir)
        
        # Update config
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read('config.ini')
        
        if ticker != 'ALL':
            config['data']['ticker'] = ticker
        if hour != 'ALL':
            config['data']['hour'] = str(hour)
        config['data']['target'] = target
        config['model']['model_type'] = direction
        config['data']['features'] = ','.join(features)
        
        # Write updated config
        with open('config.ini', 'w') as f:
            config.write(f)
        
        # Run walk-forward in this isolated directory
        cmd = [sys.executable, os.path.join(original_dir, 'OMtree_walkforward.py'), 'config.ini']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists('OMtree_results.csv'):
            # Load results
            df = pd.read_csv('OMtree_results.csv')
            
            # Process and save
            output_file = f"{ticker}_{direction}_{target}_{hour}.csv"
            output_path = os.path.join(original_dir, output_dir, output_file)
            
            # Create daily returns format
            daily_returns = pd.DataFrame()
            if 'date' in df.columns:
                daily_returns['Date'] = pd.to_datetime(df['date'])
            if 'target_value' in df.columns:
                daily_returns['Return'] = df['target_value']
            if 'prediction' in df.columns:
                daily_returns['TradeFlag'] = df['prediction'].astype(int)
            
            daily_returns.to_csv(output_path, index=False)
            
            # Return success
            print(json.dumps({'success': True, 'file': output_file}))
        else:
            print(json.dumps({'success': False, 'error': result.stderr[:500]}))
            
    finally:
        # Clean up
        os.chdir(original_dir)
        try:
            shutil.rmtree(work_dir)
        except:
            pass


if __name__ == "__main__":
    main()