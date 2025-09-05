"""
Enhanced main.py with mplfinance integration
Provides both PyQtGraph (real-time) and mplfinance (publication-quality) dashboards
Handles 7M+ datapoints efficiently through viewport management
"""
import numpy as np
import yaml
import argparse
import time
import warnings
import importlib
import inspect
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

warnings.filterwarnings('ignore')

from src.data.parquet_loader import ParquetLoader
from src.data.parquet_converter import ParquetConverter
from src.data.array_validator import ArrayValidator
from src.backtest.vbt_engine import VectorBTEngine
from strategies.base_strategy import BaseStrategy
import numpy as np  # Required for optimized decimation

# Dashboard imports - ensure src path is available for sub-imports
import sys
import os
from pathlib import Path

# Add src directory to Python path BEFORE any dashboard imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Dashboard imports with improved error handling
try:
    from src.dashboard.dashboard_manager import launch_dashboard, get_dashboard_manager
    from src.dashboard.hybrid_mplfinance_dashboard import HybridDashboardManager
    DASHBOARD_AVAILABLE = True
    MPLFINANCE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Dashboard imports failed: {e}")
    DASHBOARD_AVAILABLE = False
    MPLFINANCE_AVAILABLE = False


def load_strategy(strategy_name: str) -> BaseStrategy:
    """
    Dynamically load a strategy from the strategies folder.
    Supports both module names and specific class names.
    
    Args:
        strategy_name: Name of the strategy file or specific class name
        
    Returns:
        BaseStrategy: An instance of the loaded strategy
    """
    
    # Try to find the strategy file
    strategies_dir = Path("strategies")
    strategy_files = list(strategies_dir.glob("*.py"))
    strategy_files = [f for f in strategy_files if f.name != "__init__.py"]
    
    # Remove .py extension if provided
    if strategy_name.endswith('.py'):
        strategy_name = strategy_name[:-3]
    
    strategy_module = None
    strategy_class = None
    
    # First, try to import the module directly
    try:
        module_name = f"strategies.{strategy_name}"
        strategy_module = importlib.import_module(module_name)
        print(f"Successfully imported strategy module: {module_name}")
    except ImportError:
        # If direct import fails, search through all strategy files
        for strategy_file in strategy_files:
            module_name = f"strategies.{strategy_file.stem}"
            try:
                temp_module = importlib.import_module(module_name)
                # Check if this module contains a class with the desired name
                for name, obj in inspect.getmembers(temp_module, inspect.isclass):
                    if name.lower() == strategy_name.lower() and issubclass(obj, BaseStrategy):
                        strategy_module = temp_module
                        strategy_class = obj
                        print(f"Found strategy class {name} in module {module_name}")
                        break
                if strategy_module:
                    break
            except ImportError:
                continue
    
    if not strategy_module:
        available_strategies = [f.stem for f in strategy_files]
        raise ValueError(f"Strategy '{strategy_name}' not found. Available strategies: {available_strategies}")
    
    # If we haven't found a specific class, look for strategy classes in the module
    if not strategy_class:
        strategy_classes = []
        for name, obj in inspect.getmembers(strategy_module, inspect.isclass):
            if issubclass(obj, BaseStrategy) and obj != BaseStrategy:
                strategy_classes.append((name, obj))
        
        if not strategy_classes:
            raise ValueError(f"No strategy classes found in module '{strategy_module.__name__}'")
        
        if len(strategy_classes) == 1:
            strategy_class = strategy_classes[0][1]
            print(f"Using strategy class: {strategy_classes[0][0]}")
        else:
            # Multiple classes found, try to find one that matches the module name
            for name, obj in strategy_classes:
                if name.lower() == strategy_name.lower():
                    strategy_class = obj
                    break
            
            if not strategy_class:
                class_names = [name for name, _ in strategy_classes]
                raise ValueError(f"Multiple strategy classes found: {class_names}. Please specify the exact class name.")
    
    # Create and return an instance of the strategy
    try:
        strategy_instance = strategy_class()
        print(f"Successfully created strategy instance: {strategy_class.__name__}")
        return strategy_instance
    except Exception as e:
        raise ValueError(f"Failed to create strategy instance: {e}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def run_backtest(config: Dict[str, Any], 
                strategy: BaseStrategy,
                dashboard_type: str = 'pyqtgraph') -> Dict[str, Any]:
    """
    Run backtest with specified configuration and strategy
    
    Args:
        config: Configuration dictionary
        strategy: Strategy instance to use
        dashboard_type: Type of dashboard ('pyqtgraph', 'mplfinance', 'hybrid')
        
    Returns:
        Dict containing backtest results
    """
    
    print(f"\n=== ENHANCED BACKTEST EXECUTION (Dashboard: {dashboard_type}) ===")
    
    # Load data
    print(f"Loading data from: {config['data']['source']}")
    loader = ParquetLoader()
    
    start_time = time.time()
    
    if config['data']['source'].endswith('.parquet'):
        # Single file
        data = loader.load_parquet_file(config['data']['source'])
    else:
        # Directory of files
        data = loader.load_parquet_directory(config['data']['source'])
    
    load_time = time.time() - start_time
    print(f"Data loaded: {len(data):,} bars in {load_time:.2f} seconds")
    
    # Validate data
    validator = ArrayValidator()
    data = validator.validate_and_clean(data)
    print(f"Data validated and cleaned: {len(data):,} bars")
    
    # Run backtest
    print(f"Running backtest with strategy: {strategy.__class__.__name__}")
    
    engine = VectorBTEngine()
    
    backtest_start = time.time()
    results = engine.run_backtest(data, strategy)
    backtest_time = time.time() - backtest_start
    
    print(f"Backtest completed in {backtest_time:.2f} seconds")
    
    # Display basic results
    if 'portfolio_stats' in results:
        stats = results['portfolio_stats']
        print(f"\n=== BACKTEST RESULTS ===")
        print(f"Total Return: {stats.get('total_return', 'N/A'):.2%}")
        print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 'N/A'):.2f}")
        print(f"Max Drawdown: {stats.get('max_drawdown', 'N/A'):.2%}")
        print(f"Total Trades: {stats.get('total_trades', 'N/A')}")
    
    # Launch dashboard if available
    if DASHBOARD_AVAILABLE and config.get('dashboard', {}).get('enabled', True):
        print(f"\n=== LAUNCHING {dashboard_type.upper()} DASHBOARD ===")
        
        try:
            # Prepare data for dashboard
            price_data = {
                'open': data['open'].values if 'open' in data.columns else data['close'].values,
                'high': data['high'].values if 'high' in data.columns else data['close'].values,
                'low': data['low'].values if 'low' in data.columns else data['close'].values,
                'close': data['close'].values,
                'volume': data['volume'].values if 'volume' in data.columns else np.ones(len(data)),
                'datetime': data.index if isinstance(data.index, pd.DatetimeIndex) else pd.date_range('2020-01-01', periods=len(data), freq='1min')
            }
            
            trade_data = results.get('trades')
            portfolio_data = results.get('portfolio_stats')
            
            if dashboard_type == 'hybrid' and MPLFINANCE_AVAILABLE:
                # Use hybrid dashboard with both PyQtGraph and mplfinance
                launch_hybrid_dashboard(price_data, trade_data, portfolio_data)
            elif dashboard_type == 'mplfinance' and MPLFINANCE_AVAILABLE:
                # Use mplfinance-only dashboard
                launch_mplfinance_dashboard(price_data, trade_data, portfolio_data)
            else:
                # Use original PyQtGraph dashboard
                launch_pyqtgraph_dashboard(price_data, trade_data, portfolio_data)
                
        except Exception as e:
            print(f"Dashboard launch failed: {e}")
            print("Continuing without dashboard...")
    
    return results

def launch_pyqtgraph_dashboard(price_data, trade_data, portfolio_data):
    """Launch original PyQtGraph dashboard"""
    
    print("Launching PyQtGraph dashboard...")
    
    async def launch_async():
        dashboard = await launch_dashboard(price_data, trade_data, portfolio_data)
        if dashboard:
            print("PyQtGraph dashboard launched successfully")
            # Keep dashboard running
            while dashboard.is_dashboard_running():
                await asyncio.sleep(1)
        return dashboard
    
    # Run in event loop
    try:
        asyncio.run(launch_async())
    except KeyboardInterrupt:
        print("Dashboard closed by user")

def launch_mplfinance_dashboard(price_data, trade_data, portfolio_data):
    """Launch mplfinance-only dashboard"""
    
    print("Launching mplfinance dashboard...")
    
    try:
        from src.dashboard.mplfinance_dashboard_optimized import OptimizedMplfinanceDashboard
        from PyQt5 import QtWidgets
        
        app = QtWidgets.QApplication([])
        dashboard = OptimizedMplfinanceDashboard()
        
        # Convert price data to DataFrame
        df = pd.DataFrame({
            'Open': price_data['open'],
            'High': price_data['high'],
            'Low': price_data['low'],
            'Close': price_data['close'],
            'Volume': price_data['volume']
        }, index=price_data['datetime'])
        
        if dashboard.load_data({'close': price_data['close'], 'open': price_data['open'], 
                               'high': price_data['high'], 'low': price_data['low'],
                               'volume': price_data['volume'], 'datetime': price_data['datetime']}, 
                              trade_data):
            dashboard.show()
            print("mplfinance dashboard launched successfully")
            app.exec_()
        else:
            print("Failed to load data into mplfinance dashboard")
            
    except Exception as e:
        print(f"mplfinance dashboard launch failed: {e}")

def launch_hybrid_dashboard(price_data, trade_data, portfolio_data):
    """Launch hybrid dashboard with both PyQtGraph and mplfinance"""
    
    print("Launching hybrid dashboard...")
    
    try:
        from PyQt5 import QtWidgets
        
        app = QtWidgets.QApplication([])
        dashboard = HybridDashboardManager()
        
        # Initialize and create window
        if dashboard.initialize_qt_app():
            dashboard.create_main_window()
            
            # Load data asynchronously
            async def load_and_show():
                await dashboard.load_backtest_data(price_data, trade_data, portfolio_data)
                dashboard.show()
                return dashboard
            
            # Run async data loading
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                dashboard = loop.run_until_complete(load_and_show())
                print("Hybrid dashboard launched successfully")
                print("- PyQtGraph tab: Real-time interactive charts")
                print("- mplfinance tab: Publication-quality charts with 7M+ bar support")
                
                app.exec_()
            finally:
                loop.close()
        else:
            print("Failed to initialize hybrid dashboard")
            
    except Exception as e:
        print(f"Hybrid dashboard launch failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Enhanced Trading Backtest with mplfinance Integration')
    parser.add_argument('--config', '-c', default='config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--strategy', '-s', required=True,
                       help='Strategy name or class')
    parser.add_argument('--dashboard', '-d', choices=['pyqtgraph', 'mplfinance', 'hybrid'], 
                       default='hybrid', help='Dashboard type to use')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Run without dashboard')
    
    args = parser.parse_args()
    
    try:
        print("ENHANCED TRADING SYSTEM WITH MPLFINANCE")
        print("=" * 50)
        print(f"Configuration: {args.config}")
        print(f"Strategy: {args.strategy}")
        print(f"Dashboard: {args.dashboard}")
        
        # Load configuration
        config = load_config(args.config)
        
        # Override dashboard setting if --no-dashboard specified
        if args.no_dashboard:
            config.setdefault('dashboard', {})['enabled'] = False
        
        # Load strategy
        strategy = load_strategy(args.strategy)
        
        # Run backtest
        results = run_backtest(config, strategy, args.dashboard)
        
        print(f"\n=== EXECUTION COMPLETE ===")
        print("Results saved and dashboard launched (if enabled)")
        
        return results
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        return None
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()