"""
Performance Test for Optimized 5-Minute Range Bars - Windows Compatible

This script runs a complete performance test of all 5 range bar types.
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add module paths
sys.path.append('common')
sys.path.append('raw_price')
sys.path.append('percentage')
sys.path.append('atr')

from common.performance import PerformanceBenchmark, measure_time
from common.data_loader import TickDataLoader, load_tick_data
from raw_price.range_bars_raw import create_raw_range_bars
from percentage.range_bars_percent import create_percentage_range_bars
from atr.range_bars_atr import create_atr_range_bars

class RangeBarPerformanceTest:
    """
    Comprehensive performance test for all range bar types
    """
    
    def __init__(self, csv_file_path: str, output_dir: str = "performance_test_output"):
        self.csv_file_path = Path(csv_file_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.benchmark = PerformanceBenchmark()
        self.results = {}
        
        print("RANGE BAR PERFORMANCE TEST INITIALIZED")
        print("="*60)
        print(f"Input CSV: {self.csv_file_path}")
        print(f"Output Dir: {self.output_dir}")
        print(f"Testing 5 optimized range bar types (all ~5min/bar)")
        
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file size and basic info"""
        if not file_path.exists():
            return {"exists": False}
        
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_mb / 1024
        
        return {
            "exists": True,
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "size_gb": size_gb
        }
    
    def test_csv_to_parquet_conversion(self) -> Dict[str, Any]:
        """Test CSV to Parquet conversion performance"""
        print("\n" + "="*60)
        print("STEP 1: CSV TO PARQUET CONVERSION")
        print("="*60)
        
        csv_info = self.get_file_info(self.csv_file_path)
        if not csv_info["exists"]:
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
        
        print(f"Input CSV size: {csv_info['size_gb']:.2f} GB ({csv_info['size_mb']:.1f} MB)")
        
        # Perform conversion
        parquet_path = self.output_dir / f"{self.csv_file_path.stem}.parquet"
        
        conversion_start = time.time()
        
        with self.benchmark.measure_operation("csv_to_parquet_conversion"):
            loader = TickDataLoader(chunk_size=5_000_000)  # 5M rows per chunk for large file
            result_path = loader.csv_to_parquet_chunked(
                self.csv_file_path,
                parquet_path=parquet_path,
                compression='snappy'
            )
        
        conversion_time = time.time() - conversion_start
        
        # Get parquet file info
        parquet_info = self.get_file_info(Path(result_path))
        
        # Get detailed parquet metadata
        parquet_details = loader.get_parquet_info(result_path)
        
        conversion_results = {
            "parquet_path": result_path,
            "input_size_gb": csv_info['size_gb'],
            "output_size_gb": parquet_info['size_gb'],
            "compression_ratio": csv_info['size_gb'] / parquet_info['size_gb'],
            "total_rows": parquet_details['num_rows'],
            "total_columns": parquet_details['num_columns'],
            "conversion_time": conversion_time
        }
        
        print(f"Conversion completed:")
        print(f"   Rows: {conversion_results['total_rows']:,}")
        print(f"   Output size: {conversion_results['output_size_gb']:.2f} GB")
        print(f"   Compression: {conversion_results['compression_ratio']:.1f}:1")
        print(f"   Time: {conversion_time:.1f}s")
        print(f"   Rate: {conversion_results['total_rows'] / conversion_time:,.0f} rows/sec")
        
        self.results['csv_to_parquet'] = conversion_results
        return conversion_results
    
    def test_range_bar_type(self, 
                           parquet_path: str,
                           range_type: str, 
                           create_func,
                           **kwargs) -> Dict[str, Any]:
        """Test a specific range bar type"""
        print(f"\nTesting {range_type} Range Bars...")
        
        load_start = time.time()
        tick_data = load_tick_data(parquet_path)
        load_time = time.time() - load_start
        input_ticks = len(tick_data)
        
        print(f"   Loaded: {input_ticks:,} ticks in {load_time:.1f}s")
        
        # Generate range bars
        gen_start = time.time()
        with self.benchmark.measure_operation(f"{range_type}_range_bar_generation"):
            range_bars = create_func(tick_data, **kwargs)
        gen_time = time.time() - gen_start
        output_bars = len(range_bars)
        
        # Save to parquet
        save_start = time.time()
        output_path = self.output_dir / f"{range_type.lower()}_range_bars.parquet"
        range_bars.to_parquet(output_path, compression='snappy')
        save_time = time.time() - save_start
        
        # Get output file size
        output_info = self.get_file_info(output_path)
        
        # Calculate metrics
        compression_ratio = input_ticks / output_bars
        
        # Estimate bars per day (assuming typical ES trading)
        estimated_days = input_ticks / 50000  # Rough estimate: 50k ticks per day
        bars_per_day = output_bars / estimated_days if estimated_days > 0 else 0
        minutes_per_bar = (17 * 60) / bars_per_day if bars_per_day > 0 else 0
        
        results = {
            "range_type": range_type,
            "input_ticks": input_ticks,
            "output_bars": output_bars,
            "compression_ratio": compression_ratio,
            "estimated_bars_per_day": bars_per_day,
            "estimated_minutes_per_bar": minutes_per_bar,
            "load_time": load_time,
            "generation_time": gen_time,
            "save_time": save_time,
            "total_time": load_time + gen_time + save_time,
            "output_file_size_mb": output_info['size_mb'],
            "output_path": str(output_path),
            "generation_rate": output_bars / gen_time if gen_time > 0 else 0
        }
        
        print(f"   Output: {output_bars:,} bars")
        print(f"   Compression: {compression_ratio:.0f}:1")
        print(f"   Est. frequency: {minutes_per_bar:.1f} min/bar ({bars_per_day:.0f} bars/day)")
        print(f"   Generation time: {gen_time:.1f}s ({results['generation_rate']:,.0f} bars/sec)")
        print(f"   File size: {output_info['size_mb']:.1f} MB")
        print(f"   Total time: {results['total_time']:.1f}s")
        
        return results
    
    def run_complete_performance_test(self) -> Dict[str, Any]:
        """Run complete performance test for all range bar types"""
        print("\nSTARTING COMPLETE PERFORMANCE TEST")
        print("="*60)
        
        overall_start = time.time()
        
        # Step 1: CSV to Parquet conversion
        conversion_results = self.test_csv_to_parquet_conversion()
        parquet_path = conversion_results['parquet_path']
        
        # Step 2: Test all range bar types
        range_bar_tests = [
            ("Raw_Price", create_raw_range_bars, {"range_size": 0.22}),
            ("Percentage", create_percentage_range_bars, {"percentage": 0.0051}),
            ("ATR_14d", create_atr_range_bars, {"atr_lookback_days": 14, "atr_multiplier": 0.0088}),
            ("ATR_30d", create_atr_range_bars, {"atr_lookback_days": 30, "atr_multiplier": 0.0074}),
            ("ATR_90d", create_atr_range_bars, {"atr_lookback_days": 90, "atr_multiplier": 0.0063})
        ]
        
        range_bar_results = []
        
        for i, (range_type, create_func, kwargs) in enumerate(range_bar_tests, 2):
            print(f"\n" + "="*60)
            print(f"STEP {i}: {range_type.upper()} RANGE BARS")
            print("="*60)
            
            try:
                result = self.test_range_bar_type(parquet_path, range_type, create_func, **kwargs)
                range_bar_results.append(result)
                self.results[f'range_bars_{range_type.lower()}'] = result
            except Exception as e:
                print(f"Error testing {range_type}: {str(e)}")
                range_bar_results.append({
                    "range_type": range_type,
                    "error": str(e)
                })
        
        total_time = time.time() - overall_start
        
        # Generate comprehensive summary
        summary = self.generate_performance_summary(conversion_results, range_bar_results, total_time)
        
        return summary
    
    def generate_performance_summary(self, 
                                   conversion_results: Dict[str, Any],
                                   range_bar_results: List[Dict[str, Any]],
                                   total_time: float) -> Dict[str, Any]:
        """Generate comprehensive performance summary with tables"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE SUMMARY")
        print("="*80)
        
        # CSV to Parquet Summary
        print("\nCSV TO PARQUET CONVERSION")
        print("-" * 50)
        print(f"Input Size:        {conversion_results['input_size_gb']:.2f} GB")
        print(f"Output Size:       {conversion_results['output_size_gb']:.2f} GB") 
        print(f"Compression:       {conversion_results['compression_ratio']:.1f}:1")
        print(f"Total Rows:        {conversion_results['total_rows']:,}")
        print(f"Conversion Time:   {conversion_results['conversion_time']:.1f}s")
        print(f"Processing Rate:   {conversion_results['total_rows'] / conversion_results['conversion_time']:,.0f} rows/sec")
        
        # Range Bar Performance Table
        print("\nRANGE BAR GENERATION PERFORMANCE")
        print("-" * 95)
        print(f"{'Type':<12} {'Ticks In':<12} {'Bars Out':<10} {'Compress':<9} {'Load':<6} {'Gen':<6} {'Save':<6} {'Total':<7} {'Min/Bar':<8}")
        print("-" * 95)
        
        total_generation_time = 0
        total_save_time = 0
        total_load_time = 0
        
        for result in range_bar_results:
            if 'error' in result:
                print(f"{result['range_type']:<12} {'ERROR':<12}")
                continue
                
            range_type = result['range_type']
            load_time = result['load_time']
            gen_time = result['generation_time']
            save_time = result['save_time']
            total_test_time = result['total_time']
            
            total_generation_time += gen_time
            total_save_time += save_time
            total_load_time += load_time
            
            print(f"{range_type:<12} {result['input_ticks']:>10,}  {result['output_bars']:>8,}  "
                  f"{result['compression_ratio']:>6.0f}:1   {load_time:>4.1f}s  {gen_time:>4.1f}s  "
                  f"{save_time:>4.1f}s  {total_test_time:>5.1f}s  {result['estimated_minutes_per_bar']:>6.1f}")
        
        print("-" * 95)
        print(f"{'TOTALS':<12} {'':<12} {'':<10} {'':<9} {total_load_time:>4.1f}s  "
              f"{total_generation_time:>4.1f}s  {save_time:>4.1f}s")
        
        # Overall Performance Summary
        print(f"\nOVERALL PERFORMANCE SUMMARY")
        print("-" * 50)
        print(f"Total Processing Time:   {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"CSV Conversion:          {conversion_results['conversion_time']:.1f}s ({conversion_results['conversion_time']/total_time*100:.1f}%)")
        print(f"Data Loading:            {total_load_time:.1f}s ({total_load_time/total_time*100:.1f}%)")
        print(f"Range Bar Generation:    {total_generation_time:.1f}s ({total_generation_time/total_time*100:.1f}%)")  
        print(f"Parquet Saves:           {total_save_time:.1f}s ({total_save_time/total_time*100:.1f}%)")
        
        # Frequency Analysis
        print(f"\nFREQUENCY ANALYSIS (5-Minute Target)")
        print("-" * 50)
        for result in range_bar_results:
            if 'error' not in result:
                deviation = abs(result['estimated_minutes_per_bar'] - 5.0)
                status = "ON TARGET" if deviation < 1.0 else "OFF TARGET"
                print(f"{result['range_type']:<12}: {result['estimated_minutes_per_bar']:>6.1f} min/bar ({status})")
        
        # Create summary dictionary
        summary = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": str(self.csv_file_path),
            "conversion_results": conversion_results,
            "range_bar_results": range_bar_results,
            "total_time": total_time,
            "total_generation_time": total_generation_time,
            "total_save_time": total_save_time,
            "total_load_time": total_load_time
        }
        
        # Save results to JSON
        import json
        results_file = self.output_dir / f"performance_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nDetailed results saved: {results_file}")
        
        return summary

def main():
    """Run performance test"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Range Bar Performance Test")
    parser.add_argument("csv_file", help="Path to input CSV tick data file")
    parser.add_argument("-o", "--output", default="performance_test_output", 
                       help="Output directory for test results")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        return 1
    
    try:
        # Run performance test
        test = RangeBarPerformanceTest(args.csv_file, args.output)
        results = test.run_complete_performance_test()
        
        print(f"\nPERFORMANCE TEST COMPLETED SUCCESSFULLY!")
        print(f"All results saved in: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Performance test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())