"""
Ultra-Fast Range Bar Performance Benchmarking Module

This module provides comprehensive performance measurement tools for 
monitoring processing times, memory usage, and I/O performance during
range bar conversion operations.
"""

import time
import psutil
import functools
import pandas as pd
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

@dataclass
class PerformanceMetric:
    """Container for performance measurement data"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    peak_memory_mb: float
    start_memory_mb: float
    end_memory_mb: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta_mb(self) -> float:
        """Memory usage change during operation"""
        return self.end_memory_mb - self.start_memory_mb

class PerformanceBenchmark:
    """
    High-performance benchmarking class for measuring ETL operations.
    
    Features:
    - Precise timing with nanosecond resolution
    - Memory usage tracking (RSS, VMS, Peak)
    - I/O performance metrics
    - CPU usage monitoring
    - Nested operation tracking
    - Export to multiple formats (JSON, CSV, DataFrame)
    """
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.current_operations: Dict[str, Dict[str, Any]] = {}
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage"""
        return self.process.cpu_percent()
    
    @contextmanager
    def measure_operation(self, operation_name: str, **kwargs):
        """
        Context manager for measuring operation performance.
        
        Args:
            operation_name: Name of the operation being measured
            **kwargs: Additional metrics to track
        """
        start_time = time.perf_counter()
        start_memory = self.get_memory_usage()
        peak_memory = start_memory
        
        # Store operation start data
        self.current_operations[operation_name] = {
            'start_time': start_time,
            'start_memory': start_memory,
            'peak_memory': peak_memory,
            'additional_metrics': kwargs
        }
        
        try:
            yield self
            
            # Update peak memory during operation
            current_memory = self.get_memory_usage()
            if current_memory > peak_memory:
                peak_memory = current_memory
                self.current_operations[operation_name]['peak_memory'] = peak_memory
                
        finally:
            end_time = time.perf_counter()
            end_memory = self.get_memory_usage()
            duration = end_time - start_time
            
            # Create performance metric
            metric = PerformanceMetric(
                operation=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                peak_memory_mb=self.current_operations[operation_name]['peak_memory'],
                start_memory_mb=start_memory,
                end_memory_mb=end_memory,
                additional_metrics=kwargs
            )
            
            self.metrics.append(metric)
            
            # Clean up current operations
            del self.current_operations[operation_name]
            
            # Print real-time results
            self._print_metric(metric)
    
    def _print_metric(self, metric: PerformanceMetric):
        """Print performance metric in real-time"""
        print(f"\n{metric.operation}")
        print(f"   Duration: {metric.duration:.3f}s")
        print(f"   Memory: {metric.start_memory_mb:.1f} -> {metric.end_memory_mb:.1f} MB "
              f"(Peak: {metric.peak_memory_mb:.1f} MB, Delta: {metric.memory_delta_mb:+.1f} MB)")
        
        if metric.additional_metrics:
            for key, value in metric.additional_metrics.items():
                print(f"   {key}: {value}")
    
    def benchmark_function(self, operation_name: str, **kwargs):
        """
        Decorator for benchmarking function performance.
        
        Usage:
            @benchmark.benchmark_function("my_operation", rows=1000000)
            def my_function():
                # function code
                pass
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **func_kwargs):
                with self.measure_operation(operation_name, **kwargs):
                    return func(*args, **func_kwargs)
            return wrapper
        return decorator
    
    def measure_io_operation(self, operation_name: str, file_path: str, 
                           operation_type: str = "unknown"):
        """
        Measure I/O operation performance including file size and throughput.
        
        Args:
            operation_name: Name of the I/O operation
            file_path: Path to the file being processed
            operation_type: Type of operation (read/write)
        """
        file_size_mb = 0
        if os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / 1024 / 1024
        
        additional_metrics = {
            'file_path': file_path,
            'file_size_mb': file_size_mb,
            'operation_type': operation_type
        }
        
        return self.measure_operation(operation_name, **additional_metrics)
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics of all measured operations"""
        if not self.metrics:
            return pd.DataFrame()
        
        data = []
        for metric in self.metrics:
            row = {
                'operation': metric.operation,
                'duration_seconds': metric.duration,
                'start_memory_mb': metric.start_memory_mb,
                'end_memory_mb': metric.end_memory_mb,
                'peak_memory_mb': metric.peak_memory_mb,
                'memory_delta_mb': metric.memory_delta_mb,
                'timestamp': datetime.fromtimestamp(metric.start_time)
            }
            row.update(metric.additional_metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def print_summary(self):
        """Print comprehensive performance summary"""
        if not self.metrics:
            print("No performance metrics recorded.")
            return
        
        df = self.get_summary_stats()
        total_time = df['duration_seconds'].sum()
        
        print(f"\n" + "="*60)
        print(f"PERFORMANCE SUMMARY")
        print(f"="*60)
        print(f"Total Operations: {len(self.metrics)}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Peak Memory: {df['peak_memory_mb'].max():.1f} MB")
        print(f"Memory Delta: {df['memory_delta_mb'].sum():+.1f} MB")
        
        print(f"\nOperation Breakdown:")
        print(f"{'-'*60}")
        
        for _, row in df.iterrows():
            throughput = ""
            if 'file_size_mb' in row and row['file_size_mb'] > 0:
                rate = row['file_size_mb'] / row['duration_seconds']
                throughput = f" ({rate:.1f} MB/s)"
            
            print(f"{row['operation']:<30} {row['duration_seconds']:>8.3f}s{throughput}")
        
        print(f"="*60)
    
    def export_results(self, format_type: str = "json", 
                      file_path: Optional[str] = None) -> str:
        """
        Export performance results to file.
        
        Args:
            format_type: Export format ("json", "csv", "parquet")
            file_path: Output file path (auto-generated if None)
        
        Returns:
            Path to exported file
        """
        if not self.metrics:
            raise ValueError("No performance metrics to export")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if file_path is None:
            file_path = f"performance_results_{timestamp}.{format_type}"
        
        if format_type.lower() == "json":
            data = []
            for metric in self.metrics:
                data.append({
                    'operation': metric.operation,
                    'start_time': metric.start_time,
                    'end_time': metric.end_time,
                    'duration': metric.duration,
                    'peak_memory_mb': metric.peak_memory_mb,
                    'start_memory_mb': metric.start_memory_mb,
                    'end_memory_mb': metric.end_memory_mb,
                    'memory_delta_mb': metric.memory_delta_mb,
                    'additional_metrics': metric.additional_metrics
                })
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format_type.lower() in ["csv", "parquet"]:
            df = self.get_summary_stats()
            
            if format_type.lower() == "csv":
                df.to_csv(file_path, index=False)
            else:
                df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        print(f"📁 Performance results exported to: {file_path}")
        return file_path

# Global benchmark instance for easy access
benchmark = PerformanceBenchmark()

# Convenience functions
def measure_time(operation_name: str, **kwargs):
    """Convenience function for measuring operations"""
    return benchmark.measure_operation(operation_name, **kwargs)

def measure_io(operation_name: str, file_path: str, operation_type: str = "unknown"):
    """Convenience function for measuring I/O operations"""  
    return benchmark.measure_io_operation(operation_name, file_path, operation_type)

def benchmark_func(operation_name: str, **kwargs):
    """Convenience decorator for benchmarking functions"""
    return benchmark.benchmark_function(operation_name, **kwargs)

def print_summary():
    """Print performance summary"""
    benchmark.print_summary()

def export_results(format_type: str = "json", file_path: Optional[str] = None):
    """Export performance results"""
    return benchmark.export_results(format_type, file_path)

def clear_metrics():
    """Clear all recorded metrics"""
    benchmark.metrics.clear()
    benchmark.current_operations.clear()

# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    import time
    
    # Test the performance measurement
    with measure_time("test_array_operation", array_size=1000000):
        arr = np.random.random(1000000)
        result = np.sum(arr)
        time.sleep(0.1)  # Simulate processing
    
    with measure_time("test_memory_intensive", memory_test=True):
        large_list = [i for i in range(500000)]
        time.sleep(0.05)
    
    print_summary()
    export_results("json", "test_performance.json")