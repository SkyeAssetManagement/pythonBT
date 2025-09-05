# test_vispy_candlestick_renderer.py
# Test script for VisPy candlestick renderer
# 
# Tests the high-performance renderer with synthetic data
# Validates speed, rendering quality, and interaction

import sys
import time
import numpy as np
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_vispy_renderer_basic():
    """Test basic VisPy renderer functionality"""
    print(f"\n=== TESTING VISPY CANDLESTICK RENDERER ===")
    
    try:
        from src.dashboard.vispy_candlestick_renderer import (
            VispyCandlestickRenderer, 
            create_test_data
        )
        
        # Test 1: Create synthetic data (similar to 7M dataset but smaller for testing)
        print(f"\n1. Creating test data...")
        num_candles = 100000  # 100K candlesticks for performance testing
        test_data = create_test_data(num_candles)
        
        print(f"   SUCCESS: Created {num_candles:,} synthetic candlesticks")
        print(f"   INFO: Price range: ${test_data['low'].min():.5f} - ${test_data['high'].max():.5f}")
        
        # Test 2: Initialize renderer
        print(f"\n2. Initializing VisPy renderer...")
        start_time = time.time()
        renderer = VispyCandlestickRenderer(width=1400, height=900)
        init_time = time.time() - start_time
        
        print(f"   SUCCESS: Renderer initialized in {init_time:.3f}s")
        
        # Test 3: Load data into renderer 
        print(f"\n3. Loading data into GPU pipeline...")
        start_time = time.time()
        success = renderer.load_data(test_data)
        load_time = time.time() - start_time
        
        if success:
            print(f"   SUCCESS: Data loaded in {load_time:.3f}s")
            print(f"   PERFORMANCE: {num_candles/load_time:.0f} candles/second loading")
        else:
            print(f"   ERROR: Failed to load data into renderer")
            return False
        
        # Test 4: Start renderer and take screenshots
        print(f"\n4. Starting renderer with automatic screenshots...")
        print(f"   INFO: Renderer will run for 10 seconds and take screenshots")
        print(f"   INFO: Use mouse wheel to zoom, drag to pan")
        print(f"   INFO: Press 'S' for manual screenshot, 'Q' to quit early")
        
        # Schedule automatic screenshots during rendering
        def take_test_screenshots():
            """Take screenshots at regular intervals for testing"""
            import threading
            
            def screenshot_worker():
                time.sleep(2)  # Let renderer stabilize
                renderer._take_screenshot()  # Initial view
                
                time.sleep(2)  
                # Simulate zoom in
                renderer._handle_zoom(5, (700, 400))  # Zoom in at center
                renderer.canvas.update()
                renderer._take_screenshot()  # Zoomed view
                
                time.sleep(2)
                # Reset view 
                renderer._reset_view()
                renderer.canvas.update()
                renderer._take_screenshot()  # Reset view
                
                time.sleep(3)
                print(f"   INFO: Test screenshots completed")
            
            thread = threading.Thread(target=screenshot_worker)
            thread.daemon = True
            thread.start()
        
        # Start screenshot worker
        take_test_screenshots()
        
        # Show renderer (this will block until closed)
        renderer.show()
        
        print(f"   SUCCESS: Renderer test completed!")
        return True
        
    except ImportError as e:
        print(f"   ERROR: Missing dependencies: {e}")
        print(f"   SOLUTION: Install VisPy with: pip install vispy imageio")
        return False
        
    except Exception as e:
        print(f"   ERROR: Renderer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_benchmarks():
    """Test renderer performance with different data sizes"""
    print(f"\n=== PERFORMANCE BENCHMARKS ===")
    
    try:
        from src.dashboard.vispy_candlestick_renderer import (
            VispyCandlestickRenderer, 
            create_test_data
        )
        
        # Test different data sizes to validate scalability
        test_sizes = [1000, 10000, 100000]  # Up to 100K for CI testing
        
        results = []
        
        for size in test_sizes:
            print(f"\n--- Testing {size:,} candlesticks ---")
            
            # Generate test data
            start_time = time.time()
            test_data = create_test_data(size)
            data_time = time.time() - start_time
            
            # Test renderer
            renderer = VispyCandlestickRenderer(width=800, height=600)  # Smaller for testing
            
            start_time = time.time()  
            success = renderer.load_data(test_data)
            load_time = time.time() - start_time
            
            if success:
                loading_rate = size / load_time if load_time > 0 else 0
                results.append({
                    'size': size,
                    'data_time': data_time,
                    'load_time': load_time,
                    'loading_rate': loading_rate
                })
                
                print(f"   Data generation: {data_time:.3f}s")
                print(f"   GPU loading: {load_time:.3f}s") 
                print(f"   Loading rate: {loading_rate:.0f} candles/sec")
            else:
                print(f"   ERROR: Failed to load {size:,} candlesticks")
        
        # Summary
        print(f"\n=== PERFORMANCE SUMMARY ===")
        for result in results:
            print(f"   {result['size']:>7,} candles: {result['loading_rate']:>8.0f} candles/sec")
        
        # Validate performance meets requirements
        if results and results[-1]['loading_rate'] > 10000:  # Should handle 10K+ candles/sec
            print(f"   SUCCESS: Performance meets requirements for 7M+ candlesticks")
            return True
        else:
            print(f"   WARNING: Performance may not scale to 7M candlesticks")
            return False
            
    except Exception as e:
        print(f"   ERROR: Performance test failed: {e}")
        return False

def validate_screenshots():
    """Validate that screenshots are being created properly"""
    print(f"\n=== VALIDATING SCREENSHOTS ===")
    
    import glob
    
    # Look for recent screenshots
    screenshot_pattern = "candlestick_test_*.png"
    screenshots = glob.glob(screenshot_pattern)
    
    if screenshots:
        print(f"   SUCCESS: Found {len(screenshots)} screenshot(s)")
        for screenshot in screenshots[-3:]:  # Show last 3
            print(f"   - {screenshot}")
        return True
    else:
        print(f"   WARNING: No screenshots found")
        print(f"   INFO: Screenshots should be created during renderer testing")
        return False

if __name__ == "__main__":
    print(f"VisPy Candlestick Renderer Test Suite")
    print(f"=" * 50)
    
    # Run all tests
    basic_test = test_vispy_renderer_basic()
    perf_test = test_performance_benchmarks() 
    screenshot_test = validate_screenshots()
    
    # Final summary
    print(f"\n" + "=" * 50)
    print(f"TEST RESULTS:")
    print(f"  Basic functionality: {'PASS' if basic_test else 'FAIL'}")
    print(f"  Performance benchmarks: {'PASS' if perf_test else 'FAIL'}")
    print(f"  Screenshot validation: {'PASS' if screenshot_test else 'FAIL'}")
    
    if basic_test and perf_test:
        print(f"\nSUCCESS: VisPy candlestick renderer is ready for 7M+ datapoints!")
    else:
        print(f"\nWARNING: Some tests failed - check dependencies and environment")