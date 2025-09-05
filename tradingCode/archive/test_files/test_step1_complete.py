# test_step1_complete.py
# Final validation test for Step 1: Candlestick OHLCV chart renderer
# 
# Validates that the VisPy renderer meets all Step 1 requirements

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_step1_requirements():
    """Test all Step 1 requirements are met"""
    print(f"\n=== STEP 1 REQUIREMENT VALIDATION ===")
    print(f"Requirements:")
    print(f"  1. Candlestick OHLCV chart renderer")
    print(f"  2. Load 7M+ datapoints into memory")  
    print(f"  3. Render current view window of last 500 bars")
    print(f"  4. Quick initial render")
    print(f"  5. Smooth pan and zoom from memory")
    print(f"  6. Screenshot capability for testing")
    print(f"")
    
    results = {}
    
    try:
        from src.dashboard.vispy_candlestick_renderer import (
            VispyCandlestickRenderer, 
            create_test_data,
            DataPipeline
        )
        
        # Requirement 1: Candlestick OHLCV chart renderer
        print(f"1. Testing candlestick OHLCV chart renderer...")
        renderer = VispyCandlestickRenderer(width=1400, height=900)
        results['candlestick_renderer'] = True
        print(f"   PASS: VisPy candlestick renderer created")
        
        # Requirement 2: Load 7M+ datapoints into memory
        print(f"\n2. Testing 7M+ datapoint loading...")
        
        # Create 7M test dataset
        large_data = create_test_data(7000000)  # Exactly 7 million
        
        start_time = time.time()
        pipeline = DataPipeline()
        success = pipeline.load_ohlcv_data(large_data)
        load_time = time.time() - start_time
        
        if success and pipeline.data_length >= 7000000:
            results['large_dataset'] = True
            print(f"   PASS: Loaded {pipeline.data_length:,} candlesticks in {load_time:.3f}s")
            print(f"   INFO: Performance: {pipeline.data_length/load_time:.0f} candles/second")
        else:
            results['large_dataset'] = False
            print(f"   FAIL: Could not load 7M+ datapoints")
        
        # Requirement 3: Render current view window of last 500 bars
        print(f"\n3. Testing viewport rendering (last 500 bars)...")
        
        # Set viewport to last 500 bars
        pipeline.set_viewport_to_recent(500)
        viewport_data = pipeline.get_viewport_data(
            pipeline.viewport_start, 
            pipeline.viewport_end
        )
        
        # Check that only ~500 bars are in viewport (plus buffer)
        viewport_size = len(viewport_data)
        expected_size = 500 + (2 * pipeline.viewport_buffer)  # 500 + 2*50 buffer
        
        if 400 <= viewport_size <= 700:  # Allow reasonable range
            results['viewport_rendering'] = True
            print(f"   PASS: Viewport contains {viewport_size} bars (expected ~{expected_size})")
            print(f"   INFO: Viewport range: [{pipeline.viewport_start}, {pipeline.viewport_end}]")
        else:
            results['viewport_rendering'] = False
            print(f"   FAIL: Viewport size {viewport_size} outside expected range")
        
        # Requirement 4: Quick initial render
        print(f"\n4. Testing quick initial render performance...")
        
        # Create smaller dataset for render testing
        test_data = create_test_data(100000)
        renderer = VispyCandlestickRenderer(width=1200, height=800)
        
        start_time = time.time()
        success = renderer.load_data(test_data)
        render_time = time.time() - start_time
        
        if success and render_time < 1.0:  # Should render in under 1 second
            results['quick_render'] = True
            print(f"   PASS: Initial render completed in {render_time:.3f}s")
        else:
            results['quick_render'] = False
            print(f"   FAIL: Initial render too slow ({render_time:.3f}s)")
        
        # Requirement 5: Smooth pan and zoom from memory
        print(f"\n5. Testing pan and zoom performance...")
        
        # Test viewport updates (simulating pan/zoom)
        viewport_tests = []
        
        for i in range(10):  # Test 10 viewport changes
            start_idx = i * 1000
            end_idx = start_idx + 500
            
            start_time = time.time()
            renderer.viewport_x_range = [start_idx, end_idx]
            renderer._update_projection_matrix()
            renderer._update_gpu_buffers()
            update_time = time.time() - start_time
            
            viewport_tests.append(update_time)
        
        avg_update_time = np.mean(viewport_tests)
        max_update_time = np.max(viewport_tests)
        
        # Should update in < 10ms for smooth 60+ FPS
        if avg_update_time < 0.01 and max_update_time < 0.02:  
            results['smooth_interaction'] = True
            print(f"   PASS: Viewport updates avg {avg_update_time*1000:.1f}ms, max {max_update_time*1000:.1f}ms")
            print(f"   INFO: Estimated FPS: {1.0/avg_update_time:.0f}")
        else:
            results['smooth_interaction'] = False
            print(f"   FAIL: Viewport updates too slow (avg {avg_update_time*1000:.1f}ms)")
        
        # Requirement 6: Screenshot capability
        print(f"\n6. Testing screenshot capability...")
        
        try:
            # Test screenshot method exists and works
            renderer.screenshot_counter = 0
            renderer._take_screenshot()  # This will try to save a screenshot
            results['screenshots'] = True
            print(f"   PASS: Screenshot capability working")
        except Exception as e:
            results['screenshots'] = False
            print(f"   FAIL: Screenshot capability failed: {e}")
        
        # Summary
        print(f"\n=== STEP 1 VALIDATION SUMMARY ===")
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for requirement, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {requirement}: {status}")
        
        print(f"\nOVERALL: {passed_tests}/{total_tests} requirements met")
        
        if passed_tests == total_tests:
            print(f"SUCCESS: Step 1 completed - All requirements satisfied!")
            print(f"INFO: Ready to proceed to Step 2 (Trade List Integration)")
            return True
        else:
            print(f"WARNING: {total_tests - passed_tests} requirements still need work")
            return False
            
    except Exception as e:
        print(f"ERROR: Step 1 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def performance_benchmark_7m():
    """Benchmark performance with actual 7M dataset"""
    print(f"\n=== 7 MILLION CANDLESTICK BENCHMARK ===")
    
    try:
        from src.dashboard.vispy_candlestick_renderer import create_test_data, DataPipeline
        
        print(f"Creating 7,000,000 candlesticks...")
        start_time = time.time()
        data_7m = create_test_data(7000000)
        create_time = time.time() - start_time
        print(f"Data creation: {create_time:.3f}s")
        
        print(f"Loading into GPU pipeline...")
        pipeline = DataPipeline()
        start_time = time.time()
        success = pipeline.load_ohlcv_data(data_7m)
        load_time = time.time() - start_time
        
        if success:
            print(f"SUCCESS: 7M candlesticks loaded in {load_time:.3f}s")
            print(f"Performance: {7000000/load_time:.0f} candles/second")
            
            # Test memory efficiency
            memory_mb = pipeline.full_data.nbytes / 1024 / 1024
            print(f"Memory usage: {memory_mb:.1f} MB ({memory_mb/7:.3f} MB per million)")
            
            # Test viewport extraction
            start_time = time.time()
            viewport_data = pipeline.get_viewport_data(6999500, 7000000)  # Last 500
            viewport_time = time.time() - start_time
            
            print(f"Viewport extraction: {viewport_time*1000:.1f}ms for {len(viewport_data)} bars")
            print(f"SUCCESS: 7M candlestick pipeline ready for rendering!")
            
            return True
        else:
            print(f"FAIL: Could not load 7M candlesticks")
            return False
            
    except Exception as e:
        print(f"ERROR: 7M benchmark failed: {e}")
        return False

if __name__ == "__main__":
    print(f"Step 1 Completion Validation Test")
    print(f"=" * 50)
    
    # Run validation tests
    step1_complete = test_step1_requirements()
    benchmark_7m = performance_benchmark_7m()
    
    print(f"\n" + "=" * 50)
    print(f"FINAL RESULTS:")
    print(f"  Step 1 Requirements: {'COMPLETE' if step1_complete else 'INCOMPLETE'}")
    print(f"  7M Performance Test: {'PASS' if benchmark_7m else 'FAIL'}")
    
    if step1_complete and benchmark_7m:
        print(f"\n[SUCCESS] STEP 1 SUCCESSFULLY COMPLETED!")
        print(f"[OK] Candlestick OHLCV chart renderer with 7M+ datapoint support")
        print(f"[OK] Viewport rendering of last 500 bars") 
        print(f"[OK] Quick initial render and smooth pan/zoom")
        print(f"[OK] Screenshot system for testing")
        print(f"\n[LAUNCH] READY TO PROCEED TO STEP 2: Trade List Integration")
    else:
        print(f"\n[WARNING]  Step 1 needs additional work before proceeding")