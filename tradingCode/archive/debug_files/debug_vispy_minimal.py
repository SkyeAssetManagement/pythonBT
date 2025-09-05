# debug_vispy_minimal.py
# Minimal VisPy test to identify why the original implementation hangs
# This will help us fix the VisPy renderer properly

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_vispy_basic():
    """Test basic VisPy functionality to identify hanging issues"""
    print("=== DEBUGGING VISPY HANGING ISSUE ===")
    print()
    
    try:
        print("1. Testing VisPy import...")
        import vispy
        print(f"   [OK] VisPy version: {vispy.__version__}")
        
        print("2. Testing VisPy app backends...")
        from vispy import app
        available_backends = app.backends.BACKEND_NAMES
        print(f"   [OK] Available backends: {available_backends}")
        
        print("3. Testing PyQt5 backend specifically...")
        try:
            app_instance = app.use_app('PyQt5')
            print(f"   [OK] PyQt5 backend loaded successfully")
            print(f"   INFO: Backend: {app_instance}")
        except Exception as e:
            print(f"   [X] PyQt5 backend failed: {e}")
            print("   Trying alternate backends...")
            
            for backend_name in ['PyQt6', 'PySide2', 'PySide6', 'Glfw']:
                try:
                    app_instance = app.use_app(backend_name)
                    print(f"   [OK] {backend_name} backend works!")
                    break
                except:
                    print(f"   [X] {backend_name} not available")
        
        print("4. Testing canvas creation...")
        canvas = app.Canvas(title='VisPy Test', size=(400, 300), show=False)
        print(f"   [OK] Canvas created successfully")
        
        print("5. Testing OpenGL context...")
        try:
            from vispy import gloo
            
            # Try to create a simple GL context test
            canvas.show()  # This might hang if GL context fails
            print(f"   [OK] Canvas shown - checking GL context...")
            
            # Simple GL test
            gloo.clear(color='black')
            print(f"   [OK] OpenGL context working")
            
            canvas.close()
            print(f"   [OK] Canvas closed successfully")
            
        except Exception as e:
            print(f"   [X] OpenGL context failed: {e}")
            return False
        
        print("6. Testing our candlestick renderer initialization...")
        try:
            from src.dashboard.vispy_candlestick_renderer import VispyCandlestickRenderer
            renderer = VispyCandlestickRenderer(width=800, height=600)
            print(f"   [OK] Candlestick renderer created successfully")
            
            print("7. Testing with small dataset...")
            from src.dashboard.vispy_candlestick_renderer import create_test_data
            test_data = create_test_data(1000)  # Small dataset first
            
            success = renderer.load_data(test_data)
            if success:
                print(f"   [OK] Data loaded successfully")
                print(f"   INFO: Ready to show renderer...")
                
                # This is where it might hang
                print(f"   ATTEMPTING TO SHOW RENDERER (this might hang)...")
                print(f"   Press Ctrl+C if it hangs, or close window if it works")
                
                # Test showing for just 2 seconds
                import threading
                import time
                
                def timeout_quit():
                    time.sleep(3)
                    print("\n   TIMEOUT: Forcing quit after 3 seconds")
                    renderer.app.quit()
                
                # Start timeout thread
                timeout_thread = threading.Thread(target=timeout_quit, daemon=True)
                timeout_thread.start()
                
                # Try to show
                renderer.show()
                
                print(f"   [OK] SUCCESS: VisPy renderer worked!")
                return True
                
            else:
                print(f"   [X] Failed to load test data")
                return False
                
        except Exception as e:
            print(f"   [X] Candlestick renderer failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except ImportError as e:
        print(f"[X] VisPy import failed: {e}")
        print("   Install with: pip install vispy imageio")
        return False
    
    except Exception as e:
        print(f"[X] VisPy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("This test will identify why VisPy hangs and help fix it")
    print("If it hangs, press Ctrl+C to interrupt")
    print()
    
    success = test_vispy_basic()
    
    if success:
        print("\n[OK] VISPY WORKS! The hanging issue is resolved")
    else:
        print("\n[X] VISPY HAS ISSUES - need to investigate further")