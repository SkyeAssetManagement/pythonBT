# debug_vispy_simple.py
# Simple VisPy test without unicode characters

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_vispy_simple():
    """Simple VisPy test to identify hanging issues"""
    print("=== DEBUGGING VISPY HANGING ISSUE ===")
    
    try:
        print("1. Testing VisPy import...")
        import vispy
        print(f"   OK VisPy version: {vispy.__version__}")
        
        print("2. Testing VisPy app creation...")
        from vispy import app
        
        # Try different backends
        backend_found = False
        for backend_name in ['PyQt5', 'PyQt6', 'PySide2', 'PySide6']:
            try:
                print(f"   Trying {backend_name} backend...")
                app_instance = app.use_app(backend_name)
                print(f"   OK {backend_name} backend works")
                backend_found = True
                break
            except Exception as e:
                print(f"   FAIL {backend_name}: {e}")
        
        if not backend_found:
            print("   ERROR: No viable backend found")
            return False
            
        print("3. Testing canvas creation...")
        canvas = app.Canvas(title='Test', size=(400, 300), show=False)
        print("   OK Canvas created")
        
        print("4. Testing basic show/hide (might hang here)...")
        try:
            canvas.show()
            print("   OK Canvas shown")
            canvas.close()
            print("   OK Canvas closed")
        except Exception as e:
            print(f"   FAIL Canvas show/close: {e}")
            return False
        
        print("5. Testing our renderer...")
        from src.dashboard.vispy_candlestick_renderer import VispyCandlestickRenderer, create_test_data
        
        print("   Creating test data...")
        test_data = create_test_data(100)  # Very small dataset
        print("   OK Test data created")
        
        print("   Creating renderer...")
        renderer = VispyCandlestickRenderer(width=400, height=300)
        print("   OK Renderer created")
        
        print("   Loading data...")
        success = renderer.load_data(test_data)
        if not success:
            print("   FAIL Data loading failed")
            return False
        print("   OK Data loaded")
        
        print("   Testing show (THIS IS WHERE IT MIGHT HANG)...")
        print("   If this hangs, the issue is in renderer.show() or app.run()")
        
        # Don't actually show - just validate we got this far
        print("   SKIP Actually showing renderer for now")
        print("   SUCCESS: Reached end without hanging!")
        
        return True
        
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vispy_simple()
    
    if success:
        print("\nSUCCESS: VisPy renderer can be initialized without hanging")
        print("The hanging issue might be in the event loop or specific rendering calls")
    else:
        print("\nFAILED: VisPy has fundamental issues on this system")