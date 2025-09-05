"""
Comprehensive display environment diagnostics
"""

import sys
import os
import platform
from pathlib import Path

def diagnose_environment():
    """Diagnose the current environment for GUI support"""
    
    print("=== DISPLAY ENVIRONMENT DIAGNOSTICS ===")
    print()
    
    # System info
    print("1. SYSTEM INFORMATION:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.architecture()[0]}")
    print(f"   Python: {sys.version}")
    print(f"   Working Directory: {os.getcwd()}")
    print()
    
    # Environment variables
    print("2. DISPLAY ENVIRONMENT VARIABLES:")
    display_vars = ['DISPLAY', 'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE', 'DESKTOP_SESSION']
    for var in display_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    print()
    
    # Windows specific
    if platform.system() == 'Windows':
        print("3. WINDOWS DISPLAY INFO:")
        try:
            import win32api
            import win32con
            monitors = win32api.EnumDisplayMonitors()
            print(f"   Monitors detected: {len(monitors)}")
            for i, monitor in enumerate(monitors):
                print(f"   Monitor {i+1}: {monitor[2]}")
        except ImportError:
            print("   win32api not available - cannot check monitors")
        except Exception as e:
            print(f"   Error checking monitors: {e}")
    
    # WSL detection
    print()
    print("4. WSL/CONTAINER DETECTION:")
    is_wsl = os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()
    print(f"   WSL detected: {is_wsl}")
    
    if is_wsl:
        print("   WSL GUI requires X11 forwarding or WSLg")
        wslg_check = os.environ.get('WAYLAND_DISPLAY') or os.environ.get('DISPLAY')
        print(f"   WSLg/X11 available: {bool(wslg_check)}")
    
    # Container detection
    is_container = os.path.exists('/.dockerenv') or os.environ.get('container') == 'docker'
    print(f"   Container detected: {is_container}")
    print()
    
    # Qt diagnostics
    print("5. QT DIAGNOSTICS:")
    try:
        from PyQt5 import QtCore, QtWidgets
        print("   PyQt5 import: SUCCESS")
        
        # Check Qt platform plugins
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(['--platform', 'offscreen'])  # Safe creation
        
        print(f"   Qt Version: {QtCore.qVersion()}")
        print(f"   Available platforms: {QtWidgets.QApplication.platformName()}")
        
        # Check available platform plugins
        try:
            platform_plugins = QtCore.QCoreApplication.libraryPaths()
            print(f"   Qt library paths: {platform_plugins}")
        except:
            print("   Could not get Qt library paths")
            
    except Exception as e:
        print(f"   PyQt5 error: {e}")
    print()
    
    # Final recommendations
    print("6. RECOMMENDATIONS:")
    if is_wsl and not os.environ.get('DISPLAY'):
        print("   - Install WSLg: wsl --update")
        print("   - Or setup X11 forwarding with VcXsrv/Xming")
    elif is_container:
        print("   - GUI not supported in containers without X11 forwarding")
    elif platform.system() == 'Windows':
        print("   - Windows should support GUI - checking Qt platform...")
    else:
        print("   - Check if X11/Wayland display server is running")

def test_qt_platforms():
    """Test different Qt platform options"""
    
    print("\n=== TESTING QT PLATFORMS ===")
    
    platforms_to_test = ['windows', 'offscreen', 'minimal']
    
    for platform in platforms_to_test:
        try:
            print(f"\nTesting platform: {platform}")
            
            from PyQt5 import QtWidgets, QtCore
            
            # Try creating app with specific platform
            app = QtWidgets.QApplication(['--platform', platform])
            
            # Create simple widget
            widget = QtWidgets.QLabel(f"Test with {platform} platform")
            widget.setWindowTitle(f"Test - {platform}")
            widget.resize(300, 100)
            
            if platform != 'offscreen':
                widget.show()
                print(f"   Widget shown with {platform} platform")
                
                # Process events briefly
                QtCore.QTimer.singleShot(1000, app.quit)
                app.exec_()
            else:
                print(f"   {platform} platform created successfully (no display)")
            
            app.quit()
            del app
            
        except Exception as e:
            print(f"   Platform {platform} failed: {e}")

if __name__ == "__main__":
    diagnose_environment()
    test_qt_platforms()