#!/usr/bin/env python3
"""
Step 4: Add keyboard controls to main dashboard for testing
Goal: Add keyboard controls so we can test pan/zoom functionality without a mouse

This will add the keyboard control methods to the TradingChart class
"""

keyboard_controls_code = '''
    def keyPressEvent(self, event):
        """Handle keyboard controls for pan/zoom testing"""
        key = event.key()
        
        if key == QtCore.Qt.Key_Left:
            self.pan_left()
        elif key == QtCore.Qt.Key_Right:
            self.pan_right()
        elif key == QtCore.Qt.Key_Up:
            self.zoom_in()
        elif key == QtCore.Qt.Key_Down:
            self.zoom_out()
        elif key == QtCore.Qt.Key_R:
            self.reset_view()
        elif key == QtCore.Qt.Key_Q:
            # Close the dashboard
            if hasattr(self, 'parent') and self.parent():
                self.parent().close()
        else:
            super().keyPressEvent(event)
    
    def pan_left(self):
        """Pan chart left"""
        x_range = self.viewRange()[0]
        span = x_range[1] - x_range[0]
        pan_amount = span * 0.15  # 15% of visible range
        new_range = [x_range[0] - pan_amount, x_range[1] - pan_amount]
        self.setXRange(*new_range, padding=0)
        print(f"Panned left to bars {new_range[0]:.0f}-{new_range[1]:.0f}")
    
    def pan_right(self):
        """Pan chart right"""
        x_range = self.viewRange()[0]
        span = x_range[1] - x_range[0]
        pan_amount = span * 0.15  # 15% of visible range
        new_range = [x_range[0] + pan_amount, x_range[1] + pan_amount]
        self.setXRange(*new_range, padding=0)
        print(f"Panned right to bars {new_range[0]:.0f}-{new_range[1]:.0f}")
    
    def zoom_in(self):
        """Zoom in on chart"""
        x_range = self.viewRange()[0]
        center = (x_range[0] + x_range[1]) / 2
        span = x_range[1] - x_range[0]
        new_span = span / 1.2  # Zoom factor
        new_range = [center - new_span/2, center + new_span/2]
        self.setXRange(*new_range, padding=0)
        print(f"Zoomed in to bars {new_range[0]:.0f}-{new_range[1]:.0f} (span: {new_span:.0f})")
    
    def zoom_out(self):
        """Zoom out on chart"""
        x_range = self.viewRange()[0]
        center = (x_range[0] + x_range[1]) / 2
        span = x_range[1] - x_range[0]
        new_span = span * 1.2  # Zoom factor
        new_range = [center - new_span/2, center + new_span/2]
        self.setXRange(*new_range, padding=0)
        print(f"Zoomed out to bars {new_range[0]:.0f}-{new_range[1]:.0f} (span: {new_span:.0f})")
    
    def reset_view(self):
        """Reset to full view"""
        if self.data_buffer:
            self.setXRange(0, len(self.data_buffer) - 1, padding=0)
            print(f"Reset view to full range: 0-{len(self.data_buffer)-1}")
        else:
            self.autoRange()
            print("Reset view with autoRange")
'''

def create_enhanced_main():
    """Create an enhanced main.py that shows keyboard controls"""
    
    enhanced_main_code = '''
    # Add keyboard controls information to dashboard launch
    print(f"   INFO: Dashboard keyboard controls available:")
    print(f"     - Left Arrow: Pan left")
    print(f"     - Right Arrow: Pan right")
    print(f"     - Up Arrow: Zoom in")
    print(f"     - Down Arrow: Zoom out")
    print(f"     - R: Reset view")
    print(f"     - Q: Quit dashboard")
    print(f"   INFO: Click on chart area first to enable keyboard focus")
    '''
    
    return enhanced_main_code

def main():
    """Generate keyboard controls for dashboard testing"""
    
    print("Step 4: Adding keyboard controls to dashboard")
    print("=" * 50)
    
    print("Keyboard controls code to add to TradingChart class:")
    print(keyboard_controls_code)
    
    print()
    print("Enhanced main.py instructions:")
    enhanced = create_enhanced_main()
    print(enhanced)
    
    print()
    print("NEXT STEPS:")
    print("1. Add the keyboard controls code to TradingChart class in chart_widget.py")
    print("2. Add setFocusPolicy(QtCore.Qt.StrongFocus) to TradingChart.__init__()")
    print("3. Add the keyboard instructions to main.py dashboard launch")
    print("4. Test the dashboard with keyboard controls")

if __name__ == "__main__":
    main()