#!/usr/bin/env python3
"""
Immediate GUI Test - Run this to verify the fix works
Save as test_gui_fix.py and run: python test_gui_fix.py
"""

import tkinter as tk
import time

def test_gui_visibility():
    """Test if the GUI visibility fix works"""
    print("Testing GUI visibility fix...")
    
    # Simulate simStart.py pattern
    root = tk.Tk()
    root.withdraw()  # Hide root like simStart.py does
    
    # Create window like EnhancedSimulationInputGUI
    window = tk.Toplevel(root)
    window.title("GUI Visibility Test")
    window.geometry("400x300")
    window.transient(root)
    
    # Add content
    tk.Label(window, text="GUI Visibility Test", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Label(window, text="If you can see this window, the fix works!").pack(pady=10)
    
    result = {"clicked": False}
    
    def on_success():
        result["clicked"] = True
        print("✓ SUCCESS: User can see the window!")
        window.destroy()
    
    def on_fail():
        print("✗ FAIL: User cannot see the window")
        window.destroy()
    
    tk.Button(window, text="✓ I CAN SEE IT!", command=on_success, 
              bg="green", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Button(window, text="✗ I CANNOT SEE IT", command=on_fail,
              bg="red", fg="white").pack(pady=5)
    
    # Apply the EXACT visibility fix
    print("Applying visibility fix...")
    window.deiconify()
    window.lift()
    window.focus_force()
    window.attributes('-topmost', True)
    window.state('normal')
    window.update()
    window.update_idletasks()
    window.grab_set()
    window.after(100, lambda: window.attributes('-topmost', False))
    
    print("Window should now be visible...")
    
    # Auto-timeout after 15 seconds
    def timeout():
        if window.winfo_exists():
            print("TIMEOUT: Closing window automatically")
            window.destroy()
    
    window.after(15000, timeout)
    
    # Wait for window (this is where simStart.py hangs)
    try:
        root.wait_window(window)
        print("wait_window completed successfully")
    except Exception as e:
        print(f"wait_window failed: {e}")
    
    root.destroy()
    return result["clicked"]

if __name__ == "__main__":
    success = test_gui_visibility()
    
    if success:
        print("\n" + "="*50)
        print("✓ GUI FIX WORKS!")
        print("Now apply the fix to your simInput.py file")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("✗ GUI FIX DOESN'T WORK")
        print("Use emergency bypass instead")
        print("="*50)