#!/usr/bin/env python3
"""
GUI Hang Diagnostic Script for simStart.py
This script will help identify the exact cause of the GUI hanging issue.
"""

import sys
import traceback
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_tkinter():
    """Test basic tkinter functionality."""
    print("="*60)
    print("TEST 1: Basic Tkinter Functionality")
    print("="*60)
    
    try:
        import tkinter as tk
        print("✓ tkinter imported successfully")
        
        # Test basic window creation
        root = tk.Tk()
        root.title("Basic Test Window")
        root.geometry("300x200")
        print("✓ Basic window created")
        
        # Test immediate destruction
        root.destroy()
        print("✓ Basic window destroyed successfully")
        
        return True
    except Exception as e:
        print(f"✗ Basic tkinter test failed: {e}")
        return False

def test_toplevel_window():
    """Test Toplevel window creation and destruction."""
    print("\n" + "="*60)
    print("TEST 2: Toplevel Window Creation")
    print("="*60)
    
    try:
        import tkinter as tk
        
        # Create master window
        root = tk.Tk()
        root.withdraw()  # Hide it like simStart.py does
        print("✓ Master window created and hidden")
        
        # Create toplevel window
        toplevel = tk.Toplevel(root)
        toplevel.title("Test Toplevel")
        toplevel.geometry("300x200")
        print("✓ Toplevel window created")
        
        # Make it modal
        toplevel.transient(root)
        toplevel.grab_set()
        print("✓ Toplevel made modal")
        
        # Test immediate destruction
        toplevel.destroy()
        root.destroy()
        print("✓ Toplevel and master windows destroyed")
        
        return True
    except Exception as e:
        print(f"✗ Toplevel window test failed: {e}")
        return False

def test_wait_window_mechanism():
    """Test the wait_window mechanism that's hanging."""
    print("\n" + "="*60)
    print("TEST 3: wait_window Mechanism")
    print("="*60)
    
    try:
        import tkinter as tk
        
        # Simulate the exact pattern from simStart.py
        root = tk.Tk()
        root.withdraw()
        print("✓ Master window created and hidden")
        
        # Create a simple toplevel that auto-closes
        toplevel = tk.Toplevel(root)
        toplevel.title("Auto-closing Test")
        toplevel.geometry("300x200")
        toplevel.transient(root)
        toplevel.grab_set()
        print("✓ Toplevel created and made modal")
        
        # Add a button that closes the window
        def close_window():
            toplevel.destroy()
        
        button = tk.Button(toplevel, text="Close Window", command=close_window)
        button.pack(pady=50)
        print("✓ Close button added")
        
        # Schedule auto-close after 2 seconds
        def auto_close():
            if toplevel.winfo_exists():
                toplevel.destroy()
        
        toplevel.after(2000, auto_close)
        print("✓ Auto-close scheduled for 2 seconds")
        
        # Test wait_window (should not hang now)
        start_time = time.time()
        root.wait_window(toplevel)
        end_time = time.time()
        
        print(f"✓ wait_window completed in {end_time - start_time:.2f} seconds")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"✗ wait_window test failed: {e}")
        traceback.print_exc()
        return False

def test_simInput_import():
    """Test importing simInput and its dependencies."""
    print("\n" + "="*60)
    print("TEST 4: simInput Module Import")
    print("="*60)
    
    try:
        # Test individual imports
        print("Attempting to import simInput...")
        from simInput import get_simulation_parameters, EnhancedSimulationInputGUI
        print("✓ simInput imported successfully")
        
        print("Attempting to import simUtils...")
        from simUtils import work_days_per_year
        print("✓ simUtils imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import test failed: {e}")
        print("This may be the root cause of the hanging issue!")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        traceback.print_exc()
        return False

def test_simInput_gui_creation():
    """Test creating the actual simInput GUI."""
    print("\n" + "="*60)
    print("TEST 5: simInput GUI Creation")
    print("="*60)
    
    try:
        import tkinter as tk
        from simInput import EnhancedSimulationInputGUI
        
        # Create master window
        root = tk.Tk()
        root.withdraw()
        print("✓ Master window created")
        
        # Try to create the GUI (this is where it likely hangs)
        print("Creating EnhancedSimulationInputGUI...")
        gui = EnhancedSimulationInputGUI(root, mode="single")
        print("✓ GUI object created successfully")
        
        # Check if the GUI window exists
        if gui.root.winfo_exists():
            print("✓ GUI window exists")
        else:
            print("✗ GUI window does not exist")
        
        # Check if the GUI window is visible
        if gui.root.winfo_viewable():
            print("✓ GUI window is viewable")
        else:
            print("⚠ GUI window is not viewable (this might be the issue)")
        
        # Clean up immediately
        gui.root.destroy()
        root.destroy()
        print("✓ GUI cleaned up successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ GUI creation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("Phalanx C-sUAS Simulation - GUI Hang Diagnostic")
    print("This script will identify why simStart.py hangs after GUI initialization")
    print()
    
    tests = [
        test_basic_tkinter,
        test_toplevel_window,
        test_wait_window_mechanism,
        test_simInput_import,
        test_simInput_gui_creation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    test_names = [
        "Basic Tkinter",
        "Toplevel Window",
        "wait_window Mechanism", 
        "simInput Import",
        "simInput GUI Creation"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {name}: {status}")
    
    if all(results):
        print("\n✓ All tests passed - The issue may be environment-specific")
        print("Recommendations:")
        print("- Try running simStart.py with --batch flag to bypass GUI")
        print("- Check for display/X11 forwarding issues if using SSH")
        print("- Verify all required dependencies are installed")
    else:
        print("\n✗ Some tests failed - These are likely causes of the hang")
        failed_tests = [name for name, result in zip(test_names, results) if not result]
        print("Failed tests:", ", ".join(failed_tests))
        
        if not results[3]:  # simInput import failed
            print("\nCRITICAL: simInput import failed - check missing dependencies")
        if not results[4]:  # GUI creation failed
            print("\nCRITICAL: GUI creation failed - this is the likely cause of the hang")

if __name__ == "__main__":
    main()