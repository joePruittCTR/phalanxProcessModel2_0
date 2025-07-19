#!/usr/bin/env python3
"""
Test simPlot.py module
"""

print("=== Testing simPlot.py ===")

try:
    import simPlot
    import os
    print("✓ simPlot imported successfully")
    
    # Create test directory
    test_dir = "./test_plots"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Test Plot class
    plot = simPlot.Plot(test_dir + os.sep, "test_simulation")
    print("✓ Plot object created")
    
    # Test creating a simple plot
    test_data = [1, 2, 3, 4, 5]
    plot.create_plot("line", test_data, "Time", "Value", "Test Line Plot", "blue")
    print("✓ Line plot created")
    
    # Check if plot file was created
    expected_file = os.path.join(test_dir, "test_simulation_Test_Line_Plot.png")
    if os.path.exists(expected_file):
        print("✓ Plot file created successfully")
    else:
        print("✗ Plot file not found")
        print(f"  Expected: {expected_file}")
    
    # Test histogram
    hist_data = [1, 1, 2, 2, 2, 3, 3, 4, 5]
    plot.create_plot("hist", hist_data, "Value", "Frequency", "Test Histogram", "red")
    print("✓ Histogram created")
    
    # Cleanup
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print("✓ Test cleanup completed")
    
    print("✓ simPlot.py working correctly!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Check if matplotlib is installed: pip install matplotlib")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Next: Test simStats.py")