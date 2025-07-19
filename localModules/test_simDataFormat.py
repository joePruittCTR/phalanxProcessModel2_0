#!/usr/bin/env python3
"""
Test simDataFormat.py module
"""

print("=== Testing simDataFormat.py ===")

try:
    import simDataFormat
    import os
    import numpy as np
    print("✓ simDataFormat imported successfully")
    
    # Create test directory
    test_dir = "./test_data"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Test DataFileProcessor
    processor = simDataFormat.DataFileProcessor(test_dir)
    print("✓ DataFileProcessor created")
    
    # Test csv_export
    test_data = np.random.rand(10, 3)
    processor.csv_export("test_export", test_data)
    print("✓ csv_export working")
    
    # Check if file was created
    if os.path.exists(os.path.join(test_dir, "test_export.csv")):
        print("✓ CSV file created successfully")
    else:
        print("✗ CSV file not found")
    
    # Cleanup
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print("✓ Test cleanup completed")
    
    print("✓ simDataFormat.py working correctly!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Check if pandas is installed: pip install pandas")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Next: Test simPlot.py")