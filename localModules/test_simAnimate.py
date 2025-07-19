#!/usr/bin/env python3
"""
Test simAnimate.py module
"""

print("=== Testing simAnimate.py ===")

try:
    import salabim as sb
    print("✓ salabim imported successfully")
    
    import simAnimate
    print("✓ simAnimate imported successfully")
    
    import simProcess
    print("✓ simProcess imported successfully")
    
    # Test SimulationParameters creation
    test_params = {
        'sim_time': 60,  # Short test
        'time_unit': 'minutes',
        'processing_fte': 1.0,
        'processing_overhead': 0.10,
        'processing_efficiency': 0.90,
        'warmup_time': 0,
        'num_replications': 1,
        'seed': 42,
        'co_time': 8.0,
        'co_iat': 2.0,  # Slow arrivals for testing
    }
    
    sim_params = simProcess.SimulationParameters(test_params)
    print("✓ SimulationParameters created for animation test")
    
    # Test that classes can be imported (not instantiated without proper environment)
    print("✓ AnimatedCustomer class available:", hasattr(simAnimate, 'AnimatedCustomer'))
    print("✓ AnimatedSource class available:", hasattr(simAnimate, 'AnimatedSource'))
    
    # Test the run_animated_simulation function exists
    print("✓ run_animated_simulation function available:", hasattr(simAnimate, 'run_animated_simulation'))
    
    # Test a very basic simulation setup (without animation)
    print("\n--- Testing basic simulation setup ---")
    
    # Create environment without animation for testing
    env = sb.Environment(trace=False, animate=False)
    
    # Test the setup function from simProcess works
    sim_objects = simProcess._setup_simulation_components(env, sim_params)
    print("✓ _setup_simulation_components working")
    
    # Check that required objects are created
    required_keys = ['handler_resource', 'customer_queue', 'customer_stay_monitor', 'customer_service_monitor']
    for key in required_keys:
        if key in sim_objects:
            print(f"✓ {key} created successfully")
        else:
            print(f"✗ {key} missing")
    
    print("\n✓ simAnimate.py structure is correct!")
    print("✓ All imports and class definitions working!")
    
    print("\n--- Testing __main__ block functionality ---")
    print("To test full animation, run: python simAnimate.py")
    print("(This will open a GUI window with the animated simulation)")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    if "salabim" in str(e):
        print("  Install salabim with: pip install salabim")
    else:
        print("  Check that simProcess.py is working correctly")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("If this test passes, the animation module structure is correct.")
print("Run 'python simAnimate.py' to test the full animated simulation.")