#!/usr/bin/env python3
"""
Diagnostic Test for simProcess.py - Pinpoint Parameter Issues
This will help us understand exactly what's happening with the parameter validation.
"""

import sys
import traceback

# Import the module being tested
try:
    from simProcess import SimulationParameters, SensorParameters
    print("✓ Successfully imported simProcess modules")
except ImportError as e:
    print(f"✗ Failed to import simProcess: {e}")
    sys.exit(1)

def diagnose_sensor_parameters():
    """Diagnose SensorParameters initialization issues."""
    print("\n" + "="*60)
    print("DIAGNOSING SENSOR PARAMETERS")
    print("="*60)
    
    try:
        print("1. Testing minimal SensorParameters initialization...")
        sensor = SensorParameters(
            name="test_sensor",
            display_name="Test Sensor"
        )
        
        print(f"   ✓ Created sensor: {sensor.name}")
        print(f"   ✓ Display name: {sensor.display_name}")
        print(f"   ✓ Active: {sensor.active}")
        print(f"   ✓ Processing time: {sensor.processing_time}")
        print(f"   ✓ Files per month: {sensor.files_per_month}")
        print(f"   ✓ Batch size: {sensor.batch_size}")
        print(f"   ✓ Distribution type: {sensor.distribution_type}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ SensorParameters failed: {e}")
        print(f"   ✗ Traceback: {traceback.format_exc()}")
        return False

def diagnose_simulation_parameters():
    """Diagnose SimulationParameters initialization and validation."""
    print("\n" + "="*60)
    print("DIAGNOSING SIMULATION PARAMETERS")
    print("="*60)
    
    # Test with minimal valid parameters
    test_params = {
        'simFileName': 'diagnostic_test',
        'nservers': 3.0,
        'sim_time': 100.0,
        'seed': 42
    }
    
    try:
        print("1. Testing parameter initialization...")
        print(f"   Input nservers: {test_params['nservers']}")
        print(f"   Input sim_time: {test_params['sim_time']}")
        
        params = SimulationParameters(test_params)
        
        print(f"   ✓ Created SimulationParameters")
        print(f"   ✓ simFileName: {params.simFileName}")
        print(f"   ✓ nservers after init: {params.nservers}")
        print(f"   ✓ simTime after init: {getattr(params, 'simTime', 'NOT_SET')}")
        print(f"   ✓ sim_time after init: {getattr(params, 'sim_time', 'NOT_SET')}")
        print(f"   ✓ seed: {params.seed}")
        
        # Check if validation method exists
        if hasattr(params, '_validate_all_parameters'):
            print("2. Testing validation method...")
            try:
                params._validate_all_parameters()
                print("   ✓ Validation passed")
            except Exception as ve:
                print(f"   ✗ Validation failed: {ve}")
                print(f"   ✗ nservers during validation: {params.nservers}")
                print(f"   ✗ simTime during validation: {getattr(params, 'simTime', 'NOT_SET')}")
                return False
        else:
            print("2. No _validate_all_parameters method found")
        
        return True
        
    except Exception as e:
        print(f"   ✗ SimulationParameters failed: {e}")
        print(f"   ✗ Traceback: {traceback.format_exc()}")
        return False

def diagnose_parameter_attributes():
    """Diagnose parameter attribute mapping."""
    print("\n" + "="*60)
    print("DIAGNOSING PARAMETER ATTRIBUTE MAPPING")
    print("="*60)
    
    test_params = {
        'simFileName': 'attr_test',
        'nservers': 5.0,
        'sim_time': 200.0,
        'timeWindow': 2.0,
        'timeWindowYears': 1.5,
        'siprTransfer': 0.5,
        'seed': 123
    }
    
    try:
        params = SimulationParameters(test_params)
        
        print("Parameter mapping results:")
        print(f"   Input 'nservers': {test_params['nservers']} → params.nservers: {params.nservers}")
        print(f"   Input 'sim_time': {test_params['sim_time']} → params.sim_time: {getattr(params, 'sim_time', 'NOT_FOUND')}")
        print(f"   Input 'timeWindow': {test_params['timeWindow']} → params.timeWindowYears: {params.timeWindowYears}")
        print(f"   Input 'siprTransfer': {test_params['siprTransfer']} → params.siprTransferTime: {params.siprTransferTime}")
        
        # Check for simTime calculation
        if hasattr(params, 'simTime'):
            print(f"   Calculated simTime: {params.simTime}")
        
        # Check for any calculation methods
        calc_methods = [method for method in dir(params) if 'calculate' in method.lower()]
        if calc_methods:
            print(f"   Found calculation methods: {calc_methods}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Attribute mapping failed: {e}")
        return False

def diagnose_validation_logic():
    """Test the validation logic step by step."""
    print("\n" + "="*60)
    print("DIAGNOSING VALIDATION LOGIC")
    print("="*60)
    
    # Import validation function if it exists
    try:
        from simProcess import validate_simulation_setup
        print("✓ Found validate_simulation_setup function")
    except ImportError:
        print("✗ validate_simulation_setup function not found")
        return False
    
    test_params = {
        'simFileName': 'validation_test',
        'nservers': 4.0,
        'sim_time': 150.0,
        'seed': 42
    }
    
    try:
        params = SimulationParameters(test_params)
        print(f"   Created params with nservers: {params.nservers}")
        
        # Call validation
        result = validate_simulation_setup(params)
        
        print(f"   Validation result: {result}")
        print(f"   Valid: {result.get('valid', 'UNKNOWN')}")
        print(f"   Errors: {result.get('errors', [])}")
        print(f"   Warnings: {result.get('warnings', [])}")
        
        return result.get('valid', False)
        
    except Exception as e:
        print(f"   ✗ Validation logic failed: {e}")
        return False

def run_full_diagnosis():
    """Run complete diagnostic suite."""
    print("PHALANX C-sUAS SIMULATION - DIAGNOSTIC TEST")
    print("="*60)
    print("This will help identify the exact parameter validation issues")
    
    results = {
        'sensor_params': diagnose_sensor_parameters(),
        'simulation_params': diagnose_simulation_parameters(),
        'attribute_mapping': diagnose_parameter_attributes(),
        'validation_logic': diagnose_validation_logic()
    }
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} diagnostic tests passed")
    
    if total_passed == total_tests:
        print("🎉 All diagnostics passed - the issues may be test-specific!")
    else:
        print("🔍 Issues found - check the failed diagnostics above")
    
    return results

if __name__ == "__main__":
    try:
        results = run_full_diagnosis()
        
        # Exit with status based on results
        if all(results.values()):
            sys.exit(0)  # All diagnostics passed
        else:
            sys.exit(1)  # Some diagnostics failed
            
    except Exception as e:
        print(f"\nUnexpected diagnostic error: {e}")
        traceback.print_exc()
        sys.exit(1)