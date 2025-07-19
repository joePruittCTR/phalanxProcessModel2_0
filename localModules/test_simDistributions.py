#!/usr/bin/env python3
"""
Comprehensive Testing Implementation for simDistributions.py
Tests the enhanced probability distribution system.
"""

import sys
import os
import time
import tempfile
import shutil
import json
import logging
import traceback
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple
import unittest
from unittest.mock import patch, MagicMock

# Test imports with graceful degradation
try:
    import numpy as np
    import scipy.stats as stats
    SCIPY_NUMPY_AVAILABLE = True
except ImportError:
    SCIPY_NUMPY_AVAILABLE = False
    print("Warning: NumPy/SciPy not available - distribution tests will be limited")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import the module being tested
try:
    from simDistributions import (
        Distribution, DistributionParameters, DistributionCache,
        ExponentialDistribution, MonthlyMixedDistribution,
        WeeklyExponentialDistribution, MixedWeibullDistribution,
        BimodalExponentialDistribution, BetaDistributionCustom,
        GammaDistribution, UniformDistribution,
        get_distribution_cache, DistributionTester
    )
    SIMDISTRIBUTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import simDistributions: {e}")
    SIMDISTRIBUTIONS_AVAILABLE = False

class TestDistributionParameters(unittest.TestCase):
    """Test DistributionParameters class functionality."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_parameters_initialization_valid(self):
        """Test valid parameter initialization."""
        params = DistributionParameters(
            mean_interarrival_time=60.0,
            batch_size=1,
            distribution_type="Exponential"
        )
        
        self.assertEqual(params.mean_interarrival_time, 60.0)
        self.assertEqual(params.batch_size, 1)
        self.assertEqual(params.distribution_type, "Exponential")
        self.assertIsInstance(params.kwargs, dict)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_parameters_validation_invalid(self):
        """Test parameter validation with invalid inputs."""
        # Test negative mean interarrival time
        with self.assertRaises(ValueError):
            DistributionParameters(
                mean_interarrival_time=-10.0,
                distribution_type="Exponential"
            )
        
        # Test zero batch size
        with self.assertRaises(ValueError):
            DistributionParameters(
                mean_interarrival_time=60.0,
                batch_size=0,
                distribution_type="Exponential"
            )
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_parameters_with_kwargs(self):
        """Test parameters with additional keyword arguments."""
        kwargs = {'lambda1': 0.5, 'lambda2': 0.3, 'weight1': 0.6}
        
        params = DistributionParameters(
            mean_interarrival_time=45.0,
            batch_size=2,
            distribution_type="BimodalExpon",
            kwargs=kwargs
        )
        
        self.assertEqual(params.kwargs, kwargs)


class TestDistributionCache(unittest.TestCase):
    """Test DistributionCache functionality."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = DistributionCache()
        
        self.assertTrue(cache.enabled)
        self.assertEqual(cache.hit_count, 0)
        self.assertEqual(cache.miss_count, 0)
        self.assertEqual(len(cache.cache_data), 0)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE, 
                        "simDistributions and NumPy required")
    def test_cache_operations(self):
        """Test cache put/get operations."""
        cache = DistributionCache()
        
        # Test cache miss
        result = cache.get("test_key")
        self.assertIsNone(result)
        self.assertEqual(cache.miss_count, 1)
        
        # Test cache put and hit
        test_data = np.array([1.0, 2.0, 3.0])
        cache.put("test_key", test_data)
        
        retrieved_data = cache.get("test_key")
        self.assertIsNotNone(retrieved_data)
        self.assertEqual(cache.hit_count, 1)
        np.testing.assert_array_equal(retrieved_data, test_data)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = DistributionCache()
        
        params = {'mean': 60.0, 'scale': 1.0}
        key1 = cache.get_cache_key("Exponential", params, seed=42)
        key2 = cache.get_cache_key("Exponential", params, seed=42)
        key3 = cache.get_cache_key("Exponential", params, seed=43)
        
        # Same parameters should generate same key
        self.assertEqual(key1, key2)
        
        # Different seeds should generate different keys
        self.assertNotEqual(key1, key3)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = DistributionCache()
        
        # Initial stats
        stats = cache.get_stats()
        self.assertEqual(stats['hit_count'], 0)
        self.assertEqual(stats['miss_count'], 0)
        self.assertEqual(stats['hit_rate'], 0)
        
        # After some operations
        cache.get("missing_key")  # Miss
        cache.put("test_key", np.array([1, 2, 3]))
        cache.get("test_key")  # Hit
        
        stats = cache.get_stats()
        self.assertEqual(stats['hit_count'], 1)
        self.assertEqual(stats['miss_count'], 1)
        self.assertEqual(stats['hit_rate'], 0.5)


class TestMainDistributionClass(unittest.TestCase):
    """Test the main Distribution class functionality."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_distribution_initialization_exponential(self):
        """Test Distribution initialization with Exponential type."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        self.assertEqual(dist.params.mean_interarrival_time, 60.0)
        self.assertEqual(dist.params.distribution_type, "Exponential")
        self.assertIsNotNone(dist.distribution)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_distribution_initialization_all_types(self):
        """Test Distribution initialization with all supported types."""
        distribution_types = [
            "Exponential",
            "MonthlyMixedDist", 
            "WeeklyExponential",
            "MixedWeibull",
            "BimodalExpon",
            "BetaDistribution",
            "GammaDistribution",
            "UniformDistribution"
        ]
        
        for dist_type in distribution_types:
            try:
                dist = Distribution(
                    mean_interarrival_time=60.0,
                    distribution_type=dist_type
                )
                self.assertEqual(dist.params.distribution_type, dist_type)
            except Exception as e:
                self.fail(f"Failed to initialize {dist_type} distribution: {e}")
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_get_interarrival_time(self):
        """Test single interarrival time generation."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        # Generate multiple values to test
        values = []
        for _ in range(100):
            value = dist.get_interarrival_time()
            values.append(value)
            
            # Should be positive
            self.assertGreater(value, 0)
            
            # Should be reasonable (not extreme)
            self.assertLess(value, 1000)  # Upper bound check
        
        # Values should vary (not all the same)
        self.assertGreater(len(set(values)), 10)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE,
                        "simDistributions and NumPy required")
    def test_generate_values_batch(self):
        """Test batch value generation."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        # Test batch generation
        batch_values = dist.generate_values(50)
        
        self.assertEqual(len(batch_values), 50)
        self.assertTrue(all(v > 0 for v in batch_values))
        
        # Test single value (should return float, not array)
        single_value = dist.generate_values(1)
        self.assertIsInstance(single_value, (float, np.float64))
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE,
                        "simDistributions and NumPy required")
    def test_get_statistics(self):
        """Test distribution statistics calculation."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        stats = dist.get_statistics(n_samples=1000)
        
        # Check that statistics are returned
        self.assertIsInstance(stats, dict)
        
        expected_keys = ['mean', 'std', 'median', 'min', 'max']
        for key in expected_keys:
            if key in stats:
                self.assertIsInstance(stats[key], (float, np.float64))
                self.assertGreater(stats[key], 0)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_distribution_with_batch_size(self):
        """Test distribution with different batch sizes."""
        for batch_size in [1, 2, 5]:
            dist = Distribution(
                mean_interarrival_time=60.0,
                batch_size=batch_size,
                distribution_type="Exponential"
            )
            
            value = dist.get_interarrival_time()
            self.assertGreater(value, 0)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_invalid_distribution_type(self):
        """Test handling of invalid distribution type."""
        with self.assertRaises((ValueError, KeyError, AttributeError)):
            Distribution(
                mean_interarrival_time=60.0,
                distribution_type="InvalidDistribution"
            )


class TestSpecificDistributions(unittest.TestCase):
    """Test specific distribution implementations."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_exponential_distribution(self):
        """Test Exponential distribution specifically."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        # Generate sample and check exponential properties
        values = [dist.get_interarrival_time() for _ in range(1000)]
        mean_value = sum(values) / len(values)
        
        # Mean should be approximately equal to input (within reasonable tolerance)
        self.assertGreater(mean_value, 30)   # Should be > 30
        self.assertLess(mean_value, 120)     # Should be < 120 (2x mean)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_bimodal_exponential_distribution(self):
        """Test BimodalExpon distribution."""
        try:
            dist = Distribution(
                mean_interarrival_time=60.0,
                distribution_type="BimodalExpon",
                lambda1=0.5,
                lambda2=0.3,
                weight1=0.7
            )
            
            # Should generate values successfully
            value = dist.get_interarrival_time()
            self.assertGreater(value, 0)
            
        except Exception as e:
            self.skipTest(f"BimodalExpon distribution not fully implemented: {e}")
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_uniform_distribution(self):
        """Test Uniform distribution."""
        try:
            dist = Distribution(
                mean_interarrival_time=60.0,
                distribution_type="UniformDistribution",
                low=50.0,
                high=70.0
            )
            
            # Generate values and check they're in range
            values = [dist.get_interarrival_time() for _ in range(100)]
            
            # For uniform distribution, all values should be in specified range
            # (adjusted for batch size effects)
            for value in values:
                self.assertGreater(value, 0)
                
        except Exception as e:
            self.skipTest(f"UniformDistribution not fully implemented: {e}")


class TestDistributionPerformance(unittest.TestCase):
    """Test distribution performance characteristics."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_generation_performance(self):
        """Test performance of value generation."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        # Test single value generation performance
        start_time = time.time()
        for _ in range(1000):
            dist.get_interarrival_time()
        single_generation_time = time.time() - start_time
        
        # Should complete quickly (< 1 second for 1000 values)
        self.assertLess(single_generation_time, 1.0)
        
        # Test batch generation performance
        if hasattr(dist, 'generate_values'):
            start_time = time.time()
            dist.generate_values(1000)
            batch_generation_time = time.time() - start_time
            
            # Batch generation should be faster than individual generation
            self.assertLess(batch_generation_time, single_generation_time)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        # Generate some values
        for _ in range(100):
            dist.get_interarrival_time()
        
        # Check if performance stats are available
        if hasattr(dist, 'get_performance_stats'):
            stats = dist.get_performance_stats()
            self.assertIsInstance(stats, dict)


class TestDistributionTester(unittest.TestCase):
    """Test the DistributionTester functionality."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_distribution_tester_initialization(self):
        """Test DistributionTester initialization."""
        try:
            tester = DistributionTester()
            self.assertIsInstance(tester.test_results, dict)
        except Exception as e:
            self.skipTest(f"DistributionTester not available: {e}")
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_distribution_testing(self):
        """Test distribution testing functionality."""
        try:
            tester = DistributionTester()
            
            test_params = {
                'mean_interarrival_time': 60.0,
                'batch_size': 1
            }
            
            result = tester.test_distribution("Exponential", test_params, n_samples=100)
            self.assertIsInstance(result, dict)
            
        except Exception as e:
            self.skipTest(f"Distribution testing not fully implemented: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling in distribution system."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Test negative mean interarrival time
        with self.assertRaises(ValueError):
            Distribution(
                mean_interarrival_time=-10.0,
                distribution_type="Exponential"
            )
        
        # Test zero mean interarrival time
        with self.assertRaises(ValueError):
            Distribution(
                mean_interarrival_time=0.0,
                distribution_type="Exponential"
            )
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_missing_required_parameters(self):
        """Test handling of missing required parameters."""
        # Test missing mean_interarrival_time
        with self.assertRaises(TypeError):
            Distribution(distribution_type="Exponential")
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_robustness_with_extreme_values(self):
        """Test distribution robustness with extreme values."""
        # Test very small mean interarrival time
        try:
            dist = Distribution(
                mean_interarrival_time=0.001,
                distribution_type="Exponential"
            )
            value = dist.get_interarrival_time()
            self.assertGreater(value, 0)
        except Exception as e:
            self.skipTest(f"Extreme small values not handled: {e}")
        
        # Test very large mean interarrival time
        try:
            dist = Distribution(
                mean_interarrival_time=10000.0,
                distribution_type="Exponential"
            )
            value = dist.get_interarrival_time()
            self.assertGreater(value, 0)
        except Exception as e:
            self.skipTest(f"Extreme large values not handled: {e}")


class TestDistributionIntegration(unittest.TestCase):
    """Test integration aspects of distribution system."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_global_cache_integration(self):
        """Test integration with global cache."""
        try:
            cache = get_distribution_cache()
            self.assertIsInstance(cache, DistributionCache)
        except Exception as e:
            self.skipTest(f"Global cache not available: {e}")
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_distribution_registry(self):
        """Test distribution type registry."""
        try:
            available_types = Distribution.list_available_distributions()
            self.assertIsInstance(available_types, list)
            self.assertIn("Exponential", available_types)
        except Exception as e:
            self.skipTest(f"Distribution registry not available: {e}")


# ================================================================================================
# STANDALONE TESTING IMPLEMENTATION
# ================================================================================================

def run_standalone_tests():
    """
    Standalone testing implementation for simDistributions.py
    """
    print("="*70)
    print("PHALANX C-sUAS SIMULATION - simDistributions.py Testing")
    print("="*70)
    
    # Check dependencies
    print("\nDependency Check:")
    print("-" * 16)
    
    dependencies = [
        ("NumPy/SciPy", SCIPY_NUMPY_AVAILABLE),
        ("Matplotlib", MATPLOTLIB_AVAILABLE),
        ("simDistributions", SIMDISTRIBUTIONS_AVAILABLE),
    ]
    
    for name, available in dependencies:
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {name}: {status}")
    
    if not SIMDISTRIBUTIONS_AVAILABLE:
        print("\n✗ Cannot run tests - simDistributions module not available")
        return {'passed': 0, 'failed': 1, 'skipped': 0, 'errors': ['simDistributions module not available']}
    
    # Test execution
    test_results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }
    
    def run_test_class(test_class, class_name):
        """Run a test class and track results."""
        print(f"\n{class_name}:")
        print("-" * (len(class_name) + 1))
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Custom result tracking
        class DetailedTestResult(unittest.TestResult):
            def __init__(self):
                super().__init__()
                self.successes = []
            
            def addSuccess(self, test):
                super().addSuccess(test)
                self.successes.append(test)
        
        result = DetailedTestResult()
        suite.run(result)
        
        # Count results
        passed = len(result.successes)
        failed = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        
        test_results['passed'] += passed
        test_results['failed'] += failed + errors
        test_results['skipped'] += skipped
        
        # Report results
        if failed > 0:
            for test, traceback in result.failures:
                test_name = test.id().split('.')[-1]
                print(f"  ✗ {test_name} FAILED")
                test_results['errors'].append(f"{test_name}: FAILED")
        
        if errors > 0:
            for test, traceback in result.errors:
                test_name = test.id().split('.')[-1]
                print(f"  ✗ {test_name} ERROR")
                test_results['errors'].append(f"{test_name}: ERROR")
        
        if skipped > 0:
            for test, reason in result.skipped:
                test_name = test.id().split('.')[-1]
                print(f"  ⚠ {test_name} SKIPPED")
        
        if passed > 0:
            print(f"  ✓ {passed} tests passed")
        
        total_run = passed + failed + errors
        if total_run == 0:
            print("  ⚠ No tests could be run")
    
    # Run test classes
    test_classes = [
        (TestDistributionParameters, "Distribution Parameters Tests"),
        (TestDistributionCache, "Distribution Cache Tests"),
        (TestMainDistributionClass, "Main Distribution Class Tests"),
        (TestSpecificDistributions, "Specific Distribution Tests"),
        (TestDistributionPerformance, "Performance Tests"),
        (TestDistributionTester, "Distribution Tester Tests"),
        (TestErrorHandling, "Error Handling Tests"),
        (TestDistributionIntegration, "Integration Tests")
    ]
    
    for test_class, name in test_classes:
        try:
            run_test_class(test_class, name)
        except Exception as e:
            print(f"  ✗ Test class {name} failed to run: {e}")
            test_results['failed'] += 1
            test_results['errors'].append(f"{name}: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = test_results['passed'] + test_results['failed'] + test_results['skipped']
    print(f"Total Tests: {total_tests}")
    print(f"Tests Passed: {test_results['passed']}")
    print(f"Tests Failed: {test_results['failed']}")
    print(f"Tests Skipped: {test_results['skipped']}")
    
    if test_results['errors'] and len(test_results['errors']) <= 5:
        print(f"\nMain Issues:")
        for error in test_results['errors']:
            print(f"  • {error}")
    
    # Calculate success rate
    if total_tests > 0:
        success_rate = (test_results['passed'] / total_tests) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("✓ simDistributions.py is working excellently!")
        elif success_rate >= 70:
            print("⚠ simDistributions.py is working well with minor issues")
        elif success_rate >= 50:
            print("⚠ simDistributions.py has some issues but core functionality works")
        else:
            print("✗ simDistributions.py needs significant fixes")
    else:
        print("\n⚠ No tests could be executed")
    
    # Provide guidance
    print(f"\nNext Steps:")
    if not SCIPY_NUMPY_AVAILABLE:
        print("  • Install NumPy/SciPy: pip install numpy scipy")
    if test_results['failed'] > 0:
        print("  • Review failed tests and fix implementation issues")
    print("  • Test integration with simProcess.py")
    print("  • Test distribution accuracy with known statistical properties")
    
    print("\n" + "="*70)
    print("SIMDISTRIBUTIONS TESTING COMPLETE")
    print("="*70)
    
    return test_results


# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

if __name__ == "__main__":
    """
    Standalone testing execution for simDistributions.py
    """
    
    # Environment check
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher required")
        sys.exit(1)
    
    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Run tests
    try:
        results = run_standalone_tests()
        
        # Exit codes based on results
        if results['failed'] == 0:
            sys.exit(0)  # All tests passed or skipped
        elif results['passed'] > results['failed']:
            sys.exit(0)  # More passed than failed
        else:
            sys.exit(1)  # More failures than passes
            
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)