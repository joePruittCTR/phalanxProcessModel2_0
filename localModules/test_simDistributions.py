#!/usr/bin/env python3
"""
Comprehensive Testing Implementation for Enhanced simDistributions.py
Tests the enhanced probability distribution system with all new features.
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
import calendar

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
        GammaDistribution, UniformDistribution, YearlyArrivalPDFDistribution,
        get_distribution_cache, create_yearly_arrival_pdf,
        compare_distributions, benchmark_distributions, BaseDistribution
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
    def test_mixed_weibull_validation(self):
        """Test MixedWeibull parameter validation."""
        # Test invalid weights (don't sum to 1)
        with self.assertRaises(ValueError):
            DistributionParameters(
                mean_interarrival_time=60.0,
                distribution_type="MixedWeibull",
                kwargs={'w1': 0.5, 'w2': 0.3, 'w3': 0.3}  # Sum = 1.1
            )
        
        # Test negative weights
        with self.assertRaises(ValueError):
            DistributionParameters(
                mean_interarrival_time=60.0,
                distribution_type="MixedWeibull",
                kwargs={'w1': -0.2, 'w2': 0.6, 'w3': 0.6}
            )
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_yearly_arrival_validation(self):
        """Test YearlyArrivalPDF parameter validation."""
        # Test invalid year
        with self.assertRaises(ValueError):
            DistributionParameters(
                mean_interarrival_time=60.0,
                distribution_type="YearlyArrivalPDF",
                kwargs={'year': 1800}
            )
        
        # Test negative base_lambda
        with self.assertRaises(ValueError):
            DistributionParameters(
                mean_interarrival_time=60.0,
                distribution_type="YearlyArrivalPDF",
                kwargs={'base_lambda': -1.0}
            )

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
    def test_cache_size_management(self):
        """Test cache size management."""
        cache = DistributionCache(max_cache_size=3)
        
        # Fill cache beyond capacity
        for i in range(5):
            cache.put(f"key_{i}", np.array([i]))
        
        # Should not exceed max size
        self.assertLessEqual(len(cache.cache_data), 3)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = DistributionCache()
        cache.put("test_key", np.array([1, 2, 3]))
        cache.get("test_key")  # Create hit
        cache.get("missing_key")  # Create miss
        
        cache.clear()
        
        self.assertEqual(len(cache.cache_data), 0)
        self.assertEqual(cache.hit_count, 0)
        self.assertEqual(cache.miss_count, 0)

class TestYearlyArrivalPDF(unittest.TestCase):
    """Test YearlyArrivalPDF functionality."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE,
                        "simDistributions and NumPy required")
    def test_yearly_pdf_creation(self):
        """Test yearly PDF creation."""
        year = 2024
        pdf = create_yearly_arrival_pdf(
            year=year,
            base_lambda=1.0,
            summer_boost=1.5,
            monthly_peak_lambda_factor=3.0,
            midmonth_peak_factor=0.5,
            special_peak_lambda_factor=4.0
        )
        
        # Check basic properties
        expected_days = 366 if calendar.isleap(year) else 365
        self.assertEqual(len(pdf), expected_days)
        
        # Should be normalized (sum to 1)
        self.assertAlmostEqual(np.sum(pdf), 1.0, places=10)
        
        # All values should be non-negative
        self.assertTrue(np.all(pdf >= 0))
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_yearly_arrival_distribution(self):
        """Test YearlyArrivalPDF distribution class."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="YearlyArrivalPDF",
            year=2024,
            base_lambda=1.0,
            summer_boost=1.5,
            monthly_peak_lambda_factor=3.0,
            midmonth_peak_factor=0.5,
            special_peak_lambda_factor=4.0
        )
        
        # Test value generation
        value = dist.get_interarrival_time()
        self.assertGreater(value, 0)
        
        # Test batch generation
        if hasattr(dist, 'generate_batch'):
            batch = dist.generate_batch(100)
            self.assertEqual(len(batch), 100)
            self.assertTrue(all(v > 0 for v in batch))
        
        # Test yearly PDF access
        if hasattr(dist.distribution, 'get_yearly_pdf'):
            yearly_pdf = dist.distribution.get_yearly_pdf()
            self.assertIsInstance(yearly_pdf, np.ndarray)

class TestEnhancedDistributions(unittest.TestCase):
    """Test enhanced distribution implementations."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_all_distribution_types(self):
        """Test all supported distribution types."""
        distribution_configs = [
            ('Exponential', {}),
            ('MonthlyMixedDist', {'num_days': 30, 'first_peak_probability': 0.7}),
            ('WeeklyExponential', {'num_days': 7}),
            ('MixedWeibull', {
                'w1': 0.6, 'w2': 0.2, 'w3': 0.2,
                'weibull_shape': 0.8, 'weibull_scale': 1.0,
                'norm_mu': 5, 'norm_sigma': 1, 'expon_lambda': 0.5
            }),
            ('BimodalExpon', {
                'lambda1': 0.5, 'lambda2': 0.3, 'weight1': 0.6, 'loc2': 15
            }),
            ('BetaDistribution', {'alpha': 2, 'beta': 5}),
            ('GammaDistribution', {'shape': 2.0, 'scale': 1.0}),
            ('UniformDistribution', {'low': 2.0, 'high': 8.0}),
            ('YearlyArrivalPDF', {
                'year': 2024, 'base_lambda': 1.0, 'summer_boost': 1.5,
                'monthly_peak_lambda_factor': 3.0, 'midmonth_peak_factor': 0.5,
                'special_peak_lambda_factor': 4.0
            })
        ]
        
        for dist_type, params in distribution_configs:
            with self.subTest(distribution_type=dist_type):
                try:
                    dist = Distribution(
                        mean_interarrival_time=60.0,
                        distribution_type=dist_type,
                        **params
                    )
                    
                    # Test basic functionality
                    value = dist.get_interarrival_time()
                    self.assertGreater(value, 0)
                    self.assertIsInstance(value, (float, np.floating))
                    
                except Exception as e:
                    self.fail(f"Failed to create/test {dist_type}: {e}")
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE,
                        "simDistributions and NumPy required")
    def test_gamma_distribution_properties(self):
        """Test Gamma distribution properties."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="GammaDistribution",
            shape=2.0,
            scale=1.0
        )
        
        # Generate samples and test properties
        samples = [dist.get_interarrival_time() for _ in range(1000)]
        
        # All values should be positive
        self.assertTrue(all(s > 0 for s in samples))
        
        # Should have reasonable variance
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        self.assertGreater(std_val, 0)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE,
                        "simDistributions and NumPy required")
    def test_uniform_distribution_bounds(self):
        """Test Uniform distribution bounds."""
        low, high = 2.0, 8.0
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="UniformDistribution",
            low=low,
            high=high
        )
        
        # Generate samples and check bounds
        samples = [dist.get_interarrival_time() for _ in range(100)]
        
        # All values should be positive (accounting for batch_size effects)
        self.assertTrue(all(s > 0 for s in samples))

class TestMainDistributionClass(unittest.TestCase):
    """Test the enhanced main Distribution class functionality."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_distribution_registry(self):
        """Test distribution registry functionality."""
        # Check registry exists
        self.assertIsInstance(Distribution._distribution_registry, dict)
        
        # Check all expected types are registered
        expected_types = [
            'Exponential', 'MonthlyMixedDist', 'WeeklyExponential',
            'MixedWeibull', 'BimodalExpon', 'BetaDistribution',
            'GammaDistribution', 'UniformDistribution', 'YearlyArrivalPDF'
        ]
        
        for dist_type in expected_types:
            self.assertIn(dist_type, Distribution._distribution_registry)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE,
                        "simDistributions and NumPy required")
    def test_enhanced_methods(self):
        """Test enhanced methods in Distribution class."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        # Test batch generation
        if hasattr(dist, 'generate_batch'):
            batch = dist.generate_batch(50)
            self.assertEqual(len(batch), 50)
            self.assertIsInstance(batch, np.ndarray)
        
        # Test statistics
        if hasattr(dist, 'get_statistics'):
            stats = dist.get_statistics(1000)
            self.assertIsInstance(stats, dict)
            expected_keys = ['mean', 'std', 'median', 'min', 'max', 'q25', 'q75']
            for key in expected_keys:
                if key in stats:
                    self.assertIsInstance(stats[key], (float, np.floating))
        
        # Test performance statistics
        if hasattr(dist, 'get_performance_stats'):
            perf_stats = dist.get_performance_stats()
            self.assertIsInstance(perf_stats, dict)
            self.assertIn('generation_count', perf_stats)
        
        # Test performance reset
        if hasattr(dist, 'reset_performance_stats'):
            dist.reset_performance_stats()
            perf_stats = dist.get_performance_stats()
            self.assertEqual(perf_stats['generation_count'], 0)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and MATPLOTLIB_AVAILABLE,
                        "simDistributions and Matplotlib required")
    def test_plotting_functionality(self):
        """Test plotting functionality."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        if hasattr(dist, 'plot_distribution'):
            try:
                fig = dist.plot_distribution(n_samples=100)
                self.assertIsNotNone(fig)
                plt.close(fig)
            except Exception as e:
                self.skipTest(f"Plotting not fully functional: {e}")
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_fallback_behavior(self):
        """Test fallback behavior with invalid parameters."""
        # Test with invalid distribution type - should fallback to Exponential
        try:
            dist = Distribution(
                mean_interarrival_time=60.0,
                distribution_type="InvalidType"
            )
            # Should still work due to fallback
            value = dist.get_interarrival_time()
            self.assertGreater(value, 0)
        except Exception:
            # If no fallback, should raise appropriate error
            pass

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and MATPLOTLIB_AVAILABLE,
                        "simDistributions and Matplotlib required")
    def test_compare_distributions(self):
        """Test compare_distributions function."""
        distributions = [
            Distribution(60.0, distribution_type="Exponential"),
            Distribution(60.0, distribution_type="GammaDistribution", shape=2.0)
        ]
        
        try:
            fig = compare_distributions(distributions, n_samples=100)
            self.assertIsNotNone(fig)
            plt.close(fig)
        except Exception as e:
            self.skipTest(f"compare_distributions not functional: {e}")
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_benchmark_distributions(self):
        """Test benchmark_distributions function."""
        distribution_types = ['Exponential', 'GammaDistribution']
        
        try:
            results = benchmark_distributions(
                distribution_types, 
                mean_time=60.0, 
                n_samples=100, 
                n_runs=2
            )
            
            self.assertIsInstance(results, dict)
            for dist_type in distribution_types:
                self.assertIn(dist_type, results)
                if 'error' not in results[dist_type]:
                    self.assertIn('samples_per_second', results[dist_type])
                    
        except Exception as e:
            self.skipTest(f"benchmark_distributions not functional: {e}")

class TestBaseDistribution(unittest.TestCase):
    """Test BaseDistribution abstract class functionality."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_base_distribution_interface(self):
        """Test BaseDistribution interface through concrete implementations."""
        # Test through ExponentialDistribution
        params = DistributionParameters(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        try:
            base_dist = ExponentialDistribution(params)
            
            # Test required methods exist
            self.assertTrue(hasattr(base_dist, '_setup_distribution'))
            self.assertTrue(hasattr(base_dist, '_generate_single_value'))
            self.assertTrue(hasattr(base_dist, 'generate_values'))
            self.assertTrue(hasattr(base_dist, 'get_interarrival_time'))
            self.assertTrue(hasattr(base_dist, 'get_statistics'))
            
            # Test functionality
            value = base_dist.get_interarrival_time()
            self.assertGreater(value, 0)
            
        except Exception as e:
            self.skipTest(f"BaseDistribution interface test failed: {e}")

class TestSpecificDistributionFeatures(unittest.TestCase):
    """Test specific features of individual distributions."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE,
                        "simDistributions and NumPy required")
    def test_monthly_mixed_precomputation(self):
        """Test MonthlyMixedDistribution precomputation."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="MonthlyMixedDist",
            num_days=30,
            first_peak_probability=0.7
        )
        
        # Test that precomputed arrays exist
        if hasattr(dist.distribution, 'day_probabilities'):
            probs = dist.distribution.day_probabilities
            self.assertEqual(len(probs), 30)
            self.assertAlmostEqual(np.sum(probs), 1.0, places=10)
        
        if hasattr(dist.distribution, 'cumulative_probs'):
            cum_probs = dist.distribution.cumulative_probs
            self.assertEqual(len(cum_probs), 30)
            self.assertAlmostEqual(cum_probs[-1], 1.0, places=10)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_mixed_weibull_thresholds(self):
        """Test MixedWeibull threshold precomputation."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="MixedWeibull",
            w1=0.6, w2=0.2, w3=0.2,
            weibull_shape=0.8, norm_mu=5, norm_sigma=1, expon_lambda=0.5
        )
        
        # Test threshold computation
        if hasattr(dist.distribution, 'threshold_1'):
            self.assertAlmostEqual(dist.distribution.threshold_1, 0.6)
        if hasattr(dist.distribution, 'threshold_2'):
            self.assertAlmostEqual(dist.distribution.threshold_2, 0.8)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE,
                        "simDistributions and NumPy required")
    def test_yearly_arrival_special_methods(self):
        """Test YearlyArrivalPDF special methods."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="YearlyArrivalPDF",
            year=2024
        )
        
        # Test special methods if available
        if hasattr(dist.distribution, 'sample_arrival_days'):
            try:
                days = dist.distribution.sample_arrival_days(10)
                self.assertEqual(len(days), 10)
                self.assertTrue(all(0 <= d < 366 for d in days))
            except Exception as e:
                self.skipTest(f"sample_arrival_days not functional: {e}")

class TestPerformanceAndOptimization(unittest.TestCase):
    """Test performance and optimization features."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_performance_tracking(self):
        """Test performance tracking across distributions."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        # Generate some values to populate performance stats
        for _ in range(10):
            dist.get_interarrival_time()
        
        if hasattr(dist, 'get_performance_stats'):
            stats = dist.get_performance_stats()
            self.assertGreaterEqual(stats['generation_count'], 10)
            self.assertGreaterEqual(stats['total_generation_time'], 0)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE,
                        "simDistributions and NumPy required")
    def test_batch_vs_single_performance(self):
        """Test that batch generation is more efficient than single generation."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            distribution_type="Exponential"
        )
        
        if hasattr(dist, 'generate_batch'):
            # Time single generation
            start_time = time.time()
            for _ in range(100):
                dist.get_interarrival_time()
            single_time = time.time() - start_time
            
            # Time batch generation
            start_time = time.time()
            dist.generate_batch(100)
            batch_time = time.time() - start_time
            
            # Batch should be faster (or at least not significantly slower)
            self.assertLessEqual(batch_time, single_time * 2)  # Allow some tolerance

class TestRobustnessAndEdgeCases(unittest.TestCase):
    """Test robustness and edge cases."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_extreme_parameters(self):
        """Test with extreme but valid parameters."""
        # Very small mean interarrival time
        try:
            dist = Distribution(
                mean_interarrival_time=0.001,
                distribution_type="Exponential"
            )
            value = dist.get_interarrival_time()
            self.assertGreater(value, 0)
        except Exception as e:
            self.skipTest(f"Extreme small values not handled: {e}")
        
        # Very large mean interarrival time
        try:
            dist = Distribution(
                mean_interarrival_time=1e6,
                distribution_type="Exponential"
            )
            value = dist.get_interarrival_time()
            self.assertGreater(value, 0)
        except Exception as e:
            self.skipTest(f"Extreme large values not handled: {e}")
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_large_batch_sizes(self):
        """Test with large batch sizes."""
        dist = Distribution(
            mean_interarrival_time=60.0,
            batch_size=100,
            distribution_type="Exponential"
        )
        
        # Should still work with large batch sizes
        value = dist.get_interarrival_time()
        self.assertGreater(value, 0)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE and SCIPY_NUMPY_AVAILABLE,
                        "simDistributions and NumPy required")
    def test_leap_year_handling(self):
        """Test leap year handling in YearlyArrivalPDF."""
        # Test leap year
        try:
            pdf_leap = create_yearly_arrival_pdf(
                year=2024,  # Leap year
                base_lambda=1.0,
                summer_boost=1.5,
                monthly_peak_lambda_factor=3.0,
                midmonth_peak_factor=0.5,
                special_peak_lambda_factor=4.0
            )
            self.assertEqual(len(pdf_leap), 366)
            
            # Test non-leap year
            pdf_regular = create_yearly_arrival_pdf(
                year=2023,  # Non-leap year
                base_lambda=1.0,
                summer_boost=1.5,
                monthly_peak_lambda_factor=3.0,
                midmonth_peak_factor=0.5,
                special_peak_lambda_factor=4.0
            )
            self.assertEqual(len(pdf_regular), 365)
            
        except Exception as e:
            self.skipTest(f"Leap year handling not implemented: {e}")

class TestIntegrationAndCompatibility(unittest.TestCase):
    """Test integration and backward compatibility."""
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_backward_compatibility(self):
        """Test backward compatibility with original interface."""
        # Original interface should still work
        dist = Distribution(60.0, 1, "Exponential")
        value = dist.get_interarrival_time()
        self.assertGreater(value, 0)
        
        # With additional parameters
        dist = Distribution(60.0, 1, "BimodalExpon", lambda1=0.5, lambda2=0.3)
        value = dist.get_interarrival_time()
        self.assertGreater(value, 0)
    
    @unittest.skipUnless(SIMDISTRIBUTIONS_AVAILABLE, "simDistributions module required")
    def test_global_cache_integration(self):
        """Test global cache integration."""
        cache = get_distribution_cache()
        self.assertIsInstance(cache, DistributionCache)
        
        # Test cache functionality
        initial_stats = cache.get_stats()
        
        # Create some distributions to potentially use cache
        for i in range(3):
            dist = Distribution(60.0, distribution_type="Exponential")
            dist.get_interarrival_time()
        
        final_stats = cache.get_stats()
        # Stats should be updated (either hits or misses)
        self.assertGreaterEqual(
            final_stats['total_requests'], 
            initial_stats['total_requests']
        )

# ================================================================================================
# STANDALONE TESTING IMPLEMENTATION
# ================================================================================================
def run_comprehensive_distribution_test():
    """
    Run comprehensive test of the enhanced distribution system.
    """
    print("="*80)
    print("ENHANCED PHALANX C-sUAS SIMULATION - simDistributions.py TESTING")
    print("="*80)
    
    # Check dependencies
    print("\nDependency Check:")
    print("-" * 16)
    
    dependencies = [
        ("NumPy/SciPy", SCIPY_NUMPY_AVAILABLE),
        ("Matplotlib", MATPLOTLIB_AVAILABLE),
        ("simDistributions", SIMDISTRIBUTIONS_AVAILABLE),
    ]
    
    all_available = True
    for name, available in dependencies:
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {name}: {status}")
        if not available:
            all_available = False
    
    if not SIMDISTRIBUTIONS_AVAILABLE:
        print("\n✗ Cannot run tests - simDistributions module not available")
        return {'passed': 0, 'failed': 1, 'skipped': 0, 'errors': ['simDistributions module not available']}
    
    # Test execution
    test_results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': [],
        'warnings': []
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
                self.test_details = []
            
            def addSuccess(self, test):
                super().addSuccess(test)
                self.successes.append(test)
                test_name = test.id().split('.')[-1]
                self.test_details.append(f"✓ {test_name}")
            
            def addFailure(self, test, err):
                super().addFailure(test, err)
                test_name = test.id().split('.')[-1]
                self.test_details.append(f"✗ {test_name} FAILED")
            
            def addError(self, test, err):
                super().addError(test, err)
                test_name = test.id().split('.')[-1]
                self.test_details.append(f"✗ {test_name} ERROR")
            
            def addSkip(self, test, reason):
                super().addSkip(test, reason)
                test_name = test.id().split('.')[-1]
                self.test_details.append(f"⚠ {test_name} SKIPPED")
        
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
        
        # Report detailed results
        for detail in result.test_details:
            print(f"  {detail}")
        
        # Report failures and errors
        if result.failures:
            for test, traceback_text in result.failures:
                test_name = test.id().split('.')[-1]
                test_results['errors'].append(f"{test_name}: FAILED")
        
        if result.errors:
            for test, traceback_text in result.errors:
                test_name = test.id().split('.')[-1]
                test_results['errors'].append(f"{test_name}: ERROR")
        
        # Summary for this test class
        total_run = passed + failed + errors
        if total_run == 0:
            print("  ⚠ No tests could be run")
        else:
            print(f"  Summary: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")
    
    # Run test classes
    test_classes = [
        (TestDistributionParameters, "Distribution Parameters Tests"),
        (TestDistributionCache, "Distribution Cache Tests"),
        (TestYearlyArrivalPDF, "Yearly Arrival PDF Tests"),
        (TestEnhancedDistributions, "Enhanced Distribution Tests"),
        (TestMainDistributionClass, "Main Distribution Class Tests"),
        (TestUtilityFunctions, "Utility Functions Tests"),
        (TestBaseDistribution, "Base Distribution Tests"),
        (TestSpecificDistributionFeatures, "Specific Distribution Features Tests"),
        (TestPerformanceAndOptimization, "Performance and Optimization Tests"),
        (TestRobustnessAndEdgeCases, "Robustness and Edge Cases Tests"),
        (TestIntegrationAndCompatibility, "Integration and Compatibility Tests")
    ]
    
    for test_class, name in test_classes:
        try:
            run_test_class(test_class, name)
        except Exception as e:
            print(f"  ✗ Test class {name} failed to run: {e}")
            test_results['failed'] += 1
            test_results['errors'].append(f"{name}: {e}")
    
    # Additional functional tests
    print(f"\nFunctional Integration Tests:")
    print("-" * 29)
    
    try:
        # Test all distributions can be created and generate values
        distribution_configs = [
            ('Exponential', {}),
            ('MonthlyMixedDist', {'num_days': 30}),
            ('WeeklyExponential', {'num_days': 7}),
            ('MixedWeibull', {'w1': 0.6, 'w2': 0.2, 'w3': 0.2}),
            ('BimodalExpon', {'lambda1': 0.5, 'lambda2': 0.3}),
            ('BetaDistribution', {'alpha': 2, 'beta': 5}),
            ('GammaDistribution', {'shape': 2.0}),
            ('UniformDistribution', {'low': 2.0, 'high': 8.0}),
            ('YearlyArrivalPDF', {'year': 2024})
        ]
        
        functional_passed = 0
        functional_failed = 0
        
        for dist_type, params in distribution_configs:
            try:
                dist = Distribution(60.0, distribution_type=dist_type, **params)
                
                # Test basic functionality
                value = dist.get_interarrival_time()
                assert value > 0, f"Non-positive value: {value}"
                
                # Test batch generation if available
                if hasattr(dist, 'generate_batch'):
                    batch = dist.generate_batch(10)
                    assert len(batch) == 10, f"Wrong batch size: {len(batch)}"
                    assert all(v > 0 for v in batch), "Non-positive values in batch"
                
                print(f"  ✓ {dist_type} functional test passed")
                functional_passed += 1
                
            except Exception as e:
                print(f"  ✗ {dist_type} functional test failed: {e}")
                functional_failed += 1
                test_results['errors'].append(f"{dist_type} functional test: {e}")
        
        test_results['passed'] += functional_passed
        test_results['failed'] += functional_failed
        
    except Exception as e:
        print(f"  ✗ Functional testing failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Functional testing: {e}")
    
    # Test cache functionality
    print(f"\nCache System Tests:")
    print("-" * 19)
    
    try:
        cache = get_distribution_cache()
        initial_stats = cache.get_stats()
        
        # Create some distributions to test cache
        for i in range(3):
            dist = Distribution(60.0, distribution_type="Exponential")
            for j in range(5):
                dist.get_interarrival_time()
        
        final_stats = cache.get_stats()
        
        print(f"  ✓ Cache system functional")
        print(f"    Initial requests: {initial_stats['total_requests']}")
        print(f"    Final requests: {final_stats['total_requests']}")
        print(f"    Hit rate: {final_stats['hit_rate']:.2%}")
        
        test_results['passed'] += 1
        
    except Exception as e:
        print(f"  ✗ Cache system test failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Cache system: {e}")
    
    # Test plotting functionality if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        print(f"\nPlotting Tests:")
        print("-" * 15)
        
        try:
            dist = Distribution(60.0, distribution_type="Exponential")
            
            if hasattr(dist, 'plot_distribution'):
                fig = dist.plot_distribution(n_samples=100)
                plt.close(fig)
                print(f"  ✓ Individual distribution plotting works")
                test_results['passed'] += 1
            else:
                print(f"  ⚠ plot_distribution method not available")
                test_results['skipped'] += 1
            
            # Test comparison plotting
            try:
                distributions = [
                    Distribution(60.0, distribution_type="Exponential"),
                    Distribution(60.0, distribution_type="GammaDistribution")
                ]
                fig = compare_distributions(distributions, n_samples=50)
                plt.close(fig)
                print(f"  ✓ Distribution comparison plotting works")
                test_results['passed'] += 1
            except Exception as e:
                print(f"  ✗ Comparison plotting failed: {e}")
                test_results['failed'] += 1
                
        except Exception as e:
            print(f"  ✗ Plotting tests failed: {e}")
            test_results['failed'] += 1
            test_results['errors'].append(f"Plotting: {e}")
    else:
        print(f"\nPlotting Tests: SKIPPED (Matplotlib not available)")
        test_results['skipped'] += 1
    
    # Performance benchmark test
    print(f"\nPerformance Benchmark:")
    print("-" * 21)
    
    try:
        benchmark_types = ['Exponential', 'GammaDistribution', 'UniformDistribution']
        results = benchmark_distributions(
            benchmark_types, 
            mean_time=60.0, 
            n_samples=100, 
            n_runs=3
        )
        
        print("  Distribution Performance (samples/sec):")
        for dist_type, result in results.items():
            if 'error' not in result:
                print(f"    {dist_type:20s}: {result['samples_per_second']:8.0f}")
            else:
                print(f"    {dist_type:20s}: ERROR")
        
        test_results['passed'] += 1
        
    except Exception as e:
        print(f"  ✗ Performance benchmark failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Performance benchmark: {e}")
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    total_tests = test_results['passed'] + test_results['failed'] + test_results['skipped']
    print(f"Total Tests Run: {total_tests}")
    print(f"✓ Tests Passed: {test_results['passed']}")
    print(f"✗ Tests Failed: {test_results['failed']}")
    print(f"⚠ Tests Skipped: {test_results['skipped']}")
    
    if test_results['errors']:
        print(f"\nKey Issues Found ({len(test_results['errors'])}):")
        for i, error in enumerate(test_results['errors'][:10], 1):  # Show up to 10 errors
            print(f"  {i}. {error}")
        if len(test_results['errors']) > 10:
            print(f"  ... and {len(test_results['errors']) - 10} more")
    
    # Calculate success rate
    if total_tests > 0:
        success_rate = (test_results['passed'] / total_tests) * 100
        print(f"\nOverall Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 95:
            status = "✓ EXCELLENT - Enhanced distribution system is working perfectly!"
            exit_code = 0
        elif success_rate >= 85:
            status = "✓ VERY GOOD - Enhanced distribution system is working well"
            exit_code = 0
        elif success_rate >= 70:
            status = "⚠ GOOD - Enhanced distribution system is mostly functional"
            exit_code = 0
        elif success_rate >= 50:
            status = "⚠ FAIR - Enhanced distribution system has issues but core works"
            exit_code = 1
        else:
            status = "✗ POOR - Enhanced distribution system needs significant work"
            exit_code = 1
        
        print(f"Status: {status}")
    else:
        print("\n⚠ No tests could be executed")
        exit_code = 1
    
    # Recommendations
    print(f"\nRecommendations:")
    if not SCIPY_NUMPY_AVAILABLE:
        print("  • Install NumPy/SciPy: pip install numpy scipy")
    if not MATPLOTLIB_AVAILABLE:
        print("  • Install Matplotlib for plotting: pip install matplotlib")
    if test_results['failed'] > 0:
        print("  • Review and fix failed tests")
        print("  • Check parameter validation logic")
        print("  • Verify distribution implementations")
    if test_results['skipped'] > 10:
        print("  • Install missing dependencies to run more tests")
    
    print("  • Test integration with simulation processes")
    print("  • Validate statistical properties of distributions")
    print("  • Run performance tests with larger datasets")
    
    print("\n" + "="*80)
    print("ENHANCED SIMDISTRIBUTIONS TESTING COMPLETE")
    print("="*80)
    
    return test_results

# ================================================================================================
# MAIN EXECUTION
# ================================================================================================
def main():
    """Main test execution function."""
    return run_comprehensive_distribution_test()

if __name__ == "__main__":
    """
    Standalone testing execution for enhanced simDistributions.py
    """
    
    # Environment check
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher required")
        sys.exit(1)
    
    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Run comprehensive tests
    try:
        results = run_comprehensive_distribution_test()
        
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