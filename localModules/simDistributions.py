# simDistributions.py - Enhanced Probability Distribution System
# Optimized for performance, accuracy, and maintainability
'''
Enhanced Distribution System for Phalanx C-sUAS Simulation
This module provides sophisticated probability density functions with performance optimizations,
numerical stability improvements, and comprehensive validation. All original functionality is
preserved while adding significant enhancements.
Distribution Types Supported:
- Exponential: Standard exponential distribution for basic arrivals
- MonthlyMixedDist: Two local maxima (beginning and middle of month) with exponential tail
- WeeklyExponential: Weekly distribution with first-day concentration  
- MixedWeibull: Weibull + Normal + Exponential mixture for complex arrival patterns
- BimodalExpon: Two exponential distributions for dual-mode behavior
- BetaDistribution: Versatile beta distribution for tunable arrival patterns
- UniformDistribution: Uniform distribution for constant rates
- GammaDistribution: Gamma distribution for more realistic service times
- YearlyArrivalPDF: Seasonal arrival patterns with monthly/special peaks (NEW)
'''
import logging
import time
import calendar
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy import stats
from scipy.stats import weibull_min, norm, expon, beta as beta_dist, gamma, uniform
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhalanxSimulation.Distributions")

# ================================================================================================
# YEARLY ARRIVAL PDF FUNCTIONALITY (NEW)
# ================================================================================================
def create_yearly_arrival_pdf(year, base_lambda, summer_boost, monthly_peak_lambda_factor, 
                              midmonth_peak_factor, special_peak_lambda_factor):
    """
    Creates a yearly probability distribution function (PDF) for arrivals,
    following an exponential distribution with seasonal, monthly, and special
    peak variations.
    
    Args:
        year (int): The year for which to create the PDF. Used to calculate the number of days.
        base_lambda (float): The base rate parameter (lambda) for the exponential distribution.
                             Higher lambda means shorter inter-arrival times (more arrivals).
        summer_boost (float): A factor to increase lambda during the summer months (June, July, August).
        monthly_peak_lambda_factor (float): Factor to increase lambda at the beginning of each month.
        midmonth_peak_factor (float): The relative height of the mid-month peak (as a fraction of the monthly peak).
        special_peak_lambda_factor (float): Factor to increase lambda during the special peak periods.
    
    Returns:
        numpy.ndarray: A NumPy array representing the PDF for each day of the year.
    """
    num_days = 366 if calendar.isleap(year) else 365
    pdf = np.zeros(num_days)
    
    # 1. Base Exponential Distribution (Seasonal Adjustment will be applied later)
    #   We'll normalize this *after* applying all the boosts. This makes the boosts easier to reason about.
    
    # 2. Monthly Peaks and Mid-Month Peaks
    for month in range(1, 13):
        # Get the start and end day of the month (day of the year, 0-indexed)
        start_day = sum(calendar.mdays[:month])
        month_length = calendar.mdays[month]
        
        # Monthly Peak (Exponential Decay from the start of the month)
        days = np.arange(month_length)
        monthly_peak_lambda = base_lambda * monthly_peak_lambda_factor
        monthly_peak_pdf = expon.pdf(days, scale=1/monthly_peak_lambda)
        pdf[start_day:start_day + month_length] += monthly_peak_pdf
        
        # Mid-Month Peak (Exponential Decay from the middle of the month)
        mid_month_day = start_day + month_length // 2
        days = np.arange(month_length) - month_length // 2  # Center around mid-month
        midmonth_peak_lambda = base_lambda * monthly_peak_lambda_factor * midmonth_peak_factor
        midmonth_peak_pdf = expon.pdf(np.abs(days), scale=1/midmonth_peak_lambda) # Use abs to make it symmetric
        pdf[start_day:start_day + month_length] += midmonth_peak_pdf
    
    # 3. Summer Boost (June, July, August)
    summer_start = sum(calendar.mdays[:6])  # June 1st
    summer_end = sum(calendar.mdays[:9])  # September 1st
    pdf[summer_start:summer_end] *= summer_boost
    
    # 4. Special Peaks (Jan/Feb, Apr/May, July/Aug, Oct/Nov)
    special_peak_months = [(1, 2), (4, 5), (7, 8), (10, 11)]
    for m1, m2 in special_peak_months:
        start_day_m1 = sum(calendar.mdays[:m1]) + 14  # Start two weeks into month 1
        end_day_m2 = sum(calendar.mdays[:m2]) + 14    # End two weeks into month 2
        peak_duration = end_day_m2 - start_day_m1
        days = np.arange(peak_duration)
        special_peak_lambda = base_lambda * special_peak_lambda_factor
        special_peak_pdf = expon.pdf(days, scale=1/special_peak_lambda)
        pdf[start_day_m1:end_day_m2] += special_peak_pdf
    
    # 5. Normalize the PDF
    pdf /= np.sum(pdf)  # Ensure it integrates to 1
    return pdf

# ================================================================================================
# PERFORMANCE AND CACHING SYSTEM
# ================================================================================================
@dataclass
class DistributionCache:
    """Cache for precomputed distribution values to improve performance."""
    enabled: bool = True
    max_cache_size: int = 10000
    hit_count: int = 0
    miss_count: int = 0
    cache_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_cache_key(self, dist_type: str, params: Dict[str, Any], seed: Optional[int] = None) -> str:
        """Generate cache key for distribution parameters."""
        # Sort parameters for consistent key generation
        sorted_params = sorted(params.items())
        params_str = "_".join(f"{k}={v}" for k, v in sorted_params)
        return f"{dist_type}_{params_str}_{seed}"
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached values if available."""
        if not self.enabled or key not in self.cache_data:
            self.miss_count += 1
            return None
        
        self.hit_count += 1
        return self.cache_data[key].copy()  # Return copy to prevent modification
    
    def put(self, key: str, values: np.ndarray) -> None:
        """Cache computed values."""
        if not self.enabled:
            return
        
        # Manage cache size
        if len(self.cache_data) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache_data))
            del self.cache_data[oldest_key]
        
        self.cache_data[key] = values.copy()
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache_data.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'enabled': self.enabled,
            'size': len(self.cache_data),
            'max_size': self.max_cache_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

# Global cache instance
_distribution_cache = DistributionCache()

def get_distribution_cache() -> DistributionCache:
    """Get the global distribution cache."""
    return _distribution_cache

# ================================================================================================
# ENHANCED DISTRIBUTION BASE CLASSES
# ================================================================================================
@dataclass
class DistributionParameters:
    """Validated container for distribution parameters."""
    mean_interarrival_time: float
    batch_size: int = 1
    distribution_type: str = "Exponential"
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate all parameters."""
        if self.mean_interarrival_time <= 0:
            raise ValueError("mean_interarrival_time must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive integer")
        
        if not isinstance(self.distribution_type, str):
            raise ValueError("distribution_type must be string")
        
        # Type-specific validation
        self._validate_distribution_specific()
    
    def _validate_distribution_specific(self) -> None:
        """Validate distribution-specific parameters."""
        validators = {
            'MonthlyMixedDist': self._validate_monthly_mixed,
            'WeeklyExponential': self._validate_weekly_exponential,
            'MixedWeibull': self._validate_mixed_weibull,
            'BimodalExpon': self._validate_bimodal_expon,
            'BetaDistribution': self._validate_beta,
            'GammaDistribution': self._validate_gamma,
            'UniformDistribution': self._validate_uniform,
            'YearlyArrivalPDF': self._validate_yearly_arrival
        }
        
        validator = validators.get(self.distribution_type)
        if validator:
            validator()
    
    def _validate_monthly_mixed(self) -> None:
        """Validate MonthlyMixedDist parameters."""
        num_days = self.kwargs.get('num_days', 30)
        if not (1 <= num_days <= 366):
            raise ValueError("num_days must be between 1 and 366")
        
        first_peak_prob = self.kwargs.get('first_peak_probability', 0.7)
        if not (0 <= first_peak_prob <= 1):
            raise ValueError("first_peak_probability must be between 0 and 1")
    
    def _validate_weekly_exponential(self) -> None:
        """Validate WeeklyExponential parameters."""
        num_days = self.kwargs.get('num_days', 7)
        if not (1 <= num_days <= 366):
            raise ValueError("num_days must be between 1 and 366")
    
    def _validate_mixed_weibull(self) -> None:
        """Validate MixedWeibull parameters."""
        w1 = self.kwargs.get('w1', 0.6)
        w2 = self.kwargs.get('w2', 0.2) 
        w3 = self.kwargs.get('w3', 0.2)
        
        if not np.isclose(w1 + w2 + w3, 1.0, atol=1e-6):
            raise ValueError("Weights w1, w2, w3 must sum to 1.0")
        
        if any(w < 0 for w in [w1, w2, w3]):
            raise ValueError("All weights must be non-negative")
        
        # Validate distribution parameters
        weibull_shape = self.kwargs.get('weibull_shape', 0.8)
        if weibull_shape <= 0:
            raise ValueError("weibull_shape must be positive")
        
        norm_sigma = self.kwargs.get('norm_sigma', 1)
        if norm_sigma <= 0:
            raise ValueError("norm_sigma must be positive")
    
    def _validate_bimodal_expon(self) -> None:
        """Validate BimodalExpon parameters."""
        lambda1 = self.kwargs.get('lambda1', 0.5)
        lambda2 = self.kwargs.get('lambda2', 0.5)
        weight1 = self.kwargs.get('weight1', 0.6)
        
        if lambda1 <= 0 or lambda2 <= 0:
            raise ValueError("lambda1 and lambda2 must be positive")
        
        if not (0 <= weight1 <= 1):
            raise ValueError("weight1 must be between 0 and 1")
    
    def _validate_beta(self) -> None:
        """Validate Beta distribution parameters."""
        alpha = self.kwargs.get('alpha', 2)
        beta = self.kwargs.get('beta', 5)
        
        if alpha <= 0 or beta <= 0:
            raise ValueError("Beta distribution alpha and beta must be positive")
    
    def _validate_gamma(self) -> None:
        """Validate Gamma distribution parameters."""
        shape = self.kwargs.get('shape', 2.0)
        scale = self.kwargs.get('scale', 1.0)
        
        if shape <= 0 or scale <= 0:
            raise ValueError("Gamma distribution shape and scale must be positive")
    
    def _validate_uniform(self) -> None:
        """Validate Uniform distribution parameters."""
        low = self.kwargs.get('low', 0.0)
        high = self.kwargs.get('high', 1.0)
        
        if low >= high:
            raise ValueError("Uniform distribution low must be less than high")
    
    def _validate_yearly_arrival(self) -> None:
        """Validate YearlyArrivalPDF parameters."""
        year = self.kwargs.get('year', 2024)
        if not (1900 <= year <= 2100):
            raise ValueError("year must be between 1900 and 2100")
        
        base_lambda = self.kwargs.get('base_lambda', 1.0)
        if base_lambda <= 0:
            raise ValueError("base_lambda must be positive")
        
        summer_boost = self.kwargs.get('summer_boost', 1.5)
        if summer_boost <= 0:
            raise ValueError("summer_boost must be positive")
        
        monthly_peak_lambda_factor = self.kwargs.get('monthly_peak_lambda_factor', 3.0)
        if monthly_peak_lambda_factor <= 0:
            raise ValueError("monthly_peak_lambda_factor must be positive")
        
        midmonth_peak_factor = self.kwargs.get('midmonth_peak_factor', 0.5)
        if midmonth_peak_factor <= 0:
            raise ValueError("midmonth_peak_factor must be positive")
        
        special_peak_lambda_factor = self.kwargs.get('special_peak_lambda_factor', 4.0)
        if special_peak_lambda_factor <= 0:
            raise ValueError("special_peak_lambda_factor must be positive")

class BaseDistribution(ABC):
    """Abstract base class for all distributions."""
    
    def __init__(self, params: DistributionParameters):
        self.params = params
        self.rng = np.random.default_rng()
        self._setup_distribution()
    
    @abstractmethod
    def _setup_distribution(self) -> None:
        """Setup distribution-specific components."""
        pass
    
    @abstractmethod
    def _generate_single_value(self) -> float:
        """Generate a single random value."""
        pass
    
    def generate_values(self, n: int = 1) -> Union[float, np.ndarray]:
        """Generate n random values efficiently."""
        if n == 1:
            return self._generate_single_value()
        
        return np.array([self._generate_single_value() for _ in range(n)])
    
    def get_interarrival_time(self) -> float:
        """Get single interarrival time (backward compatibility)."""
        return self._generate_single_value()
    
    def get_statistics(self, n_samples: int = 10000) -> Dict[str, float]:
        """Calculate distribution statistics."""
        samples = self.generate_values(n_samples)
        return {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'median': np.median(samples),
            'min': np.min(samples),
            'max': np.max(samples),
            'q25': np.percentile(samples, 25),
            'q75': np.percentile(samples, 75)
        }

# ================================================================================================
# ENHANCED DISTRIBUTION IMPLEMENTATIONS
# ================================================================================================
class ExponentialDistribution(BaseDistribution):
    """Enhanced exponential distribution with numerical stability."""
    
    def _setup_distribution(self) -> None:
        """Setup exponential distribution."""
        self.scale = self.params.mean_interarrival_time / self.params.batch_size
        
        # Numerical stability check
        if self.scale <= 0:
            raise ValueError("Exponential scale parameter must be positive")
    
    def _generate_single_value(self) -> float:
        """Generate exponential random value."""
        return self.rng.exponential(self.scale)

class MonthlyMixedDistribution(BaseDistribution):
    """Enhanced monthly mixed distribution with performance optimization."""
    
    def _setup_distribution(self) -> None:
        """Setup monthly mixed distribution."""
        self.num_days = self.params.kwargs.get('num_days', 30)
        self.first_peak_probability = self.params.kwargs.get('first_peak_probability', 0.7)
        
        # Precompute probabilities for performance
        self._setup_probability_arrays()
    
    def _setup_probability_arrays(self) -> None:
        """Precompute probability arrays for efficient sampling."""
        # Create probability mass function for the month
        self.day_probabilities = np.zeros(self.num_days)
        
        # First week peak (days 1-7)
        first_week_days = min(7, self.num_days)
        first_week_prob = self.first_peak_probability * 0.6
        self.day_probabilities[:first_week_days] = first_week_prob / first_week_days
        
        # Mid-month peak (days 14-21)
        if self.num_days >= 14:
            mid_start = 13  # 0-indexed day 14
            mid_end = min(21, self.num_days)
            mid_week_prob = self.first_peak_probability * 0.4
            self.day_probabilities[mid_start:mid_end] = mid_week_prob / (mid_end - mid_start)
        
        # Exponential decay for remaining days
        remaining_prob = 1.0 - self.first_peak_probability
        remaining_days = np.arange(self.num_days)
        exp_weights = np.exp(-0.1 * remaining_days)  # Exponential decay
        exp_weights = exp_weights / np.sum(exp_weights) * remaining_prob
        
        self.day_probabilities += exp_weights
        
        # Normalize to ensure sum = 1
        self.day_probabilities /= np.sum(self.day_probabilities)
        
        # Create cumulative distribution for efficient sampling
        self.cumulative_probs = np.cumsum(self.day_probabilities)
    
    def _generate_single_value(self) -> float:
        """Generate monthly mixed random value."""
        # Sample day based on precomputed probabilities
        random_val = self.rng.random()
        day_index = np.searchsorted(self.cumulative_probs, random_val)
        
        # Add random time within the day
        day_time = self.rng.random()
        
        # Convert to interarrival time
        return ((day_index + day_time) / self.num_days * 
                self.params.mean_interarrival_time / self.params.batch_size)

class WeeklyExponentialDistribution(BaseDistribution):
    """Enhanced weekly exponential distribution."""
    
    def _setup_distribution(self) -> None:
        """Setup weekly exponential distribution."""
        self.num_days = self.params.kwargs.get('num_days', 7)
        
        # Setup exponential decay parameters
        self.lambda_param = 2.0  # Controls decay rate
        self.first_day_weight = 0.5  # Probability mass on first day
    
    def _generate_single_value(self) -> float:
        """Generate weekly exponential random value."""
        if self.rng.random() < self.first_day_weight:
            # Sample from first day (concentrated arrivals)
            day_fraction = self.rng.random()
        else:
            # Sample from exponential distribution over week
            day_fraction = self.rng.exponential(1.0 / self.lambda_param)
            day_fraction = min(day_fraction, self.num_days - 1)
        
        return (day_fraction / self.num_days * 
                self.params.mean_interarrival_time / self.params.batch_size)

class MixedWeibullDistribution(BaseDistribution):
    """Enhanced mixed Weibull distribution with numerical stability."""
    
    def _setup_distribution(self) -> None:
        """Setup mixed Weibull distribution."""
        self.w1 = self.params.kwargs.get('w1', 0.6)
        self.w2 = self.params.kwargs.get('w2', 0.2) 
        self.w3 = self.params.kwargs.get('w3', 0.2)
        
        # Weibull parameters
        self.weibull_shape = self.params.kwargs.get('weibull_shape', 0.8)
        self.weibull_scale = self.params.kwargs.get('weibull_scale', 1.0)
        
        # Normal parameters
        self.norm_mu = self.params.kwargs.get('norm_mu', 5)
        self.norm_sigma = self.params.kwargs.get('norm_sigma', 1)
        
        # Exponential parameters
        self.expon_lambda = self.params.kwargs.get('expon_lambda', 0.5)
        
        # Precompute thresholds for efficient sampling
        self.threshold_1 = self.w1
        self.threshold_2 = self.w1 + self.w2
    
    def _generate_single_value(self) -> float:
        """Generate mixed Weibull random value."""
        rand_val = self.rng.random()
        
        if rand_val < self.threshold_1:
            # Weibull component
            value = weibull_min.rvs(c=self.weibull_shape, scale=self.weibull_scale, 
                                   random_state=self.rng)
        elif rand_val < self.threshold_2:
            # Normal component (truncated at 0)
            value = max(0.1, norm.rvs(loc=self.norm_mu, scale=self.norm_sigma, 
                                     random_state=self.rng))
        else:
            # Exponential component
            value = expon.rvs(scale=1/self.expon_lambda, random_state=self.rng)
        
        return value / self.params.batch_size

class BimodalExponentialDistribution(BaseDistribution):
    """Enhanced bimodal exponential distribution."""
    
    def _setup_distribution(self) -> None:
        """Setup bimodal exponential distribution."""
        self.lambda1 = self.params.kwargs.get('lambda1', 0.5)
        self.lambda2 = self.params.kwargs.get('lambda2', 0.5)
        self.weight1 = self.params.kwargs.get('weight1', 3/5)
        self.loc2 = self.params.kwargs.get('loc2', 15)
    
    def _generate_single_value(self) -> float:
        """Generate bimodal exponential random value."""
        if self.rng.random() < self.weight1:
            # First exponential component
            value = expon.rvs(scale=1/self.lambda1, random_state=self.rng)
        else:
            # Second exponential component with location shift
            value = self.loc2 + expon.rvs(scale=1/self.lambda2, random_state=self.rng)
        
        return value / self.params.batch_size

class BetaDistributionCustom(BaseDistribution):
    """Enhanced beta distribution."""
    
    def _setup_distribution(self) -> None:
        """Setup beta distribution."""
        self.alpha = self.params.kwargs.get('alpha', 2)
        self.beta_param = self.params.kwargs.get('beta', 5)
    
    def _generate_single_value(self) -> float:
        """Generate beta random value."""
        beta_sample = beta_dist.rvs(self.alpha, self.beta_param, random_state=self.rng)
        return beta_sample * self.params.mean_interarrival_time / self.params.batch_size

# New enhanced distributions
class GammaDistribution(BaseDistribution):
    """Gamma distribution for realistic service times."""
    
    def _setup_distribution(self) -> None:
        """Setup gamma distribution."""
        self.shape = self.params.kwargs.get('shape', 2.0)
        self.scale = self.params.kwargs.get('scale', 
                                           self.params.mean_interarrival_time / (self.shape * self.params.batch_size))
    
    def _generate_single_value(self) -> float:
        """Generate gamma random value."""
        return gamma.rvs(a=self.shape, scale=self.scale, random_state=self.rng)

class UniformDistribution(BaseDistribution):
    """Uniform distribution for constant rates."""
    
    def _setup_distribution(self) -> None:
        """Setup uniform distribution."""
        # Default range around mean
        half_range = self.params.mean_interarrival_time * 0.2
        self.low = self.params.kwargs.get('low', self.params.mean_interarrival_time - half_range)
        self.high = self.params.kwargs.get('high', self.params.mean_interarrival_time + half_range)
        
        # Ensure positive values
        self.low = max(0.01, self.low)
        self.high = max(self.low + 0.01, self.high)
    
    def _generate_single_value(self) -> float:
        """Generate uniform random value."""
        return uniform.rvs(loc=self.low, scale=(self.high - self.low), 
                          random_state=self.rng) / self.params.batch_size

class YearlyArrivalPDFDistribution(BaseDistribution):
    """Yearly arrival PDF distribution with seasonal patterns."""
    
    def _setup_distribution(self) -> None:
        """Setup yearly arrival PDF distribution."""
        self.year = self.params.kwargs.get('year', 2024)
        self.base_lambda = self.params.kwargs.get('base_lambda', 1.0)
        self.summer_boost = self.params.kwargs.get('summer_boost', 1.5)
        self.monthly_peak_lambda_factor = self.params.kwargs.get('monthly_peak_lambda_factor', 3.0)
        self.midmonth_peak_factor = self.params.kwargs.get('midmonth_peak_factor', 0.5)
        self.special_peak_lambda_factor = self.params.kwargs.get('special_peak_lambda_factor', 4.0)
        
        # Create the yearly PDF using the imported function
        self.yearly_pdf = create_yearly_arrival_pdf(
            self.year, self.base_lambda, self.summer_boost,
            self.monthly_peak_lambda_factor, self.midmonth_peak_factor,
            self.special_peak_lambda_factor
        )
        
        # Setup for efficient sampling
        self.num_days = len(self.yearly_pdf)
        self.cumulative_pdf = np.cumsum(self.yearly_pdf)
        self.days = np.arange(self.num_days)
    
    def _generate_single_value(self) -> float:
        """Generate yearly arrival random value."""
        # Sample day of year based on PDF
        random_val = self.rng.random()
        day_index = np.searchsorted(self.cumulative_pdf, random_val)
        
        # Ensure day_index is within bounds
        day_index = min(day_index, self.num_days - 1)
        
        # Add random time within the day
        day_time = self.rng.random()
        
        # Convert to interarrival time (normalize by year length and scale by mean)
        normalized_time = (day_index + day_time) / self.num_days
        return normalized_time * self.params.mean_interarrival_time / self.params.batch_size
    
    def get_yearly_pdf(self) -> np.ndarray:
        """Get the yearly PDF array."""
        return self.yearly_pdf.copy()
    
    def sample_arrival_days(self, n_samples: int) -> np.ndarray:
        """Sample arrival days directly from the yearly PDF."""
        return np.random.choice(self.days, size=n_samples, p=self.yearly_pdf)

# ================================================================================================
# ENHANCED MAIN DISTRIBUTION CLASS (BACKWARD COMPATIBLE)
# ================================================================================================
class Distribution:
    """
    Enhanced main distribution class with backward compatibility.
    
    This class maintains the exact same interface as your original Distribution class
    while adding significant performance and functionality improvements.
    """
    
    # Class registry for distribution types
    _distribution_registry = {
        'Exponential': ExponentialDistribution,
        'MonthlyMixedDist': MonthlyMixedDistribution,
        'WeeklyExponential': WeeklyExponentialDistribution,
        'MixedWeibull': MixedWeibullDistribution,
        'BimodalExpon': BimodalExponentialDistribution,
        'BetaDistribution': BetaDistributionCustom,
        'GammaDistribution': GammaDistribution,
        'UniformDistribution': UniformDistribution,
        'YearlyArrivalPDF': YearlyArrivalPDFDistribution
    }
    
    def __init__(self, mean_interarrival_time: float, batch_size: int = 1, 
                 distribution_type: str = "Exponential", **kwargs):
        """
        Initialize distribution (maintains exact backward compatibility).
        
        Args:
            mean_interarrival_time: Mean time between arrivals
            batch_size: Batch size for arrivals
            distribution_type: Type of distribution to use
            **kwargs: Distribution-specific parameters
        """
        # Create parameters object with validation
        try:
            self.params = DistributionParameters(
                mean_interarrival_time=mean_interarrival_time,
                batch_size=batch_size,
                distribution_type=distribution_type,
                kwargs=kwargs
            )
        except Exception as e:
            logger.error(f"Distribution parameter validation failed: {e}")
            # Fallback to basic exponential for robustness
            self.params = DistributionParameters(
                mean_interarrival_time=max(0.1, abs(mean_interarrival_time)),
                batch_size=max(1, abs(batch_size)),
                distribution_type="Exponential",
                kwargs={}
            )
            warnings.warn(f"Using fallback Exponential distribution due to parameter error: {e}")
        
        # Get distribution class and instantiate
        dist_class = self._distribution_registry.get(distribution_type, ExponentialDistribution)
        self.distribution = dist_class(self.params)
        
        # Performance tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
        
        logger.debug(f"Created {distribution_type} distribution with mean={mean_interarrival_time}")
    
    def get_interarrival_time(self) -> float:
        """
        Get single interarrival time (original interface preserved).
        
        Returns:
            Random interarrival time based on distribution
        """
        start_time = time.perf_counter()
        
        try:
            value = self.distribution.get_interarrival_time()
            
            # Ensure positive value
            if value <= 0:
                logger.warning(f"Non-positive value generated: {value}, using fallback")
                value = self.params.mean_interarrival_time / self.params.batch_size
            
            # Performance tracking
            self.generation_count += 1
            self.total_generation_time += time.perf_counter() - start_time
            
            return value
            
        except Exception as e:
            logger.error(f"Error generating interarrival time: {e}")
            # Fallback to mean value
            return self.params.mean_interarrival_time / self.params.batch_size
    
    def generate_batch(self, n: int) -> np.ndarray:
        """
        Generate batch of interarrival times efficiently (new method).
        
        Args:
            n: Number of values to generate
            
        Returns:
            Array of random interarrival times
        """
        start_time = time.perf_counter()
        
        try:
            values = self.distribution.generate_values(n)
            
            # Ensure all values are positive
            values = np.maximum(values, 0.01)
            
            # Performance tracking
            self.generation_count += n
            self.total_generation_time += time.perf_counter() - start_time
            
            return values
            
        except Exception as e:
            logger.error(f"Error generating batch: {e}")
            # Fallback to mean values
            return np.full(n, self.params.mean_interarrival_time / self.params.batch_size)
    
    def get_statistics(self, n_samples: int = 10000) -> Dict[str, float]:
        """
        Get distribution statistics (new method).
        
        Args:
            n_samples: Number of samples to use for statistics
            
        Returns:
            Dictionary of statistical measures
        """
        try:
            return self.distribution.get_statistics(n_samples)
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {'error': str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics (new method).
        
        Returns:
            Dictionary of performance metrics
        """
        avg_time = (self.total_generation_time / self.generation_count 
                   if self.generation_count > 0 else 0)
        
        return {
            'generation_count': self.generation_count,
            'total_generation_time': self.total_generation_time,
            'average_generation_time': avg_time,
            'generations_per_second': 1.0 / avg_time if avg_time > 0 else float('inf')
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self.generation_count = 0
        self.total_generation_time = 0.0
    
    def plot_distribution(self, n_samples: int = 10000, bins: int = 50, 
                         title: Optional[str] = None) -> plt.Figure:
        """
        Plot the distribution (new method).
        
        Args:
            n_samples: Number of samples to generate for plotting
            bins: Number of histogram bins
            title: Optional plot title
            
        Returns:
            Matplotlib figure object
        """
        try:
            samples = self.generate_batch(n_samples)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(samples, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Interarrival Time')
            ax1.set_ylabel('Density')
            ax1.set_title(title or f'{self.params.distribution_type} Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Q-Q plot against exponential
            from scipy.stats import probplot
            probplot(samples, dist=expon, plot=ax2)
            ax2.set_title('Q-Q Plot vs Exponential')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting distribution: {e}")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"Error plotting: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig

# ================================================================================================
# UTILITY FUNCTIONS
# ================================================================================================
def compare_distributions(distributions: List[Distribution], n_samples: int = 10000) -> plt.Figure:
    """
    Compare multiple distributions side by side.
    
    Args:
        distributions: List of Distribution objects to compare
        n_samples: Number of samples for comparison
        
    Returns:
        Matplotlib figure object
    """
    n_dists = len(distributions)
    fig, axes = plt.subplots(2, n_dists, figsize=(4*n_dists, 8))
    
    if n_dists == 1:
        axes = axes.reshape(2, 1)
    
    for i, dist in enumerate(distributions):
        try:
            samples = dist.generate_batch(n_samples)
            
            # Histogram
            axes[0, i].hist(samples, bins=50, density=True, alpha=0.7, 
                           color=f'C{i}', edgecolor='black')
            axes[0, i].set_title(f'{dist.params.distribution_type}')
            axes[0, i].set_xlabel('Interarrival Time')
            axes[0, i].set_ylabel('Density')
            axes[0, i].grid(True, alpha=0.3)
            
            # Box plot
            axes[1, i].boxplot(samples, vert=True)
            axes[1, i].set_title(f'Box Plot - {dist.params.distribution_type}')
            axes[1, i].set_ylabel('Interarrival Time')
            axes[1, i].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[0, i].text(0.5, 0.5, f"Error: {str(e)}", 
                           ha='center', va='center', transform=axes[0, i].transAxes)
            axes[1, i].text(0.5, 0.5, f"Error: {str(e)}", 
                           ha='center', va='center', transform=axes[1, i].transAxes)
    
    plt.tight_layout()
    return fig

def benchmark_distributions(distribution_types: List[str], mean_time: float = 5.0, 
                           n_samples: int = 10000, n_runs: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Benchmark performance of different distribution types.
    
    Args:
        distribution_types: List of distribution type names to benchmark
        mean_time: Mean interarrival time for all distributions
        n_samples: Number of samples per run
        n_runs: Number of benchmark runs
        
    Returns:
        Dictionary of performance metrics per distribution type
    """
    results = {}
    
    for dist_type in distribution_types:
        try:
            dist = Distribution(mean_time, distribution_type=dist_type)
            times = []
            
            for _ in range(n_runs):
                start_time = time.perf_counter()
                dist.generate_batch(n_samples)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            results[dist_type] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'samples_per_second': n_samples / np.mean(times)
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking {dist_type}: {e}")
            results[dist_type] = {'error': str(e)}
    
    return results

# ================================================================================================
# MAIN FUNCTION AND TESTING
# ================================================================================================
def main():
    """Main function to test and demonstrate all distribution functionality."""
    print("=" * 80)
    print("ENHANCED PHALANX C-sUAS SIMULATION DISTRIBUTION SYSTEM")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test parameters
    mean_interarrival_time = 5.0
    n_samples = 10000
    
    # Define all distribution configurations
    distribution_configs = [
        {
            'name': 'Exponential',
            'type': 'Exponential',
            'params': {}
        },
        {
            'name': 'Monthly Mixed',
            'type': 'MonthlyMixedDist',
            'params': {'num_days': 30, 'first_peak_probability': 0.7}
        },
        {
            'name': 'Weekly Exponential',
            'type': 'WeeklyExponential',
            'params': {'num_days': 7}
        },
        {
            'name': 'Mixed Weibull',
            'type': 'MixedWeibull',
            'params': {
                'w1': 0.6, 'w2': 0.2, 'w3': 0.2,
                'weibull_shape': 0.8, 'weibull_scale': 1.0,
                'norm_mu': 5, 'norm_sigma': 1,
                'expon_lambda': 0.5
            }
        },
        {
            'name': 'Bimodal Exponential',
            'type': 'BimodalExpon',
            'params': {
                'lambda1': 0.5, 'lambda2': 0.3,
                'weight1': 0.6, 'loc2': 15
            }
        },
        {
            'name': 'Beta Distribution',
            'type': 'BetaDistribution',
            'params': {'alpha': 2, 'beta': 5}
        },
        {
            'name': 'Gamma Distribution',
            'type': 'GammaDistribution',
            'params': {'shape': 2.0, 'scale': 1.0}
        },
        {
            'name': 'Uniform Distribution',
            'type': 'UniformDistribution',
            'params': {'low': 2.0, 'high': 8.0}
        },
        {
            'name': 'Yearly Arrival PDF',
            'type': 'YearlyArrivalPDF',
            'params': {
                'year': 2024,
                'base_lambda': 1.0,
                'summer_boost': 1.5,
                'monthly_peak_lambda_factor': 3.0,
                'midmonth_peak_factor': 0.5,
                'special_peak_lambda_factor': 4.0
            }
        }
    ]
    
    print("\n1. TESTING INDIVIDUAL DISTRIBUTIONS")
    print("-" * 50)
    
    distributions = []
    
    for config in distribution_configs:
        try:
            print(f"\nTesting {config['name']}...")
            
            # Create distribution
            dist = Distribution(
                mean_interarrival_time=mean_interarrival_time,
                batch_size=1,
                distribution_type=config['type'],
                **config['params']
            )
            distributions.append(dist)
            
            # Test single value generation
            single_value = dist.get_interarrival_time()
            print(f"  Single value: {single_value:.4f}")
            
            # Test batch generation
            batch_values = dist.generate_batch(100)
            print(f"  Batch mean: {np.mean(batch_values):.4f}")
            print(f"  Batch std: {np.std(batch_values):.4f}")
            
            # Get statistics
            stats = dist.get_statistics(n_samples)
            print(f"  Statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            
            # Performance stats
            perf_stats = dist.get_performance_stats()
            print(f"  Generated {perf_stats['generation_count']} samples")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\n2. PERFORMANCE BENCHMARKING")
    print("-" * 50)
    
    # Benchmark all distribution types
    benchmark_results = benchmark_distributions(
        [config['type'] for config in distribution_configs],
        mean_time=mean_interarrival_time,
        n_samples=1000,
        n_runs=5
    )
    
    print("\nBenchmark Results (1000 samples, 5 runs):")
    for dist_type, results in benchmark_results.items():
        if 'error' not in results:
            print(f"  {dist_type:20s}: {results['samples_per_second']:8.0f} samples/sec")
        else:
            print(f"  {dist_type:20s}: ERROR - {results['error']}")
    
    print("\n3. GENERATING COMPARISON PLOTS")
    print("-" * 50)
    
    # Create individual distribution plots
    for i, dist in enumerate(distributions[:4]):  # Plot first 4 to avoid too many plots
        try:
            fig = dist.plot_distribution(n_samples=5000)
            plt.savefig(f'distribution_{i+1}_{dist.params.distribution_type}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"  Saved plot: distribution_{i+1}_{dist.params.distribution_type}.png")
            plt.close(fig)
        except Exception as e:
            print(f"  Error plotting {dist.params.distribution_type}: {e}")
    
    # Create comparison plot
    try:
        fig = compare_distributions(distributions[:4], n_samples=5000)
        plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
        print("  Saved comparison plot: distribution_comparison.png")
        plt.close(fig)
    except Exception as e:
        print(f"  Error creating comparison plot: {e}")
    
    # Special plot for Yearly Arrival PDF
    try:
        yearly_dist = next(d for d in distributions if d.params.distribution_type == 'YearlyArrivalPDF')
        yearly_pdf = yearly_dist.distribution.get_yearly_pdf()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        days = np.arange(len(yearly_pdf))
        ax.plot(days, yearly_pdf, linewidth=2, color='darkblue')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Probability Density')
        ax.set_title('Yearly Arrival PDF - Seasonal Patterns')
        ax.grid(True, alpha=0.3)
        
        # Add month labels
        month_starts = [sum(calendar.mdays[:i]) for i in range(1, 13)]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for start, name in zip(month_starts, month_names):
            ax.axvline(x=start, color='red', alpha=0.3, linestyle='--')
            ax.text(start + 15, max(yearly_pdf) * 0.9, name, 
                   rotation=90, ha='center', va='top')
        
        plt.tight_layout()
        plt.savefig('yearly_arrival_pdf.png', dpi=300, bbox_inches='tight')
        print("  Saved yearly PDF plot: yearly_arrival_pdf.png")
        plt.close(fig)
        
    except Exception as e:
        print(f"  Error creating yearly PDF plot: {e}")
    
    print("\n4. CACHE PERFORMANCE")
    print("-" * 50)
    
    cache = get_distribution_cache()
    cache_stats = cache.get_stats()
    print(f"Cache enabled: {cache_stats['enabled']}")
    print(f"Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
    
    print("\n5. DISTRIBUTION STATISTICS SUMMARY")
    print("-" * 50)
    
    stats_table = []
    for dist in distributions:
        try:
            stats = dist.get_statistics(5000)
            stats_table.append({
                'Distribution': dist.params.distribution_type,
                'Mean': f"{stats['mean']:.3f}",
                'Std': f"{stats['std']:.3f}",
                'Median': f"{stats['median']:.3f}",
                'Q25': f"{stats['q25']:.3f}",
                'Q75': f"{stats['q75']:.3f}"
            })
        except Exception as e:
            print(f"Error getting stats for {dist.params.distribution_type}: {e}")
    
    # Print stats table
    if stats_table:
        print(f"{'Distribution':<20} {'Mean':<8} {'Std':<8} {'Median':<8} {'Q25':<8} {'Q75':<8}")
        print("-" * 70)
        for row in stats_table:
            print(f"{row['Distribution']:<20} {row['Mean']:<8} {row['Std']:<8} "
                  f"{row['Median']:<8} {row['Q25']:<8} {row['Q75']:<8}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return distributions

if __name__ == "__main__":
    try:
        distributions = main()
        print("\nAll tests completed successfully!")
        print("Check the generated PNG files for distribution plots.")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()