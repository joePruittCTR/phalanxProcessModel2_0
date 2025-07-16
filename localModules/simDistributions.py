'''
Distribution Class: Provide useful probability density functions with accompanying 
pdf and sample attributes for use in simulation applications.  
    Contains:
        -     MonthlyMixedDistPDF: Provides a probability distribution that establishes two local maxima; one 
        near the beginning of the month, and one near the middle of the month, with the remaining 
        distribution occuring exponentially over the entirety of the month.
        -   WeeklyExponentialPDF: Provides a weekly probability distribution, with the majority of arrivals 
        occuring on the first day of the week and remaining arrivals exponentially distributed over the 
        entirety of the week.
        -   MixedWeibullPDF: Uses Weibull and Exponential distributions to achieve a distribution with 
        one sharp maximum and one rounded local maxima.
        -   BimodalExponPDF:  Uses two Exponential distributions to create one probability distribution with 
        two local maxima.
        -   BetaDistributionPDF: Versatile (and tunable) beta distribution class to conform the shape of the 
        PDF to mean interarrival probabilities.
'''

# Global imports
import numpy as np
from scipy.stats import weibull_min, norm, expon, beta as beta_dist
import matplotlib.pyplot as plt

class Distribution:
    def __init__(self, mean_interarrival_time, batch_size, distribution_type, **kwargs):
        self.mean_interarrival_time = mean_interarrival_time
        self.batch_size = batch_size # This parameter is problematic in the methods below
        self.distribution_type = distribution_type
        self.kwargs = kwargs

    def get_interarrival_time(self):
        if self.distribution_type == "MonthlyMixedDist":
            return self._monthly_mixed_dist()
        elif self.distribution_type == "WeeklyExponential":
            return self._weekly_exponential()
        elif self.distribution_type == "MixedWeibull":
            return self._mixed_weibull()
        elif self.distribution_type == "BimodalExpon":
            return self._bimodal_expon()
        elif self.distribution_type == "BetaDistribution":
            return self._beta_distribution()
        elif self.distribution_type == "Exponential": # Added for standard exponential
            return expon.rvs(scale=self.mean_interarrival_time) # No division by batch_size here
        else:
            raise ValueError("Invalid distribution type")

    def _monthly_mixed_dist(self):
        num_days = self.kwargs.get("num_days", 30)
        first_peak_fraction = self.kwargs.get("first_peak_fraction", 0.05)
        second_peak_fraction = self.kwargs.get("second_peak_fraction", 0.10)
        first_peak_probability = self.kwargs.get("first_peak_probability", 0.50)
        second_peak_probability = self.kwargs.get("second_peak_probability", 0.25)

        rand_val = np.random.rand()
        if rand_val < first_peak_probability:
            day = np.random.normal(loc=num_days * first_peak_fraction / 2, scale=num_days * first_peak_fraction / 4)
        elif rand_val < first_peak_probability + second_peak_probability:
            day = np.random.normal(loc=num_days * (0.5), scale=num_days * second_peak_fraction / 4)
        else:
            day = expon.rvs(scale=num_days)

        # PROBLEM: This division by self.batch_size is problematic if Source also scales by batch_size.
        # It implies 'day' is the time for a batch, and we want individual item time.
        # If Source wants time between batches, this should be removed.
        # For now, I'm assuming Source passes batch_size=1 to this Distribution.
        return day / self.batch_size 

    def _weekly_exponential(self):
        num_days = self.kwargs.get("num_days", 7)
        first_day_probability = self.kwargs.get("first_day_probability", 0.90)

        if np.random.rand() < first_day_probability:
            day = np.random.normal(loc=0, scale=0.5)
        else:
            day = 1 + expon.rvs(scale=num_days - 1)

        # PROBLEM: Same batch_size issue here.
        return day / self.batch_size

    def _mixed_weibull(self):
        w1 = self.kwargs.get("w1", 0.6)
        w2 = self.kwargs.get("w2", 0.2)
        w3 = self.kwargs.get("w3", 0.2)
        if not np.isclose(w1 + w2 + w3, 1.0):
            raise ValueError("Weights w1, w2, and w3 must sum to 1.")
        weibull_shape = self.kwargs.get("weibull_shape", 0.8)
        weibull_scale = self.kwargs.get("weibull_scale", 1.0)
        norm_mu = self.kwargs.get("norm_mu", 5)
        norm_sigma = self.kwargs.get("norm_sigma", 1)
        expon_lambda = self.kwargs.get("expon_lambda", 0.5)
        rand_val = np.random.rand()
        if rand_val < w1:
            return weibull_min.rvs(c=weibull_shape, scale=weibull_scale) / self.batch_size # PROBLEM
        elif rand_val < w1 + w2:
            return norm.rvs(loc=norm_mu, scale=norm_sigma) / self.batch_size # PROBLEM
        else:
            return expon.rvs(scale=1/expon_lambda) / self.batch_size # PROBLEM
    
    def _bimodal_expon(self):
        lambda1 = self.kwargs.get("lambda1", 0.5)
        lambda2 = self.kwargs.get("lambda2", 0.5)
        weight1 = self.kwargs.get("weight1", 3/5)
        loc2 = self.kwargs.get("loc2", 15)

        if np.random.rand() < weight1:
            return expon.rvs(scale=1/lambda1) / self.batch_size # PROBLEM
        else:
            return loc2 + expon.rvs(scale=1/lambda2) / self.batch_size # PROBLEM

    def _beta_distribution(self):
        alpha = self.kwargs.get("alpha", 2)
        beta = self.kwargs.get("beta", 5)
        beta_distribution = beta_dist(alpha, beta)
        return beta_distribution.rvs() / self.batch_size # PROBLEM
    
def main():
    distributions = [
        {"mean_interarrival_time": 10, "batch_size": 1, "distribution_type": "MonthlyMixedDist", "num_days": 30},
        {"mean_interarrival_time": 5, "batch_size": 1, "distribution_type": "WeeklyExponential", "num_days": 7},
        {"mean_interarrival_time": 8, "batch_size": 1, "distribution_type": "MixedWeibull", "w1": 0.6, "w2": 0.2, "w3": 0.2, "weibull_shape": 0.8, "weibull_scale": 1.0, "norm_mu": 5, "norm_sigma": 1, "expon_lambda": 0.5},
        {"mean_interarrival_time": 12, "batch_size": 1, "distribution_type": "BimodalExpon", "lambda1": 0.5, "lambda2": 0.5, "weight1": 3/5, "loc2": 15},
        {"mean_interarrival_time": 6, "batch_size": 1, "distribution_type": "BetaDistribution", "alpha": 2, "beta": 5},
        {"mean_interarrival_time": 7, "batch_size": 1, "distribution_type": "Exponential"} # Added for testing
    ]

    num_plots = len(distributions)
    fig, axes = plt.subplots(1, num_plots, figsize=(20, 5))
    for i, dist_info in enumerate(distributions):
        dist = Distribution(**dist_info)
        interarrival_times = [dist.get_interarrival_time() for _ in range(1000)]
        # Filter out negative values if any distribution can produce them (e.g., Normal)
        interarrival_times = [t for t in interarrival_times if t >= 0] 
        if not interarrival_times: # Skip plot if no valid data
            print(f"No valid data for {dist_info['distribution_type']}")
            continue

        axes[i].hist(interarrival_times, bins=50, density=True, alpha=0.6)
        axes[i].set_title(dist_info["distribution_type"])
        axes[i].set_xlabel("Interarrival Time")
        axes[i].set_ylabel("Density")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
