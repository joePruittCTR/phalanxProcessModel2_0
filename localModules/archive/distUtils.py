# Global imports
import numpy as np
from scipy.stats import weibull_min, norm, expon, beta
import matplotlib.pyplot as plt

'''
The distUtils module contains useful probability density functions with accompanying sample attributes
for use in simulation applications.
'''

class MonthlyMixedDistPDF:
    def __init__(self, num_days=30, first_peak_fraction=0.05, second_peak_fraction=0.10,
                 first_peak_probability=0.50, second_peak_probability=0.25):
        """
        Custom probability distribution over a month.
        Args:
            num_days: Number of days in the month.
            first_peak_fraction: Fraction of the x-axis for the first peak.
            second_peak_fraction: Fraction of the x-axis for the second peak.
            first_peak_probability: Probability of the first peak.
            second_peak_probability: Probability of the second peak.
        """
        self.num_days = num_days
        self.first_peak_fraction = first_peak_fraction
        self.second_peak_fraction = second_peak_fraction
        self.first_peak_probability = first_peak_probability
        self.second_peak_probability = second_peak_probability
        self.exponential_probability = 1 - (first_peak_probability + second_peak_probability)

    def pdf(self, x):
        """
        Calculates the probability density function (PDF) at x.
        Args:
            x: The value at which to evaluate the PDF.
        Returns:
            The PDF value at x.
        """
        first_peak = self.first_peak_probability * norm.pdf(x, loc=self.num_days * self.first_peak_fraction / 2, scale=self.num_days * self.first_peak_fraction / 4)
        second_peak = self.second_peak_probability * norm.pdf(x, loc=self.num_days * (0.5), scale=self.num_days * self.second_peak_fraction / 4)
        exponential = self.exponential_probability * expon.pdf(x, loc=0, scale=self.num_days)
        return first_peak + second_peak + exponential

    def sample(self, num_samples=1):
        """
        Generates samples from the distribution.
        Args:
            num_samples: Number of samples to generate.
        Returns:
            A NumPy array of samples.
        """
        samples = []
        for _ in range(num_samples):
            rand_val = np.random.rand()
            if rand_val < self.first_peak_probability:
                day = np.random.normal(loc=self.num_days * self.first_peak_fraction / 2, scale=self.num_days * self.first_peak_fraction / 4)
            elif rand_val < self.first_peak_probability + self.second_peak_probability:
                day = np.random.normal(loc=self.num_days * (0.5), scale=self.num_days * self.second_peak_fraction / 4)
            else:
                day = expon.rvs(scale=self.num_days)
            samples.append(day)
        return np.array(samples)


class WeeklyExponentialPDF:
    def __init__(self, num_days=7, first_day_probability=0.90):  # Updated default probability
        """
        Weekly cycle distribution with a large peak on the first day.
        Args:
            num_days: Number of days in the week (default 7).
            first_day_probability: Probability of an event occurring on the first day.
        """
        self.num_days = num_days
        self.first_day_probability = first_day_probability
        self.remaining_probability = 1 - first_day_probability

    def pdf(self, x):
        """
        Calculates the probability density function (PDF) at x.
        Args:
            x: The value at which to evaluate the PDF.
        Returns:
            The PDF value at x.
        """
        first_day = self.first_day_probability * norm.pdf(x, loc=0, scale=0.5)
        remaining = self.remaining_probability * expon.pdf(x, loc=1, scale=self.num_days - 1)
        return first_day + remaining

    def sample(self, num_samples=1):
        """
        Generates samples from the weekly distribution.
        Args:
            num_samples: Number of samples to generate.
        Returns:
            A NumPy array of samples.
        """
        samples = []
        for _ in range(num_samples):
            if np.random.rand() < self.first_day_probability:
                day = np.random.normal(loc=0, scale=0.5)
            else:
                day = 1 + expon.rvs(scale=self.num_days - 1)
            samples.append(day)
        return np.array(samples)
    
class MixedWeibullPDF:
    def __init__(self, w1, w2, w3, weibull_shape, weibull_scale, norm_mu, norm_sigma, expon_lambda, num_samples=1000):  # Added num_samples
        """
        Represents a mixture distribution with Weibull, Normal, and Exponential components.

        Args:
            w1, w2, w3: Mixing coefficients (weights). Must sum to 1.
            weibull_shape, weibull_scale: Parameters for the Weibull distribution.
            norm_mu, norm_sigma: Parameters for the Normal distribution.
            expon_lambda: Parameter for the Exponential distribution.
            num_samples: Number of samples to generate when calling sample().
        """
        if not np.isclose(w1 + w2 + w3, 1.0):
            raise ValueError("Mixing coefficients (w1, w2, w3) must sum to 1.")

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.weibull_shape = weibull_shape
        self.weibull_scale = weibull_scale
        self.norm_mu = norm_mu
        self.norm_sigma = norm_sigma
        self.expon_lambda = expon_lambda
        self.num_samples = num_samples  # Store num_samples

    def pdf(self, x):
        """Calculates the PDF at x."""
        term1 = self.w1 * weibull_min.pdf(x, c=self.weibull_shape, scale=self.weibull_scale)
        term2 = self.w2 * norm.pdf(x, loc=self.norm_mu, scale=self.norm_sigma)
        term3 = self.w3 * expon.pdf(x, loc=0, scale=1/self.expon_lambda)
        return term1 + term2 + term3

    def sample(self, n_samples=None):  # sample method
        """Generates random samples from the mixture distribution."""
        if n_samples is None:
            n_samples = self.num_samples  # use default if not provided

        # Determine number of samples from each distribution
        n_weibull = int(self.w1 * n_samples)
        n_normal = int(self.w2 * n_samples)
        n_expon = n_samples - n_weibull - n_normal # ensure total samples is correct

        # Generate samples
        weibull_samples = weibull_min.rvs(c=self.weibull_shape, scale=self.weibull_scale, size=n_weibull)
        normal_samples = norm.rvs(loc=self.norm_mu, scale=self.norm_sigma, size=n_normal)
        expon_samples = expon.rvs(scale=1/self.expon_lambda, size=n_expon)

        # Combine and shuffle samples
        all_samples = np.concatenate([weibull_samples, normal_samples, expon_samples])
        np.random.shuffle(all_samples)
        return all_samples

class BimodalExponPDF:
    def __init__(self, lambda1, lambda2, weight1, loc2, num_samples=10000):
        """
        Represents a bimodal distribution composed of two exponential distributions.

        Args:
            lambda1: Rate parameter for the first exponential distribution.
            lambda2: Rate parameter for the second exponential distribution.
            weight1: Weight of the first distribution (weight2 is calculated as 1 - weight1).
            loc2: Location parameter (shift) for the second distribution.
            num_samples: Default number of samples to generate.
        """
        if not 0 <= weight1 <= 1:
            raise ValueError("weight1 must be between 0 and 1.")

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.weight1 = weight1
        self.weight2 = 1 - weight1
        self.loc2 = loc2
        self.num_samples = num_samples

    def pdf(self, x):
        """Calculates the probability density function (PDF) at x."""
        pdf1 = self.weight1 * expon.pdf(x, scale=1/self.lambda1)
        pdf2 = self.weight2 * expon.pdf(x, loc=self.loc2, scale=1/self.lambda2)
        return pdf1 + pdf2

    def sample(self, n_samples=None):
        """Generates random samples from the bimodal distribution."""

        if n_samples is None:
            n_samples = self.num_samples

        n1 = int(n_samples * self.weight1)
        n2 = n_samples - n1  # Ensure correct total samples

        samples1 = expon.rvs(scale=1/self.lambda1, size=n1)
        samples2 = expon.rvs(loc=self.loc2, scale=1/self.lambda2, size=n2)
        all_samples = np.concatenate([samples1, samples2])
        np.random.shuffle(all_samples)
        return all_samples

class BetaDistributionPDF:
    def __init__(self, alpha, beta):
        """
        Represents a Beta distribution.

        Args:
            alpha: Shape parameter (alpha > 0).
            beta: Shape parameter (beta > 0).
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta parameters must be greater than 0.")
        self.alpha = alpha
        self.beta = beta

    def pdf(self, x):
        """Calculates the probability density function (PDF) at x."""
        return beta.pdf(x, self.alpha, self.beta)

    def sample(self, n_samples=1000):  # sample method
        """Generates random samples from the Beta distribution."""
        return beta.rvs(self.alpha, self.beta, size=n_samples)

def main():
    """Plots each distribution in a separate subplot."""

    distributions = [
        {"class": MonthlyMixedDistPDF, "params": {"num_days": 30}, "kwargs": {"num_samples": 5000}, "x_range": (0, 30)},
        {"class": WeeklyExponentialPDF, "params": {}, "kwargs": {"num_samples": 5000}, "x_range": (0, 7)},
        {"class": MixedWeibullPDF, "params": {"w1": 0.6, "w2": 0.2, "w3": 0.2, "weibull_shape": 0.8, "weibull_scale": 1.0, "norm_mu": 5, "norm_sigma": 1, "expon_lambda": 0.5, "num_samples": 5000}, "kwargs": {"n_samples": 5000}, "x_range": (0, 15)}, 
        {"class": BimodalExponPDF, "params": {"lambda1": 0.5, "lambda2": 0.5, "weight1": 3/5, "loc2": 15, "num_samples": 5000}, "kwargs": {"n_samples": 5000}, "x_range": (0, 30)}, 
        {"class": BetaDistributionPDF, "params": {"alpha": 2, "beta": 5}, "kwargs": {"n_samples": 5000}, "x_range": (0, 1)},
        {"class": BetaDistributionPDF, "params": {"alpha": 5, "beta": 2}, "kwargs": {"n_samples": 5000}, "x_range": (0, 1)}, # Added another Beta distribution
        {"class": BetaDistributionPDF, "params": {"alpha": 0.5, "beta": 0.5}, "kwargs": {"n_samples": 5000}, "x_range": (0, 1)} # Add a U shaped Beta distribution
    ]

    num_plots = len(distributions)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create a 2x4 grid of subplots

    for i, dist_info in enumerate(distributions):
        dist_class = dist_info["class"]
        params = dist_info["params"]
        kwargs = dist_info["kwargs"]
        x_min, x_max = dist_info["x_range"]

        dist = dist_class(**params)
        x = np.linspace(x_min, x_max, 200)

        if hasattr(dist, 'pdf'):
            samples = dist.sample(**kwargs)
            axes[i // 4, i % 4].plot(x, dist.pdf(x), label="PDF")
            axes[i // 4, i % 4].hist(samples, bins=50, density=True, alpha=0.6, label="Samples")

        else:
            # Calculate a discrete PDF for monthly and weekly distributions
            pdf = np.zeros(int(x_max - x_min + 1))
            samples = dist.sample(**kwargs)
            for day in range(int(x_min), int(x_max + 1)):
                count = np.sum(samples == day)
                pdf[day] = count / len(samples)
            axes[i // 4, i % 4].plot(range(int(x_min), int(x_max + 1)), pdf, label="PDF", drawstyle='steps-mid')
            axes[i // 4, i % 4].hist(samples, bins=int(x_max - x_min + 1), density=True, alpha=0.6, label="Samples", range=(x_min, x_max), align='left', rwidth=0.8)

        axes[i // 4, i % 4].set_title(dist_class.__name__, fontsize=10)
        axes[i // 4, i % 4].set_xlabel("x", fontsize=8)
        axes[i // 4, i % 4].set_ylabel("Density/Probability", fontsize=8)
        axes[i // 4, i % 4].legend(fontsize=8)
        axes[i // 4, i % 4].set_xlim(x_min, x_max)
        axes[i // 4, i % 4].grid(True)

    plt.tight_layout() # Improves subplot spacing
    plt.show()

if __name__ == "__main__":
    main()