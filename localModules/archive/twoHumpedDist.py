import numpy as np
from scipy.stats import weibull_min, norm, expon
import matplotlib.pyplot as plt

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



# Example usage:
params = {
    "w1": 0.6, "w2": 0.2, "w3": 0.2,
    "weibull_shape": 0.8, "weibull_scale": 1.0,
    "norm_mu": 5, "norm_sigma": 1, "expon_lambda": 0.5, "num_samples":5000
}
mixed_pdf = MixedWeibullPDF(**params)

# Generate samples
samples = mixed_pdf.sample()


# Plotting (separate call)
x = np.linspace(0, 15, 200)
pdf_values = mixed_pdf.pdf(x)
plt.plot(x, pdf_values, label="PDF")
plt.hist(samples, bins=50, density=True, alpha=0.5, label="Samples") # Histogram of samples
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Mixture Distribution")
plt.legend()
plt.grid(True)
plt.show()