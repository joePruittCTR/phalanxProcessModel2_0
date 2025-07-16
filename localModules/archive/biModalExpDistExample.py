import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

class BimodalExpon:
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



# Example Usage
bimodal = BimodalExpon(lambda1=0.5, lambda2=0.5, weight1=3/5, loc2=15, num_samples=10000)

# Generate Samples (using the default num_samples)
samples = bimodal.sample()

# Or generate a different number of samples:
more_samples = bimodal.sample(n_samples=20000)

# Plotting (separate call)
x = np.linspace(0, 35, 500)  # Adjust range as needed
pdf_values = bimodal.pdf(x)

plt.hist(samples, bins=50, density=True, alpha=0.7, label="Samples (10000)")
plt.hist(more_samples, bins=50, density=True, alpha=0.5, label="Samples (20000)")  # Plot both sample sets
plt.plot(x, pdf_values, 'r-', label="PDF")

plt.xlabel('Day of the Month')
plt.ylabel('Probability Density')
plt.title('Bimodal Exponential Distribution')
plt.legend()
plt.show()