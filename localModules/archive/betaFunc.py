import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

class BetaDistribution:
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


# Example usage:
# Create instances of the BetaDistribution class with different parameters:

distributions = [
    BetaDistribution(0.5, 0.5),    # U-shaped
    BetaDistribution(1, 1),        # Uniform
    BetaDistribution(2, 2),        # Bell-shaped, symmetric
    BetaDistribution(5, 2),        # Right-skewed
    BetaDistribution(2, 5),        # Left-skewed
    BetaDistribution(0.5, 5),      # J-shaped
    BetaDistribution(5, 0.5)     # Reverse J-shaped

]


# Plotting
x = np.linspace(0, 1, 200)  # Adjust as needed


plt.figure(figsize=(10, 6))
for dist in distributions:
    plt.plot(x, dist.pdf(x), label=f"α={dist.alpha}, β={dist.beta}")
    samples = dist.sample(n_samples=5000) # Example of generating samples.
    plt.hist(samples, bins=30, density=True, alpha=0.4, label=f"Samples (α={dist.alpha}, β={dist.beta})") # Histogram of samples

plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Beta Distributions")
plt.xlim(0, 1)
plt.ylim(0, None)  # Dynamically adjust y-axis limits
plt.grid(True)
plt.legend()
plt.show()