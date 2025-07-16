import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

class CustomDistribution:
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
                day = int(np.random.uniform(0, self.num_days * self.first_peak_fraction))
            elif rand_val < self.first_peak_probability + self.second_peak_probability:
                day = int(np.random.uniform(self.num_days * (0.5-self.second_peak_fraction/2), 
                                            self.num_days * (0.5+self.second_peak_fraction/2) ))

            else:
                day = int(expon.rvs(scale=self.num_days) % self.num_days)
            samples.append(day)
        return np.array(samples)



# Example usage:
my_dist = CustomDistribution(num_days=30)  # Create an instance of the distribution

# Generate 10000 samples
samples = my_dist.sample(10000)

# Plot
plt.hist(samples, bins=30, density=True, alpha=0.7, label="Custom Distribution")
plt.xlabel("Day of the Month")
plt.ylabel("Probability Density")
plt.title("Custom Probability Distribution over a Month")
plt.show()




# Generate a single sample
single_sample = my_dist.sample()
print(f"Single sample: {single_sample}")  # Output a single sample


# Generate 5 samples:
five_samples = my_dist.sample(5)
print(f"Five samples: {five_samples}")
