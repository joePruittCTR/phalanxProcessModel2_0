import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

class WeeklyCycleDistribution:
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

    def sample(self, num_samples=1):
        """Generates samples from the weekly distribution."""
        samples = []
        for _ in range(num_samples):
            if np.random.rand() < self.first_day_probability:
                day = 0  # First day of the week (index 0)
            else:
                # Distribute remaining probability exponentially over the rest of the week
                day = int(expon.rvs(scale=(self.num_days - 1))+1 ) % self.num_days  # Shifted exponential, wrapped

            samples.append(day)

        return np.array(samples)

# Example Usage (as a function call):
def sample_weekly_distribution(num_samples=1, first_day_probability=0.90, num_days=7):
    """
    Generates samples from the weekly distribution using a function.

    This function encapsulates the class functionality for a simplified call.

    """
    dist = WeeklyCycleDistribution(num_days, first_day_probability)
    return dist.sample(num_samples)




# Generate samples using the function:
samples = sample_weekly_distribution(num_samples=10000)  # Use function directly



# Plotting (same as before):
plt.hist(samples, bins=np.arange(8)-0.5, density=True, alpha=0.7, rwidth=0.8, align='mid', label="Weekly Distribution")
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.xlabel("Day of the Week")
plt.ylabel("Probability Density")
plt.title("Weekly Cycle Distribution (90% on First Day)") # Updated title
plt.legend()
plt.show()

# Example single/multiple samples (using the function):
single = sample_weekly_distribution()
print(f"Single Sample: {single}")

five = sample_weekly_distribution(5)
print(f"Five Samples: {five}")
