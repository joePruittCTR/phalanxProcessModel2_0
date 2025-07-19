import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import calendar

def create_yearly_arrival_pdf(year, base_lambda, summer_boost, monthly_peak_lambda_factor, midmonth_peak_factor, special_peak_lambda_factor):
    """
    Creates a yearly probability distribution function (PDF) for arrivals,
    following an exponential distribution with seasonal, monthly, and special
    peak variations.

    Args:
        year (int): The year for which to create the PDF.  Used to calculate the number of days.
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
    #   We'll normalize this *after* applying all the boosts.  This makes the boosts easier to reason about.

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


if __name__ == '__main__':
    # --- Parameter Tuning ---
    year = 2024  # Example year
    base_lambda = 1.0  # Base arrival rate
    summer_boost = 1.5  # Increase arrivals by 50% in summer
    monthly_peak_lambda_factor = 3.0 # Monthly peak lambda multiplier
    midmonth_peak_factor = 0.5  # Mid-month peak is 50% of the monthly peak
    special_peak_lambda_factor = 4.0  # Special peak lambda multiplier

    # Create the PDF
    yearly_pdf = create_yearly_arrival_pdf(year, base_lambda, summer_boost, monthly_peak_lambda_factor, midmonth_peak_factor, special_peak_lambda_factor)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_pdf)
    plt.title('Yearly Arrival Probability Distribution')
    plt.xlabel('Day of the Year')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()

    # --- Basic Statistics (Example) ---
    # You can use this PDF to sample arrival times in your simulation
    # For example:
    num_samples = 1000
    days = np.arange(len(yearly_pdf))
    arrival_days = np.random.choice(days, size=num_samples, p=yearly_pdf)

    print(f"Generated {num_samples} arrival days.")
    print(f"Mean arrival day: {np.mean(arrival_days)}")
    print(f"Std dev arrival day: {np.std(arrival_days)}")
    plt.hist(arrival_days, bins=30)
    plt.title('Histogram of Sampled Arrival Days')
    plt.xlabel('Day of the Year')
    plt.ylabel('Frequency')
    plt.show()
