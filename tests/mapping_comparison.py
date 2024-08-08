"""
Script for comparing the different mapping types for rate
coding, in terms of the rate - pixel intensity tuning curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pixel_to_rate_exponential(pixel_value, max_rate=800, min_rate=0, scale=5):
    normalized_pixel = pixel_value / 255.0
    exponential_values = np.exp(scale * normalized_pixel) - 1
    normalized_exponential_values = exponential_values / np.max(exponential_values)
    rate = min_rate + (max_rate - min_rate) * normalized_exponential_values
    return rate


def pixel_to_rate_linear(pixel_value, max_rate=800, min_rate=0):
    normalized_pixel = pixel_value / 255.0
    rate = min_rate + (max_rate - min_rate) * normalized_pixel
    return rate


def pixel_to_rate_constant(pixel_value, max_rate=1000, min_rate=0):
    normalized_pixel = pixel_value / 255.0
    isi = 1 / normalized_pixel
    rate = 1 / (isi / 1000)
    return rate


# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

pixel_values = np.linspace(0, 255, 256)
linear_rates = pixel_to_rate_linear(pixel_values)
exponential_rates = pixel_to_rate_exponential(pixel_values, scale=5)
constant_rates = pixel_to_rate_constant(pixel_values)

plt.figure(figsize=(8, 6))
plt.plot(pixel_values, linear_rates, label="Linear Mapping", linewidth=2)
plt.plot(
    pixel_values,
    exponential_rates,
    label="Exponential Mapping",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    pixel_values, constant_rates, label="Constant Mapping", linestyle="-.", linewidth=2
)

# Add label for max rate on linear mapping
max_rate_linear = pixel_to_rate_linear(255)


# Add label for max rate on exponential mapping
max_rate_exponential = pixel_to_rate_exponential(255)


plt.xlabel("Pixel Intensity", fontsize=14, fontweight="bold")
plt.ylabel("Rate", fontsize=14, fontweight="bold")

plt.legend(fontsize=16)
plt.grid(True)

# Adjust the gridlines and background for better readability
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

plt.savefig("tests/plots/comparison.svg", format="svg")
plt.show()
