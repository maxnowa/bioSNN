"""
Script for plotting the distribution of pixel intensities 
in the MNIST data set, with and without zeros.
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist


def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)

def plot_pixel_distribution(data, remove_zeros=False, ax=None):
    pixel_values = data.flatten()
    if remove_zeros:
        pixel_values = pixel_values[pixel_values > 0]

    bins_range = (1, 255) if remove_zeros else (0, 255)
    ax.hist(pixel_values, bins=256, range=bins_range, density=True, color='blue', alpha=0.7)
    title = 'MNIST Non-Zero Pixel Values' if remove_zeros else 'MNIST Pixel Values'
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel('Pixel Value', fontsize=14, fontweight="bold")
    ax.set_ylabel('Frequency', fontsize=14, fontweight="bold")
    ax.grid(True)

def main():
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    data = np.concatenate((X_train, X_test), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot with zeros
    plot_pixel_distribution(data, remove_zeros=False, ax=axes[0])

    # Plot without zeros
    plot_pixel_distribution(data, remove_zeros=True, ax=axes[1])

    plt.tight_layout()
    plt.savefig("tests/plots/mnist_distribution.svg", format="svg")
    plt.show()

if __name__ == "__main__":
    main()
