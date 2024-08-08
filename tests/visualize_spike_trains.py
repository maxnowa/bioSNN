import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from network.neuronal_coding import rate_coding_constant, rate_coding_poisson
import tensorflow as tf

### --------------- LOAD DATA ---------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0
flattened_data = x_train.reshape((x_train.shape[0], -1))


def plot_spike_trains_3d(spike_trains, image_size=28):
    """
    Plot the generated spike trains in 3D space.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_pixels, num_timesteps = spike_trains.shape

    for i in range(num_pixels):
        for j in range(num_timesteps):
            if spike_trains[i, j] == 1:
                x = i % image_size  # Column index
                y = i // image_size  # Row index
                z = j  # Time step
                ax.scatter(x, y, z, color='b', marker='o')  # Plot spike

    ax.set_xlabel('X (Image Width)')
    ax.set_ylabel('Y (Image Height)')
    ax.set_zlabel('Time Step')

    plt.show()

# Example usage
# Assuming 'dataset' is your flattened MNIST dataset
# dataset = load_mnist_dataset()  # Load your MNIST dataset here

# Generate spike trains for the second sample
spike_trains = rate_coding_poisson(flattened_data, sample_idx=1, duration=100, max_rate=200, rest=0, coding_type="linear")

# Plot the spike trains in 3D
plot_spike_trains_3d(spike_trains)