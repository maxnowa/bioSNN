import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from network.logger import configure_logger


logger = configure_logger()
label_size = 20

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.widgets import Slider

# Set global font size
plt.rcParams.update({"font.size": 14})


# ------------- WEIGHT RECONSTRUCTION -----------------
def plot_weight_image(weights_array, path, label_size=18):
    num_neurons = weights_array.shape[1]
    reshaped_array = weights_array.reshape(28, 28, num_neurons)

    cols = 8  # Fixed number of columns
    rows = (num_neurons // cols) + int(
        num_neurons % cols != 0
    )  # Calculate the number of rows

    plt.figure(figsize=(cols * 2, rows * 2))  # Adjust figure size as needed

    for i in range(num_neurons):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(reshaped_array[:, :, i], cmap="gray")
        plt.title(f"Image {i + 1}", fontsize=label_size)
        plt.axis("off")
    plt.tight_layout()

    out_png = Path(path) / "weight_images.png"
    out_svg = Path(path) / "weight_images.svg"
    plt.savefig(out_png)
    plt.savefig(out_svg)
    plt.show()


def plot_weight_distribution(weights_array, path, label_size=18):
    def plot_and_save(weights_array, suffix, exclude_zeros, layout=None):
        num_neurons = weights_array.shape[1]
        if layout is None:
            rows = int(
                np.ceil(num_neurons / 4)
            )  # Adjust the number of columns as needed
            cols = min(num_neurons, 4)
        else:
            rows, cols = layout
        figsize = (5 * cols, 5 * rows)  # Adjust figure size for readability

        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < num_neurons:
                    data = weights_array[:, idx]
                    if exclude_zeros:
                        data = data[data != 0]  # Exclude zero weights
                    axs[i, j].hist(data, bins=30, color="skyblue", alpha=0.7)
                    axs[i, j].set_title(
                        f"Neuron {idx + 1}", fontsize=label_size
                    )
                    # Set x and y labels only for the first column and last row
                    if j == 0:
                        axs[i, j].set_ylabel("Frequency", fontsize=label_size)
                    if i == rows - 1:
                        axs[i, j].set_xlabel("Weight Value", fontsize=label_size)
                else:
                    axs[i, j].axis("off")  # Hide empty subplots

        plt.tight_layout()
        out_png = Path(path) / f"weight_distribution_{suffix}.png"
        out_svg = Path(path) / f"weight_distribution_{suffix}.svg"
        plt.savefig(out_png)
        plt.savefig(out_svg)
        plt.show()

    # Plot and save with zeros included for all neurons
    plot_and_save(weights_array, "with_zeros", exclude_zeros=False)
    # Plot and save with zeros excluded for all neurons
    plot_and_save(weights_array, "no_zeros", exclude_zeros=True)

    # # Plot and save with zeros included for first four neurons
    # first_four_weights_array = weights_array[:, :4]  # Select only the first 4 neurons
    # plot_and_save(
    #     first_four_weights_array,
    #     "first_four_with_zeros",
    #     exclude_zeros=False,
    #     layout=(2, 2),
    # )
    # # Plot and save with zeros excluded for first four neurons
    # plot_and_save(
    #     first_four_weights_array,
    #     "first_four_no_zeros",
    #     exclude_zeros=True,
    #     layout=(2, 2),
    # )


def plot_training_metrics(average_weight, total_rates, path, label_size=16):
    average_weight = np.array(average_weight)
    total_rates = np.array(total_rates)

    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # Plot average weights
    axs[0].plot(np.mean(average_weight, axis=1))
    # axs[0].set_title("Average Weights Over Time", fontsize=label_size)
    # axs[0].set_xlabel("Batch", fontsize=label_size)
    axs[0].set_ylabel("Average Weight", fontsize=label_size)

    # Plot total firing rates for each neuron
    for neuron_idx in range(total_rates.shape[1]):
        axs[1].plot(total_rates[:, neuron_idx], label=f"Neuron {neuron_idx + 1}")
    # axs[1].set_title("Firing Rates Over Time for Each Neuron", fontsize=label_size)
    # axs[1].set_xlabel("Batch", fontsize=label_size)
    axs[1].set_ylabel("Firing Rate (Hz)", fontsize=label_size)
    # axs[1].legend(loc="upper right")

    # Plot total firing rates excluding the first neuron
    for neuron_idx in range(1, total_rates.shape[1]):
        axs[2].plot(total_rates[:, neuron_idx], label=f"Neuron {neuron_idx + 1}")
    # axs[2].set_title(
    #    "Firing Rates Over Time for Each Neuron (Excluding First Neuron)",
    #    fontsize=label_size,
    # )
    axs[2].set_xlabel("Batch", fontsize=label_size)
    axs[2].set_ylabel("Firing Rate (Hz)", fontsize=label_size)
    # axs[2].legend(loc="upper right")

    plt.tight_layout()

    out_png = Path(path) / "training_metrics.png"
    out_svg = Path(path) / "training_metrics.svg"
    plt.savefig(out_png)
    plt.savefig(out_svg)
    plt.show()


def plot_training_metrics_per_neuron(
    average_weight, total_rates, path, label_size=18, plot_first_four=False
):
    average_weight = np.array(average_weight)
    total_rates = np.array(total_rates)

    if plot_first_four:
        num_neurons = 4
        rows, cols = 2, 2  # 2x2 layout for first 4 neurons
        figsize = (8, 6)  # Adjust figure size for readability
    else:
        num_neurons = total_rates.shape[1]
        rows = int(
            np.ceil(np.sqrt(num_neurons))
        )  # Adjust rows to be the square root of num_neurons
        cols = int(np.ceil(num_neurons / rows))  # Adjust cols accordingly
        figsize = (20, 14)  # Original figure size

    fig, axs = plt.subplots(
        rows, cols, figsize=figsize
    )  # Adjust figsize for better fit

    for neuron_idx in range(num_neurons):
        row = neuron_idx // cols
        col = neuron_idx % cols

        ax = axs[row, col]

        # Plot average weights for the neuron
        ax.plot(average_weight[:, neuron_idx], label="Average Weight")
        ax.set_xlabel("Batch", fontsize=label_size)
        ax.set_ylabel("Average Weight", fontsize=label_size)

        # Plot total firing rates for the neuron
        ax2 = ax.twinx()
        ax2.plot(total_rates[:, neuron_idx], label="Firing Rate", color="orange")
        ax2.set_ylabel("Firing Rate (Hz)", fontsize=label_size)

        ax.set_title(f"Neuron {neuron_idx + 1}", fontsize=label_size)

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    # Hide any unused subplots
    for neuron_idx in range(num_neurons, rows * cols):
        fig.delaxes(axs.flatten()[neuron_idx])

    plt.tight_layout()
    out_png = Path(path) / "training_metrics_per_neuron.png"
    out_svg = Path(path) / "training_metrics_per_neuron.svg"
    plt.savefig(out_png)
    plt.savefig(out_svg)
    plt.show()


def plot_selectivity(spike_counts, path, label_size=16):
    num_neurons = spike_counts.shape[0]
    num_digits = spike_counts.shape[1]

    # Determine the grid size (rows and columns) for the subplots
    cols = 5  # Number of columns for the grid
    rows = int(np.ceil(num_neurons / cols))  # Calculate the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))  # Adjust the figsize accordingly

    for neuron_idx in range(num_neurons):
        row = neuron_idx // cols
        col = neuron_idx % cols

        ax = axs[row, col]
        ax.bar(
            range(num_digits),
            spike_counts[neuron_idx],
            tick_label=list(range(num_digits)),
        )
        ax.set_title(f"Neuron {neuron_idx + 1}", fontsize=label_size)

        # Set x and y labels only for the first column and last row
        if col == 0:
            ax.set_ylabel("Spike Count", fontsize=label_size)
        if row == rows - 1:
            ax.set_xlabel("Digit", fontsize=label_size)

        # Highlight the highest value
        max_idx = np.argmax(spike_counts[neuron_idx])
        max_value = spike_counts[neuron_idx, max_idx]
        ax.plot(max_idx, max_value, "ro")  # 'ro' means red circle

        # Annotate the highest value
        ax.annotate(
            f"{max_value}",
            xy=(max_idx, max_value),
            xytext=(max_idx, max_value + 1),
            arrowprops=dict(facecolor="red", shrink=0.05),
        )

    # Hide any unused subplots
    for neuron_idx in range(num_neurons, rows * cols):
        if neuron_idx >= num_neurons:
            fig.delaxes(axs.flatten()[neuron_idx])

    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    out_png = Path(path) / "selectivity.png"
    out_svg = Path(path) / "selectivity.svg"
    plt.savefig(out_png)
    plt.savefig(out_svg)
    plt.show()

# visualization.py


def plot_weights_over_time(saved_weights, label_size=18):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    num_neurons = saved_weights[0].shape[1]

    # Initial plot
    weight_img = ax.imshow(saved_weights[0], aspect="auto", cmap="viridis")
    ax.set_title("Network Weights Over Time", fontsize=label_size)
    ax.set_xlabel("Neurons", fontsize=label_size)
    ax.set_ylabel("Weights", fontsize=label_size)
    cbar = fig.colorbar(weight_img, ax=ax)
    cbar.set_label("Weight Value", fontsize=label_size)

    # Slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(
        ax_slider, "Sample", 0, len(saved_weights) - 1, valinit=0, valstep=1
    )

    def update(val):
        sample_idx = int(slider.val)
        weight_img.set_data(saved_weights[sample_idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def plot_weight_image_change(weights_array, label_size=18):
    num_timesteps = len(weights_array)
    num_neurons = weights_array[0].shape[1]
    image_shape = (28, 28)

    # Determine the number of rows and columns for subplots
    cols = 8  # Fixed number of columns
    rows = (num_neurons // cols) + int(
        num_neurons % cols != 0
    )  # Calculate the number of rows

    # Create a figure and axis for the slider and plots
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    # Initial plot setup
    initial_weights = weights_array[0].reshape(
        image_shape[0], image_shape[1], num_neurons
    )
    images = []
    for i in range(num_neurons):
        im = axs[i].imshow(initial_weights[:, :, i], cmap="gray")
        axs[i].set_title(f"Image {i + 1}", fontsize=label_size)
        axs[i].axis("off")
        images.append(im)
    for j in range(num_neurons, len(axs)):
        fig.delaxes(axs[j])  # Remove unused axes

    # Slider setup
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax_slider, "Time", 0, num_timesteps - 1, valinit=0, valstep=1)

    def update(val):
        timestep = int(slider.val)
        new_weights = weights_array[timestep].reshape(
            image_shape[0], image_shape[1], num_neurons
        )
        for i in range(num_neurons):
            images[i].set_data(new_weights[:, :, i])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def exclude_first(training_parameters, network):
    average_weights = np.mean(network.weights, axis=0)
    if training_parameters["coding"] == "Constant":
        exclusion = np.argmax(average_weights)
        del network.neurons[exclusion]
        network.weights = np.delete(network.weights, exclusion, axis=1)
        logger.info(f"Removed Neuron {exclusion}")


def exclude_highest(training_parameters, network):
    average_weights = np.mean(network.weights, axis=0)
    if training_parameters["coding"] == "Constant":
        # Find the indices of the three highest average weights
        exclusions = np.argsort(average_weights)[-3:]
        exclusions = sorted(
            exclusions, reverse=True
        )  # Sort in reverse order for safe deletion

        # Remove the neurons and their corresponding weights
        for exclusion in exclusions:
            del network.neurons[exclusion]
            network.weights = np.delete(network.weights, exclusion, axis=1)
            logger.info(f"Removed Neuron {exclusion}")

def plot_spike_counts_per_class(spike_counts, neuron_selectivity):
    assigned_indices = np.where(neuron_selectivity != -1)[0]
    filtered_spike_counts = spike_counts[assigned_indices]
    total_spikes_per_class = np.sum(filtered_spike_counts, axis=0)
    classes = np.arange(10)

    plt.figure(figsize=(10, 6))
    plt.bar(classes, total_spikes_per_class)
    plt.xlabel('Class Label')
    plt.ylabel('Total Spike Count')
    plt.title('Spike Counts per Class (After Removing Unassigned Neurons)')
    plt.xticks(classes)
    plt.show()

if __name__ == "__main__":

    # Example data for testing
    weights_array = np.random.rand(784, 16)  # Example weights array (28*28, 32 neurons)
    average_weight = np.random.rand(50, 16)  # 50 batches, 32 neurons
    total_rates = np.random.rand(50, 16)  # 50 batches, 32 neurons
    spike_counts = np.random.randint(0, 100, (10, 10))  # 10 neurons, 10 digits
    saved_weights = [
        np.random.rand(32, 32) for _ in range(10)
    ]  # 10 timesteps, 32x32 weights

    # Create a path to save the plots
    path = "plots"
    Path(path).mkdir(parents=True, exist_ok=True)

    # Plot functions
    plot_weight_image(weights_array, path, label_size=label_size)
    plot_weight_distribution(weights_array, path, label_size=label_size)
    plot_training_metrics(average_weight, total_rates, path)
    plot_training_metrics_per_neuron(average_weight, total_rates, path)
    plot_selectivity(spike_counts, path, label_size=label_size)
