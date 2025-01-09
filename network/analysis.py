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
    """
    Plots the weights as images in a grid layout with approximately equal columns and rows.

    Parameters:
        weights_array: A 2D numpy array of weights (pixels x neurons).
        path: Path to save the output images.
        label_size: Font size for the image titles.
    """
    num_neurons = weights_array.shape[1]
    reshaped_array = weights_array.reshape(28, 28, num_neurons)

    # Dynamically calculate the number of columns and rows for a square layout
    cols = int(np.ceil(np.sqrt(num_neurons)))
    rows = int(np.ceil(num_neurons / cols))

    plt.figure(figsize=(cols * 2, rows * 2))  # Adjust figure size based on layout

    for i in range(num_neurons):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(reshaped_array[:, :, i], cmap="gray")
        plt.title(f"Neuron {i + 1}", fontsize=label_size)
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

# def plot_selectivity(spike_counts, path):
#     num_neurons = spike_counts.shape[0]
#     num_digits = spike_counts.shape[1]

#     # Determine the grid size (rows and columns) for the subplots
#     cols = 8  # Number of columns for the grid
#     rows = int(np.ceil(num_neurons / cols))  # Calculate the number of rows needed

#     fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))  # Adjust the figsize accordingly
#     axs = axs.flatten()

#     for neuron_idx in range(num_neurons):
#         ax = axs[neuron_idx]
#         ax.bar(range(num_digits), spike_counts[neuron_idx], tick_label=list(range(num_digits)))
#         ax.set_title(f"Neuron {neuron_idx + 1}")

#         # Highlight the highest value
#         max_idx = np.argmax(spike_counts[neuron_idx])
#         max_value = spike_counts[neuron_idx, max_idx]
#         ax.plot(max_idx, max_value, "ro")  # 'ro' means red circle

#     # Hide any unused subplots
#     for neuron_idx in range(num_neurons, len(axs)):
#         fig.delaxes(axs[neuron_idx])

#     plt.subplots_adjust(wspace=0.4, hspace=0.5)
#     out_png = Path(path) / "selectivity.png"
#     plt.savefig(out_png)
#     plt.close()
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_selectivity(spike_counts, neuron_selectivity, path, num_classes=10):
    """
    Efficiently plots the average spike count and selectivity of neurons assigned to each class.

    Args:
        spike_counts (np.ndarray): Spike counts for each neuron (rows) and each class (columns).
        neuron_selectivity (array-like): Assigned class or weights from the assign_classes function.
                                         For shared methods, pass the top class indices or weights.
        path (str): Directory path to save the plot.
        num_classes (int): Total number of classes. Default is 10.
    """
    # Initialize result arrays
    avg_spike_counts = np.zeros(num_classes)
    avg_selectivity_differences = np.zeros(num_classes)

    # Determine neuron assignments for each class
    if len(neuron_selectivity.shape) > 1:  # Shared method
        neuron_classes = np.argmax(neuron_selectivity, axis=1)
    else:  # Non-shared method
        neuron_classes = neuron_selectivity

    # Compute metrics for each class in a vectorized manner
    for cls in range(num_classes):
        # Get indices of neurons assigned to this class
        assigned_neurons = neuron_classes == cls
        if np.any(assigned_neurons):  # If any neurons are assigned
            class_spike_counts = spike_counts[assigned_neurons]
            avg_spike_counts[cls] = np.mean(class_spike_counts[:, cls])
            max_selectivities = np.max(class_spike_counts, axis=1)
            mean_selectivities = np.mean(class_spike_counts, axis=1)
            avg_selectivity_differences[cls] = np.mean(max_selectivities - mean_selectivities)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot average spike counts
    bar_width = 0.6
    ax.bar(range(num_classes), avg_spike_counts, width=bar_width, color="skyblue", label="Avg Spike Count")

    # Plot selectivity differences as red dots
    ax.plot(range(num_classes), avg_selectivity_differences, 'ro-', label="Avg Selectivity Difference")

    # Annotate bars with exact values
    for i, (spike_count, selectivity_diff) in enumerate(zip(avg_spike_counts, avg_selectivity_differences)):
        ax.text(i, spike_count + 0.5, f"{spike_count:.2f}", ha="center", va="bottom", fontsize=10, color="blue")
        ax.text(i, selectivity_diff + 0.5, f"{selectivity_diff:.2f}", ha="center", va="bottom", fontsize=10, color="red")

    # Customize the plot
    ax.set_xlabel("Class")
    ax.set_ylabel("Spike Count / Selectivity")
    ax.set_title("Average Spike Count and Selectivity by Class")
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(range(num_classes))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot
    out_png = Path(path) / "selectivity_by_class.png"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# def plot_selectivity(spike_counts, path, label_size=14):
#     num_neurons = spike_counts.shape[0]
#     num_digits = spike_counts.shape[1]

#     # Determine the grid size (rows and columns) for the subplots
#     cols = 5  # Number of columns for the grid
#     rows = int(np.ceil(num_neurons / cols))  # Calculate the number of rows needed

#     fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))  # Adjust the figsize accordingly

#     for neuron_idx in range(num_neurons):
#         row = neuron_idx // cols
#         col = neuron_idx % cols

#         ax = axs[row, col]
#         ax.bar(
#             range(num_digits),
#             spike_counts[neuron_idx],
#             tick_label=list(range(num_digits)),
#         )
#         ax.set_title(f"Neuron {neuron_idx + 1}", fontsize=label_size)

#         # Set x and y labels only for the first column and last row
#         if col == 0:
#             ax.set_ylabel("Spike Count", fontsize=label_size)
#         if row == rows - 1:
#             ax.set_xlabel("Digit", fontsize=label_size)

#         # Highlight the highest value
#         max_idx = np.argmax(spike_counts[neuron_idx])
#         max_value = spike_counts[neuron_idx, max_idx]
#         ax.plot(max_idx, max_value, "ro")  # 'ro' means red circle

#         # Annotate the highest value
#         ax.annotate(
#             f"{max_value}",
#             xy=(max_idx, max_value),
#             xytext=(max_idx, max_value + 1),
#         )

#     # Hide any unused subplots
#     for neuron_idx in range(num_neurons, rows * cols):
#         if neuron_idx >= num_neurons:
#             fig.delaxes(axs.flatten()[neuron_idx])

#     plt.subplots_adjust(wspace=0.4, hspace=0.5)
#     out_png = Path(path) / "selectivity.png"
#     out_svg = Path(path) / "selectivity.svg"
#     plt.savefig(out_png)
#     plt.savefig(out_svg)
#     plt.show()

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
        logger.info(f"Removed teacher neuron {exclusion}")


def exclude_highest(training_parameters, network):
    average_weights = np.mean(network.weights, axis=0)
    if training_parameters["coding"] == "Constant":
        # Find the indices of the three highest average weights
        exclusions = np.argsort(average_weights)[-2:]
        exclusions = sorted(
            exclusions, reverse=True
        )  # Sort in reverse order for safe deletion

        # Remove the neurons and their corresponding weights
        for exclusion in exclusions:
            del network.neurons[exclusion]
            network.weights = np.delete(network.weights, exclusion, axis=1)
            logger.info(f"Removed Neuron {exclusion}")

def exclude_outliers(training_parameters, network):
    """
    Excludes neurons whose average weight deviates significantly from the mean.
    
    Parameters:
        training_parameters (dict): Training parameters, including "coding".
        network: The network model object, which contains neurons and weights.
    """
    # Compute the average weights for each neuron
    average_weights = np.mean(network.weights, axis=0)
    
    # Compute the mean and standard deviation of the average weights
    mean_weight = np.mean(average_weights)
    std_weight = np.std(average_weights)
    
    # Identify neurons whose average weights deviate beyond 2 standard deviations
    threshold = 2 * std_weight
    outliers = np.where(np.abs(average_weights - mean_weight) > threshold)[0]
    
    # Sort outliers in reverse order for safe deletion
    outliers = sorted(outliers, reverse=True)
    
    # Log the exclusions
    logger.info(f"Excluding {len(outliers)} neurons: {outliers}")
    
    # Remove the outlier neurons and their corresponding weights
    for exclusion in outliers:
        del network.neurons[exclusion]
        network.weights = np.delete(network.weights, exclusion, axis=1)
        logger.info(f"Removed Neuron {exclusion}")

def  plot_spike_counts_per_class(spike_counts, neuron_selectivity, path):
    total_spikes_per_class = np.sum(spike_counts, axis=0)
    classes = np.arange(10)

    # # Plot and save spike counts per class
    # plt.figure(figsize=(10, 6))
    # plt.bar(classes, total_spikes_per_class)
    # plt.xlabel('Class Label')
    # plt.ylabel('Total Spike Count')
    # plt.title('Spike Counts per Class')
    # plt.xticks(classes)
    # plt.savefig(f"{path}/spike_counts_per_class.png")
    # plt.close()

    # Plot and save the distribution of classes assigned to neurons
    assigned_class_counts = np.bincount(neuron_selectivity, minlength=10)
    plt.figure(figsize=(10, 6))
    plt.plot(classes, assigned_class_counts, marker='o')
    plt.xlabel('Class Label')
    plt.ylabel('Number of Neurons Assigned')
    plt.title('Distribution of Classes Assigned to Neurons')
    plt.xticks(classes)
    plt.grid(True)
    plt.savefig(f"{path}/distribution_classes_assigned.png")
    plt.close()

    # Plot and save the number of spikes per class divided by the number of neurons having that class
    average_spikes_per_neuron = total_spikes_per_class / (assigned_class_counts + (assigned_class_counts == 0))
    plt.figure(figsize=(10, 6))
    plt.bar(classes, average_spikes_per_neuron)
    plt.xlabel('Class Label')
    plt.ylabel('Average Spike Count per Neuron')
    plt.title('Average Spike Count per Neuron for Each Class')
    plt.xticks(classes)
    plt.savefig(f"{path}/average_spike_count_per_neuron.png")
    plt.close()


def plot_spike_count_heatmap(spike_counts, title="Neuron Spike Count Heatmap"):
    plt.figure(figsize=(10, 8))
    plt.imshow(spike_counts, aspect="auto", cmap="viridis")
    plt.colorbar(label="Spike Count")
    plt.xlabel("Class")
    plt.ylabel("Neuron Index")
    plt.title(title)
    plt.show()

def plot_assigned_class_distribution(neuron_selectivity, num_classes=10, title="Assigned Class Distribution", path=None):
    """
    Plots the distribution of assigned classes based on neuron selectivity.

    Parameters:
        neuron_selectivity (array-like): Array of class assignments (one per neuron).
                                         For shared methods, pass the top classes or weights.
                                         Use -1 for unassigned neurons.
        num_classes (int): Total number of classes. Default is 10.
        title (str, optional): Title of the plot. Default is "Assigned Class Distribution".

    Returns:
        None
    """
    # Handle the case where neuron_selectivity is a weight matrix (shared method)
    if len(neuron_selectivity.shape) > 1:  # Shared method with weights
        assigned_classes = np.argmax(neuron_selectivity, axis=1)
    else:
        assigned_classes = neuron_selectivity

    # Exclude unassigned neurons (-1)
    valid_assignments = assigned_classes[assigned_classes >= 0]

    # Count occurrences of each class
    unique_classes, class_counts = np.unique(valid_assignments, return_counts=True)

    # Prepare a full list of class counts for all classes (including missing)
    full_class_counts = np.zeros(num_classes, dtype=int)
    full_class_counts[unique_classes] = class_counts

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), full_class_counts, tick_label=range(num_classes))
    plt.xlabel("Class")
    plt.ylabel("Number of Neurons")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate bars with counts
    for i, count in enumerate(full_class_counts):
        plt.text(i, count + 0.5, str(count), ha="center", va="bottom", fontsize=10)
    plt.savefig(path / "class_distribution.png")
    plt.close()
    #plt.show()


if __name__ == "__main__":

    pass

