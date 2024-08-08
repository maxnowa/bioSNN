from pathlib import Path
import json
import numpy as np
from network.logger import configure_logger

logger = configure_logger()
def save_scores(path, scores, matches):
    with open(path / "scores.json", "w") as file:
        json.dump(scores, file, indent=4)
    with open(path / "matches.json", "w") as file:
        json.dump(matches, file, indent=4)
        
def save_network(path, weights, neuron, network, stdp, training, weight_change):
    logger.info(f"Saving network to {path}")
    #save parametrs as json
    try:
        para_path = Path(path) / "parameters/"
        with open(para_path / 'neuron_parameters.json', 'w') as json_file:
            json.dump(neuron, json_file, indent=4)
        with open(para_path / 'network_parameters.json', 'w') as json_file:
            json.dump(network, json_file, indent=4)
        with open(para_path / 'stdp_parameters.json', 'w') as json_file:
            json.dump(stdp, json_file, indent=4)
        with open(para_path / "training_parameters.json", "w") as json_file:
            json.dump(training, json_file, indent=4)
        logger.info(" > Parameters saved")

        #save weights
        weight_path = Path(path) / "weights/weights.npy"
        np.save(weight_path, weights)
        saved_weights = Path(path) / "weights/weight_change.npy"
        np.save(saved_weights, weight_change)
        logger.info(" > Weights saved")
        script_path = Path(path) / "plot_weight_changes.py"
        with open(script_path, 'w') as script_file:
            script_file.write("""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path


def plot_weights_over_time(saved_weights):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    num_neurons = saved_weights[0].shape[1]

    # Initial plot
    weight_img = ax.imshow(saved_weights[0], aspect='auto', cmap='viridis')
    ax.set_title('Network Weights Over Time')
    ax.set_xlabel('Neurons')
    ax.set_ylabel('Weights ')
    cbar = fig.colorbar(weight_img, ax=ax)
    cbar.set_label('Weight Value')

    # Slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax_slider, 'Sample', 0, len(saved_weights) - 1, valinit=0, valstep=1)

    def update(val):
        sample_idx = int(slider.val)
        weight_img.set_data(saved_weights[sample_idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def plot_weight_image_change(weights_array):
    num_timesteps = len(weights_array)
    num_neurons = weights_array[0].shape[1]
    image_shape = (28, 28)

    # Determine the number of rows and columns for subplots
    cols = 8  # Fixed number of columns
    rows = (num_neurons // cols) + int(num_neurons % cols != 0)  # Calculate the number of rows

    # Create a figure and axis for the slider and plots
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    # Initial plot setup
    initial_weights = weights_array[0].reshape(image_shape[0], image_shape[1], num_neurons)
    images = []
    for i in range(num_neurons):
        im = axs[i].imshow(initial_weights[:, :, i], cmap="gray")
        axs[i].set_title(f"Image {i + 1}")
        axs[i].axis("off")
        images.append(im)
    for j in range(num_neurons, len(axs)):
        fig.delaxes(axs[j])  # Remove unused axes

    # Slider setup
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax_slider, 'Time', 0, num_timesteps - 1, valinit=0, valstep=1)

    def update(val):
        timestep = int(slider.val)
        new_weights = weights_array[timestep].reshape(image_shape[0], image_shape[1], num_neurons)
        for i in range(num_neurons):
            images[i].set_data(new_weights[:, :, i])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
if __name__ == "__main__":
    current_script_path = Path(__file__).resolve().parent
    # Construct absolute paths to the weight files
    weight_change_path = current_script_path / 'weights' / 'weight_change.npy'

    # load weights and display plots
    weights_array = np.load(weight_change_path)
    plot_weights_over_time(weights_array)
    plot_weight_image_change(weights_array)
""")
        logger.info(" > Python script saved")
    except Exception as e:
        logger.error(f"An error occurred: {e}")