# inference.py
import numpy as np
from network.logger import configure_logger
from network.neuronal_coding import (
    random_time_coding,
    exact_time_coding,
    generate_inhibitory,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from pathlib import Path
from sklearn.metrics import mean_squared_error


logger = configure_logger()


def run_inference(
    network,
    data,
    labels,
    coding,
    coding_type="linear",
    t_present=100,
    t_rest=0,
    max_rate=200,
    EX_ONLY=False,
):
    logger = configure_logger()
    logger.info("Running network in inference mode")
    # create array to store input currents
    I = np.zeros(network.weights.shape[1])
    n_inhib = 128
    if coding == "Constant":
        spike_train = exact_time_coding(
            dataset=data, duration=t_present, rest=t_rest
        )
    elif coding == "Poisson":
        spike_train = random_time_coding(
            dataset=data,
            duration=t_present,
            rest=t_rest,
            max_rate=max_rate,
            coding_type=coding_type,
        )

    if not EX_ONLY:
        inhib_spikes = generate_inhibitory(10, n_inhib, spike_train.shape[1])

    spike_counts = np.zeros(
        (len(network.neurons), 10)
    )  # to store spike counts for each neuron and digit

    num_samples = len(labels)
    for sample_idx in tqdm(range(num_samples), desc="Running inference"):
        start_time = sample_idx * (t_present + t_rest)
        end_time = start_time + t_present
        for timestep in range(start_time, end_time):
            for index, neuron in enumerate(network.neurons):

                if neuron.check_spike(network.dt):
                    spike_counts[index, labels[sample_idx]] += 1
                if EX_ONLY:
                    I_neg = 0
                else:
                    I_neg = inhib_spikes[:, timestep].sum()
                I_pos = (
                    np.dot(network.weights[:, index], spike_train[:, timestep].T)
                    * neuron.spike_strength
                )
                I[index] = I_pos + I_neg
                neuron.update_state(network.dt, I[index])

    # Determine neuron selectivity based on spike counts
    neuron_selectivity = np.argmax(spike_counts, axis=1)
    return spike_counts, neuron_selectivity


def predict_labels(
    network,
    data,
    neuron_selectivity,
    coding="Constant",
    coding_type="linear",
    t_present=100,
    t_rest=0,
    max_rate=200,
    EX_ONLY=False,
):
    num_samples = len(data)
    predicted_labels = np.zeros(num_samples)
    I = np.zeros(network.weights.shape[1])
    n_inhib = 128
    if coding == "Constant":
        spike_train = exact_time_coding(
            dataset=data, duration=t_present, rest=t_rest
        )
    elif coding == "Poisson":
        spike_train = random_time_coding(
            dataset=data,
            duration=t_present,
            rest=t_rest,
            max_rate=max_rate,
            coding_type=coding_type,
        )

    if not EX_ONLY:
        inhib_spikes = generate_inhibitory(10, n_inhib, spike_train.shape[1])

    for sample_idx in tqdm(range(num_samples), desc="Predicting labels"):
        start_time = sample_idx * (t_present + t_rest)
        end_time = start_time + t_present
        response_rates = np.zeros(len(network.neurons))
        for timestep in range(start_time, end_time):
            for index, neuron in enumerate(network.neurons):
                if neuron.check_spike(network.dt):
                    response_rates[index] += 1
                if EX_ONLY:
                    I_neg = 0
                else:
                    I_neg = inhib_spikes[:, timestep].sum()
                I_pos = (
                    np.dot(network.weights[:, index], spike_train[:, timestep].T)
                    * neuron.spike_strength
                )
                I[index] = I_pos + I_neg
                neuron.update_state(network.dt, I[index])
        highest_response_neuron = np.argmax(response_rates)
        predicted_labels[sample_idx] = neuron_selectivity[highest_response_neuron]
    return predicted_labels


### Modified `evaluate_model` Function


def evaluate_model(true_labels, predicted_labels, path):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted", zero_division=np.nan
    )
    results_dict = {"Precision": precision, "Recall": recall, "F1": f1}
    cm = confusion_matrix(true_labels, predicted_labels)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    out_path = Path(path) / "confusion_matrix.png"
    plt.savefig(out_path)
    out_svg = Path(path) / "confusion_matrix.svg"
    plt.savefig(out_svg)
    plt.show()
    return results_dict


def average_images_per_digit(test_images, test_labels):
    """Average all test images per pixel that have the same digit."""
    unique_digits = np.unique(test_labels)
    averaged_images = {}

    for digit in unique_digits:
        digit_indices = np.where(test_labels == digit)[0]
        digit_images = test_images[digit_indices]
        averaged_image = np.mean(digit_images, axis=0)
        averaged_images[digit] = averaged_image

    return averaged_images


def evaluate_rmse_and_selectivity(
    test_images, weights, test_labels, selectivity_vector
):
    """Evaluate RMSE for each neuron and compare with selectivity."""
    # Average test images per digit
    averaged_images = average_images_per_digit(test_images, test_labels)

    neuron_results = []

    for neuron_index, neuron_specific_digit in enumerate(selectivity_vector):
        neuron_reconstructed_vector = weights[:, neuron_index]
        rmse_per_digit = {}

        for digit, averaged_image in averaged_images.items():
            flattened_averaged_image = averaged_image.flatten()
            rmse = np.sqrt(
                mean_squared_error(
                    flattened_averaged_image, neuron_reconstructed_vector
                )
            )
            rmse_per_digit[digit] = rmse

        best_matching_digit = min(rmse_per_digit, key=rmse_per_digit.get)
        best_rmse = rmse_per_digit[best_matching_digit]

        logger.info(
            f"Neuron {neuron_index}: specific to {neuron_specific_digit} | best matching to {best_matching_digit} | RMSE {best_rmse}"
        )

        neuron_results.append(
            (
                int(neuron_index),
                int(neuron_specific_digit),
                int(best_matching_digit),
                float(best_rmse),
            )
        )

    return neuron_results
