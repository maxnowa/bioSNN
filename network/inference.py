# inference.py
import numpy as np
from network.logger import configure_logger
from network.neuronal_coding import (
    random_time_coding,
    exact_time_coding,
    generate_inhibitory,
)
from network.utils import create_batches
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

def temperature_softmax(x, temperature=1.0):
    """
    Applies temperature scaling to the softmax function.

    Parameters:
        x (ndarray): Input array (e.g., spike counts).
        temperature (float): Temperature parameter. Higher values smoothen the output.

    Returns:
        ndarray: Temperature-scaled softmax probabilities.
    """
    scaled_x = x / temperature
    e_x = np.exp(scaled_x - np.max(scaled_x, axis=1, keepdims=True))  # Stability trick
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def remove_unassigned_neurons(network, neuron_selectivity):
    # count unique classes before pruning
    unique_classes = np.unique(neuron_selectivity)
    unique_classes = unique_classes[unique_classes != -1]  # Exclude class -1
    logger.info(f"Number of different classes before pruning: {len(unique_classes)}")
    # remove -1 and duplicate classes
    assigned_indices = np.where(neuron_selectivity != -1)[0]
    network.neurons = [network.neurons[i] for i in assigned_indices]
    network.weights = network.weights[:, assigned_indices]
    neuron_selectivity = neuron_selectivity[assigned_indices]
    # count unique classes after pruning
    after_prune = np.unique(neuron_selectivity)
    logger.info(f"Output layer pruned. Remaining unique classes: {len(after_prune)}")
    return neuron_selectivity


def assign_and_predict(
    network,
    assign_data,
    assign_labels,
    predict_data,
    assign_params,
    predict_params,
):
    """
    Combines class assignment and prediction into one function.
    
    Parameters:
        network: The network model object.
        assign_data: Data for assigning classes.
        assign_labels: Labels corresponding to `assign_data`.
        predict_data: Data for prediction.
        assign_params (dict): Parameters for class assignment:
            - coding
            - coding_type
            - t_present
            - t_rest
            - max_rate
            - EX_ONLY
            - WTA
            - num_epochs
            - batch_size
            - method
        predict_params (dict): Parameters for prediction:
            - coding
            - coding_type
            - t_present
            - t_rest
            - max_rate
            - EX_ONLY
            - WTA
            - batch_size
    
    Returns:
        spike_counts: The spike counts from the assignment phase.
        neuron_selectivity: The neuron-to-class assignments.
        predicted_labels: Predicted labels for the `predict_data`.
    """
    # Extract parameters for assignment
    assign_coding = assign_params.get("coding", "Poisson")
    assign_coding_type = assign_params.get("coding_type", "linear")
    assign_t_present = assign_params.get("t_present", 100)
    assign_t_rest = assign_params.get("t_rest", 0)
    assign_max_rate = assign_params.get("max_rate", 500)
    assign_EX_ONLY = assign_params.get("EX_ONLY", True)
    assign_WTA = assign_params.get("WTA", False)
    assign_num_epochs = assign_params.get("num_epochs", 1)
    assign_batch_size = assign_params.get("batch_size", 128)
    assign_method = assign_params.get("method", "shared")

    logger.info(f"Running class assignment with method {assign_method}")
    # Assign Classes
    spike_counts, neuron_selectivity = assign_classes(
        network=network,
        data=assign_data,
        labels=assign_labels,
        coding=assign_coding,
        coding_type=assign_coding_type,
        t_present=assign_t_present,
        t_rest=assign_t_rest,
        max_rate=assign_max_rate,
        EX_ONLY=assign_EX_ONLY,
        WTA=assign_WTA,
        num_epochs=assign_num_epochs,
        batch_size=assign_batch_size,
        method=assign_method,
    )

    # Extract parameters for prediction
    predict_coding = predict_params.get("coding", "Poisson")
    predict_coding_type = predict_params.get("coding_type", "linear")
    predict_t_present = predict_params.get("t_present", 100)
    predict_t_rest = predict_params.get("t_rest", 0)
    predict_max_rate = predict_params.get("max_rate", 500)
    predict_EX_ONLY = predict_params.get("EX_ONLY", True)
    predict_WTA = predict_params.get("WTA", False)
    predict_batch_size = predict_params.get("batch_size", 128)
    assignment_method = predict_params.get("method", "shared")
    logger.info(f"Running prediction with method {assignment_method}")
    
    # Predict Labels
    predicted_labels = predict_labels(
        network=network,
        data=predict_data,
        neuron_selectivity=neuron_selectivity,
        coding=predict_coding,
        coding_type=predict_coding_type,
        t_present=predict_t_present,
        t_rest=predict_t_rest,
        max_rate=predict_max_rate,
        EX_ONLY=predict_EX_ONLY,
        WTA=predict_WTA,
        batch_size=predict_batch_size,
        assignment_method=assignment_method
    )

    return spike_counts, neuron_selectivity, predicted_labels

def assign_classes(
    network,
    data,
    labels,
    coding,
    num_epochs=2,
    coding_type="linear",
    t_present=100,
    t_rest=0,
    max_rate=200,
    EX_ONLY=False,
    dominance=2,
    WTA=False,
    batch_size=128,
    method="threshold"
):
    logger = configure_logger()
    logger.info("Running network in inference mode")
    I = np.zeros(network.weights.shape[1])
    n_inhib = 128

    spike_counts = np.zeros((len(network.neurons), 10))
    digit_counts = np.bincount(labels, minlength=10)

    # Create batches
    data_batches = create_batches(data, batch_size)
    label_batches = create_batches(labels, batch_size)

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (batch_data, batch_labels) in tqdm(
            enumerate(zip(data_batches, label_batches)),
            desc=f"Epoch {epoch + 1}/{num_epochs} - Processing batches",
            total=len(data_batches),
        ):
            spike_train = (
                exact_time_coding(dataset=batch_data, duration=t_present, rest=t_rest)
                if coding == "Constant"
                else random_time_coding(
                    dataset=batch_data,
                    duration=t_present,
                    rest=t_rest,
                    max_rate=max_rate,
                    coding_type=coding_type,
                )
            )
            inhib_spikes = (
                generate_inhibitory(10, n_inhib, spike_train.shape[1])
                if not EX_ONLY
                else None
            )

            for sample_idx in range(len(batch_labels)):
                for index, neuron in enumerate(network.neurons):
                    neuron.v = neuron.v_reset
                    neuron.tr = 0
                start_time, end_time = (
                    sample_idx * (t_present + t_rest),
                    sample_idx * (t_present + t_rest) + t_present,
                )
                for timestep in range(start_time, end_time):
                    I_neg = 0 if EX_ONLY else inhib_spikes[:, timestep].sum()
                    fired_neuron_index = None
                    for index, neuron in enumerate(network.neurons):
                        if neuron.check_spike(
                            network.dt, neuron_ind=fired_neuron_index, WTA=WTA
                        ):
                            spike_counts[index, batch_labels[sample_idx]] += 1
                            fired_neuron_index = index
                        I[index] = (
                            np.dot(
                                network.weights[:, index], spike_train[:, timestep].T
                            )
                            * neuron.spike_strength
                            + I_neg
                        )
                        neuron.update_state(network.dt, I[index])

    spike_counts /= (digit_counts[np.newaxis, :] + (digit_counts == 0)) * num_epochs

    if method == "threshold":
        normalized_spike_counts = spike_counts / spike_counts.sum(axis=1, keepdims=True)
        threshold = 0.5
        neuron_selectivity = np.argmax(normalized_spike_counts, axis=1)
        neuron_selectivity = np.where(
            np.max(normalized_spike_counts, axis=1) >= threshold, neuron_selectivity, -1
        )

    elif method == "softmax":
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        softmax_spike_counts = softmax(spike_counts.T).T
        neuron_selectivity = np.argmax(softmax_spike_counts, axis=1)
    
    # this proves to be useless
    elif method == "clustering":
        from sklearn.cluster import KMeans
        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(spike_counts)
        neuron_selectivity = kmeans.labels_

    elif method == "wta_global":
        neuron_selectivity = -1 * np.ones(len(spike_counts))
        for digit in range(10):
            max_spike_neuron = np.argmax(spike_counts[:, digit])
            if spike_counts[max_spike_neuron, digit] > 0:
                neuron_selectivity[max_spike_neuron] = digit

    elif method == "entropy":
        from scipy.stats import entropy

        spike_probabilities = spike_counts / spike_counts.sum(axis=1, keepdims=True)
        neuron_entropies = entropy(spike_probabilities.T)
        specificity_threshold = 1.5
        neuron_selectivity = np.argmax(spike_counts, axis=1)
        neuron_selectivity = np.where(
            neuron_entropies <= specificity_threshold, neuron_selectivity, -1
        )

    # elif method == "shared":
    #     # implementation for temp softmax
    #     #normalized_spike_counts = temperature_softmax(spike_counts, temperature=2.0)
    #     normalized_spike_counts = spike_counts / spike_counts.sum(axis=1, keepdims=True)
    #     neuron_selectivity = normalized_spike_counts
    elif method == "shared":
        # # Normalize spike counts for all neurons
        # normalized_spike_counts = spike_counts / spike_counts.sum(axis=1, keepdims=True)
        
        # # Get the number of neurons assigned to each class
        # class_counts = np.bincount(np.argmax(normalized_spike_counts, axis=1), minlength=normalized_spike_counts.shape[1])
        
        # # Scale the weights inversely proportional to the number of neurons in each class
        # scaling_factors = 1 / (class_counts + 1e-6)  # Add epsilon to avoid division by zero
        # scaled_spike_counts = normalized_spike_counts * scaling_factors[np.newaxis, :]
        
        # # Normalize scaled spike counts to ensure they sum to 1
        # neuron_selectivity = scaled_spike_counts / scaled_spike_counts.sum(axis=1, keepdims=True)

        ##### new code #####
        # Normalize spike counts for all neurons
        normalized_spike_counts = spike_counts / spike_counts.sum(axis=1, keepdims=True)

        # Count neurons assigned to each class
        class_counts = np.bincount(np.argmax(normalized_spike_counts, axis=1), minlength=normalized_spike_counts.shape[1])

        # Prevent scaling of classes with zero neurons assigned
        scaling_factors = np.zeros_like(class_counts, dtype=float)
        scaling_factors[class_counts > 0] = 1 / class_counts[class_counts > 0]

        # Scale spike counts inversely proportional to class counts
        scaled_spike_counts = normalized_spike_counts * scaling_factors[np.newaxis, :]

        # Normalize scaled spike counts to ensure they sum to 1 across classes
        neuron_selectivity = scaled_spike_counts / scaled_spike_counts.sum(axis=1, keepdims=True)

    else:
        raise ValueError(f"Invalid method '{method}' specified.")

    return spike_counts, neuron_selectivity



def predict_labels(
    network,
    data,
    neuron_selectivity,
    assignment_method="threshold",
    coding="Constant",
    coding_type="linear",
    t_present=100,
    t_rest=0,
    max_rate=200,
    EX_ONLY=False,
    WTA=False,
    batch_size=128,
):
    num_samples = len(data)
    predicted_labels = np.zeros(num_samples)
    I = np.zeros(network.weights.shape[1])
    n_inhib = 128

    # Create batches
    data_batches = create_batches(data, batch_size)

    for batch_idx, batch_data in tqdm(
        enumerate(data_batches),
        desc="Predicting labels - Processing batches",
        total=len(data_batches),
    ):
        spike_train = (
            exact_time_coding(dataset=batch_data, duration=t_present, rest=t_rest)
            if coding == "Constant"
            else random_time_coding(
                dataset=batch_data,
                duration=t_present,
                rest=t_rest,
                max_rate=max_rate,
                coding_type=coding_type,
            )
        )
        if not EX_ONLY:
            inhib_spikes = generate_inhibitory(10, n_inhib, spike_train.shape[1])

        for sample_idx in range(len(batch_data)):
            for index, neuron in enumerate(network.neurons):
                neuron.v = neuron.v_reset
                neuron.tr = 0

            global_sample_idx = batch_idx * batch_size + sample_idx
            start_time = sample_idx * (t_present + t_rest)
            end_time = start_time + t_present
            response_rates = np.zeros(len(network.neurons))

            for timestep in range(start_time, end_time):
                fired_neuron_index = None
                for index, neuron in enumerate(network.neurons):
                    if neuron.check_spike(
                        network.dt, neuron_ind=fired_neuron_index, WTA=WTA
                    ):
                        response_rates[index] += 1
                        fired_neuron_index = index

                    I_neg = 0 if EX_ONLY else inhib_spikes[:, timestep].sum()
                    I_pos = (
                        np.dot(network.weights[:, index], spike_train[:, timestep].T)
                        * neuron.spike_strength
                    )
                    I[index] = I_pos + I_neg
                    neuron.update_state(network.dt, I[index])

            if assignment_method in ["threshold", "wta_global", "entropy"]:
                highest_response_neuron = np.argmax(response_rates)
                predicted_labels[global_sample_idx] = neuron_selectivity[
                    highest_response_neuron
                ]

            elif assignment_method == "softmax":
                # Compute softmax probabilities over response rates
                neuron_probabilities = np.exp(response_rates) / np.sum(np.exp(response_rates))
                weighted_response_rates = response_rates[:, None] * neuron_probabilities[:, None]
                predicted_labels[global_sample_idx] = np.argmax(weighted_response_rates.sum(axis=0))

            elif assignment_method == "clustering":
                highest_response_neuron = np.argmax(response_rates)
                predicted_labels[global_sample_idx] = neuron_selectivity[
                    highest_response_neuron
                ]

            elif assignment_method == "shared":

                class_responses = np.zeros(neuron_selectivity.shape[1])
                for neuron_idx, class_weights in enumerate(neuron_selectivity):
                    class_responses += response_rates[neuron_idx] * class_weights
                predicted_labels[global_sample_idx] = np.argmax(class_responses)

            else:
                raise ValueError(f"Invalid assignment method: {assignment_method}")

    return predicted_labels



##### MEASURES FOR MODEL QUALITY #####

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
    # out_svg = Path(path) / "confusion_matrix.svg"
    # plt.savefig(out_svg)
    #plt.show()
    plt.close()  # Close the plot to prevent it from displaying
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


#### OLD CODE ####
# def predict_labels(
#     network,
#     data,
#     neuron_selectivity,
#     coding="Constant",
#     coding_type="linear",
#     t_present=100,
#     t_rest=0,
#     max_rate=200,
#     EX_ONLY=False,
#     WTA=False
# ):
#     num_samples = len(data)
#     predicted_labels = np.zeros(num_samples)
#     I = np.zeros(network.weights.shape[1])
#     n_inhib = 128

#     # parameters for calcium trace
#     calcium = np.zeros(network.weights.shape[1])
#     dC = 0.2
#     C_target = 2
#     k = 5
#     tau_C = 80

#     if coding == "Constant":
#         spike_train = exact_time_coding(
#             dataset=data, duration=t_present, rest=t_rest
#         )
#     elif coding == "Poisson":
#         spike_train = random_time_coding(
#             dataset=data,
#             duration=t_present,
#             rest=t_rest,
#             max_rate=max_rate,
#             coding_type=coding_type,
#         )

#     if not EX_ONLY:
#         inhib_spikes = generate_inhibitory(10, n_inhib, spike_train.shape[1])

#     for sample_idx in tqdm(range(num_samples), desc="Predicting labels"):
#         start_time = sample_idx * (t_present + t_rest)
#         end_time = start_time + t_present
#         response_rates = np.zeros(len(network.neurons))
#         for timestep in range(start_time, end_time):
#             fired_neuron_index = None
#             for index, neuron in enumerate(network.neurons):
#                 if neuron.check_spike(network.dt, neuron_ind=fired_neuron_index, WTA=WTA):
#                     response_rates[index] += 1
#                     fired_neuron_index = index
#                     calcium[index] += dC
#                     neuron.v += k*(C_target - calcium[index])
#                 if EX_ONLY:
#                     I_neg = 0
#                 else:
#                     I_neg = inhib_spikes[:, timestep].sum()
#                 I_pos = (
#                     np.dot(network.weights[:, index], spike_train[:, timestep].T)
#                     * neuron.spike_strength
#                 )
#                 I[index] = I_pos + I_neg
#                 neuron.update_state(network.dt, I[index])
#                 calcium[index] -= calcium[index]/tau_C

#             # # apply WTA
#             # if fired_neuron_index is not None and WTA is not False:
#             #     for index, neuron in enumerate(network.neurons):
#             #         if index != fired_neuron_index:
#             #             if WTA == "Hard":
#             #                 neuron.v = neuron.v_reset
#             #             elif WTA == "Soft":
#             #                 neuron.v = max(
#             #                     neuron.v - (0.5 * (neuron.v_thresh - neuron.v_reset)),
#             #                     neuron.v_reset,
#             #                 )
#         highest_response_neuron = np.argmax(response_rates)
#         predicted_labels[sample_idx] = neuron_selectivity[highest_response_neuron]
#     return predicted_labels


# def assign_classes(
#     network,
#     data,
#     labels,
#     coding,
#     num_epochs=2,
#     coding_type="linear",
#     t_present=100,
#     t_rest=0,
#     max_rate=200,
#     EX_ONLY=False,
#     dominance=2,
#     WTA=False
# ):
#     logger = configure_logger()
#     logger.info("Running network in inference mode")
#     I = np.zeros(network.weights.shape[1])
#     n_inhib = 128

#     # parameters for calcium trace
#     calcium = np.zeros(network.weights.shape[1])
#     dC = 0.2
#     C_target = 2
#     k = 5
#     tau_C = 20


#     spike_counts = np.zeros((len(network.neurons), 10))
#     digit_counts = np.bincount(labels, minlength=10)
    
#     for epoch in range(num_epochs):
#         logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
#         spike_train = (
#             exact_time_coding(dataset=data, duration=t_present, rest=t_rest)
#             if coding == "Constant"
#             else random_time_coding(dataset=data, duration=t_present, rest=t_rest, max_rate=max_rate, coding_type=coding_type)
#         )
#         inhib_spikes = generate_inhibitory(10, n_inhib, spike_train.shape[1]) if not EX_ONLY else None

#         for sample_idx in tqdm(range(len(labels)), desc=f"Epoch {epoch + 1}/{num_epochs} - Running inference"):
#             start_time, end_time = sample_idx * (t_present + t_rest), sample_idx * (t_present + t_rest) + t_present
#             for timestep in range(start_time, end_time):
#                 I_neg = 0 if EX_ONLY else inhib_spikes[:, timestep].sum()
                
#                 fired_neuron_index = None
#                 for index, neuron in enumerate(network.neurons):
#                     if neuron.check_spike(network.dt, neuron_ind=fired_neuron_index, WTA=WTA):
#                         spike_counts[index, labels[sample_idx]] += 1
#                         # get index of neuron that fired 
#                         fired_neuron_index = index
#                         # increase calcium trace and adjust membrane potential based on trace value at that timestep
#                         calcium[index] += dC
#                         neuron.v += k*(C_target - calcium[index])
#                     I[index] = np.dot(network.weights[:, index], spike_train[:, timestep].T) * neuron.spike_strength + I_neg
#                     neuron.update_state(network.dt, I[index])

#                     # exp decay of calcium trace
#                     calcium[index] -= calcium[index]/tau_C
#                 # # apply WTA
#                 # if fired_neuron_index is not None and WTA is not False:
#                 #     for index, neuron in enumerate(network.neurons):
#                 #         if index != fired_neuron_index:
#                 #             if WTA == "Hard":
#                 #                 neuron.v = neuron.v_reset
#                 #             elif WTA == "Soft":
#                 #                 neuron.v = max(
#                 #                     neuron.v - (0.5 * (neuron.v_thresh - neuron.v_reset)),
#                 #                     neuron.v_reset,
#                 #                 )

#     spike_counts /= (digit_counts[np.newaxis, :] + (digit_counts == 0)) * num_epochs
#     # Determine neuron selectivity based on spike counts and thresholding by mean condition
#     mean_spikes = spike_counts.mean(axis=1)
#     max_spikes = np.max(spike_counts, axis=1)
#     neuron_selectivity = np.argmax(spike_counts, axis=1)
#     # neuron_selectivity = np.where(max_spikes > dominance * mean_spikes, np.argmax(spike_counts, axis=1), -1)
#     # # Resolve multiple assignments of the same class to different neurons
#     # for digit in range(10):
#     #     assigned_neurons = np.where(neuron_selectivity == digit)[0]
#     #     if len(assigned_neurons) > 1:
#     #         max_spike_neuron = assigned_neurons[np.argmax(spike_counts[assigned_neurons, digit])]
#     #         neuron_selectivity[assigned_neurons] = -1
#     #         neuron_selectivity[max_spike_neuron] = digit

#     return spike_counts, neuron_selectivity
