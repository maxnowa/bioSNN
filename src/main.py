# main.py
import sys
from pathlib import Path
import os

os.environ["NUMEXPR_MAX_THREADS"] = "10"

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from network.setup import initialize_network
from network.training import train_network
from network.inference import (
    assign_classes,
    predict_labels,
    evaluate_model,
    evaluate_rmse_and_selectivity,
    remove_unassigned_neurons,
)
from network.analysis import (
    plot_weight_distribution,
    plot_weight_image,
    plot_training_metrics,
    plot_training_metrics_per_neuron,
    plot_selectivity,
    plot_weights_over_time,
    plot_weight_image_change,
    exclude_first,
    exclude_highest,
    plot_spike_counts_per_class,
)
from network.save_network import save_network, save_scores
from network.utils import create_experiment_folder

import tensorflow as tf
import numpy as np

### --------------- LOAD DATA ---------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0
flattened_data = x_train.reshape((x_train.shape[0], -1))
flattened_test = x_test.reshape((x_test.shape[0], -1))


### ------------- SPECIFY PARAMETERS ---------------
train = True
inference = False
data_set_size = 30000
trial_data = flattened_data[:data_set_size, :]
folder = "data/bioSNN-v1.05_MNIST60000_d5781b86_wmax-4_gamma-20_A_plus-0.012_ratio-1.06_coding-Constant_coding_type-linear_t_present-12_t_rest-0_max_rate-800_batch_size-500_EX_ONLY-True_WTA-Hard_verbose-True_epochs-5"
weight_path = Path(folder) / "weights/weights.npy"

stdp_paras = {"wmax": 4, "gamma": 1, "A_plus": 0.012, "ratio": 1.06}
network_paras = {
    "architecture": (784, 16),
    "connection_type": "SP",
    "seed": False,
    "init_mode": "uniform",
}
neuron_paras = {
    "neuron_type": "LIF",
    "spike_min": 1,
    "spike_max": 1,
    "adaptive_th": True,
    "error": False,
}
training_paras = {
    "coding": "Constant",
    "coding_type": "linear",
    "t_present": 50,
    "t_rest": 0,
    "max_rate": 800,
    "batch_size": 500,
    "EX_ONLY": True,
    "WTA": "Hard",
    "verbose": True,
    "epochs": 1,
}

### ------------- SETUP NETWORK ------------------
network = initialize_network(
    stdp_paras=stdp_paras, neuron_paras=neuron_paras, **network_paras
)


### ------------- TRAIN NETWORK -------------------
if train:
    neurons, weights, av_weights, rates, saved_weights = train_network(
        network, trial_data, **training_paras
    )

    # save stdp, network, training parameters
    data_set = "MNIST" + f"{data_set_size}"
    out_path = create_experiment_folder(data_set, **stdp_paras, **training_paras)
    plot_path = Path(out_path) / "plots/"

    save_network(
        out_path,
        weights,
        neuron_paras,
        network_paras,
        stdp_paras,
        training_paras,
        saved_weights,
    )

    ### ------------- ANALYZE NETWORK ------------------
    plot_weight_image(weights_array=weights, path=plot_path)
    if network_paras["architecture"][1] < 30:
        plot_weight_distribution(weights_array=weights, path=plot_path)
        plot_training_metrics(av_weights, rates, path=plot_path)
        plot_training_metrics_per_neuron(av_weights, rates, path=plot_path)

        plot_weights_over_time(saved_weights)
        plot_weight_image_change(saved_weights)

    exclude_first(training_paras, network)
    # exclude_highest(training_paras, network)
elif not train:
    network.weights = np.load(weight_path)  # load network here
    exclude_first(training_paras, network)
### ---------------- RUN INFERENCE ------------------
if inference:
    if not train:
        # create rerun folder
        plot_path = Path(folder) / "plots_rerun6/"
        plot_path.mkdir(parents=True, exist_ok=True)

    assignment_paras = {
        "num_epochs": 1,
        "coding": "Poisson",
        "coding_type": "linear",
        "t_present": 220,
        "t_rest": 120,
        "max_rate": 500,
        "EX_ONLY": True,
        "dominance": 2.5,
        "WTA": "Hard",
    }
    # change data here to control for selectivity measurement on test or train data
    spikes, selectivity = assign_classes(
        network, flattened_test, labels=y_test, **assignment_paras
    )
    plot_selectivity(spike_counts=spikes, path=plot_path)
    # print unique classes, remove -1 neurons and plot spike counts of cleaned network
    # selectivity = remove_unassigned_neurons(network, selectivity)
    plot_spike_counts_per_class(spikes, selectivity, path=plot_path)

    inference_paras = {
        "coding": "Poisson",
        "coding_type": "linear",
        "t_present": 250,
        "t_rest": 150,
        "max_rate": 700,
        "EX_ONLY": True,
    }
    # Test the accuracy and  plot confusion matrix
    y_pred = predict_labels(network, flattened_test, selectivity, **inference_paras)
    eval_scores = evaluate_model(y_test, y_pred, path=plot_path)
    neuron_results = evaluate_rmse_and_selectivity(
        x_test, network.weights, y_test, selectivity
    )

    # save network parameters
    score_path = Path(out_path) / "results"
    save_scores(score_path, eval_scores, neuron_results)
