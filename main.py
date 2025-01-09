# main.py
import sys
from pathlib import Path
import os

os.environ["NUMEXPR_MAX_THREADS"] = "10"

# Add the parent directory to the Python path
# sys.path.append(str(Path(__file__).resolve().parent.parent))

from network.setup import initialize_network
from network.training import train_network
from network.inference import (
    assign_and_predict,
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
    exclude_outliers,
    plot_spike_counts_per_class,
    plot_assigned_class_distribution,
    plot_spike_count_heatmap,
)
from network.save_network import save_network, save_scores
from network.utils import create_experiment_folder

import tensorflow as tf
import numpy as np

### --------------- LOAD DATA ---------------
data = "MNIST"
if data == "CIFAR10":
    dataset_object = tf.keras.datasets.cifar10
elif data == "MNIST":
    dataset_object = tf.keras.datasets.mnist
elif data == "N-MNIST":
    pass
elif data == "F-MNIST":
    dataset_object = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = dataset_object.load_data()
# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0
flattened_data = x_train.reshape((x_train.shape[0], -1))
flattened_test = x_test.reshape((x_test.shape[0], -1))


### ------------- SPECIFY PARAMETERS ---------------
train = False
inference = True
data_set_size = 60000
trial_data = flattened_data[:data_set_size, :]
folder = "data/bioSNN-v1.06_MNIST60000_e05fa553_wmax-4_A_plus-0.012_ratio-1.07_coding-Constant_coding_type-linear_t_present-4_t_rest-0_max_rate-1000_batch_size-500_EX_ONLY-True_WTA-Hard_verbose-True_epochs-3"
weight_path = Path(folder) / "weights/weights.npy"

stdp_paras = {"wmax": 4, "A_plus": 0.012, "ratio": 1.07}
network_paras = {
    "architecture": (784, 150),
    "connection_type": "FC",
    "seed": False,
    "init_mode": "uniform",
}
neuron_paras = {
    "neuron_type": "LIF",
    "spike_min": 1,
    "spike_max": 1,
    "adaptive_th": True,
    "error": False,
    "gamma": 1.2,
    "t_ref": 4,
}
training_paras = {
    "coding": "Constant",
    "coding_type": "linear",
    "t_present": 4,
    "t_rest": 0,
    "max_rate": 1000,
    "batch_size": 500,
    "EX_ONLY": True,
    "WTA": "Hard",
    "verbose": True,
    "epochs": 3,
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
    data_set = data + f"{data_set_size}"
    out_path = create_experiment_folder(
        data_set, inference, **stdp_paras, **training_paras
    )
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
    if network_paras["architecture"][1] < 50:
        # plot_weight_distribution(weights_array=weights, path=plot_path)
        plot_training_metrics(av_weights, rates, path=plot_path)
        # plot_training_metrics_per_neuron(av_weights, rates, path=plot_path)

        plot_weights_over_time(saved_weights)
        plot_weight_image_change(saved_weights)

    #exclude_first(training_paras, network)
    # exclude_first(training_paras, network)

    exclude_highest(training_paras, network)
elif not train:
    network.weights = np.load(weight_path)  # load network here
    #exclude_first(training_paras, network)
    exclude_outliers(training_paras, network)
    #exclude_highest(training_paras, network)
### ---------------- RUN INFERENCE ------------------
if inference:
    if not train:
        # create rerun folder
        plot_path = Path(folder) / "plots_rerun/"
        plot_path.mkdir(parents=True, exist_ok=True)

    assignment_paras = {
        "num_epochs": 1,
        "coding": "Constant",
        "coding_type": "linear",
        "t_present": 30,
        "t_rest": 0,
        "max_rate": 600,
        "EX_ONLY": True,
        "dominance": 2.5,
        "WTA": False,
        "method": "shared",
        "batch_size": 100,
    }
    prediction_paras = {
        "coding": "Constant",
        "coding_type": "linear",
        "t_present": 30,
        "t_rest": 0,
        "max_rate": 600,
        "EX_ONLY": True,
        "batch_size": 100,
        "method": "shared",
    }

    ## change the refractory period so that each neuron only fires once per sample
    for neuron in network.neurons:
        neuron.t_ref = 1
    
    spikes, selectivity, y_pred = assign_and_predict(
        network,
        assign_data=flattened_test,
        assign_labels=y_test,
        predict_data=flattened_test,
        assign_params=assignment_paras,
        predict_params=prediction_paras,
    )

    plot_spike_count_heatmap(spikes)
    plot_selectivity(
        spike_counts=spikes, neuron_selectivity=selectivity, path=plot_path
    )
    plot_assigned_class_distribution(selectivity, path=plot_path)
    # print unique classes, remove -1 neurons and plot spike counts of cleaned network

    # plot_spike_counts_per_class(spikes, selectivity, path=plot_path)

    # Test the accuracy and  plot confusion matrix
    eval_scores = evaluate_model(y_test, y_pred, path=plot_path)
    # neuron_results = evaluate_rmse_and_selectivity(
    #     x_test, network.weights, y_test, selectivity
    # )

    # # save network parameters
    # score_path = Path(out_path) / "results"
    # save_scores(score_path, eval_scores, neuron_results)
