# setup.py
import numpy as np
from network.network import Network
from network.logger import configure_logger
import sys
from pathlib import Path
from neurons import LIF, IZH

logger = configure_logger()
# sys.path.append(str(Path(__file__).resolve().parent.parent))


def initialize_weights(connection_matrix, wmin, wmax, seed=False, init_mode="uniform"):
    logger.info("Initializing weights")
    if seed:
        np.random.seed(0)
    pre, post = connection_matrix.shape
    if init_mode == "uniform":
        weights = np.random.uniform(wmin, wmax, size=(pre, post))
    elif init_mode == "normal":
        centre = (wmax - wmin) / 2
        weights = np.random.normal(loc=centre, scale=centre, size=(pre, post))
    # Zero out weights where there is no connection
    weights *= connection_matrix
    return weights


# neuron paras are loaded from a config file
def initialize_neurons(neuron_instances, neuron_type, spike_min, spike_max, adaptive_th, error, **kwargs):
    logger.info("Initializing neurons")

    if len(neuron_instances) > 1:
        pass
    else:
        post_neurons = []
        # initialize neurons with specified parameters
        for i in range(neuron_instances[0]):
            if neuron_type == "LIF":
                post_neurons.append(LIF.LIFNeuron(error=error, ad_th=adaptive_th, **kwargs))

            elif neuron_type == "IZH":
                post_neurons.append(IZH.IZHNeuron())

    # setup neurons for simulation
    for neuron in post_neurons:
        # for potential visualization
        neuron.v = neuron.v_init
        neuron.v_trace = []
        neuron.rec_spikes = []
        neuron.tr = 0
        neuron.spike_strength = np.random.uniform(spike_min, spike_max)
    return post_neurons


def initialize_connections(architecture, con_type):
    logger.info("Initializing connections")
    layers = len(architecture)
    input_layer_size = architecture[0]
    output_layer_size = architecture[-1]

    if con_type in ["Fully connected", "FC"]:
        # All connections exist
        connection_matrix = np.ones((input_layer_size, output_layer_size))
    elif con_type in ["Sparse", "SP"]:
        # Generate a sparse connection matrix with sparsity applied per postsynaptic neuron
        sparsity = 0.4  # Adjust the sparsity level as needed
        connection_matrix = np.zeros((input_layer_size, output_layer_size), dtype=int)

        for j in range(output_layer_size):
            # Randomly choose `sparsity * input_layer_size` connections for each postsynaptic neuron
            num_connections = int(sparsity * input_layer_size)
            if num_connections > 0:
                indices = np.random.choice(input_layer_size, num_connections, replace=False)
                connection_matrix[indices, j] = 1

    elif con_type in ["Randomly connected", "RC"]:
        # Random connections with certain probability
        connection_probability = 0.5  # Adjust the connection probability as needed
        connection_matrix = np.random.rand(input_layer_size, output_layer_size) < connection_probability
        connection_matrix = connection_matrix.astype(int)
    else:
        raise ValueError(f"Unknown connection type: {con_type}")

    return connection_matrix


def initialize_network(
    stdp_paras,
    neuron_paras,
    architecture=(784, 16),
    connection_type="FC",
    seed=True,
    init_mode="uniform"
):
    network = Network(**stdp_paras)
    # Get the connection matrix
    connection_matrix = initialize_connections(architecture, connection_type)

    network.weights = initialize_weights(
        connection_matrix, network.wmin, network.wmax, seed=seed, init_mode=init_mode
    )
    # For now, we will only initialize the output neurons
    output_layer_size = architecture[-1]
    network.neurons = initialize_neurons(
        [output_layer_size], **neuron_paras
    )
    logger.info(
        "Network initialized with configuration:"
        + "\n"
        + f"- Architecture: {architecture}"
        + "\n"
        + f"- Connection: {connection_type}"
        + "\n"
        + f"- Neuron Type: {neuron_paras['neuron_type']}"
    )
    return network
