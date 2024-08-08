# setup.py
import numpy as np
from network.network import Network
from network.logger import configure_logger
import sys
from pathlib import Path
from neurons import LIF, IZH

logger = configure_logger()
sys.path.append(str(Path(__file__).resolve().parent.parent))


def initialize_weights(pre, post, wmin, wmax, seed=False, init_mode="uniform"):
    logger.info("Initializing weights")
    if seed:
        np.random.seed(0)
    if init_mode == "uniform":
        weights = np.random.uniform(wmin, wmax, size=(pre, post))
    elif init_mode == "normal":
        centre = (wmax-wmin)/2
        weights = np.random.normal(loc=centre, scale=centre, size=(pre, post))
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
    input_layer = architecture[0]
    output_layer = architecture[-1]
    if con_type in ["Fully connected", "FC"]:
        pass
    elif con_type in ["Sparse", "SP"]:
        pass
    elif con_type in ["Randomly connected", "RC"]:
        pass
    # TODO add functionality for adding more layers here
    return input_layer, output_layer


def initialize_network(
    stdp_paras,
    neuron_paras,
    architecture=(784, 16),
    connection_type="FC",
    seed=True,
    init_mode = "uniform"
):
    network = Network(**stdp_paras)
    # only important for making connections random or sparse
    input_layer, output_layer = initialize_connections(architecture, connection_type)

    network.weights = initialize_weights(
        input_layer, output_layer, network.wmin, network.wmax, seed=seed, init_mode=init_mode
    )
    network.neurons = initialize_neurons(
        [output_layer], **neuron_paras
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
