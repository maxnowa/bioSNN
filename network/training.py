# training.py
import numpy as np
from network.utils import create_batches
from network.logger import configure_logger, log_time
from network.neuronal_coding import rate_coding_poisson, generate_inhibitory, rate_coding_constant
import time


logger = configure_logger()

def train_network(
    network,
    data,
    coding,
    coding_type="linear",
    t_present=100,
    t_rest=0,
    max_rate=200,
    batch_size=500,
    WTA="Hard",
    EX_ONLY=False,
    verbose=False,
):

    n_inhib = 128
    batches = create_batches(data, batch_size=batch_size)
    logger.info(f"Created {len(batches)} batches of size {batch_size}")

    LTP = np.zeros(network.weights.shape)
    LTD = np.zeros(network.weights.shape[1])
    I = np.zeros(network.weights.shape[1])

    average_weight = []
    total_rates = []
    saved_weights = []

    logger.info("Finished setup - training started")
    starttime = time.time()
    sample_counter = 0

    # assumes the data has the same format as mnist
    for ind, batch in enumerate(batches):
        batch_start_time = time.time()
        postsynaptic_rate = np.ones(network.weights.shape[1])

        if coding == "Poisson":
            spike_train = rate_coding_poisson(
                batch,
                duration=t_present,
                max_rate=max_rate,
                rest=t_rest,
                coding_type=coding_type,
            )
        elif coding == "Constant":
            spike_train = rate_coding_constant(batch, duration=t_present, rest=t_rest)
        if not EX_ONLY:
            inhib_spikes = generate_inhibitory(10, n_inhib, spike_train.shape[1])
        logger.info(f"Batch {ind + 1}/{len(batches)}")

        for timestep in range(spike_train.shape[1]):
            fired_neuron_index = None
            for index, neuron in enumerate(network.neurons):
                if neuron.check_spike(
                    network.dt, neuron_ind=fired_neuron_index, WTA=WTA
                ):
                    fired_neuron_index = index
                    LTD[index] -= network.A_minus
                    network.weights[:, index] += LTP[:, index] * network.wmax
                    network.weights[:, index] = np.clip(
                        network.weights[:, index],
                        a_min=network.wmin,
                        a_max=network.wmax,
                    )

                LTD[index] -= LTD[index] * network.dt / network.tau_neg
                d_ltp = (
                    -(LTP[:, index] * network.dt / network.tau_pos)
                    + network.A_plus * spike_train[:, timestep]
                )
                LTP[:, index] += d_ltp

                if fired_neuron_index is None or index == fired_neuron_index:
                    network.weights[:, index] += (
                        LTD[index] * spike_train[:, timestep]
                    ) * network.wmax
                    network.weights[:, index] = np.clip(
                        network.weights[:, index],
                        a_min=network.wmin,
                        a_max=network.wmax,
                    )

                if EX_ONLY:
                    I_neg = 0
                if not EX_ONLY:
                    I_neg = -network.wneg * inhib_spikes[:, timestep].sum()

                I_pos = (
                    np.dot(network.weights[:, index], spike_train[:, timestep].T)
                    * neuron.spike_strength
                )
                I[index] = I_pos + I_neg
                neuron.update_state(network.dt, I[index])

            sample_counter += 1
            if sample_counter % 1000 == 0 or sample_counter == 1:
                saved_weights.append(np.copy(network.weights))

            if fired_neuron_index is not None:
                for index, neuron in enumerate(network.neurons):
                    if index != fired_neuron_index:
                        if WTA == "Hard":
                            neuron.v = neuron.v_reset
                        elif WTA == "Soft":
                            neuron.v = max(
                                neuron.v - (0.5 * (neuron.v_thresh - neuron.v_reset)),
                                neuron.v_reset,
                            )

        # Estimate and log remaining time
        batch_end_time = time.time()
        hours, minutes, seconds = log_time(
            starttime, batch_end_time=batch_end_time, n_batches=len(batches), ind=ind
        )
        logger.info(
            f"Estimated remaining time: {int(hours)}h {int(minutes)}m {seconds:.0f}s"
        )

        # calculate convergence values after each batch
        # average weight
        average_weight.append(np.average(network.weights, axis=0))
        # postsynaptic rate estimation as spikes in Hz
        for n, neuron in enumerate(network.neurons):
            postsynaptic_rate[n] = (
                sum(neuron.rec_spikes) / (spike_train.shape[1] * (ind + 1)) * 1000
            )
        total_rates.append(postsynaptic_rate)

    # Save the final weights
    saved_weights.append(np.copy(network.weights))

    # Training duration
    endtime = time.time()
    hours, minutes, seconds = log_time(starttime, endtime)
    logger.info(
        "Training finished\n"
        + "\t" + f"Duration: {int(hours)}h {int(minutes)}m {seconds:.2f}s"
    )

    if verbose:
        return network.neurons, network.weights, average_weight, total_rates, saved_weights
    return network.neurons, network.weights, saved_weights