import numpy as np


def generate_inhibitory(rate, n, length, myseed=False):

    # set random seed
    if myseed:
        np.random.seed(seed=myseed)
    else:
        np.random.seed()
    dt = 1
    # generate uniformly distributed random variables
    u_rand = np.random.rand(n, length)
    # generate Poisson train
    poisson_train = 1.0 * (u_rand < rate * (dt / 1000.0))

    return poisson_train


def generate_poisson_spike_train(duration, rate):

    num_bins = int(duration)
    p_spike = rate / 1000  # Convert rate to probability per bin (dt in seconds)
    spike_train = np.random.rand(num_bins) < p_spike
    return spike_train.astype(int)


def generate_poisson_events(time, events):
    # Generate the number of events using a Poisson distribution
    num_events = np.random.poisson(events)
    # Ensure num_events does not exceed n
    num_events = min(num_events, time)

    spike_train = np.zeros(time)
    # Randomly select `num_events` timesteps to place the spikes
    if num_events > 0:
        spike_times = np.random.choice(time, num_events, replace=False)
        spike_train[spike_times] = 1

    return spike_train


def rate_coding_poisson(
    dataset, duration=100, max_rate=200, rest=None, coding_type="linear"
):
    num_samples, num_pixels = dataset.shape
    rest = rest if rest is not None else 0
    single_train_length = duration + rest
    total_length = num_samples * single_train_length
    spike_trains = np.zeros((num_pixels, total_length), dtype=int)

    for i in range(num_pixels):
        # Generate spike trains for each sample
        pixel_spike_trains = np.zeros((num_samples, single_train_length), dtype=int)
        for j in range(num_samples):
            pixel_intensity = dataset[j, i]
            if coding_type == "linear":
                converted_rate = pixel_intensity * max_rate
            elif coding_type == "exponential":
                # this parameter has been tuned for
                scale = 5
                exponential_values = np.exp(scale * pixel_intensity) - 1
                converted_rate = max_rate * exponential_values
            spike_train = generate_poisson_spike_train(duration, converted_rate)
            if rest > 0:
                spike_train = np.concatenate([spike_train, np.zeros(rest, dtype=int)])
            pixel_spike_trains[j, :] = spike_train

        # Flatten the spike trains into a single array for this pixel
        concatenated_spike_train = pixel_spike_trains.flatten()

        if concatenated_spike_train.shape[0] != total_length:
            print(
                f"Error: concatenated spike train length ({concatenated_spike_train.shape[0]}) does not match total_length ({total_length}) for pixel {i}"
            )

        spike_trains[i, :] = concatenated_spike_train

    return spike_trains


def rate_coding_constant(dataset, duration=100, rest=None):
    """
    Generates spike trains where the ISI is determined by 1/normalized pixel value.

    Args:
        dataset (numpy.ndarray): 2D array where each row is a sample and each column is a pixel.
        duration (int): Duration of the spike train for each sample.
        rest (int): Rest period after each spike train.

    Returns:
        numpy.ndarray: Generated spike trains.
    """
    num_samples, num_pixels = dataset.shape
    rest = rest if rest is not None else 0
    single_train_length = duration + rest
    total_length = num_samples * single_train_length
    spike_trains = np.zeros((num_pixels, total_length), dtype=int)

    for i in range(num_pixels):
        # Generate spike trains for each sample
        pixel_spike_trains = np.zeros((num_samples, single_train_length), dtype=int)
        for j in range(num_samples):
            pixel_intensity = dataset[j, i]
            if pixel_intensity > 0:
                isi = int(round(1.0 / pixel_intensity))
                if isi == 0:
                    isi = 1  # Ensure ISI is at least 1
            else:
                isi = duration + rest  # No spikes if pixel intensity is zero

            spike_train = np.zeros(duration, dtype=int)
            spike_times = np.arange(0, duration, isi)
            spike_train[spike_times] = 1

            if rest > 0:
                spike_train = np.concatenate([spike_train, np.zeros(rest, dtype=int)])

            pixel_spike_trains[j, :] = spike_train

        # Flatten the spike trains into a single array for this pixel
        concatenated_spike_train = pixel_spike_trains.flatten()

        if concatenated_spike_train.shape[0] != total_length:
            print(
                f"Error: concatenated spike train length ({concatenated_spike_train.shape[0]}) does not match total_length ({total_length}) for pixel {i}"
            )

        spike_trains[i, :] = concatenated_spike_train

    return spike_trains
