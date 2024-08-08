import matplotlib.pyplot as plt
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neurons.LIF import LIFNeuron

def test_adaptive_threshold():
    # Simulation parameters
    dt = 1
    time = np.arange(0, 1000, dt)
    burst_intervals = [(100, 200), (400, 500), (700, 800)]
    base_current = 200  # Base current applied during non-burst periods
    burst_current = 1000  # High current during burst periods
    input_current = np.full_like(time, base_current)

    for start, end in burst_intervals:
        input_current[int(start / dt) : int(end / dt)] = burst_current

    # Initialize neuron with adaptive threshold enabled
    neuron = LIFNeuron(ad_th=True)

    # Run the simulation
    spikes = []
    for t in range(len(time)):
        neuron.update_state(dt, input_current[t])
        spike = neuron.check_spike(dt)
        spikes.append(spike)

    # Plot membrane potential and firing rate
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot membrane potential
    axs[0].plot(time, neuron.v_trace, label="Membrane Potential")
    axs[0].set_ylabel("Membrane Potential (mV)")
    axs[0].legend(loc="upper right")

    # Mark burst onset times
    for start, _ in burst_intervals:
        axs[0].axvline(
            x=start,
            color="red",
            linestyle="--",
            label="Burst Onset" if start == burst_intervals[0][0] else "",
        )

    # Plot firing rate
    spike_train = np.array(spikes)
    window_size = int(100 / dt)  # 100 ms window
    firing_rate = np.convolve(spike_train, np.ones(window_size), "same") / window_size
    axs[1].plot(time, firing_rate, label="Firing Rate")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Firing Rate (Hz)")
    axs[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

# Run the test
test_adaptive_threshold()
