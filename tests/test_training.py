import unittest
import numpy as np
from unittest.mock import MagicMock
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from network.utils import create_batches
from network.neuronal_coding import rate_coding_poisson, generate_inhibitory
from network.training import train_network

class TestTrainingProcedure(unittest.TestCase):
    
    def setUp(self):
        # Set up a mock network with necessary attributes
        self.network = MagicMock()
        self.network.dt = 1
        self.network.weights = np.random.rand(784, 10)
        self.network.w_min = 0
        self.network.w_max = 4
        self.network.A_plus = 0.01
        self.network.A_minus = 0.01
        self.network.tau_pos = 20
        self.network.tau_neg = 20
        self.network.gamma = 5
        self.network.neurons = [MagicMock() for _ in range(10)]
        for neuron in self.network.neurons:
            neuron.check_spike = MagicMock(return_value=False)
            neuron.update_state = MagicMock()
            neuron.v_reset = 0
            neuron.v_thresh = 1
            neuron.strength = 1

        self.data = np.random.rand(1000, 784)  # Smaller dataset for testing

    def test_create_batches(self):
        # Verify that batches are created correctly
        batch_size = 200
        batches = create_batches(self.data, batch_size=batch_size)
        self.assertEqual(len(batches), 5)
        for batch in batches:
            self.assertEqual(batch.shape, (batch_size, 784))

    def test_rate_coding_poisson(self):
        # Verify the spike train generation
        spike_train = rate_coding_poisson(self.data[:10], duration=100, max_rate=200, rest=20)
        self.assertEqual(spike_train.shape, (784, 1200))

    def test_train_network(self):
        # Test the overall training function
        neurons, weights = train_network(
            self.network,
            self.data,
            coding="Poisson",
            t_present=60,
            t_rest=20,
            max_rate=200,
            batch_size=500,
        )
        self.assertEqual(weights.shape, (784, 10))
        for neuron in neurons:
            self.assertIsInstance(neuron, MagicMock)
        
    def test_weight_updates(self):
        # Ensure weights are being updated
        initial_weights = np.copy(self.network.weights)
        train_network(
            self.network,
            self.data,
            coding="Constant",
            t_present=60,
            t_rest=20,
            max_rate=800,
            batch_size=500,
        )
        self.assertFalse(np.array_equal(self.network.weights, initial_weights))

    def test_neuron_state_updates(self):
        # Ensure neuron states are being updated
        train_network(
            self.network,
            self.data,
            coding="Poisson",
            t_present=60,
            t_rest=20,
            max_rate=200,
            batch_size=500,
        )
        for neuron in self.network.neurons:
            neuron.update_state.assert_called()

if __name__ == "__main__":
    unittest.main()
