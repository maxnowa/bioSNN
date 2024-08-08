import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from network.neuronal_coding import generate_inhibitory, generate_poisson_spike_train, generate_poisson_events, rate_coding_poisson

class TestSpikeGenerationFunctions(unittest.TestCase):
    
    def test_generate_inhibitory(self):
        rate = 10
        n = 100
        length = 1000
        poisson_train = generate_inhibitory(rate, n, length, myseed=42)
        
        # Check the shape of the output
        self.assertEqual(poisson_train.shape, (n, length))
        
        # Check that the output is binary
        self.assertTrue(np.all((poisson_train == 0) | (poisson_train == 1)))
        
        # Check the rate approximately
        self.assertAlmostEqual(np.mean(poisson_train), rate / 1000, delta=0.005)
    
    def test_generate_poisson_spike_train(self):
        duration = 100
        rate = 50
        spike_train = generate_poisson_spike_train(duration, rate)
        
        # Check the length of the output
        self.assertEqual(spike_train.shape[0], duration)
        
        # Check that the output is binary
        self.assertTrue(np.all((spike_train == 0) | (spike_train == 1)))
        
        # Check the rate approximately
        self.assertAlmostEqual(np.mean(spike_train), rate / 1000, delta=0.01)
    
    # def test_generate_poisson_events(self):
    #     time = 100
    #     events = 50
    #     spike_train = generate_poisson_events(time, events)
        
    #     # Check the length of the output
    #     self.assertEqual(spike_train.shape[0], time)
        
    #     # Check that the output is binary
    #     self.assertTrue(np.all((spike_train == 0) | (spike_train == 1)))
        
    #     # Check the number of events
    #     self.assertEqual(np.sum(spike_train), min(events, time))
    
    def test_rate_coding_poisson_linear(self):
        dataset = np.random.rand(10, 5)
        duration = 100
        max_rate = 200
        rest = 20
        spike_trains = rate_coding_poisson(dataset, duration, max_rate, rest, coding_type="linear")
        
        # Check the shape of the output
        self.assertEqual(spike_trains.shape, (5, 1200))
        
        # Check that the output is binary
        self.assertTrue(np.all((spike_trains == 0) | (spike_trains == 1)))
        
    def test_rate_coding_poisson_exponential(self):
        dataset = np.random.rand(10, 5)
        duration = 100
        max_rate = 200
        rest = 20
        spike_trains = rate_coding_poisson(dataset, duration, max_rate, rest, coding_type="exponential")
        
        # Check the shape of the output
        self.assertEqual(spike_trains.shape, (5, 1200))
        
        # Check that the output is binary
        self.assertTrue(np.all((spike_trains == 0) | (spike_trains == 1)))

if __name__ == "__main__":
    unittest.main()
