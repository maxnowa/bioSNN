import json
import numpy as np

class Network:
    def __init__(
        self, dt=1, ratio=1.05, A_plus=0.008, wmax=2, neg_ratio=2, gamma=1, tau_neg=20, tau_pos=20
    ) -> None:
        # simulation parameters
        self.dt = dt
        # STDP parameters
        self.tau_neg = tau_neg
        self.tau_pos = tau_pos
        # Ratio of A-/A+
        self.ratio = ratio
        self.A_plus = A_plus
        self.A_minus = round(self.A_plus * self.ratio, 7)
        self.gamma = gamma
        # weight parameters
        self.wmax = wmax
        self.wmin = 0
        self.neg_ratio = neg_ratio
        self.wneg = self.wmax * self.neg_ratio
