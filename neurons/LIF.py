import numpy as np


class LIFNeuron():

    def __init__(self, v_th=-55, v_res=-65, v_leak=-70, v_init=-65, tau_m=20, t_ref=2, error=False, ad_th=False, gamma=0.1) -> None:
        self.v_thresh = v_th
        self.v_reset = v_res
        self.v_leak = v_leak
        self.v_init = v_init
        self.tau_m = tau_m
        self.t_ref = t_ref
        # gamma is determined according to R * spike_strength, normal case its between 0.1 and 10
        self.gamma = gamma
        self.v = 0
        self.v_trace = []
        self.rec_spikes = []
        self.tr = 0

        # parameters for noise
        self.error = error
        # Code for adaptive threshold
        self.ad_th = ad_th
        # value to track adaptive threshold
        self.theta_th = 0
        # value to be added each time neuron spikes
        self.delta_theta = 6
        # time constant for decay of adaptive threshold
        self.theta_tau = 6

        if ad_th:
            # TODO find good value for this 
            self.theta_th = 0
        # just for experimentation, saves the threshold at which the neuron fires
        self.sp_thresh = []

        
    def update_state(self, dt, input_current):
        epsilon = 0
        if self.error:
            # random noise can lead to the neuron randomly spiking TODO: investigate suitable parameters
            epsilon = np.random.normal(0, 1)
        dv = (-(self.v - self.v_leak) + self.gamma * (input_current + epsilon)) * dt/self.tau_m
        self.v = self.v + dv
        # calculate decay of adaptive threshold
        if self.ad_th:
            self.theta_th -= self.theta_th * dt / self.theta_tau
        # keep track of mebrane potential
        self.v_trace.append(self.v)

    
    def check_spike(self, dt, neuron_ind=None, WTA=False):
        # neuron is in refractory period
        if self.tr > 0:
            self.v = self.v_reset
            self.tr -= 1
        # membrane potential is ovet threshhold -> spike occurs
        elif WTA in ["Hard", "Soft"]:
            if self.v >= self.v_thresh + self.theta_th and neuron_ind is None:
                self.rec_spikes.append(1)
                self.v = self.v_reset
                self.tr = self.t_ref / dt
                if self.ad_th:
                    self.theta_th += self.delta_theta
                return True
        else:
            if self.v >= self.v_thresh + self.theta_th:
                # append spike times
                self.rec_spikes.append(1)
                self.sp_thresh.append(self.v_thresh + self.theta_th)
                self.v = self.v_reset
                # set refractory period counter
                self.tr = self.t_ref / dt
                if self.ad_th:
                    self.theta_th += self.delta_theta
                return True
        self.rec_spikes.append(0)
        return False