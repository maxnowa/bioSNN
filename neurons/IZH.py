## File for describing the neuron of Izhikevich type

class IZHNeuron():

    def __init__(self, v_init=-70, u_init=-10, v_thresh=30, parameters=((0.02, 0.2, -65.0, 8), "Regular Spiking")):
        self.a,self.b,self.c,self.d = parameters[0]
        self.type = parameters[1]
        self.v = v_init
        self.u = u_init
        self.v_thresh = v_thresh

        self.v_trace = []
        self.rec_spikes = []

    def update_state(self, timestep, dt, I):
        dv = (0.04*self.v**2) + (5*self.v) + 140 - self.u + I
        du = self.a*(self.b*self.v - self.u)

        self.v += dv * dt
        self.u += du * dt
        self.v_trace.append(self.v)


    def check_spike(self, timestep):
        if self.v >= self.v_thresh:
            self.v = self.c
            self.u = self.u + self.d
            self.rec_spikes.append(timestep)

    