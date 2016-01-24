"""Kuramoto network: N simple phase oscillators coupled together.

This example also shows how a model parameter can be a probability distribution
instead of a number. Then each time an instance is created from the model class
recipe it will be given a specific value drawn from the distribution. 
"""

import nsim
import numpy as np
from scipy import stats

class PhaseOscillator(nsim.ODEModel):
    """A simple oscillator. It just has a phase y that increases at a
    constant angular velocity omega.  dy/dt = omega """
    # each oscillator's natural frequency is drawn from a distribution:
    omega = stats.norm(20.0, 0.1)
    # each oscillator's initial phase is chosen at random:
    y0 = stats.uniform(-np.pi, scale=2*np.pi)

    def f(self, y, t):
        return self.omega

    def coupling(self, source_y, target_y, weight):
        return weight * np.sin(source_y - target_y)


# now couple together N of those oscillators:
N = 100
nodes = [PhaseOscillator() for i in range(N)]

# network is all-to-all weak coupling:
coupling_strength = 0.24
network = (np.ones((N, N)) - np.identity(N)) * coupling_strength / N

model = nsim.NetworkModel(nodes, network)
sim = nsim.Simulation(model, T=60.0)
ts = sim.timeseries.mod2pi() # interpret y as a phase variable (mod 2pi)
ts[:,:,53:58].plot(title='viewing five of the %d oscillators' % N)
ts.order_param().plot(title='phase synchronization order parameter')
ts.phase_histogram()
