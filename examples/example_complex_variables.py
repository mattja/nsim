"""example using an SDE model with complex variables

dz = (-z + j (delta + g cos(omega t)) z + j epsilon) dt + sigma exp(4jt) dW
"""

import nsim
import numpy as np


class Osc(nsim.SDEModel):
    delta = 2.0
    epsilon = 100.0
    sigma = 10.0
    g = 1.0
    omega = 2.0
    y0 = np.array([0.01 + 0.01j])

    def f(self, y, t):
        return (-1 + 1j*(self.delta + 
                         self.g*np.cos(self.omega*t))*y + self.epsilon*1j

    def G(self, y, t):
        return np.array([[self.sigma * np.exp(4j*t)]])
    

sims = nsim.RepeatedSim(Osc, T=1440.0, repeat=20)

ts = sims.timeseries

means = np.array([ts.mean(axis=0) for s in sims]).mean()
phases = (ts - means).angle()
phases.plot()
