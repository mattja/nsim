"""1D Ornstein-Uhlenbeck processes
dy = lam y + sigma dW
"""

import nsim
import numpy as np


class OU(nsim.SDEModel):
    lam = -1.0
    sigma = 0.8
    y0 = np.array([0.])

    def f(self, y, t):
        return self.lam * y

    def G(self, y, t):
        return np.array([[self.sigma]])


sims = nsim.RepeatedSim(OU, T=60.0, repeat=20

sims.plot()
