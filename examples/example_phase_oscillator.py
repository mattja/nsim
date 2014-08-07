"""Nonlinear 1D phase oscillator.

dphi = (1 + epsilon cos(phi)) dt + sigma dW
"""

import nsim
import numpy as np


class Oscillator1D(nsim.SDEModel):
    epsilon = 0.6
    sigma = 0.03
    y0 = np.array([0.])

    def f(self, y, t):
        return 1 + self.epsilon*np.cos(y)

    def G(self, y, t):
        return np.array([[self.sigma]])


sims = nsim.RepeatedSim(Oscillator1D, T=3600.0, repeat=256)


print('mean period is %f s' % sims.periods().mean())


mean_series = sims.phase_mean()
mean_series.plot()

diffusion_series = sims.phase_std()
diffusion_series.plot()

sims.phase_histogram([20, 998, 1500, 3200], nbins=60)
