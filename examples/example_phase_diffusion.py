"""A nonlinear 1D phase oscillator.
dphi = (1 + epsilon cos(phi)) dt + sigma dW

Plots some figures showing how the circular standard deviation of the phase 
does not necessarily increase monotonically.

N.B. this one uses a fair bit of CPU to compute 1024 sample paths in parallel.
"""

import nsim
import numpy as np

class Oscillator1D(nsim.SDEModel):
    epsilon = 0.6
    sigma = 0.03
    y0 = 0.0

    def f(self, y, t):
        return 1 + self.epsilon*np.cos(y)

    def G(self, y, t):
        return self.sigma


sims = nsim.RepeatedSim(Oscillator1D, T=2400.0, repeat=1024)

sims[0].output.mod2pi().t[0:50].plot() # show example 50 secs of oscillations

print('mean period is %g s' % sims.periods().mean())

mean_series = sims.phase_mean()
mean_series.plot(title='circular mean')

diffusion_series = sims.phase_std()
diffusion_series.plot(title='circular std')

sims.phase_histogram([20, 998, 1500, 2400], nbins=50)
