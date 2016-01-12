"""A nonlinear 1D phase oscillator.
dphi = (1 + epsilon cos(phi)) dt + sigma dW

Plots some figures showing how the circular standard deviation of the phase 
does not necessarily increase monotonically.

N.B. this one uses a fair bit of CPU to compute 256 sample paths in parallel.
"""

import nsim
import numpy as np

class Oscillator1D(nsim.StratonovichModel):
    epsilon = 0.6
    sigma = 0.03
    y0 = 0.0

    def f(self, y, t):
        return 1 + self.epsilon*np.cos(y)

    def G(self, y, t):
        return self.sigma


sims = nsim.RepeatedSim(Oscillator1D, T=2400.0, repeat=256)

first = sims[0].output # timeseries output of first realization: shape 480000x1
first.t[100:150].mod2pi().plot(title='50 secs output of first realization')

ts = sims.output  # timeseries output of all 256 realizations: 480000 x 1 x 256

p = ts.periods().mean()
print('mean period is %g s' % p)

snaps = ts.t[0.0::p] # Discrete time series, taking snapshot each mean period

mean_series = snaps.circmean()
mean_series.plot(title='circular mean')

diffusion_series = snaps.circstd()
diffusion_series.plot(title='circular std')

snaps.phase_histogram([20, 998, 1500, 2400], nbins=30)
