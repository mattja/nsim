"""two-dimensional Ornstein-Uhlenbeck process     dy = A y dt + B dW
"""

import nsim
import numpy as np


class OU2D(nsim.ItoModel):
    A = np.array([[-0.5, -2.0],
                  [ 2.0, -1.0]])
    B = np.diag([0.5, 0.5])

    y0 = np.array([0.3, 0.3])

    def f(self, y, t):
        return self.A.dot(y)

    def G(self, y, t):
        return self.B


sims = nsim.RepeatedSim(OU2D, T=120.0, dt=0.005, repeat=4)

# Timeseries of a single simulation has shape 24001 x 2 (model has 2 variables)
sims[0].timeseries.plot(title='plot timeseries of simulation 0 only')

# Timeseries for all 4 simulations has shape  24001 x 2 x 4
ts = sims.timeseries  

# Plotting a distributed timeseries: can downsample to plot only N time points
ts.plot(points=1000, title='timeseries of all 4 repetitions, rough view')

# Get a phase from the x and y variables. resulting array has shape  24001 x 4
phase = np.arctan2(ts[:,0,:], ts[:,1,:])   

# If you have the latest numpy (>=1.10.0) then arctan2 will have been computed
# in parallel on the distributed inputs and `phase` will be a distributed array

# If you have an older numpy,  arctan2 will have done the computation locally,
# and `phase` will be a local array.

print('\nIdentify individual periods:\n%s' % phase.periods())

print('\nmean period is %g' % phase.periods().mean())

phase.plot(title='phases for each simulation')
