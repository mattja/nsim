"""two-dimensional Ornstein-Uhlenbeck process     dy = A y dt + B dW
"""

import nsim
import numpy as np


class OU2D(nsim.SDEModel):
    A = np.array([[-0.5, -2.0],
                  [ 2.0, -1.0]])
    B = np.diag([0.5, 0.5])

    y0 = np.array([0.3, 0.3])

    def f(self, y, t):
        return self.A.dot(y)

    def G(self, y, t):
        return self.B


sims = nsim.RepeatedSim(OU2D, T=120.0, repeat=4)
sims.timeseries.plot()

ts = sims[0].timeseries
ts.plot()
phase = np.arctan2(ts[:, 0], ts[:, 1])
phase.plot(title='phase')
print('mean period is %g' % phase.periods().mean())
