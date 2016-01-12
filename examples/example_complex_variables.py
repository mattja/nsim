"""Example of using a model defined with a single complex variable.

dz = (-z + j (delta + g cos(omega t)) z + j epsilon) dt + sigma exp(4jt) dW

With parameters below the system has noise driven oscillations around a stable
fixed point.

Using complex vector variables is ok, too. (numpy arrays of complex128)
nsim will look at y0 to determine what type of variable is being used.
"""

import nsim
import numpy as np
import matplotlib.pyplot as plt


class Osc(nsim.StratonovichModel):
    delta = 2.0
    epsilon = 100.0
    sigma = 10.0
    g = 1.0
    omega = 2.0
    y0 = 0.0 + 0.0j

    def f(self, y, t):
        return (-y + 1j*(self.delta + self.g*np.cos(self.omega*t))*y +
                1j*self.epsilon)

    def G(self, y, t):
        return self.sigma * np.exp(4j*t)
    

sims = nsim.RepeatedSim(Osc, T=1440.0, repeat=10)
ts = sims.timeseries
means = ts.mean(axis=0)

phases = (ts - means).angle()

phases.plot(title='phase at each node')
phases[:,:,3].t[100:160].plot(title='phase') # show 60 seconds of node 3

print('mean period is %g seconds' % phases.periods().mean())

r = (ts - means).abs()
r.plot(title='amplitude at each node')

# show distribution of mean reversion times
rtimes = r.first_return_times(r.mean())
plt.figure()
plt.hist(rtimes, bins=150, range=(0.05, 2.9))
plt.title('Distribution of mean reversion times for amplitude')
plt.show(block=False)

print('\nmean reversion time is %g seconds' % rtimes.mean())
