# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Simple oscillator models

classes:
  Oscillator
"""

from nsim import ItoModel
import numpy as np
from scipy import stats

class Oscillator(ItoModel):
    dimension = 2
    output_vars = [0]

    lam = -10.0
    omega = 2.0*np.pi*20.0
    sigma1 = 0.01 
    sigma2 = 0.01

    y0 = np.array([1.0, 1.0])

    def f(self, y, t):
        ret = np.zeros(2)
        ret[0] = self.lam*y[0] - self.omega*y[1]
        ret[1] = self.lam*y[1] + self.omega*y[0]
        return ret

    def G(self, y, t):
        # 2x2 matrix, with uncorrelated noise to the two variables
        return np.diag([self.sigma1, self.sigma2])


class Oscillator1D(ItoModel):
    epsilon = 0.6
    sigma = 0.03
    y0 = np.array([0.])

    def f(self, y, t):
        return 1 + self.epsilon*np.cos(y)

    def G(self, y, t):
        return np.array([[self.sigma]])
