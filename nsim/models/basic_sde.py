# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Simple Stochastic Differential Equation models

classes:
  OU   1D Ornstein Uhlenbeck model
"""

from nsim import ItoModel
import numpy as np
from scipy import stats


class OU(ItoModel):

    dimension = 1
    output_vars = [0]

    lam = -1.0
    sigma = 0.8
    y0 = np.array([0.])

    def f(self, y, t):
        return self.lam * y

    def G(self, y, t):
        return np.array([[self.sigma]])



