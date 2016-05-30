# Copyright 2015 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Large scale population models for neuroscience. 

classes:
  JansenRit
"""

from nsim import StratonovichModel
import numpy as np
from scipy import stats


class JansenRit(StratonovichModel):
    """Jansen-Rit neural mass model of a small cortical region.

    By default, it simulates the model of Jansen and Rit (1995)
    
    It also implements the extended equations given by Aburn et al. (2012)
    allowing input to both pyramidal cells and spiny stellate cells.
    (If you set u_mean and u_sdev to nonzero values)

    See also:
      Jansen, B. Rit, V. (1995) Electroencephalogram and visual evoked 
      potential generation in a mathematical model of coupled cortical columns

      Aburn et al. (2012) Critical fluctuations in cortical models near
      instability
    """
    dimension = 8 # the number of state variables

    labels = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    # List of integers to indicate which variables are output of the model. For
    # this model, v[1]-v[2] is usually the output (average pyramidal potential)
    output_vars = [1, 2]
    input_vars = [5]

    # standard parameters for Jansen and Rit 1995
    rho1 = 0.56 # mV^-1
    rho2 = 6.0  # mV
    e0 = 2.5 # s^-1
    He1 = 3.25 # mV
    He2 = 3.25 # mV
    He3 = 3.25 # mV
    Hi = 22.0 # mV
    ke1 = 1/0.01 # == 100 s^-1
    ke2 = 1/0.01 # == 100 s^-1
    ke3 = 1/0.01 # == 100 s^-1
    ki = 1/0.02 # ==  50 s^-1
    g1 = 135.0
    g2 = 0.80*g1
    g3 = 0.25*g1
    g4 = 0.25*g1

    u_sdev = 0.0 # s^-1  No noise input to Spiny cells in Jansen1995
    u_mean = 0.0 # s^-1
    p_sdev = 100.0/np.sqrt(3.0) # s^-1  Noise input to Pyramidal cells
    p_mean = 220.0 # s^-1
    # Now reduce noise level to correct for Jansen-Rit failure to scale noise, 
    # in order to get similar average effective noise variance to theirs.
    average_timestep_used_by_jr = 0.0012 
    # (the value above is a guess based on runs of our own RKF45 ODE integrator)
    p_sdev = p_sdev*np.sqrt(average_timestep_used_by_jr) 
    # i.e. reduce noise by a factor of about 29

    # default initial conditions: near stable equilibrium
    y0 = np.array([12.214, 23.925, 16.841, 3.0534, 13.564, -11.803, -109.62,
                    3.3909])

    def S(self, y):
        return (2.0*self.e0)/(1.0 + np.exp(self.rho1*(self.rho2 - y)))

    def f(self, v, t):
        """Aburn2012 equations right hand side, noise free term
        Args: 
          v: (8,) array 
             state vector
          t: number
             scalar time
        Returns:
          (8,) array
        """
        ret = np.zeros(8)
        ret[0] = v[4]
        ret[4] = (self.He1*self.ke1*(self.g1*self.S(v[1]-v[2]) + self.u_mean) -
                  2*self.ke1*v[4] - self.ke1*self.ke1*v[0])

        ret[1] = v[5]
        ret[5] = (self.He2*self.ke2*(self.g2*self.S(v[0]) + self.p_mean) -
                  2*self.ke2*v[5] - self.ke2*self.ke2*v[1])

        ret[2] = v[6]
        ret[6] = (self.Hi*self.ki*self.g4*self.S(v[3]) - 2*self.ki*v[6] -
                  self.ki*self.ki*v[2])

        ret[3] = v[7]
        ret[7] = (self.He3*self.ke3*self.g3*self.S(v[1]-v[2]) -
                  2*self.ke3*v[7] - self.ke3*self.ke3*v[3])
        return ret

    def G(self, v, t):
        """Aburn2012 equations right hand side, noise term
        Args: 
          v: (8,) array 
             state vector
          t: number
             scalar time
        Returns:
          (8,1) array
          Only one matrix column, meaning that in this example we are modelling
          the noise input to pyramidal and spiny populations as fully 
          correlated.  To simulate uncorrelated inputs instead, use an array of
          shape (8, 2) with the second noise element [5,1] instead of [5,0].
        """
        ret = np.zeros((8, 1))
        ret[4,0] = self.ke1 * self.He1 * self.u_sdev
        ret[5,0] = self.ke2 * self.He2 * self.p_sdev
        return ret

    def coupling(self, source_y, target_y, weight):
        """How to couple the output of one node to the input of another.
        Args:
          source_y (array of shape (8,)): state of the source node
          target_y (array of shape (8,)): state of the target node
          weight (float): the connection strength
        Returns:
          input (array of shape (8,)): value to drive each variable of the
            target node.
        """
        v_pyramidal = source_y[1] - source_y[2]
        return (np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) *
                (weight*self.g1*self.He2*self.ke2*self.S(v_pyramidal)))
