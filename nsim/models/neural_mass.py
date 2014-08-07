# Copyright 2014 Matthew J. Aburn
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

from nsim import SDEModel
import numpy as np
from scipy import stats


class JansenRit(SDEModel):
    """Jansen-Rit neural mass model of a small cortical region.

    By default, it simulates the model of Jansen and Rit (1995)
    
    It also implements the extended equations given by Moran et al (2007) 
    allowing input to both pyramidal cells and spiny stellate cells. 
    (If you set u_mean and u_sdev to nonzero values)

    The default settings below, when put into the Moran2007 model, 
    turn it into the simpler Jansen1995 model. That is the default.
    There is a small difference between the sigmoid functions of the two papers.
    Sm(v) =  Sj(v) - 0.03 s^-1, which is a negligible difference for v>0.
    The Jansen1995 sigmoid is used by default. 

    See also:
      Jansen, B. Rit, V. (1995) Electroencephalogram and visual evoked 
      potential generation in a mathematical model of coupled cortical columns

      Moran et al (2007) A neural mass model of spectral responses 
      in electrophysiology
    """
    dimension = 13 # the number of state variables

    # List of integers to indicate which variables are output of the model.
    # For this model, x[5] is usually the output (average pyramidal potential)
    output_vars = [5]

    # standard parameters for Jansen and Rit 1995
    rho1 = 0.56 # mV^-1
    rho2 = 6.0  # mV
    He = 3.25 # mV
    Hi = 22.0 # mV
    ke = 1/0.01 # == 100 s^-1
    ki = 1/0.02 # ==  50 s^-1
    ka = 0 # == 0 s^-1 (i.e. no change ever)
    ma = 0 # mV.s
    g = 135*5.0 # This takes into account extra factor of 2*e0 = 5.0 s^-1, 
                # which is missing in Moran2007.
    g1 = g
    g2 = 0.80*g
    g3 = 0.25*g
    g4 = 0.25*g
    g5 = 0

    u_sdev = 0 # s^-1  No noise input to Spiny cells in Jansen1995
    u_mean = 0 # s^-1
    p_sdev = 100.0/np.sqrt(3.0) # s^-1  Noise input to Pyramidal cells
    p_mean = 220.0 # s^-1
    # Now reduce noise level to correct for Jansen-Rit failure to scale noise, 
    # in order to get similar average effective noise variance to theirs.
    average_timestep_used_by_jr = 0.0012 
    # (the value above is a guess based on runs of our own RKF45 ODE integrator)
    p_sdev = p_sdev*np.sqrt(average_timestep_used_by_jr) 
    # i.e. reduce noise by a factor of about 29

    # initial conditions
    y0 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    #Jansen1995 sigmoid
    def S(self, y):
        return 1/(1 + np.exp(-self.rho1 * (y - self.rho2)))

    ## alternatively, Moran2007 sigmoid
    #def S(self, y):
    #    return (1/(1 + np.exp(-self.rho1 * (y - self.rho2))) - 
    #            1/(1 + np.exp(self.rho1 * self.rho2))); 

    def f(self, x, t):
        """Moran2007 equations right hand side, noise free term
        Args: 
          x: (13,) array 
             state vector
          t: number
             scalar time
        Returns:
          (13,) array

        Dynamical variables x[n], n=0..6 correspond to v_1 to v_7 in Moran2007.
        In particular, x[5] corresponds to v_6, the average pyramidal potential.
        x[n], n=7..11 correspond to i_1 to i_5 in Moran2007.
        x[12] corresponds to 'a' in Moran2007.
        """
        ret = np.zeros(13)
        ret[3]  = x[10];
        ret[10] = (self.ke*self.He*self.g3*self.S(x[5]) - 2*self.ke*x[10] - 
                   self.ke*self.ke*x[3]);

        ret[4]  = x[11];
        ret[11] = (self.ki*self.Hi*self.g5*self.S(x[6]) - 2*self.ki*x[11] - 
                   self.ki*self.ki*x[4]);
        
        ret[6]  = x[10] - x[11];

        ret[0]  = x[7];
        ret[7]  = (self.ke*self.He*(self.g1*self.S(x[5]) + self.u_mean) - 
                   2*self.ke*x[7] - self.ke*self.ke*x[0]);

        ret[1]  = x[8];
        ret[8]  = (self.ke*self.He*(self.g2*self.S(x[0]-x[12]) + self.p_mean) -
                   2*self.ke*x[8] - self.ke*self.ke*x[1]);

        ret[2]  = x[9];
        ret[9]  = (self.ki*self.Hi*self.g4*self.S(x[6]) - 2*self.ki*x[9] - 
                   self.ki*self.ki*x[2]);

        ret[5]  = x[8] - x[9];

        ret[12] = self.ka*(self.ma*self.S(x[0] - x[12]) - x[12]);
        return ret

    def G(self, x, t):
        """Moran2007 equations right hand side, noise term
        Args: 
          x: (13,) array 
             state vector
          t: number
             scalar time
        Returns:
          (13,13) array
          Only one matrix column is non-zero, meaning that in this example
          we are modelling the noise input to pyramidal and spiny populations 
          as fully correlated.
          To simulate uncorrelated inputs instead, change [8,0] to [8,1].
        """
        ret = np.zeros((13, 13))
        ret[7,0] = self.ke * self.He * self.u_sdev
        ret[8,0] = self.ke * self.He * self.p_sdev
        return ret

