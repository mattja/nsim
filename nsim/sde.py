# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""Provides functions to integrate systems of stochastic differential equations

sodeint  integrates Stratonovich SDE systems using the Heun algorithm
"""

import numpy as np
import numbers


class Error(Exception):
    pass


class SDEValueError(Error):
    """Thrown if integration arguments fail some basic sanity checks"""
    pass


def sodeint(f, G, y0, tspan):
    """Integrate an (ordinary) stochastic differential equation
    dy = f(y,t)dt + G(y,t).dW(t)  (which must be given in Stratonovich form) 

    where y is the n-dimensional state vector, f is a vector-valued function, 
    G is an nxm matrix-valued function giving the noise coefficients and 
    dW(t) = (dW_1, dW_2, ... dW_m) is a vector of independent Weiner increments.

    Integration is currently done using the Heun algorithm. 

    Args:
      f: callable(y, t) returning (n,) array
         Vector-valued function to define the deterministic part of the system 
      G: callable(y, t) returning (n,m) array
         Matrix-valued function to define the noise coefficients of the system
      y0: array of shape (n,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y. 
        These must be equally spaced, e.g. np.arange(0,10,0.005) 
        tspan[0] is the intial time corresponding to the initial state y0.

    Returns:
      y: array, with shape (len(tspan), len(y0)) 
         With the initial value y0 in the first row

    Raises:
      SDEValueError

    See also: 
      R. Mannella (2002) Integration of Stochastic Differential Equations 
         on a Computer
      W. Rumelin (1982) Numerical Treatment of Stochastic Differential Equations
    """
    # do some validation
    if not np.isclose(min(np.diff(tspan)), max(np.diff(tspan))):
        raise SDEValueError('Time steps must be equally spaced.')
    # be flexible to allow scalar equations. convert them to a 1D vector system
    if isinstance(y0, numbers.Number):
        if isinstance(y0, numbers.Integral):
            numtype = np.float64
        else:
            numtype = type(y0)
        y0_orig = y0
        y0 = np.array([y0], dtype=numtype)
        def make_vector_fn(fn):
            def newfn(y, t):
                return np.array([fn(y[0], t)], dtype=numtype)
            newfn.__name__ = fn.__name__
            return newfn
        def make_matrix_fn(fn):
            def newfn(y, t):
                return np.array([[fn(y[0], t)]], dtype=numtype)
            newfn.__name__ = fn.__name__
            return newfn
        if isinstance(f(y0_orig, 0.0), numbers.Number):
            f = make_vector_fn(f)
        if isinstance(G(y0_orig, 0.0), numbers.Number):
            G = make_matrix_fn(G)
    # dimension of system
    n = len(y0)
    if len(f(y0, tspan[0])) != n or len(G(y0, tspan[0])) != n:
        raise SDEValueError('y0, f and G have incompatible shapes.')
    # number of independent noise processes m (may be different from n)
    m = G(y0, tspan[0]).shape[1]
    # preallocate space for result
    y = np.zeros((len(tspan), len(y0)), dtype=type(y0[0]))
    y[0] = y0;
    for i in range(1, len(tspan)):
        t1 = tspan[i - 1]
        t2 = tspan[i]
        dt = t2 - t1
        y1 = y[i - 1]
        # Vector of m independent Weiner increments (variance scales with time) 
        dW = np.random.normal(0.0, np.sqrt(dt), m)
        ybar = y1 + f(y1, t1)*dt + G(y1, t1).dot(dW)
        y[i] = (y1 + 0.5*(f(y1, t1) + f(ybar, t2))*dt +
                0.5*(G(y1, t1) + G(ybar, t2)).dot(dW))
    return y
