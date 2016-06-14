# Copyright 2016 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""

functions:
"""

import numpy as np
import distob
from nsim.timeseries import Timeseries


def crossing_times(ts, c=0.0, d=0.0):
    """For a single variable timeseries, find the times at which the
    value crosses ``c`` from above or below. Can optionally set a non-zero
    ``d`` to impose the condition that the value must wander at least ``d`` 
    units away from ``c`` between crossings.

    If the timeseries begins (or ends) exactly at ``c``, then time zero 
    (or the ending time) is also included as a crossing event, 
    so that the boundaries of the first and last excursions are included.

    If the actual crossing time falls between two time steps, linear
    interpolation is used to estimate the crossing time.

    Args:
      ts: Timeseries (single variable)

      c (float): Critical value at which to report crossings.

      d (float): Optional min distance from c to be attained between crossings.

    Returns:
      array of float
    """
    #TODO support multivariate time series
    ts = ts.squeeze()
    if ts.ndim is not 1:
        raise ValueError('Currently can only use on single variable timeseries')

    # Translate to put the critical value at zero:
    ts = ts - c

    tsa = ts[0:-1]
    tsb = ts[1:]
    # Time indices where phase crosses or reaches zero from below or above
    zc = np.nonzero((tsa < 0) & (tsb >= 0) | (tsa > 0) & (tsb <= 0))[0] + 1
    # Estimate crossing time interpolated linearly within a single time step
    va = ts[zc-1]
    vb = ts[zc]
    ct = (np.abs(vb)*ts.tspan[zc-1] +
          np.abs(va)*ts.tspan[zc]) / np.abs(vb - va) # denominator always !=0
    # Also include starting time if we started exactly at zero
    if ts[0] == 0.0:
        zc = np.r_[np.array([0]), zc]
        ct = np.r_[np.array([ts.tspan[0]]), ct]

    if d == 0.0 or ct.shape[0] is 0:
        return ct

    # Time indices where value crosses c+d or c-d:
    dc = np.nonzero((tsa < d) & (tsb >= d) | (tsa > -d) & (tsb <= -d))[0] + 1
    # Select those zero-crossings separated by at least one d-crossing
    splice = np.searchsorted(dc, zc)
    which_zc = np.r_[np.array([0]), np.nonzero(splice[0:-1] - splice[1:])[0] +1]
    return ct[which_zc]


def first_return_times(ts, c=None, d=0.0):
    """For a single variable time series, first wait until the time series
    attains the value c for the first time. Then record the time intervals 
    between successive returns to c. If c is not given, the default is the mean
    of the time series.
    
    Args:
      ts: Timeseries (single variable)

      c (float): Optional target value (default is the mean of the time series)

      d (float): Optional min distance from c to be attained between returns

    Returns:
      array of time intervals (Can take the mean of these to estimate the
      expected first return time)
    """
    ts = np.squeeze(ts)
    if c is None:
        c = ts.mean()
    if ts.ndim <= 1:
        return np.diff(ts.crossing_times(c, d))
    else:
        return np.hstack(
            ts[..., i].first_return_times(c, d) for i in range(ts.shape[-1]))


def autocorrelation(ts, normalized=False, unbiased=False):
    """
    Returns the discrete, linear convolution of a time series with itself, 
    optionally using unbiased normalization. 

    N.B. Autocorrelation estimates are necessarily inaccurate for longer lags,
    as there are less pairs of points to convolve separated by that lag.
    Therefore best to throw out the results except for shorter lags, e.g. 
    keep lags from tau=0 up to one quarter of the total time series length.

    Args:
      normalized (boolean): If True, the time series will first be normalized
        to a mean of 0 and variance of 1. This gives autocorrelation 1 at
        zero lag.

      unbiased (boolean): If True, the result at each lag m will be scaled by
        1/(N-m). This gives an unbiased estimation of the autocorrelation of a
        stationary process from a finite length sample.

    Ref: S. J. Orfanidis (1996) "Optimum Signal Processing", 2nd Ed.
    """
    ts = np.squeeze(ts)
    if ts.ndim <= 1:
        if normalized:
            ts = (ts - ts.mean())/ts.std()
        N = ts.shape[0]
        ar = np.asarray(ts)
        acf = np.correlate(ar, ar, mode='full')
        outlen = (acf.shape[0] + 1) / 2
        acf = acf[(outlen - 1):]
        if unbiased:
            factor = np.array([1.0/(N - m) for m in range(0, outlen)])
            acf = acf * factor
        dt = (ts.tspan[-1] - ts.tspan[0]) / (len(ts) - 1.0)
        lags = np.arange(outlen)*dt
        return Timeseries(acf, tspan=lags, labels=ts.labels)
    else:
        # recursively handle arrays of dimension > 1
        lastaxis = ts.ndim - 1
        m = ts.shape[lastaxis]
        acfs = [ts[...,i].autocorrelation(normalized, unbiased)[...,np.newaxis]
                for i in range(m)]
        res = distob.concatenate(acfs, axis=lastaxis)
        res.labels[lastaxis] = ts.labels[lastaxis]
        return res
