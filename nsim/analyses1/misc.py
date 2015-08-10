# Copyright 2015 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""

functions:
"""

import numpy as np


def crossing_indices(ts, c=0.0, d=0.0):
    """For a single variable timeseries, find the time indices each time the
    value crosses ``c`` from above or below. Can optionally set a non-zero
    ``d`` to impose the condition that the value must wander at least ``d`` 
    units away from ``c`` between crossings.

    If the timeseries begins (or ends) exactly at ``c``, then time zero 
    (or the ending time) is also included as a crossing event, 
    so that the boundaries of the first and last excursions are included.

    Args:
      ts: Timeseries (single variable)

      c (float): Critical value at which to report crossings.

      d (float): Optional min distance from c to be attained between crossings.

    Returns:
      array of indices
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

    # Also include index of initial point if we started exactly at zero
    if ts[0] == 0.0:
        zc = np.r_[np.array([0]), zc]

    if d == 0.0 or zc.shape[0] is 0:
        return zc

    # Time indices where value crosses c+d or c-d:
    dc = np.nonzero((tsa < d) & (tsb >= d) | (tsa > -d) & (tsb <= -d))[0] + 1
    # Select those zero-crossings separated by at least one d-crossing
    splice = np.searchsorted(dc, zc)
    which_zc = np.r_[np.array([0]), np.nonzero(splice[0:-1] - splice[1:])[0] +1]
    return zc[which_zc]


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
        return np.diff(ts.tspan[ts.crossing_indices(c, d)])
    else:
        return np.hstack(
            ts[..., i].first_return_times(c, d) for i in range(ts.shape[-1]))
