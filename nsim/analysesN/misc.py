# Copyright 2015 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Miscelaneous analyses that apply to an ensemble of many time series
"""
import distob
from nsim import analyses1
import numpy as np


def first_return_times(dts, c=None, d=0.0):
    """For an ensemble of time series, return the set of all time intervals
    between successive returns to value c for all instances in the ensemble.
    If c is not given, the default is the mean across all times and across all
    time series in the ensemble.

    Args:
      dts (DistTimeseries)

      c (float): Optional target value (default is the ensemble mean value)

      d (float): Optional min distance from c to be attained between returns

    Returns:
      array of time intervals (Can take the mean of these to estimate the
      expected first return time for the whole ensemble)
    """
    if c is None:
        c = dts.mean()
    vmrt = distob.vectorize(analyses1.first_return_times)
    all_intervals = vmrt(dts, c, d)
    if hasattr(type(all_intervals), '__array_interface__'):
        return np.ravel(all_intervals)
    else:
        return np.hstack([distob.gather(ilist) for ilist in all_intervals])
