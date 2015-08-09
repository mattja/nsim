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


def mean_reversion_times(dts, d=0.0):
    """For an ensemble of time series, return the set of all time intervals
    between successive mean reversions of all instances in the ensemble.

    Args:
      dts (DistTimeseries)

      d (float): Optional min distance from mean to be attained between returns
    """
    vmrt = distob.vectorize(analyses1.mean_reversion_times)
    all_intervals = vmrt(dts, d)
    if hasattr(type(all_intervals), '__array_interface__'):
        return np.ravel(all_intervals)
    else:
        return np.hstack([distob.gather(ilist) for ilist in all_intervals])
