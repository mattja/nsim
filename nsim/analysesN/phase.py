# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Various phase analyses that apply to an ensemble of many oscillators
"""
import distob
from nsim import Timeseries
import numpy as np
from scipy import stats


def periods_all(dts, phi=0.0):
    """For an ensemble of oscillators, return the set of periods lengths of 
    all successive oscillations of all oscillators.

    An individual oscillation is defined to start and end when the phase 
    passes phi (by default zero) after completing a full cycle.

    If the timeseries of an oscillator phase begins (or ends) exactly at phi, 
    then the first (or last) oscillation will be included.

    Arguments:
      dts (DistTimeseries): where dts.shape[1] is 1 (single output variable
        representing phase) and axis 2 ranges over multiple realizations of
        the oscillator.

      phi=0.0: float
          A single oscillation starts and ends at phase phi (by default zero).
    """
    periods = dts.periods(phi)
    if hasattr(type(periods), '__array_interface__'):
        return np.ravel(periods)
    else:
        return np.hstack([distob.gather(plist) for plist in dts.periods()])


def phase_mean(dts):
    interval = dts.periods_all().mean()
    snapshots = dts.t[0.0::interval]
    snapshots = distob.gather(snapshots.mod2pi())
    array = stats.circmean(snapshots, high=np.pi, low=-np.pi, axis=2)
    return Timeseries(array, snapshots.tspan)


def phase_std(dts):
    interval = dts.periods_all().mean()
    snapshots = dts.t[0.0::interval]
    snapshots = distob.gather(snapshots.mod2pi())
    array = stats.circstd(snapshots, high=np.pi, low=-np.pi, axis=2)
    return Timeseries(array, snapshots.tspan)
