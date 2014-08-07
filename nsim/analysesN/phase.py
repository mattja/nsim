# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Various phase analyses that apply to an ensemble of many oscillators

(This interface will be changed after the distributed array class
 is implemented)
"""

# TODO refactor code so these apply to 'distributed timeseries' rather than Sims

from nsim import Timeseries
import numpy as np
from scipy import stats


def periods(msim, phi=0.0):
    """For an ensemble of oscillators, return the set of periods lengths of 
    all successive oscillations of all oscillators.

    An individual oscillation is defined to start and end when the phase 
    passes phi (by default zero) after completing a full cycle.

    If the timeseries of an oscillator phase begins (or ends) exactly at phi, 
    then the first (or last) oscillation will be included.

    Arguments: 
      sim MultipleSimulation 
          with each simulation having single output variable representing phase

      phi=0.0: float 
          A single oscillation starts and ends at phase phi (by default zero).
    """
    return np.hstack((sim.output.periods(phi) for sim in msim.sims))


def snapshots(msim, start, interval):
    snaps = [sim.output.t[start::interval] for sim in msim.sims]
    ndim = snaps[0].ndim
    if ndim <= 1:
        array = np.dstack(tuple(ts[:,np.newaxis, np.newaxis] for ts in snaps))
    else:
        array = np.concatenate(tuple(ts[...,np.newaxis] for ts in snaps), 
                               axis=ndim)
    return Timeseries(array, snaps[0].tspan._ob)


def phase_mean(msim):
    snaps = msim.snapshots(0.0, msim.periods().mean()).mod2pi()
    array = stats.circmean(snaps, high=np.pi, low=-np.pi, axis=2)
    return Timeseries(array, snaps.tspan)


def phase_std(msim):
    snaps = msim.snapshots(0.0, msim.periods().mean()).mod2pi()
    array = stats.circstd(snaps, high=np.pi, low=-np.pi, axis=2)
    return Timeseries(array, snaps.tspan)
