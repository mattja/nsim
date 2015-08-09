# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Various analyses that apply to the phase time-series of an oscillator 

functions:
"""

import numpy as np


def mod2pi(ts):
    """For a timeseries where all variables represent phases (in radians),
    return an equivalent timeseries where all values are in the range (-pi, pi]
    """
    return np.pi - np.mod(np.pi - ts, 2*np.pi)


def phase_crossings(ts, phi=0.0):
    """For a single variable timeseries representing the phase of an oscillator,
    find the time indices each time the phase crosses angle phi,
    with the condition that the phase must visit phi+pi between crossings.

    (Thus if noise causes the phase to wander back and forth across angle phi 
    without the oscillator doing a full revolution, then this is recorded as 
    a single crossing event, giving the index of the earliest arrival.)

    If the timeseries begins (or ends) exactly at phi, then time zero 
    (or the ending time) is also included as a crossing event, 
    so that the boundaries of the first and last oscillations are included.

    Arguments:
      ts: Timeseries (single variable)
          The timeseries of an angle variable (radians)

      phi (float): Critical phase angle (radians) at which to report crossings.

    Returns:
      array of indices
    """
    #TODO support multivariate time series
    ts = ts.squeeze()
    if ts.ndim is not 1:
        raise ValueError('Currently can only use on single variable timeseries')

    # Interpret the timeseries as belonging to a phase variable. 
    # Map its range to the interval (-pi, pi] with critical angle at zero:
    ts = mod2pi(ts - phi)

    tsa = ts[0:-1]
    tsb = ts[1:]
    p2 = np.pi/2
    # Time indices where phase crosses or reaches zero from below or above
    zc = np.nonzero((tsa > -p2) & (tsa < 0) & (tsb >= 0) & (tsb <  p2) | 
                    (tsa <  p2) & (tsa > 0) & (tsb <= 0) & (tsb > -p2))[0] + 1
    # Also include index of initial point if we started exactly at zero
    if ts[0] == 0.0:
        zc = np.r_[np.array([0]), zc]
    # Time indices where phase crosses pi
    pc = np.nonzero((tsa > p2) & (tsb < -p2) | (tsa < -p2) & (tsb > p2))[0] + 1

    # Select those zero-crossings separated by at least one pi-crossing
    splice = np.searchsorted(pc, zc)
    which_zc = np.r_[np.array([0]), np.nonzero(splice[0:-1] - splice[1:])[0] +1]
    if zc.shape[0] is 0:
        return zc
    else:
        return zc[which_zc]
    

def periods(ts, phi=0.0):
    """For a single variable timeseries representing the phase of an oscillator,
    measure the period of each successive oscillation.

    An individual oscillation is defined to start and end when the phase 
    passes phi (by default zero) after completing a full cycle.

    If the timeseries begins (or ends) exactly at phi, then the first
    (or last) oscillation will be included.

    Arguments: 
      ts: Timeseries (single variable)
          The timeseries of an angle variable (radians)

      phi (float): A single oscillation starts and ends at phase phi (by 
        default zero).
    """
    ts = np.squeeze(ts)
    if ts.ndim <= 1:
        return np.diff(ts.tspan[phase_crossings(ts, phi)])
    else:
        return np.hstack(ts[..., i].periods(phi) for i in range(ts.shape[-1]))


def circmean(ts, axis=2):
    """Circular mean phase"""
    return np.exp(1.0j * ts).mean(axis=axis).angle()


def circstd(ts, axis=2):
    """Circular standard deviation"""
    res = np.exp(1.0j * ts).mean(axis=axis)
    R = np.abs(res)
    return np.sqrt(-2.0 * np.log(R))
