# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Functions to identify pseudo-stationary epochs, where a timeseries has 
low 'variability'. There is flexibility to give your own function to define 
'variability' in the way you want, or you can use the default.

An example of a 'variability' function is given based on Blenkinsop2012, 
using centroid frequency and height of spectral peak to define 'variability'.

functions:
  epochs   Find pseudo-stationary epochs, based on any defn of 'variability'.
  epochs_distributed  Same as above, but compute all channels in parallel.
  epochs_joint  Find epochs where x% of channels are simultaneously stationary.
  variability_fp  Example 2D variability function, extending Blenkinsop2012.

See also: Blenkinsop et al. (2012) The dynamic evolution of focal-onset 
          epilepsies - combining theoretical and clinical observations
"""
import distob
from nsim import analyses1


def variability_fp(ts, freqs=None, ncycles=6, plot=True):
    """Example variability function.
    Gives two continuous, time-resolved measures of the variability of a
    time series, ranging between -1 and 1. 
    The two measures are based on variance of the centroid frequency and 
    variance of the height of the spectral peak, respectively.
    (Centroid frequency meaning the power-weighted average frequency)
    These measures are calculated over sliding time windows of variable size.
    See also: Blenkinsop et al. (2012) The dynamic evolution of focal-onset 
              epilepsies - combining theoretical and clinical observations
    Args:
      ts  Timeseries of m variables, shape (n, m). Assumed constant timestep.
      freqs   (optional) List of frequencies to examine. If None, defaults to
              50 frequency bands ranging 1Hz to 60Hz, logarithmically spaced.
      ncycles  Window size, in number of cycles of the centroid frequency.
      plot  bool  Whether to display the output

    Returns:
      variability   Timeseries of shape (n, m, 2)  
                    variability[:, :, 0] gives a measure of variability 
                    between -1 and 1 based on variance of centroid frequency.
                    variability[:, :, 1] gives a measure of variability 
                    between -1 and 1 based on variance of maximum power.
    """
    if ts.ndim <= 2:
        return analyses1.variability_fp(ts, freqs, ncycles, plot)
    else:
        return distob.vectorize(analyses1.variability_fp)(
                ts, freqs, ncycles, plot)


def epochs(ts, variability=None, threshold=0.0, minlength=1.0, plot=True):
    """Identify "stationary" epochs within a time series, based on a 
    continuous measure of variability.
    Epochs are defined to contain the points of minimal variability, and to 
    extend as wide as possible with variability not exceeding the threshold.

    Args:
      ts  Timeseries of m variables, shape (n, m). 
      variability  (optional) Timeseries of shape (n, m, q),  giving q scalar 
                   measures of the variability of timeseries `ts` near each 
                   point in time. (if None, we will use variability_fp())
                   Epochs require the mean of these to be below the threshold.
      threshold   The maximum variability permitted in stationary epochs.
      minlength   Shortest acceptable epoch length (in seconds)
      plot  bool  Whether to display the output

    Returns: (variability, allchannels_epochs) 
      variability: as above
      allchannels_epochs: (list of) list of tuples
      For each variable, a list of tuples (start, end) that give the 
      starting and ending indices of stationary epochs.
      (epochs are inclusive of start point but not the end point)
    """
    if ts.ndim <= 2:
        return analyses1.epochs_distributed(
                ts, variability, threshold, minlength, plot)
    else:
        return distob.vectorize(analyses1.epochs)(
                ts, variability, threshold, minlength, plot)


def epochs_distributed(ts, variability=None, threshold=0.0, minlength=1.0, 
                       plot=True):
    """Same as `epochs()`, but computes channels in parallel for speed.

    (Note: This requires an IPython cluster to be started first, 
           e.g. on a workstation type 'ipcluster start')

    Identify "stationary" epochs within a time series, based on a 
    continuous measure of variability.
    Epochs are defined to contain the points of minimal variability, and to 
    extend as wide as possible with variability not exceeding the threshold.

    Args:
      ts  Timeseries of m variables, shape (n, m). 
      variability  (optional) Timeseries of shape (n, m, q),  giving q scalar 
                   measures of the variability of timeseries `ts` near each 
                   point in time. (if None, we will use variability_fp())
                   Epochs require the mean of these to be below the threshold.
      threshold   The maximum variability permitted in stationary epochs.
      minlength   Shortest acceptable epoch length (in seconds)
      plot  bool  Whether to display the output

    Returns: (variability, allchannels_epochs) 
      variability: as above
      allchannels_epochs: (list of) list of tuples
      For each variable, a list of tuples (start, end) that give the 
      starting and ending indices of stationary epochs.
      (epochs are inclusive of start point but not the end point)
    """
    if ts.ndim <= 2:
        return analyses1.epochs_distributed(
                ts, variability, threshold, minlength, plot)
    else:
        return distob.vectorize(analyses1.epochs)(
                ts, variability, threshold, minlength, plot)


def epochs_joint(ts, variability=None, threshold=0.0, minlength=1.0,
                 proportion=0.75, plot=True):
    """Identify epochs within a multivariate time series where at least a 
    certain proportion of channels are "stationary", based on a previously 
    computed variability measure.

    (Note: This requires an IPython cluster to be started first, 
     e.g. on a workstation type 'ipcluster start')

    Args:
      ts  Timeseries of m variables, shape (n, m). 
      variability  (optional) Timeseries of shape (n, m),  giving a scalar 
                   measure of the variability of timeseries `ts` near each 
                   point in time. (if None, we will use variability_fp())
      threshold   The maximum variability permitted in stationary epochs.
      minlength   Shortest acceptable epoch length (in seconds)
      proportion  Require at least this fraction of channels to be "stationary"
      plot  bool  Whether to display the output

    Returns: (variability, joint_epochs)
      joint_epochs: list of tuples
      A list of tuples (start, end) that give the starting and ending indices 
      of time epochs that are stationary for at least `proportion` of channels.
      (epochs are inclusive of start point but not the end point)
    """
    if ts.ndim <= 2:
        return analyses1.epochs_joint(
                ts, variability, threshold, minlength, plot)
    else:
        return distob.vectorize(analyses1.epochs_joint)(
                ts, variability, threshold, minlength, plot)
