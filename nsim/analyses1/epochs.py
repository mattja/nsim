# Copyright 2016 Matthew J. Aburn
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
from nsim.analyses1._cwtmorlet import cwtmorlet
from nsim.timeseries import Timeseries
from scipy import signal
import numpy as np


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
    if freqs is None:
        freqs = np.logspace(np.log10(1.0), np.log10(60.0), 50)
    else:
        freqs = np.array(freqs)
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    channels = ts.shape[1]
    n = len(ts)
    dt = (1.0*ts.tspan[-1] - ts.tspan[0]) / (n - 1)
    fs = 1.0 / dt
    dtype = ts.dtype
    # Estimate time-resolved power spectra using continuous wavelet transform
    coefs = ts.cwt(freqs, wavelet=cwtmorlet, plot=False)
    # this is a huge array so try to do operations in place
    powers = np.square(np.abs(coefs, coefs), coefs).real.astype(dtype, 
                                                                copy=False)
    del coefs
    max_power = np.max(powers, axis=1)
    total_power = np.sum(powers, axis=1, keepdims=True)
    rel_power = np.divide(powers, total_power, powers)
    del powers
    centroid_freq = np.tensordot(freqs, rel_power, axes=(0, 1))  # shape (n, m)
    del rel_power
    # hw is half window size (in number of samples)
    hw = np.int64(np.ceil(0.5 * ncycles * fs / centroid_freq))  # shape (n, m)
    allchannels_variability = np.zeros((n, channels, 2), dtype) # output array
    for i in range(channels):
        logvar_centfreq = np.zeros(n, dtype)
        logvar_maxpower = np.zeros(n, dtype)
        for j in range(n):
            # compute variance of two chosen signal properties over a 
            # window of 2*hw+1 samples centered on sample number j
            wstart = j - hw[j, i]
            wend = j + hw[j, i]
            if wstart >= 0 and wend < n:
                logvar_centfreq[j] = np.log(centroid_freq[wstart:wend+1].var())
                logvar_maxpower[j] = np.log(max_power[wstart:wend+1].var())
            else:
                logvar_centfreq[j] = np.nan
                logvar_maxpower[j] = np.nan
        allchannels_variability[:, i, 0] = _rescale(logvar_centfreq)
        allchannels_variability[:, i, 1] = _rescale(logvar_maxpower)
    allchannels_variability = Timeseries(allchannels_variability, 
                                         ts.tspan, labels=ts.labels)
    if plot:
        _plot_variability(ts, allchannels_variability)
    return allchannels_variability


def _rescale(ar):
    """Shift and rescale array ar to the interval [-1, 1]"""
    max = np.nanmax(ar)
    min = np.nanmin(ar)
    midpoint = (max + min) / 2.0
    return 2.0 * (ar - midpoint) / (max - min)


def _get_color_list():
    """Get cycle of colors in a way compatible with all matplotlib versions"""
    if 'axes.prop_cycle' in plt.rcParams:
        return [p['color'] for p in list(plt.rcParams['axes.prop_cycle'])]
    else:
        return plt.rcParams['axes.color_cycle']


def _plot_variability(ts, variability, threshold=None, epochs=None):
    """Plot the timeseries and variability. Optionally plot epochs."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if variability.ndim is 1:
        variability = variability[:, np.newaxis, np.newaxis]
    elif variability.ndim is 2:
        variability = variability[:, np.newaxis, :]
    vmeasures = variability.shape[2]
    channels = ts.shape[1]
    dt = (1.0*ts.tspan[-1] - ts.tspan[0]) / (len(ts) - 1)
    fig = plt.figure()
    ylabelprops = dict(rotation=0, 
                       horizontalalignment='right', 
                       verticalalignment='center', 
                       x=-0.01)
    for i in range(channels):
        rect = (0.1, 0.85*(channels - i - 1)/channels + 0.1, 
                0.8, 0.85/channels)
        axprops = dict()
        if channels > 10:
            axprops['yticks'] = []
        ax = fig.add_axes(rect, **axprops)
        ax.plot(ts.tspan, ts[:, i])
        if ts.labels[1] is None:
            ax.set_ylabel(u'channel %d' % i, **ylabelprops)
        else:
            ax.set_ylabel(ts.labels[1][i], **ylabelprops)
        plt.setp(ax.get_xticklabels(), visible=False)
        if i is channels - 1:
            plt.setp(ax.get_xticklabels(), visible=True)
            ax.set_xlabel('time (s)')
        ax2 = ax.twinx()
        if vmeasures > 1:
            mean_v = np.nanmean(variability[:, i, :], axis=1)
            ax2.plot(ts.tspan, mean_v, color='g')
            colors = _get_color_list()
            for j in range(vmeasures):
                ax2.plot(ts.tspan, variability[:, i, j], linestyle='dotted',
                         color=colors[(3 + j) % len(colors)])
            if i is 0:
                ax2.legend(['variability (mean)'] + 
                          ['variability %d' % j for j in range(vmeasures)], 
                          loc='best')
        else:
            ax2.plot(ts.tspan, variability[:, i, 0])
            ax2.legend(('variability',), loc='best')
        if threshold is not None:
            ax2.axhline(y=threshold, color='Gray', linestyle='dashed')
        ax2.set_ylabel('variability')
        ymin = np.nanmin(ts[:, i])
        ymax = np.nanmax(ts[:, i])
        tstart = ts.tspan[0]
        if epochs:
            # highlight epochs using rectangular patches
            for e in epochs[i]:
                t1 = tstart + (e[0] - 1) * dt
                ax.add_patch(mpl.patches.Rectangle(
                    (t1, ymin), (e[1] - e[0])*dt, ymax - ymin, alpha=0.2,
                    color='green', ec='none'))
    fig.axes[0].set_title(u'variability (threshold = %g)' % threshold)
    fig.show()


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
    if variability is None:
        variability = ts.variability_fp(plot=False)
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    if variability.ndim is 1:
        variability = variability[:, np.newaxis, np.newaxis]
    elif variability.ndim is 2:
        variability = variability[:, np.newaxis, :]
    channels = ts.shape[1]
    n = len(ts)
    dt = (1.0*ts.tspan[-1] - ts.tspan[0]) / (n - 1)
    fs = 1.0 / dt
    allchannels_epochs = []
    for i in range(channels):
        v = variability[:, i, :]
        v = np.nanmean(v, axis=1) # mean of q different variability measures
        # then smooth the variability with a low-pass filter
        nonnan_ix = np.nonzero(~np.isnan(v))[0]
        nonnans = slice(nonnan_ix.min(), nonnan_ix.max())
        crit_freq = 1.0 # Hz
        b, a = signal.butter(3, 2.0 * crit_freq / fs)
        #v[nonnans] = signal.filtfilt(b, a, v[nonnans])
        v[nonnan_ix] = signal.filtfilt(b, a, v[nonnan_ix])
        # find all local minima of the variability not exceeding the threshold
        m = v[1:-1]
        l = v[0:-2]
        r = v[2:]
        minima = np.nonzero(~np.isnan(m) & ~np.isnan(l) & ~np.isnan(r) &
                            (m <= threshold) & (m-l < 0) & (r-m > 0))[0] + 1
        if len(minima) is 0:
            print(u'Channel %d: no epochs found using threshold %g' % (
                i, threshold))
            allchannels_epochs.append([])
        else:
            # Sort the list of minima by ascending variability
            minima = minima[np.argsort(v[minima])]
            epochs = []
            for m in minima:
                # Check this minimum is not inside an existing epoch
                overlap = False
                for e in epochs:
                    if m >= e[0] and m <= e[1]:
                        overlap = True
                        break
                if not overlap:
                    # Get largest subthreshold interval surrounding the minimum
                    startix = m - 1
                    endix = m + 1
                    for startix in range(m - 1, 0, -1):
                        if np.isnan(v[startix]) or v[startix] > threshold:
                            startix += 1
                            break
                    for endix in range(m + 1, len(v), 1):
                        if np.isnan(v[endix]) or v[endix] > threshold:
                            break
                    if (endix - startix) * dt >= minlength: 
                        epochs.append((startix, endix))
            allchannels_epochs.append(epochs)
    if plot:
        _plot_variability(ts, variability, threshold, allchannels_epochs)
    if orig_ndim is 1:
        allchannels_epochs = allchannels_epochs[0]
    return (variability, allchannels_epochs)


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
    import distob
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    if variability is None:
        dts = distob.scatter(ts, axis=1)
        vepochs = distob.vectorize(epochs)
        results = vepochs(dts, None, threshold, minlength, plot=False)
    else: 
        def f(pair):
            return epochs(pair[0], pair[1], threshold, minlength, plot=False)
        allpairs = [(ts[:, i], variability[:, i]) for i in range(ts.shape[1])]
        vf = distob.vectorize(f)
        results = vf(allpairs)
    vars, allchannels_epochs = zip(*results)
    variability = distob.hstack(vars)
    if plot:
        _plot_variability(ts, variability, threshold, allchannels_epochs)
    return (variability, allchannels_epochs)


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
    variability, allchannels_epochs = ts.epochs_distributed(
            variability, threshold, minlength, plot=False)
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
        allchannels_epochs = [allchannels_epochs]
        variability = variability[:, np.newaxis]
    channels = ts.shape[1]
    dt = (1.0*ts.tspan[-1] - ts.tspan[0]) / (len(ts) - 1)
    starts = [(e[0], 1) for channel in allchannels_epochs for e in channel]
    ends = [(e[1], -1) for channel in allchannels_epochs for e in channel]
    all = sorted(starts + ends)
    joint_epochs = []
    in_joint_epoch = False
    joint_start = 0.0
    inside_count = 0
    for bound in all:
        inside_count += bound[1]
        if not in_joint_epoch and 1.0*inside_count/channels >= proportion:
            in_joint_epoch = True
            joint_start = bound[0]
        if in_joint_epoch and 1.0*inside_count/channels < proportion:
            in_joint_epoch = False
            joint_end = bound[0]
            if (joint_end - joint_start)*dt >= minlength:
                joint_epochs.append((joint_start, joint_end))
    if plot:
        joint_epochs_repeated = [joint_epochs] * channels
        _plot_variability(ts, variability, threshold, joint_epochs_repeated)
    return (variability, joint_epochs)
