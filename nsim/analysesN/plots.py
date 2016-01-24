# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Various plotting functions for a distributed timeseries
"""
import distob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numbers


def plot(dts, title=None, points=None, show=True):
    """Plot a distributed timeseries
    Args: 
      dts (DistTimeseries)
      title (str, optional)
      points (int, optional): Limit the number of time points plotted. 
        If specified, will downsample to use this total number of time points, 
        and only fetch back the necessary points to the client for plotting.
    Returns: 
      fig
    """
    if points is not None and len(dts.tspan) > points:
        # then downsample  (TODO: use interpolation)
        ix = np.linspace(0, len(dts.tspan) - 1, points).astype(np.int64)
        dts = dts[ix, ...]
    ts = distob.gather(dts)
    return ts.plot(title, show)


def phase_histogram(dts, times=None, nbins=30, colormap=mpl.cm.Blues):
    """Plot a polar histogram of a phase variable's probability distribution
    Args:
      dts: DistTimeseries with axis 2 ranging over separate instances of an
        oscillator (time series values are assumed to represent an angle)
      times (float or sequence of floats): The target times at which 
        to plot the distribution
      nbins (int): number of histogram bins
      colormap
    """
    if times is None:
        times = np.linspace(dts.tspan[0], dts.tspan[-1], num=4)
    elif isinstance(times, numbers.Number):
        times = np.array([times], dtype=np.float64)
    indices = distob.gather(dts.tspan.searchsorted(times))
    if indices[-1] == len(dts.tspan):
        indices[-1] -= 1
    nplots = len(indices)
    fig = plt.figure()
    n = np.zeros((nbins, nplots))
    for i in xrange(nplots):
        index = indices[i]
        time = dts.tspan[index]
        phases = distob.gather(dts.mod2pi()[index, 0, :])
        ax = fig.add_subplot(1, nplots, i + 1, projection='polar')
        n[:,i], bins, patches = ax.hist(phases, nbins, (-np.pi, np.pi), 
                                        normed=True, histtype='bar')
        ax.set_title('time = %d s' % time)
        ax.set_xticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', 
                            r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{-3\pi}{4}$',
                            r'$\frac{-\pi}{2}$', r'$\frac{-\pi}{4}$'])
    nmin, nmax = n.min(), n.max()
    #TODO should make a custom colormap instead of reducing color dynamic range:
    norm = mpl.colors.Normalize(1.2*nmin - 0.2*nmax, 
                                0.6*nmin + 0.4*nmax, clip=True)
    for i in xrange(nplots):
        ax = fig.get_axes()[i]
        ax.set_ylim(0, nmax)
        for this_n, thispatch in zip(n[:,i], ax.patches):
            color = colormap(norm(this_n))
            thispatch.set_facecolor(color)
            thispatch.set_edgecolor(color)
    fig.show()
