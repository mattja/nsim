# Copyright 2016 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Various plotting routines for time series

functions:
  plot
  phase_histogram
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numbers

def _get_color_list():
    """Get cycle of colors in a way compatible with all matplotlib versions"""
    if 'axes.prop_cycle' in plt.rcParams:
        return [p['color'] for p in list(plt.rcParams['axes.prop_cycle'])]
    else:
        return plt.rcParams['axes.color_cycle']


def plot(ts, title=None, show=True):
    """Plot a Timeseries
    Args: 
      ts  Timeseries
      title  str
      show  bool whether to display the figure or just return a figure object  
      """
    ts = _remove_pi_crossings(ts)
    fig = plt.figure()
    ylabelprops = dict(rotation=0, 
                       horizontalalignment='right', 
                       verticalalignment='center', 
                       x=-0.01)
    if ts.ndim > 2: # multiple sim timeseries. collapse vars onto each subplot.
        num_subplots = ts.shape[ts.ndim - 1]
        if title is None:
            title = u'time series at each node'
        for i in range(num_subplots):
            ax = fig.add_subplot(num_subplots, 1, i+1)
            ax.plot(ts.tspan, ts[...,i])
            if ts.labels[-1] is not None:
                ax.set_ylabel(ts.labels[-1][i], **ylabelprops)
            else:
                ax.set_ylabel('node ' + str(i), **ylabelprops)
            plt.setp(ax.get_xticklabels(), visible=False)
        fig.axes[0].set_title(title)
        plt.setp(fig.axes[num_subplots-1].get_xticklabels(), visible=True)
        fig.axes[num_subplots-1].set_xlabel('time (s)')
    else: # single sim timeseries. show each variable separately.
        if ts.ndim is 1:
            ts = ts.reshape((-1, 1))
        num_ax = ts.shape[1]
        if title is None:
            title=u'time series'
        axprops = dict()
        if num_ax > 10:
            axprops['yticks'] = []
        colors = _get_color_list()
        for i in range(num_ax):
            rect = 0.1, 0.85*(num_ax - i - 1)/num_ax + 0.1, 0.8, 0.85/num_ax
            ax = fig.add_axes(rect, **axprops)
            ax.plot(ts.tspan, ts[...,i], color=colors[i % len(colors)])
            plt.setp(ax.get_xticklabels(), visible=False)
            if ts.labels[1] is not None:
                ax.set_ylabel(ts.labels[1][i], **ylabelprops)
        fig.axes[0].set_title(title)
        plt.setp(fig.axes[num_ax-1].get_xticklabels(), visible=True)
        fig.axes[num_ax-1].set_xlabel('time (s)')
    if show:
        fig.show()
    return fig


def _remove_pi_crossings(ts):
    """For each variable in the Timeseries, checks whether it represents
    a phase variable ranging from -pi to pi. If so, set all points where the
    phase crosses pi to 'nan' so that spurious lines will not be plotted.

    If ts does not need adjustment, then return ts. 
    Otherwise return a modified copy.
    """
    orig_ts = ts
    if ts.ndim is 1:
        ts = ts[:, np.newaxis, np.newaxis]
    elif ts.ndim is 2:
        ts = ts[:, np.newaxis]
    # Get the indices of those variables that have range of approx -pi to pi
    tsmax = ts.max(axis=0)
    tsmin = ts.min(axis=0)
    phase_vars = np.transpose(np.nonzero((np.abs(tsmax - np.pi) < 0.01) & 
                                         (np.abs(tsmin + np.pi) < 0.01)))
    if len(phase_vars) is 0:
        return orig_ts
    else:
        ts = ts.copy()
        for v in phase_vars:
            ts1 = np.asarray(ts[:, v[0], v[1]]) # time series of single variable
            ts1a = ts1[0:-1]
            ts1b = ts1[1:]
            p2 = np.pi/2
            # Find time indices where phase crosses pi. Set those values to nan.
            pc = np.nonzero((ts1a > p2) & (ts1b < -p2) | 
                            (ts1a < -p2) & (ts1b > p2))[0] + 1
            ts1[pc] = np.nan
            ts[:, v[0], v[1]] = ts1
        return ts


def phase_histogram(ts, times=None, nbins=30, colormap=mpl.cm.Blues):
    """Plot a polar histogram of a phase variable's probability distribution
    Args:
      ts: Timeseries with axis 2 ranging over separate instances of an
        oscillator (time series values are assumed to represent an angle)
      times (float or sequence of floats): The target times at which 
        to plot the distribution
      nbins (int): number of histogram bins
      colormap
    """
    if times is None:
        times = np.linspace(ts.tspan[0], ts.tspan[-1], num=4)
    elif isinstance(times, numbers.Number):
        times = np.array([times], dtype=np.float64)
    indices = ts.tspan.searchsorted(times)
    if indices[-1] == len(ts.tspan):
        indices[-1] -= 1
    nplots = len(indices)
    fig = plt.figure()
    n = np.zeros((nbins, nplots))
    for i in xrange(nplots):
        index = indices[i]
        time = ts.tspan[index]
        phases = ts.mod2pi()[index, 0, :]
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
