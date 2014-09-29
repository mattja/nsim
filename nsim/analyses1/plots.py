# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Various plotting routines for time series

functions:
  plot
"""

import numpy as np
import matplotlib.pyplot as plt


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
        for i in range(num_ax):
            rect = 0.1, 0.85*(num_ax - i - 1)/num_ax + 0.1, 0.8, 0.85/num_ax
            ax = fig.add_axes(rect, **axprops)
            # use i'th color in cycle
            _ = [next(ax._get_lines.color_cycle) for j in range(i)]
            ax.plot(ts.tspan, ts[...,i])
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
