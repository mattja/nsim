# Copyright 2016 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Various frequency domain analyses that apply to a single time series.

functions:
  psd
  lowpass
  highpass
  bandpass
  notch
  hilbert
  hilbert_amplitude
  hilbert_phase
  epochs
  cwt
  cwt_distributed
"""
from __future__ import absolute_import
from ._cwtmorlet import cwtmorlet, roughcwt
from nsim.timeseries import Timeseries
from scipy import signal
import numpy as np


def psd(ts, nperseg=1500, noverlap=1200, plot=True):
    """plot Welch estimate of power spectral density, using nperseg samples per
    segment, with noverlap samples overlap and Hamming window."""
    ts = ts.squeeze()
    if ts.ndim is 1:
        ts = ts.reshape((-1, 1))
    fs = (len(ts) - 1.0) / (ts.tspan[-1] - ts.tspan[0])
    window = signal.hamming(nperseg, sym=False)
    nfft = max(256, 2**np.int(np.log2(nperseg) + 1))
    freqs, pxx = signal.welch(ts, fs, window, nperseg, noverlap, nfft,
                              detrend='linear', axis=0)
    # Discard estimates for freq bins that are too low for the window size.
    # (require two full cycles to fit within the window)
    index = np.nonzero(freqs >= 2.0*fs/nperseg)[0][0]
    if index > 0:
        freqs = freqs[index:]
        pxx = pxx[index:]
    # Discard estimate for last freq bin as too high for Nyquist frequency:
    freqs = freqs[:-1]
    pxx = pxx[:-1]
    if plot is True:
        _plot_psd(ts, freqs, pxx)
    return freqs, pxx


def _plot_psd(ts, freqs, pxx):
    import matplotlib.pyplot as plt
    num_subplots = ts.shape[ts.ndim - 1]
    multinode = (ts.ndim > 2 and ts.shape[ts.ndim - 1] > 1)
    fig = plt.figure()
    ylabelprops = dict(rotation=0,
                       horizontalalignment='right',
                       verticalalignment='center',
                       x=-0.01)
    for i in range(num_subplots):
        ax = fig.add_subplot(num_subplots, 1, i+1)
        #ax.set_xscale('log')
        ax.set_xscale('linear')
        # If timeseries is identically zero, use linear scaling
        if np.count_nonzero(pxx[...,i]) is 0:
            ax.set_yscale('linear')
        else:
            ax.set_yscale('log')
        ax.plot(freqs, pxx[...,i])
        if multinode is True:
            if ts.labels[2] is not None:
                ax.set_ylabel(ts.labels[2][i], **ylabelprops)
            else:
                ax.set_ylabel(u'node %d' % i, **ylabelprops)
        elif num_subplots > 1:
            if ts.labels[1] is not None:
                ax.set_ylabel(ts.labels[1][i], **ylabelprops)
            else:
                ax.set_ylabel(u'channel %d' % i, **ylabelprops)
        else:
            pass
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.grid()
    fig.axes[0].set_title('Welch power spectrum estimate')
    plt.setp(fig.axes[num_subplots-1].get_xticklabels(), visible=True)
    fig.axes[num_subplots-1].set_xlabel('frequency (Hz)')
    fig.show()


def lowpass(ts, cutoff_hz, order=3):
    """forward-backward butterworth low-pass filter"""
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    channels = ts.shape[1]
    fs = (len(ts) - 1.0) / (ts.tspan[-1] - ts.tspan[0])
    nyq = 0.5 * fs
    cutoff = cutoff_hz/nyq
    b, a = signal.butter(order, cutoff, btype='low')
    if not np.all(np.abs(np.roots(a)) < 1.0):
        raise ValueError('Filter will not be stable with these values.')
    dtype = ts.dtype
    output = np.zeros((len(ts), channels), dtype)
    for i in range(channels):
        output[:, i] = signal.filtfilt(b, a, ts[:, i])
    if orig_ndim is 1:
        output = output[:, 0]
    return Timeseries(output, ts.tspan, labels=ts.labels)


def highpass(ts, cutoff_hz, order=3):
    """forward-backward butterworth high-pass filter"""
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    channels = ts.shape[1]
    fs = (len(ts) - 1.0) / (ts.tspan[-1] - ts.tspan[0])
    nyq = 0.5 * fs
    cutoff = cutoff_hz/nyq
    b, a = signal.butter(order, cutoff, btype='highpass')
    if not np.all(np.abs(np.roots(a)) < 1.0):
        raise ValueError('Filter will not be stable with these values.')
    dtype = ts.dtype
    output = np.zeros((len(ts), channels), dtype)
    for i in range(channels):
        output[:, i] = signal.filtfilt(b, a, ts[:, i])
    if orig_ndim is 1:
        output = output[:, 0]
    return Timeseries(output, ts.tspan, labels=ts.labels)


def bandpass(ts, low_hz, high_hz, order=3):
    """forward-backward butterworth band-pass filter"""
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    channels = ts.shape[1]
    fs = (len(ts) - 1.0) / (ts.tspan[-1] - ts.tspan[0])
    nyq = 0.5 * fs
    low = low_hz/nyq
    high = high_hz/nyq
    b, a = signal.butter(order, [low, high], btype='band')
    if not np.all(np.abs(np.roots(a)) < 1.0):
        raise ValueError('Filter will not be stable with these values.')
    dtype = ts.dtype
    output = np.zeros((len(ts), channels), dtype)
    for i in range(channels):
        output[:, i] = signal.filtfilt(b, a, ts[:, i])
    if orig_ndim is 1:
        output = output[:, 0]
    return Timeseries(output, ts.tspan, labels=ts.labels)


def notch(ts, freq_hz, bandwidth_hz=1.0):
    """notch filter to remove remove a particular frequency
    Adapted from code by Sturla Molden
    """
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    channels = ts.shape[1]
    fs = (len(ts) - 1.0) / (ts.tspan[-1] - ts.tspan[0])
    nyq = 0.5 * fs
    freq = freq_hz/nyq
    bandwidth = bandwidth_hz/nyq
    R = 1.0 - 3.0*(bandwidth/2.0)
    K = ((1.0 - 2.0*R*np.cos(np.pi*freq) + R**2) /
         (2.0 - 2.0*np.cos(np.pi*freq)))
    b, a = np.zeros(3), np.zeros(3)
    a[0] = 1.0
    a[1] = -2.0*R*np.cos(np.pi*freq)
    a[2] = R**2
    b[0] = K
    b[1] = -2*K*np.cos(np.pi*freq)
    b[2] = K
    if not np.all(np.abs(np.roots(a)) < 1.0):
        raise ValueError('Filter will not be stable with these values.')
    dtype = ts.dtype
    output = np.zeros((len(ts), channels), dtype)
    for i in range(channels):
        output[:, i] = signal.filtfilt(b, a, ts[:, i])
    if orig_ndim is 1:
        output = output[:, 0]
    return Timeseries(output, ts.tspan, labels=ts.labels)


def hilbert(ts):
    """Analytic signal, using the Hilbert transform"""
    output = signal.hilbert(signal.detrend(ts, axis=0), axis=0)
    return Timeseries(output, ts.tspan, labels=ts.labels)


def hilbert_amplitude(ts):
    """Amplitude of the analytic signal, using the Hilbert transform"""
    output = np.abs(signal.hilbert(signal.detrend(ts, axis=0), axis=0))
    return Timeseries(output, ts.tspan, labels=ts.labels)


def hilbert_phase(ts):
    """Phase of the analytic signal, using the Hilbert transform"""
    output = np.angle(signal.hilbert(signal.detrend(ts, axis=0), axis=0))
    return Timeseries(output, ts.tspan, labels=ts.labels)


def cwt(ts, freqs=np.logspace(0, 2), wavelet=cwtmorlet, plot=True):
    """Continuous wavelet transform
    Note the full results can use a huge amount of memory at 64-bit precision

    Args:
      ts: Timeseries of m variables, shape (n, m). Assumed constant timestep.
      freqs: list of frequencies (in Hz) to use for the tranform. 
        (default is 50 frequency bins logarithmic from 1Hz to 100Hz)
      wavelet: the wavelet to use. may be complex. see scipy.signal.wavelets
      plot: whether to plot time-resolved power spectrum

    Returns: 
      coefs: Continuous wavelet transform output array, shape (n,len(freqs),m)
    """
    orig_ndim = ts.ndim
    if ts.ndim is 1:
        ts = ts[:, np.newaxis]
    channels = ts.shape[1]
    fs = (len(ts) - 1.0) / (1.0*ts.tspan[-1] - ts.tspan[0])
    x = signal.detrend(ts, axis=0)
    dtype = wavelet(fs/freqs[0], fs/freqs[0]).dtype
    coefs = np.zeros((len(ts), len(freqs), channels), dtype)
    for i in range(channels):
        coefs[:, :, i] = roughcwt(x[:, i], cwtmorlet, fs/freqs).T
    if plot:
        _plot_cwt(ts, coefs, freqs)
    if orig_ndim is 1:
        coefs = coefs[:, :, 0]
    return coefs


def cwt_distributed(ts, freqs=np.logspace(0, 2), wavelet=cwtmorlet, plot=True):
    """Continuous wavelet transform using distributed computation.
    (Currently just splits the data by channel. TODO split it further.)
    Note: this function requires an IPython cluster to be started first.

    Args:
      ts: Timeseries of m variables, shape (n, m). Assumed constant timestep.
      freqs: list of frequencies (in Hz) to use for the tranform. 
        (default is 50 frequency bins logarithmic from 1Hz to 100Hz)
      wavelet: the wavelet to use. may be complex. see scipy.signal.wavelets
      plot: whether to plot time-resolved power spectrum

    Returns: 
      coefs: Continuous wavelet transform output array, shape (n,len(freqs),m)
    """
    if ts.ndim is 1 or ts.shape[1] is 1:
        return cwt(ts, freqs, wavelet, plot)
    import distob
    vcwt = distob.vectorize(cwt)
    coefs = vcwt(ts, freqs, wavelet, plot=False)
    if plot:
        _plot_cwt(ts, coefs, freqs)
    return coefs


def _plot_cwt(ts, coefs, freqs, tsize=1024, fsize=512):
    """Plot time resolved power spectral density from cwt results
    Args:
      ts: the original Timeseries
      coefs:  continuous wavelet transform coefficients as calculated by cwt()
      freqs: list of frequencies (in Hz) corresponding to coefs.
      tsize, fsize: size of the plot (time axis and frequency axis, in pixels)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy import interpolate
    channels = ts.shape[1]
    fig = plt.figure()
    for i in range(channels):
        rect = (0.1, 0.85*(channels - i - 1)/channels + 0.1, 
                0.8, 0.85/channels)
        ax = fig.add_axes(rect)
        logpowers = np.log((coefs[:, :, i] * coefs[:, :, i].conj()).real)
        tmin, tmax = ts.tspan[0], ts.tspan[-1]
        fmin, fmax = freqs[0], freqs[-1]
        tgrid, fgrid = np.mgrid[tmin:tmax:tsize*1j, fmin:fmax:fsize*1j]
        gd = interpolate.interpn((ts.tspan, freqs), logpowers, 
                                 (tgrid, fgrid)).T
        ax.imshow(gd, cmap='gnuplot2', aspect='auto', origin='lower',
                   extent=(tmin, tmax, fmin, fmax))
        ax.set_ylabel('freq (Hz)')
    fig.axes[0].set_title(u'log(power spectral density)')
    fig.axes[channels - 1].set_xlabel('time (s)')
    fig.show()
