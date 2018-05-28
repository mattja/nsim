# Copyright 2016 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
functions:
  `timeseries_from_mat()` load a Timeseries from a MATLAB .mat file
  `timeseries_from_file()` load a Timeseries from many file types
  `save_mat()`  save a Timeseries to a MATLAB .mat file
"""

from __future__ import absolute_import
from nsim import Timeseries, Error
import numpy as np
from os import path


def timeseries_from_mat(filename, varname=None, fs=1.0):
    """load a multi-channel Timeseries from a MATLAB .mat file

    Args:
      filename (str): .mat file to load
      varname (str): variable name. only needed if there is more than one
        variable saved in the .mat file
      fs (scalar): sample rate of timeseries in Hz. (constant timestep assumed)

    Returns:
      Timeseries
    """
    import scipy.io as sio
    if varname is None:
        mat_dict = sio.loadmat(filename)
        if len(mat_dict) > 1:
            raise ValueError('Must specify varname: file contains '
                             'more than one variable. ')
    else:
        mat_dict = sio.loadmat(filename, variable_names=(varname,))
        array = mat_dict.popitem()[1]
    return Timeseries(array, fs=fs)


def save_mat(ts, filename):
    """save a Timeseries to a MATLAB .mat file
    Args:
      ts (Timeseries): the timeseries to save
      filename (str): .mat filename to save to
    """
    import scipy.io as sio
    tspan = ts.tspan
    fs = (1.0*len(tspan) - 1) / (tspan[-1] - tspan[0])
    mat_dict = {'data': np.asarray(ts),
                'fs': fs,
                'labels': ts.labels[1]}
    sio.savemat(filename, mat_dict, do_compression=True)
    return


def timeseries_from_file(filename):
    """Load a multi-channel Timeseries from any file type supported by `biosig`

    Supported file formats include EDF/EDF+, BDF/BDF+, EEG, CNT and GDF.
    Full list is here: http://pub.ist.ac.at/~schloegl/biosig/TESTED

    For EDF, EDF+, BDF and BDF+ files, we will use python-edf 
    if it is installed, otherwise will fall back to python-biosig.

    Args: 
      filename

    Returns: 
      Timeseries
    """
    if not path.isfile(filename):
        raise Error("file not found: '%s'" % filename)
    is_edf_bdf = (filename[-4:].lower() in ['.edf', '.bdf'])
    if is_edf_bdf:
        try:
            import edflib
            return _load_edflib(filename)
        except ImportError:
            print('python-edf not installed. trying python-biosig instead...')
    try:
        import biosig
        return _load_biosig(filename)
    except ImportError:
        message = (
            """To load timeseries from file, ensure python-biosig is installed
            e.g. on Ubuntu or Debian type `apt-get install python-biosig`
            or get it from http://biosig.sf.net/download.html""")
        if is_edf_bdf:
            message += """\n(For EDF/BDF files, can instead install python-edf:
                       https://bitbucket.org/cleemesser/python-edf/ )"""
        raise Error(message)


def _load_biosig(filename):
    import biosig
    hdr = biosig.constructHDR(0, 0)
    hdr = biosig.sopen(filename, 'r', hdr)
    # The logic here does not match the biosig API. But it is deliberately
    # this way to work around a bug loading EDF files in python-biosig 1.3.
    # TODO: revisit this code when I finally get python-biosig 1.6 to install.
    channels = hdr.NS - 1
    fs = hdr.SampleRate / channels
    npoints = hdr.NRec * hdr.SPR / channels
    ar = np.zeros((npoints, channels), dtype=np.float64)
    channelnames = []
    for i in range(channels):
        label = hdr.CHANNEL[i].Label
        if '\x00' in label:
            label = label[:label.index('\x00')]
        channelnames.append(label)
        for j in range(hdr.NS):
            hdr.CHANNEL[j].OnOff = int(j == i)
        data = biosig.sread(0, hdr.NRec, hdr)
        ar[:, i] = data.reshape((npoints, channels))[:, 0]
    biosig.sclose(hdr)
    biosig.destructHDR(hdr)
    return Timeseries(ar, labels=[None, channelnames], fs=fs)


def _load_edflib(filename):
    """load a multi-channel Timeseries from an EDF (European Data Format) file
    or EDF+ file, using edflib.

    Args:
      filename: EDF+ file

    Returns:
      Timeseries
    """
    import edflib
    e = edflib.EdfReader(filename, annotations_mode='all')
    if np.ptp(e.get_samples_per_signal()) != 0:
        raise Error('channels have differing numbers of samples')
    if np.ptp(e.get_signal_freqs()) != 0:
        raise Error('channels have differing sample rates')
    n = e.samples_in_file(0)
    m = e.signals_in_file
    channelnames = e.get_signal_text_labels()
    dt = 1.0/e.samplefrequency(0)
    # EDF files hold <=16 bits of information for each sample. Representing as
    # double precision (64bit) is unnecessary use of memory. use 32 bit float:
    ar = np.zeros((n, m), dtype=np.float32)
    # edflib requires input buffer of float64s
    buf = np.zeros((n,), dtype=np.float64)
    for i in range(m):
        e.read_phys_signal(i, 0, n, buf)
        ar[:,i] = buf
    tspan = np.arange(0, (n - 1 + 0.5) * dt, dt, dtype=np.float32)
    return Timeseries(ar, tspan, labels=[None, channelnames])


def annotations_from_file(filename):
    """Get a list of event annotations from an EDF (European Data Format file
    or EDF+ file, using edflib.

    Args:
      filename: EDF+ file

    Returns:
      list: annotation events, each in the form [start_time, duration, text]
    """
    import edflib
    e = edflib.EdfReader(filename, annotations_mode='all')
    return e.read_annotations()
