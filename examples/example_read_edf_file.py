"""Example that does no simulation, 
instead reads in an experimental time series from an EDF file.
"""

import nsim

ts = nsim.timeseries_from_edf('data/31_before-1+.edf')
ts = ts[:, 0:19]
ts_filtered = ts.bandpass(1.5, 70.0)
ts_filtered.plot()
ts_filtered.t[800:840, 3].psd()
