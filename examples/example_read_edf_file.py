"""Instead of simulation, can read in experimental time series data
from an .mat file or .edf file and analyze it.
"""
import nsim

ts = nsim.timeseries_from_file('data/31_before-1+.edf')

ts_filtered = ts.bandpass(1.5, 70.0)
ts_filtered.psd()

ts2 = ts_filtered.t[800:840, 3] # just channel 3, time from 800 to 840 seconds
ts2.plot()
mobility, complexity = ts2.hjorth()

ts_filtered.t[800:, 0:8].epochs_distributed() # intervals of low variability
