nsim
====
Simulate systems from ODEs or SDEs, analyze timeseries.

Overview
--------
nsim is for systems in physics, biology and finance that are modelled
in continuous time with differential equations. nsim makes it easy to
define and simulate these (including proper treatment of noise in SDEs)
and to analyze the properties of the resulting timeseries.

* Automatic parallel computing / cluster computing: For multiple or repeated
  simulations, nsim distributes these across a cluster (or across the CPUs
  of one computer) without needing to do any parallel programming.

  (First start an IPython cluster e.g. by typing `ipcluster start`)
  
* Model parameters can optionally be specified as random distributions, 
  instead of fixed values, to create multiple non-identical simulations.

* nsim provides a `Timeseries` class. This is a numpy array.  
  It allows slicing the array by time instead of by array index, 
  and can keep track of channel names (or variable names) of a multivariate 
  time series.

* As well as the usual methods of numpy arrays, the `Timeseries` objects 
  have extra methods for easy filtering, plotting and analysis.
  Analyses can be chained together in a pipeline. This can easily be extended
  with your own analysis functions by calling `Timeseries.add_analyses()`

  Analyses of multiple time series are distributed on the cluster,
  without needing to do any parallel programming.

* Besides simulations, arrays of time series data can be loaded from MATLAB 
  .mat files or .EDF files for distributed analysis.

TODO
----
* Add support for models with time delays (DDEs and delay SDEs)

* Support network models of dynamical nodes, auto-generated from models of 
  node dynamics and a network graph structure. (use shared memory and 
  multiple CPU cores on each cluster host for simulation of network models,
  splitting degrees of freedom evenly across CPUs).

* Auto-generate multiple simulations covering a lattice of points in 
  parameter space, to run in parallel.

* Directly support SDEs expressed in Ito form. (Currently need to write it
  in Stratonovich form as an intermediate step before simulating in nsim)

* Optionally allow the equations to be specified and integrated in C, for speed

* Write statistical analyses applying to ensembles of repeated SDE simulations  
  (First will improve the `distob` package to add a DistArray class,
   allowing a single ndarray to be spread across the cluster)

Thanks
------
Incorporates extra time series analyses from Forrest Sheng Bao's `pyeeg` http://fsbao.net

`IPython` parallel computing, see: http://ipython.org/ipython-doc/dev/parallel/
