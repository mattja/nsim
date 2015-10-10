nsim
====

| Simulate systems from ODEs or SDEs, analyze timeseries.
|  N.B. this is a pre-release: still a lot left to be done

Overview
--------

nsim is for systems in physics, biology and finance that are modelled in
continuous time with differential equations. nsim makes it easy to
define and simulate these (including proper treatment of noise in SDEs)
and to analyze the properties of the resulting timeseries.

-  | Automatic parallel computing / cluster computing: For multiple or repeated simulations, nsim distributes these across a cluster or Amazon EC2 cloud (or across the CPUs of one computer) without needing to do any parallel programming.
   | (First configure an `IPython cluster <https://ipyparallel.readthedocs.org/en/latest/process.html#configuring-an-ipython-cluster>`_. e.g. on a single computer can type ``ipcluster start``)
   | Note: computation scales poorly in the current version of nsim if the number of simulations is much greater than the number of CPUs, but this will be handled properly in future versions.

- Now supports ODEs, scalar and vector Ito and Stratonovich SDEs (possibly with multiple driving noise processes) and can use a more recent order 1.0 strong stochastic Runge-Kutta algorithm (Rößler2010) for simulating the SDEs.

-  Model parameters can optionally be specified as random distributions,
   instead of fixed values, to create multiple non-identical
   simulations.

-  | nsim provides a ``Timeseries`` class. This is a numpy array.
   | It allows slicing the array by time instead of by array index, and can keep track of channel names (or variable names) of a multivariate time series.

-  | As well as the usual methods of numpy arrays, the ``Timeseries`` objects have extra methods for easy filtering, plotting and analysis. Analyses can be chained together in a pipeline. This can easily be extended with your own analysis functions by calling ``Timeseries.add_analyses()``
   | Analyses of multiple time series are distributed on the cluster, without needing to do any parallel programming.

-  Besides simulations, arrays of time series data can be loaded from
   MATLAB .mat files or .EDF files for distributed analysis.

-  For best results use numpy 1.11.0 or later (not yet released!) this enables us to support distributed computation when analysing the resulting time series. You can get a development snapshot of numpy here: https://github.com/numpy/numpy/archive/master.zip

TODO
----
-  Currently parallel simulation and analysis scales poorly if the number of
   simulations is much greater than the number of CPUs. This is a known issue
   in the ``distob`` library to be fixed in its next release. Until this is 
   fixed, you can use ``import distob; ts = distob.gather(ts)`` to work with 
   a timeseries object ``ts`` locally instead of leaving it distributed on the 
   cluster.

-  Auto-generate multiple simulations covering a lattice of points in
   parameter space, to run in parallel.

-  Add support for models with time delays (DDEs and delay SDEs)

-  Support network models of dynamical nodes, auto-generated from models
   of node dynamics and a network graph structure. (use shared memory
   and multiple CPU cores on each cluster host for simulation of network
   models, splitting degrees of freedom evenly across CPUs).

-  Optionally allow the equations to be specified and integrated in C,
   for speed

Thanks
------

Incorporates extra time series analyses from Forrest Sheng Bao's
``pyeeg`` http://fsbao.net

``IPython`` parallel computing, see:
http://ipython.org/ipython-doc/dev/parallel/

See also:
---------

``sdeint``: Library of SDE integration algorithms that is used by ``nsim`` to do the simulations. https://github.com/mattja/sdeint
