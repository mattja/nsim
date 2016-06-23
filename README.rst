nsim
====
| Simulate systems from ODEs or SDEs, analyze timeseries.
|  N.B. this is a pre-release: still a lot left to be done

Simulation
----------
nsim is for systems in physics, biology and finance that are modelled in continuous time with differential equations. nsim makes it easy to define and simulate these (including proper treatment of noise) and to analyze the resulting time series.

-  | Automatic parallel computing / cluster computing: For multiple or repeated simulations, nsim distributes these across a cluster or Amazon EC2 cloud (or across the CPUs of one computer) without needing to do any parallel programming.
   | (First configure an `IPython cluster <https://ipyparallel.readthedocs.org/en/latest/process.html#configuring-an-ipython-cluster>`_. e.g. on a single computer can type ``ipcluster start``)

-  To define a scalar or vector ODE system, subclass ``ODEModel``. (see `examples <https://github.com/mattja/nsim/tree/master/examples>`_) To define a scalar or vector SDE system, subclass ``ItoModel`` or ``StratonovichModel``. Multiple driving Wiener processes are now supported. Order 1.0 strong stochastic Runge-Kutta algorithms (Rößler2010) are used for SDE integration by default.

-  Model parameters can be specified as random distributions, to create multiple non-identical simulations.

-  The ``NetworkModel`` class allows you to simulate many subsystems coupled together into a network, with the network structure specified as a weighted directed graph. Sub-models can all be identical but they don't have to be. (The `networkx <http://networkx.github.io/>`_ package can optionally be used to generate various kinds of random, clustered and small world graphs useful in a NetworkModel). The sub-models in a NetworkModel can even be other NetworkModels, for simulating networks of networks.

Analyzing time series
---------------------
Besides time series from simulations, empirical time series data can also be loaded from MATLAB .mat files or .EDF files for distributed analysis.

-  | nsim provides a ``Timeseries`` class. This is a numpy array.
   | It allows slicing the array by time instead of by array index, e.g. can write ``ts.t[10.5:30]`` to slice from t=10.5 to t=30 seconds. When manipulating the array it will keep track of any channel names (or variable names) of a multivariate time series.

-  | As well as the usual methods of numpy arrays, the ``Timeseries`` objects have extra methods for easy filtering, plotting and analysis. Analyses can be chained together in a pipeline. For example with a ``Timeseries`` instance ``ts`` you can write a chain of analyses like ``ts.t[10:30].bandpass(20, 35).hilbert().abs().plot()``
   | This can be extended with your own analysis functions by calling ``Timeseries.add_analyses()``
   | Analysis of multiple time series is distributed on the cluster, without needing to do any parallel programming.

-  For best results use numpy 1.11.0 or later (not yet released!) this enables us to support distributed computation when analysing the resulting time series. You can get a development snapshot of numpy here: https://github.com/numpy/numpy/archive/master.zip

TODO
----
-  Auto-generate multiple simulations covering a lattice of points in
   parameter space, to run in parallel.

-  Optionally allow the equations to be specified and integrated in C,
   for speed

-  Add support for models with time delays (DDEs and delay SDEs)

-  Currently a single CPU core is used to simulate each single instance of a
   Model, including a NetworkModel. Ideally could use shared memory and
   multiple CPU cores on each cluster host for simulation of a Model instance,
   splitting degrees of freedom evenly across CPUs on a single host.

Thanks
------
Incorporates extra time series analyses from Forrest Sheng Bao's
``pyeeg`` http://fsbao.net

``ipyparallel`` interactive parallel computing:
https://ipyparallel.readthedocs.org/

See also:
---------
``sdeint``: Library of SDE integration algorithms that is used by ``nsim`` to do the simulations. https://github.com/mattja/sdeint
