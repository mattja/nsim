# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
Implements the core functionality of nsim.

Classes:
--------
``Model``   base class for different kinds of dynamical model
``ODEModel``   system of ordinary differential equations
``SDEModel``   system of stochastic differential equations
``DelaySDEModel``   system of delay stochastic differential equations

``Simulation``   single simulation run of a model, with simulation results

``MultipleSim``    set of simulations, distributed.
``RepeatedSim``   repeated simulations of the same model (to get statistics)
``ParameterSim``  multiple simulations of a model exploring parameter space
``NetworkSim``    simulate many instances of a model coupled in a network

functions:
----------
``quickmodel()``  Create a new model class dynamically at runtime
``quicksim()``  Create a new simulation dynamically at runtime
"""

from __future__ import absolute_import
from .timeseries import Timeseries, _Timeslice
from . import sde
from . import analysesN
import distob
from scipy import stats
from scipy import integrate
import numpy as np
import copy
import types
import numbers
import random
#from memory_profiler import profile


class Error(Exception):
    pass


class SimTypeError(Error):
    pass


class SimValueError(Error):
    pass


class SimClusterError(Error):
    """Thrown if there is a problem using the cluster that we can't fix"""
    pass


@distob.proxy_methods(Timeseries, include_underscore=(
    '__getitem__', '__setitem__', '__getslice__', '__setslice__'))
class RemoteTimeseries(distob.RemoteArray, object):
    """Local object representing a Timeseries that may be local or remote"""
    def __repr__(self):
        return self._cached_apply('__repr__').replace(
            self._ref.type.__name__, self.__class__.__name__, 1)

    def plot(self, title=None, show=True):
        self._fetch()
        return self._obcache.plot(title, show)


@distob.proxy_methods(_Timeslice, include_underscore=(
    '__getitem__', '__setitem__', '__getslice__', '__setslice__', '__repr__'))
class Remote_Timeslice(distob.Remote, object):
    pass


class Model(object):
    """Base class for different kinds of dynamical systems"""

    def __init__(self):
        """When making each new instance from the Model, the constructor will 
        convert any random-variable class attributes into fixed numbers drawn 
        from the specified distribution. Thus each individual object made from 
        the class 'recipe' can be given slightly different parameter values.
        """
        for attrib in dir(self):
            if isinstance(getattr(self, attrib), stats.distributions.rv_frozen):
                setattr(self, attrib, getattr(self, attrib).rvs())

    def integrate(self, tspan):
        """numerical integration function to use"""
        pass 


class ODEModel(Model):
    """Model defined by a system of ordinary differential equations"""
    def __init__(self):
        super(ODEModel, self).__init__()

    def integrate(self, tspan):
        return Timeseries(integrate.odeint(self.f, self.y0, tspan), tspan)

    def f(y, t):
        pass


class SDEModel(Model):
    """Model defined by a system of (ordinary) stochastic differential equations

    Attributes:
      ndim (integer): Dimension of the state space
      output_vars (list of integers): If i is in this list then y[i] is 
        considered an output variable

    Instance attributes:
      y0 (array of shape (ndim,)): Initial state vector
    """
    dimension = 1
    output_vars = [0]

    def __init__(self):
        super(SDEModel, self).__init__()
        if not hasattr(self.__class__, 'y0'):
            self.y0 = np.zeros(self.__class__.dimension)

    def integrate(self, tspan):
        return Timeseries(sde.sodeint(self.f, self.G, self.y0, tspan), tspan)

    def f(y, t):
        pass

    def G(y, t):
        pass


class DelaySDEModel(Model):
    """Model defined by a system of stochastic delay differential equations
    """
    pass


class Simulation(object):
    """Represents simulation of a single system and the resulting time series.

    Attributes:
      system (Model): The dynamical system being simulated. (Can provide either
        a Model subclass or Model instance)
      tspan (array): The sequence of time points simulated
      timeseries (array of shape (len(tspan), len(y0))): 
        Multivariate time series of full simulation results.
      output: Some function of the simulated timeseries, for example a 
        univariate time series of a single output variable. 
    """
    def __init__(self, system, T=60.0, dt=0.005):
        """
        Args:
          system (Model): The dynamical system to simulate
          T (Number, optional): Total length of time to simulate, in seconds.
          dt (Number, optional): Timestep for numerical integration.
        """
        if isinstance(system, type):
            self.system = system()
        else:
            self.system = system
        self.T = T
        self.dt = dt
        self.__timeseries = None
        self.__output_vars = tuple(self.system.__class__.output_vars)

    def compute(self):
         tspan = np.arange(0, self.T + self.dt, self.dt)
         self.__timeseries = Timeseries(self.system.integrate(tspan), tspan)

    def __get_timeseries(self):
         if self.__timeseries is None:
             self.compute()
         return self.__timeseries

    timeseries = property(fget=__get_timeseries, doc="Simulated time series")

    def __get_output(self):
        return self.timeseries[:,self.__output_vars]

    output = property(fget=__get_output, doc="Simulated model output")


# TODO can remove this class after distob proxy methods support block=False
@distob.proxy_methods(Simulation)
class RemoteSimulation(distob.Remote, Simulation):
    """Local object representing a remote Simulation"""
    def __init__(self, ref, client):
        """Make a RemoteSimulation to access an already-existing Simulation 
        object, which may be on a remote engine.

        Args:
          ref (Ref): reference to a Simulation to be controlled by this proxy
          client (IPython.parallel.client)
        """
        super(RemoteSimulation, self).__init__(ref, client)
        #self.compute()

    def compute(self):
        """Start the computation process asynchronously"""
        def remote_compute(sim_id):
            distob.engine[sim_id].compute()
        self._dv.apply_async(remote_compute, self._id)

def _with_plugins(plugin_module):
    """class decorator. Make methods from the functions in plugin_module"""
    def modify_class(cls):
        for name in plugin_module.__dict__:
            if isinstance(plugin_module.__dict__[name], types.FunctionType):
                setattr(cls, name, plugin_module.__dict__[name])
        return cls
    return modify_class


@_with_plugins(analysesN)
class MultipleSim(object):
    """Represents multiple simulations, possibly running on different hosts

    Like a list, indexing with [i] gives access to the ith simulation

    Attributes:
      sims: list of simulations
    """
    def __init__(self, systems, T=60.0, dt=0.005):
        """
        Args:
          systems: sequence of Model instances that should be simulated.
          T: total length of time to simulate, in seconds.
          dt: timestep for numerical integration.
        """
        self.T = T
        self.dt = dt
        self.sims = [Simulation(s, T, dt) for s in systems]
        distob.scatter(self.sims)
        for s in self.sims:
            s.compute()

    def __len__(self):
        return len(self.sims)

    def __getitem__(self, key):
        return self.sims[key]

    def _repr_pretty_(self, p, cycle):
        return p.pretty(self.sims)


class RepeatedSim(MultipleSim):
    """Independent simulations of the same model multiple times, with results.

    Like a list, indexing the object with [i] gives access to the ith simulation

    Attributes:
      sims: the individual simulations
      modelclass: the Model class common to all the simulations
      timeseries: resulting timeseries: all variables of all simulations
      output: resulting timeseries: output variables of all simulations
    """
    def __init__(self, model, T=60.0, dt=0.005, repeat=1, identical=True):
        """
        Args:
          model: Can be either a Model subclass or Model instance. This 
            defines the dynamical systems to simulate.
          T (optional): total length of time to simulate, in seconds.
          dt (optional): timestep for numerical integration.
          repeat (int, optional): number of repeated simulations of the model
          identical (bool, optional): Whether the repeated simulations use 
            identical parameters. If identical=False, each simulation will use 
            different parameters drawn from the random distributions defined in 
            the Model class. If identical=True, the choice will be made once 
            and then all simulations done with identical parameters. 
        """
        if isinstance(model, type):
            self.modelclass = model
            system = self.modelclass()
        else:
            system = model
            self.modelclass = type(model)
        if identical is True:
            systems = [copy.deepcopy(system) for i in range(repeat)]
        else:
            systems = [self.modelclass() for i in range(repeat)]
        super(RepeatedSim, self).__init__(systems, T, dt)

    def __get_timeseries(self):
        ndim = self.sims[0].timeseries.ndim
        if ndim is 1:
            array = np.dstack(
                tuple(np.expand_dims(s.timeseries, 1) for s in self.sims))
        else:
            array = np.concatenate(
                tuple(np.expand_dims(s.timeseries, ndim) for s in self.sims), 
                ndim)
        return Timeseries(array, self.sims[0].timeseries.tspan)

    timeseries = property(fget=__get_timeseries, doc="Rank 3 array representing"
        " multiple time series. 1st axis is time, 2nd axis ranges across all"
        " dynamical variables in a single simulation, 3rd axis ranges across"
        " different simulation instances.")

    def __get_output(self):
        ndim = self.sims[0].output.ndim
        if ndim is 1:
            array = np.dstack(
                tuple(np.expand_dims(s.output, 1) for s in self.sims))
        else:
            array = np.concatenate(
                tuple(np.expand_dims(s.output, ndim) for s in self.sims), ndim)
        return Timeseries(array, self.sims[0].output.tspan)

    output = property(fget=__get_output, doc="Rank 3 array representing"
        " output time series. 1st axis is time, 2nd axis ranges across"
        " output variables of a single simulation, 3rd axis ranges across"
        " different simulation instances.")


class ParameterSim(MultipleSim):
    """Independent simulations of a model exploring different parameters"""
    pass


class NetworkSim(MultipleSim):
    """Simulation of many coupled instances of a model connected in a network"""
    pass


def quicksim(f, G, y0, T=60.0, dt=0.005, repeat=1, identical=True):
    """Make a simulation of the system defined by functions f and G.

    dy = f(y,t)dt + G(y,t).dW with initial condition y0
    This helper function is for convenience, making it easy to define 
    one-off simulations interactively in ipython.

    Args:
      f: callable(y, t) (defined in global scope) returning (n,) array
        Vector-valued function to define the deterministic part of the system 
      G: callable(y, t) (defined in global scope) returning (n,m) array
        Optional matrix-valued function to define noise coefficients
      y0 (array):  Initial condition 
      T: Total length of time to simulate, in seconds.
      dt: Timestep for numerical integration.
      repeat (int, optional)
      identical (bool, optional)

    Returns: 
      sim (sim.Simulation)

    Raises:
      SimValueError, SimTypeError
    """
    NewModel = quickmodel(f, G, y0, 'NewModel')
    if repeat == 1:
        return Simulation(NewModel())
    else:
        return RepeatedSim(NewModel, repeat=repeat, identical=identical)


def quickmodel(f, G, y0, name='NewModel'):
    """Use the functions f and G to define a new Model class for simulations. 

    It will take functions f and G from global scope and make a new Model class
    out of them. It will automatically gather any globals used in the definition
    of f and G and turn them into attributes of the new Model.

    Args:
      f: callable(y, t) (defined in global scope) returning (n,) array
         Scalar or vector-valued function to define the deterministic part
      G: callable(y, t) (defined in global scope) returning (n,m) array
         Optional scalar or matrix-valued function to define noise coefficients
      y0 (Number or array): Initial condition
      name: Optional class name for the new model

    Returns: 
      new class (subclass of Model)

    Raises:
      SimValueError, SimTypeError
    """
    if not callable(f) or (G is not None and not callable(G)):
        raise SimTypeError('f and G must be functions of y and t.')
    if G is not None and f.__globals__ is not G.__globals__:
        raise SimValueError('f and G must be defined in the same place')
    # TODO: validate that f and G are defined at global scope.
    # TODO: Handle nonlocals used in f,G so that we can lift this restriction.
    if G is None:
        newclass = type(name, (ODEModel,), dict())
        setattr(newclass, 'f', staticmethod(__clone_function(f, 'f')))
    else:
        newclass = type(name, (SDEModel,), dict())
        setattr(newclass, 'f', staticmethod(__clone_function(f, 'f')))
        setattr(newclass, 'G', staticmethod(__clone_function(G, 'G')))
    setattr(newclass, 'y0', copy.deepcopy(y0))
    # For any global that is used by the functions f or G, create a 
    # corresponding attribute in our new class.
    globals_used = [x for x in f.__globals__ if (x in f.__code__.co_names or 
        G is not None and x in G.__code__.co_names)]
    for x in globals_used:
        if G is None:
            setattr(newclass, x, __AccessDict(x, newclass.f.__globals__))
        else:
            setattr(newclass, x, __AccessDicts(x, newclass.f.__globals__, 
                                                  newclass.G.__globals__))
    # Allow passing in scalars as well. In this case convert to arrays here.
    if isinstance(y0, numbers.Number):
        setattr(newclass, 'y0', np.array([newclass.y0]))
        scalarf, scalarG = newclass.f, newclass.G
        setattr(newclass, 'f', 
                staticmethod(lambda y, t: np.array([scalarf(y, t)])))
        if G is not None:
            setattr(newclass, 'G',
                    staticmethod(lambda y, t: np.array([[scalarG(y, t)]])))
    # Make the new class' official name visible in namespace nsim.nsim
    globals()[name] = newclass 
    return newclass


class __AccessDict(object):
    """A descriptor class representing a value held in a dict.
      k (object): The common key to access. 
      d (dict)
    """
    def __init__(self, k, d):
        self.k = k
        self.d = d
    def __get__(self, obj, objtype):
        return self.d[self.k]
    def __set__(self, obj, val):
        self.d[self.k] = val


class __AccessDicts(object):
    """A descriptor class representing a common value held in two dicts.
      k (object): The common key to access. 
      d1, d2 (dict):  (Invariant: d1[k] is d2[k])
    """
    def __init__(self, k, d1, d2):
        self.k = k
        self.d1 = d1
        self.d2 = d2
    def __get__(self, obj, objtype):
        return self.d1[self.k]
    def __set__(self, obj, val):
        self.d1[self.k] = val
        self.d2[self.k] = val


def __clone_function(f, name=None):
    """Make a new version of a function that has its own independent copy 
    of any globals that it uses directly, and has its own name. 
    All other attributes are assigned from the original function.

    Args:
      f: the function to clone
      name (str):  the name for the new function (if None, keep the same name)

    Returns:
      A copy of the function f, having its own copy of any globals used

    Raises:
      SimValueError
    """
    if not isinstance(f, types.FunctionType):
        raise SimTypeError('Given parameter is not a function.')
    if name is None:
        name = f.__name__
    newglobals = f.__globals__.copy()
    globals_used = [x for x in f.__globals__ if x in f.__code__.co_names]
    for x in globals_used:
        gv = f.__globals__[x]
        if isinstance(gv, types.FunctionType):
            # Recursively clone any global functions used by this function.
            newglobals[x] = __clone_function(gv)
        elif isinstance(gv, types.ModuleType):
            newglobals[x] = gv
        else:
            # If it is something else, deep copy it.
            newglobals[x] = copy.deepcopy(gv)
    newfunc = types.FunctionType(
        f.__code__, newglobals, name, f.__defaults__, f.__closure__)
    return newfunc
