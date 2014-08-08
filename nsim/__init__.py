from __future__ import absolute_import

from .timeseries import Timeseries, merge, timeseries_from_mat
from . import analyses1
Timeseries.add_analyses(analyses1)

from .nsim import (
        Model, ODEModel, SDEModel, DelaySDEModel, Simulation, 
        MultipleSim, RepeatedSim, ParameterSim, NetworkSim, newmodel, newsim, 
        Error, SimTypeError, SimValueError, SimClusterError)
from . import models, sde

# If python-edf is installed, provide support for loading edf files
try:
    from .edf import timeseries_from_edf
except ImportError:
    pass

__version__ = '0.1.1'

