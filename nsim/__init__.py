from __future__ import absolute_import

from .timeseries import Timeseries, merge

from . import analyses1
Timeseries.add_analyses(analyses1)

from .nsim import (
        Model, ODEModel, ItoModel, StratonovichModel, NetworkModel,
        DelayItoModel, Simulation, MultipleSim, RepeatedSim, ParameterSim,
        newmodel, newsim, DistTimeseries, Error, SimTypeError, SimValueError)

from . import analysesN
DistTimeseries.add_analyses(analyses1, vectorize=True)
DistTimeseries.add_analyses(analysesN)

from . import models
from .readfile import (
        timeseries_from_mat, timeseries_from_file, annotations_from_file,
        save_mat)

__version__ = '0.1.18'
