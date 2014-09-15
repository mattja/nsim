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

``RemoteTimeseries`` Local proxy representing a Timeseries on a remote engine 
``DistTimeseries`` Timeseries with one axis distributed onto multiple engines 

functions:
----------
``newmodel()``  Create a new model class dynamically at runtime
``newsim()``  Create a new simulation dynamically at runtime
"""

from __future__ import absolute_import
from .timeseries import Timeseries, _Timeslice
from . import sde
from . import analysesN
import distob
from scipy import stats
from scipy import integrate
import numpy as np
from collections import Sequence
import copy
import types
import numbers
import random
#from memory_profiler import profile

# types for compatibility across python 2 and 3
_SliceType = type(slice(None))
_EllipsisType = type(Ellipsis)
_TupleType = type(())
_NewaxisType = type(np.newaxis)


class Error(Exception):
    pass


class SimTypeError(Error):
    pass


class SimValueError(Error):
    pass


@distob.proxy_methods(Timeseries, exclude=('dtype',), include_underscore=(
    '__getitem__', '__setitem__', '__getslice__', '__setslice__'))
class RemoteTimeseries(distob.RemoteArray, object):
    """Local object representing a Timeseries that may be local or remote"""

    def __repr__(self):
        return self._cached_apply('__repr__').replace(
            self._ref.type.__name__, self.__class__.__name__, 1)

    def plot(self, title=None, show=True):
        self._fetch()
        return self._obcache.plot(title, show)

    def psd(self, plot=True):
        """Power spectral density
        Args:
          plot (bool)
        """
        if not plot:
            return self._cached_apply('psd', plot=False)
        else:
            self._fetch()
            return self._obcache.psd(plot=True)

    def concatenate(self, tup, axis=0):
        if not isinstance(tup, Sequence):
            tup = (tup,)
        if tup is (None,) or len(tup) is 0:
            return self
        new_array = super(RemoteTimeseries, self).concatenate(tup, axis)
        tup = (self,) + tuple(tup)
        if not all(hasattr(ts, 'tspan') and
                   hasattr(ts, 'labels') for ts in tup):
            return new_array
        if axis == 0:
            starts = [ts.tspan[0] for ts in tup]
            ends = [ts.tspan[-1] for ts in tup]
            if not all(starts[i] > ends[i-1] for i in range(1, len(starts))):
                # series being joined are not ordered in time. not a Timeseries
                return new_array
            else:
                new_tspan = np.concatenate([ts.tspan for ts in tup])
        else:
            new_tspan = self.tspan
        new_labels = [None]
        for ax in range(1, new_array.ndim):
            if ax == axis:
                axislabels = []
                for ts in tup:
                    if ts.labels[axis] is None:
                        axislabels.extend('' * ts.shape[axis])
                    else:
                        axislabels.extend(ts.labels[axis])
                if all(lab == '' for lab in axislabels):
                    new_labels.append(None)
                else:
                    new_labels.append(axislabels)
            else:
                # non-concatenation axis
                axlabels = tup[0].labels[ax]
                if not all(ts.labels[ax] == axlabels for ts in tup[1:]):
                    # series to be joined do not agree on labels for this axis
                    axlabels = None
                new_labels.append(axlabels)
        if isinstance(new_array, distob.RemoteArray):
            # TODO: could have kept tspan and labels remote in this case
            return _rts_from_ra(new_array, new_tspan, new_labels)
        else:
            assert(isinstance(new_array, distob.DistArray))
            return _dts_from_da(new_array, new_tspan, new_labels)


@distob.proxy_methods(_Timeslice, include_underscore=(
    '__getitem__', '__setitem__', '__getslice__', '__setslice__', '__repr__'))
class Remote_Timeslice(distob.Remote, object):
    pass


class DistTimeseries(distob.DistArray):
    """a Timeseries with one axis distributed onto multiple computing engines.

    For example, a multi-channel timeseries could be distributed so that each
    channel is held on a different computer (for parallel computation)

    Currently only a non-time axis can be distributed.
    """
    # TODO: consider refactoring as __getitem__ and __repr__ share much logic
    #   with the Timeseries class
    def __init__(self, subts, axis=None, axislabels=None):
        """Make a DistTimeseries from a list of existing remote Timeseries

        Args:
          subts (list of RemoteTimeseries, or list of Ref to Timeseries): 
            the sub-timeseries (possibly remote) which form the whole
            DistTimeseries when concatenated. These must all have the same 
            time points, shape and dtype. Currently must have 
            `ts.shape[axis] == 1` for each sub-timeseries `ts`.

          axis (int, optional): Position of the distributed axis, which is the
            axis along which the sub-timeseries will be concatenated. Default
            is the last axis. Cannot be 0, that is, we currently only allow a
            non-time axis to be distributed.

          axislabels (list of str, optional): names to label each position 
            along the new distributed axis. e.g. ['node1', ..., 'nodeN']
        """
        if axis == 0:
            raise SimValueError(u'Currently cannot distribute the time axis')
        if axislabels is not None and not isinstance(axislabels, Sequence):
            raise SimValueError(u'axislabels should be a list of str')
        if axislabels is not None and len(axislabels) != len(subts):
            raise SimValueError(u'mismatch: %d subarrays but %d axislabels' % (
                len(subts), len(axislabels)))
        super(DistTimeseries, self).__init__(subts, axis)
        self.tspan = distob.gather(self._subarrays[0].tspan)
        # Expensive to validate all tspans are the same. check start and end t
        # (TODO: in Timeseries class make special case for constant timestep)
        starts = [rts.tspan[0] for rts in self._subarrays]
        ends = [rts.tspan[-1] for rts in self._subarrays]
        if (not all(t == self.tspan[0] for t in starts) or 
                not all(t == self.tspan[-1] for t in ends)):
            raise SimValueError(u'timeseries must use the same time points')
        nlabels = [rts.labels for rts in self._subarrays]
        self.labels = [None]
        # only give labels to an axis if all sub-timeseries agree
        for i in range(1, self.ndim - 1):
            if len(set(labels[i] for labels in nlabels)) is 1:
                self.labels.append(nlabels[0][i])
            else:
                self.labels.append(None)
        self.labels.insert(self._distaxis, axislabels)
        self.t = _Timeslice(self)

    @classmethod
    def add_analyses(cls, source, vectorize=False):
        """Dynamically add new analysis methods to the DistTimeseries class.
        Args:
          source: Can be a function, module or the filename of a python file.
            If a filename or a module is given, then all functions defined 
            inside not starting with _ will be added as methods.
          vectorize (bool): Whether to apply `distob.vectorize()` to the 
            function before making it a method of DistTimeseries.

        The only restriction on the functions is that they can accept a 
        Timeseries as their first argument.
        """
        if isinstance(source, types.FunctionType):
            if vectorize:
                source = distob.vectorize(source)
            setattr(cls, source.__name__, source)
        else:
            if isinstance(source, types.ModuleType):
                mod = source
            elif isinstance(source, types.StringTypes):
                import os
                import imp
                path = os.path.abspath(source)
                if os.path.isfile(path) and path[-3:] == '.py':
                    dir, file = os.path.split(path)
                    name = file[:-3]
                    module_info = imp.find_module(name, [dir])
                    mod = imp.load_module('nsim.' + name, *module_info)
                elif (os.path.isdir(path) and 
                        '__init__.py' in os.listdir(path)):
                    module_info = imp.find_module('__init__', [path])
                    name = os.path.basename(path)
                    mod = imp.load_module('nsim.' + name, *module_info)
                else:
                    raise Error('"%s" is not a file or directory' % source)
            else:
                raise ValueError('`source` argument not a function or module')
            for name, obj in mod.__dict__.items():
                if name[0] != '_' and isinstance(obj, types.FunctionType):
                    if vectorize:
                        obj = distob.vectorize(obj)
                    setattr(cls, name, obj)

    def __getitem__(self, index):
        """Slice the distributed timeseries"""
        ar = super(DistTimeseries, self).__getitem__(index)
        if (isinstance(ar, RemoteTimeseries) or 
                isinstance(ar, distob.RemoteArray)):
            # slicing result is no longer distributed
            return ar
        # otherwise `ar` is a DistArray
        if not isinstance(ar._subarrays[0], RemoteTimeseries):
            return ar
        cur_labels = self.labels
        cur_shape = self.shape # current values, may be updated by np.newaxis
        cur_ndim = self.ndim
        if isinstance(index, np.ndarray) and index.dtype.type is np.bool_:
            raise Error('indexing by boolean array not yet implemented')
        if not isinstance(index, Sequence):
            index = (index,) + (slice(None),)*(self.ndim - 1)
        ix_types = tuple(type(x) for x in index)
        if (np.ndarray in ix_types or
                (not isinstance(index, _TupleType) and
                    _NewaxisType not in ix_types and
                    _EllipsisType not in ix_types and
                    _SliceType not in ix_types) or
                any(issubclass(T, Sequence) for T in ix_types)):
            basic_slicing = False
        else:
            basic_slicing = True
        # Apply any ellipses
        while _EllipsisType in ix_types:
            pos = ix_types.index(_EllipsisType)
            m = (self.ndim + ix_types.count(_NewaxisType) - len(index) + 1)
            index = index[:pos] + (slice(None),)*m + index[(pos+1):]
            ix_types = tuple(type(x) for x in index)
        # Apply any np.newaxis
        while _NewaxisType in ix_types:
            pos = ix_types.index(type(np.newaxis))
            if pos is 0:
                # prepended an axis: no longer a Timeseries
                return ar
            index = index[:pos] + (slice(None),) + index[(pos+1):]
            cur_labels = cur_labels[:pos] + [None] + cur_labels[pos:]
            cur_shape = cur_shape[:pos] + (1,) + cur_shape[pos:]
            cur_ndim = len(cur_shape)
            ix_types = tuple(type(x) for x in index)
        index = tuple(index) + (slice(None),)*(cur_ndim - len(index))
        if len(index) > cur_ndim:
            raise IndexError('too many indices for array')
        if basic_slicing:
            new_tspan = self.tspan[index[0]]
            if not isinstance(new_tspan, np.ndarray) or new_tspan.shape is ():
                # axis 0 has been sliced away: result is no longer a Timeseries
                return ar
            new_labels = [None]
            for i in range(1, cur_ndim):
                # make temp ndarray of labels to ensure correct slicing
                if cur_labels[i] is None:
                    labelarray = np.array([''] * cur_shape[i])
                else:
                    labelarray = np.array(cur_labels[i])
                newlabelarray = labelarray[index[i]]
                if newlabelarray.shape is not ():
                    if cur_labels[i] is None:
                        new_labels.append(None)
                    else:
                        new_labels.append(list(newlabelarray))
            new_subts = ar._subarrays
            new_distaxis = ar._distaxis
            new_axislabels = new_labels[ar._distaxis]
            return DistTimeseries(new_subts, new_distaxis, new_axislabels)
        else:
            # advanced integer indexing
            is_fancy = tuple(not isinstance(x, _SliceType) for x in index)
            fancy_pos = tuple(i for i in range(len(index)) if is_fancy[i])
            nonfancy_pos = tuple(
                    i for i in range(len(index)) if not is_fancy[i])
            contiguous = (fancy_pos[-1] - fancy_pos[0] == len(fancy_pos) - 1)
            index = list(index)
            ix_arrays = [index[i] for i in fancy_pos]
            ix_arrays = np.broadcast_arrays(*ix_arrays)
            for j in range(len(fancy_pos)):
                index[fancy_pos[j]] = ix_arrays[j]
            index = tuple(index)
            ishape = index[fancy_pos[0]].shape # common shape all index arrays
            idim = len(ishape)
            assert(idim > 0)
            new_tspan = None
            if contiguous and not is_fancy[0]:
                new_tspan = self.tspan[index[0]]
                if (not isinstance(new_tspan, np.ndarray) or
                        new_tspan.shape is ()):
                    # axis 0 has been sliced away: no longer a Timeseries
                    return ar
            # compute labels for the nonfancy output axes
            nonfancy_labels = []
            nonfancy_retained = []
            for i in nonfancy_pos:
                # make temp ndarray of labels to ensure correct slicing
                if cur_labels[i] is None:
                    labelarray = np.array([''] * cur_shape[i])
                else:
                    labelarray = np.array(cur_labels[i])
                newlabelarray = labelarray[index[i]]
                if newlabelarray.shape is ():
                    nonfancy_retained.append(False)
                else:
                    nonfancy_retained.append(True)
                    if cur_labels[i] is None or len(newlabelarray) is 0:
                        nonfancy_labels.append(None)
                    else:
                        nonfancy_labels.append(list(newlabelarray))
            # compute labels for the fancy output axes:
            #
            # For each fancy output axis k, call input axis i a 'candidate'
            # label source for k if k is the only nonconstant axis in the
            # indexing array for i.
            # We will give labels/tspan to output axis k from input axis i
            # only if i is the sole candidate label source for k.
            candidates = [[]] * idim
            # candidates[k] will be a list of candidate label sources for k
            for i in fancy_pos:
                nonconstant_ix_axes = []
                for k in range(idim):
                    n = ishape[k]
                    if n > 0:
                        partix = np.split(index[i], n, axis=k)
                        if not all(np.array_equal(
                                partix[0], partix[q]) for q in range(1, n)):
                            nonconstant_ix_axes.append(k)
                if len(nonconstant_ix_axes) is 1:
                    candidates[nonconstant_ix_axes[0]].append(i)
            fancy_labels = []
            for k in range(idim):
                if len(candidates[k]) is 1:
                    # then we can label this output axis
                    label_source = candidates[k][0]
                    if cur_labels[label_source] is None:
                        labelarray = np.array([''] * cur_shape[label_source])
                    else:
                        labelarray = np.array(cur_labels[label_source])
                    iix = [0] * idim
                    iix[k] = slice(None)
                    iix = tuple(iix)
                    newlabelarray = labelarray[index[label_source][iix]]
                    if newlabelarray.shape is not ():
                        if cur_labels[label_source] is None:
                            fancy_labels.append(None)
                        else:
                            fancy_labels.append(list(newlabelarray))
                    if k is 0 and (is_fancy[0] or not contiguous):
                        # then this output axis will be axis 0 of output
                        if label_source is 0:
                            new_tspan = self.tspan[index[0][iix]]
                            if (not isinstance(new_tspan, np.ndarray) or
                                    new_tspan.shape is ()):
                                # axis 0 has been sliced away: not a Timeseries
                                return ar
                            if not np.all(np.diff(new_tspan) > 0):
                                #tspan not monotonic increasing: not Timeseries
                                return ar
                        else:
                            #axis 0 no longer represents time: not a Timeseries
                            return ar
                else:
                    # not a 'sole candidate'
                    fancy_labels.append(None)
            if contiguous:
                # fancy output axes are put where the fancy input axes were:
                new_labels = []
                for i in range(0, fancy_pos[0]):
                    if nonfancy_retained.pop(0):
                        new_labels.append(nonfancy_labels.pop(0))
                new_labels.extend(fancy_labels)
                for i in range(fancy_pos[-1] + 1, cur_ndim):
                    if nonfancy_retained.pop(0):
                        new_labels.append(nonfancy_labels.pop(0))
            else:
                # not contiguous. fancy output axes move to the start:
                new_labels = fancy_labels + nonfancy_labels
            if new_tspan is None:
                return ar
            else:
                new_subts = ar._subarrays
                new_distaxis = ar._distaxis
                new_axislabels = new_labels[ar._distaxis]
                return DistTimeseries(new_subts, new_distaxis, new_axislabels)

    def __repr__(self):
        classname = self.__class__.__name__
        super_classname = super(DistTimeseries, self).__class__.__name__
        if self.tspan.shape is ():
            first = last = self.tspan
        else:
            first = self.tspan[0]
            last = self.tspan[-1]
        head = (u'<%s of shape %s from time %f to %f '
                 'with axis %d distributed>:\n') % (
                     classname, self.shape, first, last, self._distaxis)
        repr_tspan = 'tspan=' + repr(self.tspan)
        if len(repr_tspan) > 160:
            repr_tspan = 'tspan=array([ %f, ..., %f ])' % (first, last)
        # reuse DistArray repr, but replacing the heading:
        content = super(DistTimeseries, self).__repr__()
        first_newline = content.index('\n')
        content = content[(first_newline + 1):]
        content = content.replace(super_classname, classname, 1)
        content = content.rstrip(')') + ', \n' + repr_tspan
        if all(l is None for l in self.labels):
            labelsrep = ''
        else:
            labelsrep = ', \nlabels=' + repr(self.labels)
        return head + content + labelsrep + ')'

    @classmethod
    def __distob_vectorize__(cls, f):
        """Upgrades a normal function f to act on a DistTimeseries in parallel

        Args:
          f (callable): ordinary function which expects as its first
            argument a Timeseries (of the same shape as our subarrays)

        Returns:
          vf (callable): new function that takes a DistTimeseries as its first
            argument. ``vf(dist_timeseries)`` will do the computation ``f(ts)``
            on each sub-timeseries in parallel and if possible will return the
            results as a DistTimeseries or DistArray. (or if the results are
            not arrays, will return a list with the result for each
            sub-timeseries)
        """
        def vf(self, *args, **kwargs):
            remove_axis = ((slice(None),)*(self._distaxis) + (0,) + 
                           (slice(None),)*(self.ndim - self._distaxis - 1))
            refs = [ra[remove_axis]._ref for ra in self._subarrays]
            dv = distob.engine._client[:]
            def remote_f(object_id, *args, **kwargs):
                result = f(distob.engine[object_id], *args, **kwargs)
                if type(result) in distob.engine.proxy_types:
                    return distob.Ref(result)
                else:
                    return result
            results = []
            for ref in refs:
                dv.targets = ref.engine_id
                ar = dv.apply_async(remote_f, ref.object_id, *args, **kwargs)
                results.append(ar)
            for i in range(len(results)):
                ar = results[i]
                ar.wait()
                results[i] = ar.r
                if isinstance(results[i], distob.Ref):
                    ref = results[i]
                    RemoteClass = distob.engine.proxy_types[ref.type]
                    results[i] = RemoteClass(ref)
            if (all(isinstance(r, distob.RemoteArray) for r in results) and
                    all(r.shape == results[0].shape for r in results)):
                # Then we can join the results and return a DistArray.
                # To position result distaxis, match input shape where possible
                old_subshape = (self.shape[0:self._distaxis] +
                                self.shape[(self._distaxis+1):])
                res_subshape = list(results[0].shape)
                pos = len(res_subshape)
                for i in range(len(old_subshape)):
                    n = old_subshape[i]
                    if n not in res_subshape:
                        continue
                    pos = res_subshape.index(n)
                    res_subshape[pos] = None
                    if i >= self._distaxis:
                        break
                    pos += 1
                new_distaxis = pos
                results = [r.expand_dims(new_distaxis) for r in results]
                if all(isinstance(r, RemoteTimeseries) for r in results):
                    axlabels = self.labels[self._distaxis]
                    try:
                       return DistTimeseries(results, new_distaxis, axlabels)
                    except SimValueError:
                        pass
                return distob.DistArray(results, new_distaxis)
            else:
                return results
        if hasattr(f, '__name__'):
            vf.__name__ = 'v' + f.__name__
            f_str = f.__name__ + '()'
        else:
            f_str = 'callable'
        doc = u"""Apply %s in parallel to a DistTimeseries\n
               Args:
                 dts (DistTimeseries)
                 other args are the same as for %s
               """ % (f_str, f_str)
        if hasattr(f, '__doc__') and f.__doc__ is not None:
            doc = doc.rstrip() + (' detailed below:\n----------\n' + f.__doc__)
        vf.__doc__ = doc
        return vf

    def expand_dims(self, axis):
        """Insert a new axis, at a given position in the array shape
        Args:
          axis (int): Position (amongst axes) where new axis is to be inserted.
        """
        if axis <= self._distaxis:
            subaxis = axis
            new_distaxis = self._distaxis + 1
        else:
            subaxis = axis - 1
            new_distaxis = self._distaxis
        new_subts = [expand_dims(rts, subaxis) for rts in self._subarrays]
        if axis == 0:
            # prepended an axis: no longer a Timeseries
            return distob.DistArray(new_subts, new_distaxis)
        else:
            axislabels = self.labels[self._distaxis]
            return DistTimeseries(new_subts, new_distaxis, axislabels)


def _rts_from_ra(ra, tspan, labels):
    """construct a RemoteTimeseries from a RemoteArray"""
    eng_id = ra._ref.engine_id
    dv = distob.engine._client[eng_id]
    def remote_convert(ra_id, tspan, labels):
        from distob import engine
        from nsim import Timeseries
        array = engine[ra_id]
        ts = Timeseries(array, tspan, labels)
        return Ref(ts)
    ref = dv.apply_sync(remote_convert, ra._ref.object_id, tspan, labels)
    return RemoteTimeseries(ref)


def _dts_from_da(da, tspan, labels):
    """construct a DistTimeseries from a DistArray whose subarrays are already
    RemoteTimeseries 
    """
    assert(all(isinstance(ra, RemoteTimeseries) for ra in da._subarrays))
    da.__class__ = DistTimeseries
    da.tspan = tspan
    da.labels = labels
    da.t = _Timeslice(da)
    return da


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
    """Model defined by a system of ordinary differential equations

    Attributes:
      dimension (integer): Dimension of the state space
      output_vars (list of integers): If i is in this list then y[i] is 
        considered an output variable
      f(y, t): right hand side of the ODE system dy/dt = f(y, t)

    Instance attributes:
      y0 (array of shape (ndim,)): Initial state vector
    """
    dimension = 1
    output_vars = [0]

    def __init__(self):
        super(ODEModel, self).__init__()
        if not hasattr(self.__class__, 'y0'):
            self.y0 = np.zeros(self.__class__.dimension)

    def integrate(self, tspan):
        return Timeseries(integrate.odeint(self.f, self.y0, tspan), tspan)

    def f(y, t):
        pass


class SDEModel(Model):
    """Model defined by system of (ordinary) stochastic differential equations
    dy = f(y, t) dt + G(y, t) dW  (N.B. currently must be in Stratonovich form)

    Attributes:
      dimension (integer): Dimension of the state space
      output_vars (list of integers): If i is in this list then y[i] is 
        considered an output variable
      f(y, t): deterministic part of Stratonovich SDE system 
      G(y, t): noise coefficient matrix of Stratonovich SDE system 

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
    def __init__(self, ref):
        """Make a RemoteSimulation to access an already-existing Simulation 
        object, which may be on a remote engine.

        Args:
          ref (Ref): reference to a Simulation to be controlled by this proxy
        """
        super(RemoteSimulation, self).__init__(ref)
        #self.compute()

    def compute(self):
        """Start the computation process asynchronously"""
        def remote_compute(sim_id):
            distob.engine[sim_id].compute()
        self._dv.apply_async(remote_compute, self._id)


class MultipleSim(object):
    """Represents multiple simulations, possibly running on different hosts

    Like a list, indexing with [i] gives access to the ith simulation

    Attributes:
      timeseries: resulting timeseries: all variables of all simulations
      output: resulting timeseries: output variables of all simulations
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
        self.sims = distob.scatter([Simulation(s, T, dt) for s in systems])
        for s in self.sims:
            s.compute()

    def __len__(self):
        return len(self.sims)

    def __getitem__(self, key):
        return self.sims[key]

    def _repr_pretty_(self, p, cycle):
        return p.pretty(self.sims)

    def _node_labels(self):
        return ['node %d' % i for i in range(len(self.sims))]

    def __get_timeseries(self):
        subts = [s.timeseries for s in self.sims]
        sub_ndim = subts[0].ndim
        if sub_ndim is 1:
            subts = [distob.expand_dims(rts, 1) for rts in subts]
            sub_ndim += 1
        distaxis = sub_ndim
        subts = [distob.expand_dims(rts, distaxis) for rts in subts]
        return DistTimeseries(subts, distaxis, self._node_labels())

    timeseries = property(fget=__get_timeseries, doc="Rank 3 array representing"
        " multiple time series. 1st axis is time, 2nd axis ranges across all"
        " dynamical variables in a single simulation, 3rd axis ranges across"
        " different simulation instances.")

    def __get_output(self):
        subts = [s.output for s in self.sims]
        sub_ndim = subts[0].ndim
        if sub_ndim is 1:
            subts = [distob.expand_dims(rts, 1) for rts in subts]
            sub_ndim += 1
        distaxis = sub_ndim
        subts = [distob.expand_dims(rts, distaxis) for rts in subts]
        return DistTimeseries(subts, distaxis, self._node_labels())

    output = property(fget=__get_output, doc="Rank 3 array representing"
        " output time series. 1st axis is time, 2nd axis ranges across"
        " output variables of a single simulation, 3rd axis ranges across"
        " different simulation instances.")


class RepeatedSim(MultipleSim):
    """Independent simulations of the same model multiple times, with results.

    Like a list, indexing the object with [i] gives access to the ith simulation

    Attributes:
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

    def _node_labels(self):
        return ['repetition %d' % i for i in range(len(self.sims))]


class ParameterSim(MultipleSim):
    """Independent simulations of a model exploring different parameters"""
    pass


class NetworkSim(MultipleSim):
    """Simulation of many coupled instances of a model connected in a network"""
    pass


def newsim(f, G, y0, name='NewModel', T=60.0, dt=0.005, repeat=1, identical=True):
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
      name (str): Optional class name for the new model
      T: Total length of time to simulate, in seconds.
      dt: Timestep for numerical integration.
      repeat (int, optional)
      identical (bool, optional)

    Returns: 
      Simulation

    Raises:
      SimValueError, SimTypeError
    """
    NewModel = newmodel(f, G, y0, name)
    if repeat == 1:
        return Simulation(NewModel(), T, dt)
    else:
        return RepeatedSim(NewModel, T, dt, repeat, identical)


def newmodel(f, G, y0, name='NewModel'):
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
      name (str): Optional class name for the new model

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
    # Put the new class into namespace __main__ (to cause dill to pickle it)
    newclass.__module__ = '__main__'
    import __main__
    __main__.__dict__[name] = newclass 
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
