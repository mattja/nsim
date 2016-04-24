# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
classes:

  Timeseries   numpy array with extra methods for time series analyses
"""

from __future__ import absolute_import
import numpy as np
from collections import Sequence
import warnings
import copy
import types
import numbers

# types for compatibility across python 2 and 3
_SliceType = type(slice(None))
_EllipsisType = type(Ellipsis)
_TupleType = type(())
_NewaxisType = type(np.newaxis)

warnings.filterwarnings('ignore', 'comparison to `None`', FutureWarning)


class Timeseries(np.ndarray):
    """A numpy array with extra methods for applying time series analyses.
    It is an array of up to 3 dimensions.

    axis 0 ranges across values of the system at different points in time.
           For a single variable time series this is the only axis needed.

    axis 1, if present, ranges across the different variables/channels of a 
            multivariate time series, at a single node.

    axis 2, if present, ranges across different nodes of a network simulation 
            or repeated simulation.                  

    Thus the shape of a Timeseries array is (N, n, m) where N is the 
    total number of time points, n is the number of variables or channels of a
    single node and m is the number of nodes.

    Slice by array index:  timeseries[103280:104800]
    Slice by time in seconds:  timeseries.t[10.0:20.5]

    Methods:
      All functions defined in the package nsim.analyses1 are available as 
      methods of the Timeseries  (as well as the usual methods of numpy arrays)

      Your own analysis functions can also be included by calling
      Timeseries.add_analyses('file.py')
    
    Attributes:
      tspan: An array of shape (N,) giving the time in seconds at each 
        time point along axis 0 of the array. tspan will always remain sorted.
        The time points are not necessarily evenly spaced.

      t: Allows slicing by time: timeseries.t[starttime:endtime, ...]

      labels (list): For each axis>0, optionally gives names to label 
        points along the axis. (labels[0] is always None as axis 0 is the
        time axis, labeled by numbers in `tspan` rather than by strings) 
        For i>0, labels[i] is a list of str of length shape(timeseries)[i]
        giving names to each position along axis i. For example labels[1] can 
        hold names for each variable or channel in a multivariate timeseries.
    """
    def __new__(cls, input_array, tspan=None, labels=None, fs=None):
        """Create a new Timeseries from an ordinary numpy array.
        Args:
          input_array (ndarray): the timeseries data.
          tspan (ndarray, optional): An array of shape (N,) giving the time 
            in seconds at each time point along axis 0 of the array. tspan will
            always remain sorted. Time points are not necessarily evenly spaced
          labels (list, optional): For each axis>0, optionally give names 
            to label points along the axis. (labels[0] must be None).
          fs (scalar, optional): sample rate. Can be given instead of tspan if
            all time points are evenly spaced.
        """
        #print('In __new__')
        obj = np.asarray(input_array).view(cls)
        if tspan is not None:
            obj.tspan = tspan
        elif isinstance(input_array, Timeseries):
            obj.tspan = input_array.tspan
        else:
            if fs is None:
                fs = 1.0
            n = len(input_array)
            obj.tspan = np.linspace(0.0, 1.0*(n-1)/fs, n, endpoint=True)
        if isinstance(input_array, Timeseries) and labels is None:
            obj.labels = input_array.labels
        else:
            obj.labels = labels
        if obj.labels is None:
            obj.labels = [None] * obj.ndim
        if len(obj.labels) < obj.ndim:
            obj.labels.extend([None] * (obj.ndim - len(obj.labels)))
        assert(obj.labels[0] is None) # time axis should not have string labels
        for i, seq in enumerate(obj.labels):
            if seq is not None:
                if len(seq) != obj.shape[i]:
                    raise ValueError(
                        '%d labels given for axis %d of length %d' % (
                        len(seq), i, obj.shape[i]))
                obj.labels[i] = list(seq)
        obj.t = _Timeslice(obj)
        return obj

    def __init__(self, input_array, tspan=None, labels=None, fs=None):
        #print('In __init__')
        pass

    @classmethod
    def add_analyses(cls, source):
        """Dynamically add new analysis methods to the Timeseries class.
        Args:
          source: Can be a function, module or the filename of a python file.
            If a filename or a module is given, then all functions defined 
            inside not starting with _ will be added as methods.

        The only restriction on the functions is that they can accept a 
        Timeseries as their first argument. So existing functions that 
        take a ndarray or array or even a list will usually also work.
        """
        if isinstance(source, types.FunctionType):
            _add_single_method(source.__name__, source)
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
                    cls._add_single_method(name, obj)

    @classmethod
    def _add_single_method(cls, name, f):
        import sys
        setattr(cls, name, f)
        #If using RemoteTimeseries, add method to that class too:
        if 'distob' in sys.modules:
            import distob
            if distob.engine is not None and cls in distob.engine.proxy_types:
                RemoteClass = distob.engine.proxy_types[cls]
                remotemethod = distob._make_proxy_method(name, f.__doc__)
                setattr(RemoteClass, name, remotemethod)
                # also update class definitions on remote engines
                if isinstance(distob.engine, distob.ObjectHub):
                    dv = distob.engine._dv
                    dv.targets = 'all'
                    def remote_update(name, Class, f, remotemethod):
                        setattr(Class, name, f)
                        RemoteClass = distob.engine.proxy_types[Class]
                        setattr(RemoteClass, name, remotemethod)
                    ars = dv.apply(remote_update, real_type, proxy_type)
                    dv.wait(ars)
                    for ar in ars:
                        if not ar.successful():
                            raise ar.r

    def __array_finalize__(self, obj):
        #print('In __array_finalize__, obj is type ' + str(type(obj)))
        if obj is None:
            return
        if isinstance(obj, self.__class__):
            if obj.shape is () or obj.shape is not () and len(self) ==len(obj):
                self.tspan = obj.tspan
                self.labels = obj.labels
                self.t = _Timeslice(self)

    def __array_prepare__(self, in_arr, context=None):
        #print('In __array_prepare__')
        return super(Timeseries, self).__array_prepare__(in_arr, context)

    def __array_wrap__(self, out_arr, context=None):
        #print('In __array_wrap__')
        return super(Timeseries, self).__array_wrap__(out_arr, context)

    def __getitem__(self, index):
        """When a Timeseries is sliced, tspan will be sliced in the
        same way as axis 0 of the Timeseries, and labels[i] will be sliced in
        the same way as axis i of the Timeseries.
        If the resulting array is not a Timeseries (for example if the
        time axis is sliced to a scalar value) then returns an ndarray.
        """
        new_array = np.asarray(self).__getitem__(index)
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
                return np.asarray(new_array)
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
            if (not isinstance(new_tspan, np.ndarray) or 
                    new_tspan.shape is () or
                    new_tspan.shape[0] is 0):
                # axis 0 has been sliced away: result is no longer a Timeseries
                return np.asarray(new_array)
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
            return Timeseries(new_array, new_tspan, new_labels)
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
                if ix_arrays[j].shape is ():
                    ix_arrays[j] = np.expand_dims(ix_arrays[j], 0)
                index[fancy_pos[j]] = ix_arrays[j]
            index = tuple(index)
            ishape = index[fancy_pos[0]].shape # common shape all index arrays
            idim = len(ishape)
            assert(idim > 0)
            new_tspan = None
            if contiguous and not is_fancy[0]:
                new_tspan = self.tspan[index[0]]
                if (not isinstance(new_tspan, np.ndarray) or
                        new_tspan.shape is () or
                        new_tspan.shape[0] is 0):
                    # axis 0 has been sliced away: no longer a Timeseries
                    return np.asarray(new_array)
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
                                    new_tspan.shape is () or
                                    new_tspan.shape[0] is 0):
                                # axis 0 has been sliced away: not a Timeseries
                                return np.asarray(new_array)
                            if not np.all(np.diff(new_tspan) > 0):
                                #tspan not monotonic increasing: not Timeseries
                                return np.asarray(new_array)
                        else:
                            #axis 0 no longer represents time: not a Timeseries
                            return np.asarray(new_array)
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
                return np.asarray(new_array)
            else:
                return Timeseries(new_array, new_tspan, new_labels)

    def __setitem__(self, index, value):
        #print('in __setitem__')
        return super(Timeseries, self).__setitem__(index, value)

    def __getslice__(self, i, j):
        #print('in __getslice__')
        return self.__getitem__(slice(i, j, None))

    def __setslice__(self, i, j, value):
        #print('in __setslice__')
        return self.__setitem__(slice(i, j, None), value)

    def __repr__(self):
        classname = self.__class__.__name__
        if self.tspan.shape is ():
            first = last = self.tspan
        else:
            first = self.tspan[0]
            last = self.tspan[-1]
        head = u'<%s of shape %s from time %f to %f>:\n' % (
            classname, self.shape, first, last)
        repr_tspan = 'tspan=' + repr(self.tspan)
        if len(repr_tspan) > 160:
            repr_tspan = 'tspan=array([ %f, ..., %f ])' % (first, last)
        content = repr(np.asarray(self)).replace('array', 
            classname, 1).rstrip(')') + ', \n' + repr_tspan
        if all(l is None for l in self.labels):
            labelsrep = ''
        else:
            labelsrep = ', \nlabels=['
            for lab in self.labels:
                if len(repr(lab)) > 160:
                    labelsrep += '[%s, ..., %s], ' % (lab[0], lab[-1])
                else:
                    labelsrep += repr(lab) + ', '
            labelsrep = labelsrep[:-2] + ']'
        return head + content + labelsrep + ')'

    def __reduce__(self):
        """Support pickling Timeseries instances by saving __dict__"""
        recon, initargs, state = super(Timeseries, self).__reduce__()
        tsstate = tuple((state, self.__dict__)) # nested, to avoid tuple copying
        return tuple((recon, initargs, tsstate))
        
    def __setstate__(self, tsstate):
        """Support unpickling Timeseries instances by loading __dict__"""
        super(Timeseries, self).__setstate__(tsstate[0])
        self.__dict__.update(tsstate[1])

    def __distob_scatter__(self, axis=-1, destination=None, blocksize=None):
        """Turn a Timeseries into a distributed timeseries"""
        import distob
        from nsim import DistTimeseries
        if axis is None:
            return distob.distob._scatter_ndarray(self, None, destination)
        if axis == 0:
            raise ValueError(u'Currently cannot distribute the time axis')
        if axis < 0:
            axis = self.ndim + axis
        dar = distob.distob._scatter_ndarray(self, axis,
                                             destination, blocksize)
        axlabels = self.labels[axis]
        return DistTimeseries([rts for rts in dar._subarrays], axis, axlabels)

    def absolute(self):
        """Calculate the absolute value element-wise.

        Returns:
          absolute (Timeseries):
            Absolute value. For complex input (a + b*j) gives sqrt(a**a + b**2)
        """
        return Timeseries(np.absolute(self), self.tspan, self.labels)

    def abs(self):
        """Calculate the absolute value element-wise."""
        return self.absolute()

    def angle(self, deg=False):
        """Return the angle of the complex argument.

        Args:
          deg (bool, optional):
            Return angle in degrees if True, radians if False (default).

        Returns:
          angle (Timeseries):
            The counterclockwise angle from the positive real axis on
            the complex plane, with dtype as numpy.float64.
        """
        if self.dtype.str[1] != 'c':
            warnings.warn('angle() is intended for complex-valued timeseries',
                          RuntimeWarning, 1)
        return Timeseries(np.angle(self, deg=deg), self.tspan, self.labels)

    # Some of the usual array operations on Timeseries return a plain ndarray. 
    # This depends on whether a time axis is present in the result:

    def flatten(self, order='C'):
        return np.asarray(self).flatten(order)

    def ravel(self, order='C'):
        return np.asarray(self).ravel(order)

    def swapaxes(self, axis1, axis2):
        """Interchange two axes of a Timeseries."""
        if self.ndim <=1 or axis1 == axis2:
            return self
        ar = np.asarray(self).swapaxes(axis1, axis2)
        if axis1 != 0 and axis2 != 0:
            # then axis 0 is unaffected by the swap
            labels = self.labels[:]
            labels[axis1], labels[axis2] = labels[axis2], labels[axis1]
            return Timeseries(ar, self.tspan, labels)
        return ar

    def transpose(self, *axes):
        """Permute the dimensions of a Timeseries."""
        if self.ndim <= 1:
            return self
        ar = np.asarray(self).transpose(*axes)
        if axes[0] != 0:
            # then axis 0 is unaffected by the transposition
            newlabels = [self.labels[ax] for ax in axes]
            return Timeseries(ar, self.tspan, newlabels)
        else:
            return ar

    def argmin(self, axis=None, out=None):
        return np.asarray(self).argmin(axis, out)

    def argmax(self, axis=None, out=None):
        return np.asarray(self).argmax(axis, out)

    def reshape(self, newshape, order='C'):
        """If axis 0 is unaffected by the reshape, then returns a Timeseries,
        otherwise returns an ndarray. Preserves labels of axis j only if all 
        axes<=j are unaffected by the reshape.  
        See ``numpy.ndarray.reshape()`` for more information
        """
        oldshape = self.shape
        ar = np.asarray(self).reshape(newshape, order=order)
        if (newshape is -1 and len(oldshape) is 1 or
                (isinstance(newshape, numbers.Integral) and 
                    newshape == oldshape[0]) or 
                (isinstance(newshape, Sequence) and
                    (newshape[0] == oldshape[0] or
                     (newshape[0] is -1 and np.array(oldshape[1:]).prod() ==
                                            np.array(newshape[1:]).prod())))):
            # then axis 0 is unaffected by the reshape
            newlabels = [None] * ar.ndim
            i = 1
            while i < ar.ndim and i < self.ndim and ar.shape[i] == oldshape[i]:
                newlabels[i] = self.labels[i]
                i += 1
            return Timeseries(ar, self.tspan, newlabels)
        else:
            return ar

    def min(self, axis=None, out=None):
        if (axis is 0 or 
                axis is None or
                self.ndim is 1 or 
                isinstance(axis, _TupleType) and 0 in axis):
            return np.asarray(self).min(axis, out)
        else:
            ar = super(Timeseries, self).min(axis, out)
            if isinstance(axis, numbers.Number):
                axis = (axis,)
            new_labels = []
            for i in range(self.ndim):
                if i not in axis:
                    new_labels.append(self.labels[i])
            return Timeseries(ar, self.tspan, new_labels)

    def max(self, axis=None, out=None):
        if (axis is 0 or 
                axis is None or
                self.ndim is 1 or 
                isinstance(axis, _TupleType) and 0 in axis):
            return np.asarray(self).max(axis, out)
        else:
            ar = super(Timeseries, self).max(axis, out)
            if isinstance(axis, numbers.Number):
                axis = (axis,)
            new_labels = []
            for i in range(self.ndim):
                if i not in axis:
                    new_labels.append(self.labels[i])
            return Timeseries(ar, self.tspan, new_labels)

    def ptp(self, axis=None, out=None):
        if (axis is 0 or 
                axis is None or
                self.ndim is 1 or 
                isinstance(axis, _TupleType) and 0 in axis):
            return np.asarray(self).ptp(axis, out)
        else:
            ar = super(Timeseries, self).ptp(axis, out)
            if isinstance(axis, numbers.Number):
                axis = (axis,)
            new_labels = []
            for i in range(self.ndim):
                if i not in axis:
                    new_labels.append(self.labels[i])
            return Timeseries(ar, self.tspan, new_labels)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        if axis == -1:
            axis = self.ndim
        if keepdims:
            if axis is None:
                out_shape = [1] * self.ndim
            else:
                out_shape = list(self.shape)
                out_shape[axis] = 1
        if (axis is 0 or 
                axis is None or
                self.ndim is 1 or 
                isinstance(axis, _TupleType) and 0 in axis):
            res = np.asarray(self).mean(axis, dtype, out)
            if keepdims:
                res = res.reshape(out_shape)
        else:
            ar = super(Timeseries, self).mean(axis, dtype, out)
            if isinstance(axis, numbers.Number):
                axis = (axis,)
            new_labels = []
            for i in range(self.ndim):
                if i not in axis:
                    new_labels.append(self.labels[i])
            res = Timeseries(ar, self.tspan, new_labels)
            if keepdims:
                res = res.reshape(out_shape)
        return res

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        if (axis is 0 or 
                axis is None or
                self.ndim is 1 or 
                isinstance(axis, _TupleType) and 0 in axis):
            return np.asarray(self).std(axis, dtype, out, ddof)
        else:
            ar = super(Timeseries, self).std(axis, dtype, out)
            if isinstance(axis, numbers.Number):
                axis = (axis,)
            new_labels = []
            for i in range(self.ndim):
                if i not in axis:
                    new_labels.append(self.labels[i])
            return Timeseries(ar, self.tspan, new_labels)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        if (axis is 0 or 
                axis is None or
                self.ndim is 1 or 
                isinstance(axis, _TupleType) and 0 in axis):
            return np.asarray(self).var(axis, dtype, out, ddof)
        else:
            ar = super(Timeseries, self).var(axis, dtype, out)
            if isinstance(axis, numbers.Number):
                axis = (axis,)
            new_labels = []
            for i in range(self.ndim):
                if i not in axis:
                    new_labels.append(self.labels[i])
            return Timeseries(ar, self.tspan, new_labels)

    def merge(self, ts):
        """Merge another timeseries with this one
        Arguments:
          ts (Timeseries): The two timeseries being merged must have the
            same shape except for axis 0.
        Returns: 
          Resulting merged timeseries which can have duplicate time points.
        """
        if ts.shape[1:] != self.shape[1:]:
            raise ValueError('Timeseries to merge must have compatible shapes')
        indices = np.vstack((self.tspan, ts.tspan)).argsort()
        return np.vstack((self, ts))[indices]

    def expand_dims(self, axis):
        """Insert a new axis, at a given position in the array shape
        Args:
          axis (int): Position (amongst axes) where new axis is to be inserted.
        """
        if axis == -1:
            axis = self.ndim
        array = np.expand_dims(self, axis)
        if axis == 0:
            # prepended an axis: no longer a Timeseries
            return array
        else:
            new_labels = self.labels.insert(axis, None)
            return Timeseries(array, self.tspan, new_labels)

    def concatenate(self, tup, axis=0):
        """Join a sequence of Timeseries to this one
        Args: 
          tup (sequence of Timeseries): timeseries to be joined with this one.
            They must have the same shape as this Timeseries, except in the
            dimension corresponding to `axis`.
          axis (int, optional): The axis along which timeseries will be joined.
        Returns:
          res (Timeseries or ndarray)
        """
        if not isinstance(tup, Sequence):
            tup = (tup,)
        if tup is (None,) or len(tup) is 0:
            return self
        tup = (self,) + tuple(tup)
        new_array = np.concatenate(tup, axis)
        if not all(hasattr(ts, 'tspan') and 
                   hasattr(ts, 'labels') for ts in tup):
            return new_array
        if axis == 0:
            starts = [ts.tspan[0] for ts in tup]
            ends = [ts.tspan[-1] for ts in tup]
            if not all(starts[i] > ends[i-1] for i in range(1, len(starts))):
                # series being joined are not ordered in time. not Timeseries
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
        return self.__new__(self.__class__, new_array, new_tspan, new_labels)

    def split(self, indices_or_sections, axis=0):
        """Split a timeseries into multiple sub-timeseries"""
        if not isinstance(indices_or_sections, numbers.Integral):
            raise Error('splitting by array of indices is not yet implemented')
        n = indices_or_sections
        if self.shape[axis] % n != 0:
            raise ValueError("Array split doesn't result in an equal division")
        step = self.shape[axis] / n
        pieces = []
        start = 0
        while start < self.shape[axis]:
            stop = start + step
            ix = [slice(None)] * self.ndim
            ix[axis] = slice(start, stop)
            ix = tuple(ix)
            pieces.append(self[ix])
            start += step
        return pieces

    def save_mat(self, filename):
        """save a Timeseries to a MATLAB .mat file
        Args:
          ts (Timeseries): the timeseries to save
          filename (str): .mat filename to save to
        """
        from nsim import save_mat
        return save_mat(self, filename)


class _Timeslice(object):
    """Implements the `t` attribute of Timeseries that allows slicing by time
    instead of by array index. For example, timeseries.t[9.5:30:0.1] 
    resamples the time range from t=9.5 to t=30 in 0.1 second increments.
    """
    def __init__(self, ts):
        """Args: ts: Timeseries"""
        self.ts = ts

    def __getitem__(self, index):
        #print('in timeslice getitem, index type %s, index is %s' % (
        #    type(index), repr(index)))
        ts = self.ts
        dt = (ts.tspan[-1] - ts.tspan[0]) / (len(ts) - 1)
        if isinstance(index, numbers.Number):
            if not (index >= ts.tspan[0] and index <= ts.tspan[-1]):
                raise ValueError('time %s not in valid range %g to %g' % (
                        index, ts.tspan[0], ts.tspan[-1]))
            newix = ts.tspan.searchsorted(index)
            return ts[newix]
        elif isinstance(index, _SliceType):
            if index.start is None or index.start < ts.tspan[0]:
                startt = ts.tspan[0]
            else:
                startt = index.start
            if index.stop is None or index.stop > ts.tspan[-1]:
                stopt = ts.tspan[-1] + dt/2
            else:
                stopt = index.stop
            start, stop = ts.tspan.searchsorted([startt, stopt])
            if index.step is None or index.step <= dt:
                return ts[slice(start, stop, None)]
            else:
                times = np.arange(startt, stopt, index.step)
                if times[-1] == stopt:
                    times = times[:-1]
                indices = ts.tspan.searchsorted(times)
                if indices[-1] == len(ts.tspan):
                    indices = indices[:-1]
                return ts[indices]
        elif isinstance(index, _EllipsisType) or index is None:
            return ts[index]
        elif isinstance(index, np.ndarray) and index.ndim is 1:
            indices = ts.tspan.searchsorted(index)
            if indices[-1] == len(ts.tspan):
                indices = indices[:-1]
            return ts[indices]
        elif isinstance(index, _TupleType):
            timeix = index[0]
            otherix = index[1:]
            ts1 = ts.t[timeix]
            if ts1.ndim < ts.ndim:
                if len(otherix) is 1:
                    otherix = otherix[0]
                return ts1[otherix]
            else:
                return ts1[(slice(None),) + otherix]
        else:
            raise TypeError("Time slicing can't handle that type of index yet")

    def __setitem__(self, index, value):
        #TODO update logic to match __getitem__
        ts = self.ts
        dt = (ts.tspan[-1] - ts.tspan[0]) / (len(ts) - 1)
        if isinstance(index, numbers.Number):
            newix = ts.tspan.searchsorted(index)
            return ts.__setitem__(newix, value)
        elif isinstance(index, _SliceType):
            if index.step is None:
                start, stop = ts.tspan.searchsorted(index.start, index.stop)
                return ts.__setitem__(slice(start, stop, None), value)
            else:
                n = np.floor_divide(index.start - index.stop, index.step)
                times = np.linspace(index.start, index.stop, n, endpoint=False)
                indices = ts.tspan.searchsorted(times)
                if indices[-1] == len(ts.tspan):
                    indices = indices[:-1]
                return ts.__setitem__(indices, value)
        elif isinstance(index, _EllipsisType) or index is None:
            return ts.__setitem__(index, value)
        elif isinstance(index, np.ndarray) and index.ndim is 1:
            indices = ts.tspan.searchsorted(index)
            if indices[-1] == len(ts.tspan):
                indices = indices[:-1]
            return ts.__setitem__(indices, value)
        elif isinstance(index, _TupleType):
            timeix = index[0]
            ts = ts.t[timeix]
            otherix = index[1:]
            return ts.__setitem__(otherix, value)
        else:
            raise TypeError("Time slicing can't handle that type of index yet")

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j, None))

    def __setslice__(self, i, j, value):
        return self.__setitem__(slice(i, j, None), value)

    def __repr__(self):
        fs = (1.0*len(self.ts) - 1) / (self.ts.tspan[-1] - self.ts.tspan[0])
        s = u'Time range %g to %g with average sample rate %g Hz\n' % (
                self.ts.tspan[0], self.ts.tspan[-1], fs)
        return s


def merge(tup):
    """Merge several timeseries
    Arguments:
      tup: sequence of Timeseries, with the same shape except for axis 0
    Returns: 
      Resulting merged timeseries which can have duplicate time points.
    """
    if not all(tuple(ts.shape[1:] == tup[0].shape[1:] for ts in tup[1:])):
        raise ValueError('Timeseries to merge must have compatible shapes')
    indices = np.vstack(tuple(ts.tspan for ts in tup)).argsort()
    return np.vstack((tup))[indices]
