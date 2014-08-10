"""An empty file where tests should be
"""

import pytest
import nsim
import numpy as np


def test_slicing():
    ar = np.random.randn(3, 4, 5)
    labels = [None, ['c0', 'c1', 'c2', 'c3'], ['n0', 'n1', 'n2', 'n3', 'n4']]
    ts = nsim.Timeseries(ar, labels=labels)
    # slicing can add a new axis
    assert(ts[:, np.newaxis, :, :].shape == (3, 1, 4, 5))
    assert(isinstance(ts[:, np.newaxis, :, :], nsim.Timeseries))
    assert(ts[np.newaxis, :, :, :].shape == (1, 3, 4, 5))
    assert(not isinstance(ts[np.newaxis, :, :, :], nsim.Timeseries))


def test_slicing_labels():
    ar = np.random.randn(3, 4, 5)
    labels = [None, ['c0', 'c1', 'c2', 'c3'], ['n0', 'n1', 'n2', 'n3', 'n4']]
    ts = nsim.Timeseries(ar, labels=labels)
    assert(ts[:, :, :].labels == ts.labels)
    assert(ts[:, 2, :].labels[1][4] == 'n4')
    assert(len(ts[:, :, 3].labels) is 2)
    assert(ts[:, :, 3].labels[1][3] == 'c3')
    assert(ts[:, 1, 1].labels == [None])
    assert(all(np.equal(ts[:, 1, 1].tspan, ts.tspan)))
    assert(ts[..., np.newaxis].labels[3] is None)
    assert(ts[..., np.newaxis, :].labels[3][0] == 'n0')
    partial_labels = [None, None, ['n0', 'n1', 'n2', 'n3', 'n4']]
    ts1 = nsim.Timeseries(ar, labels=partial_labels)
    assert(ts1[:, 2, :].labels[1][4] == 'n4')
    assert(len(ts1[:, :, 3].labels) is 2)
    assert(ts1[:, :, 3].labels[1] is None)


def test_reshape():
    ar = np.random.randn(4, 4, 4)
    labels = [None, ['c0', 'c1', 'c2', 'c3'], ['n0', 'n1', 'n2', 'n3']]
    ts = nsim.Timeseries(ar, labels=labels)
    assert(not isinstance(ts.reshape((2, 2, 4, 4)), nsim.Timeseries))
    ts1 = ts.reshape((-1, 4, 2, 2))
    assert(all(np.equal(ts1.tspan, ts.tspan)))
    assert(ts1.labels[1] == ts.labels[1])
    assert(ts1.labels[2] is None)
