import numpy as np
from scipy import interpolate

from mesostat.utils.arrays import slice_sorted, numpy_shape_reduced_axes, numpy_move_dimension
from mesostat.stat.stat import gaussian


def zscore(x, axis=None):
    shapeNew = numpy_shape_reduced_axes(x.shape, axis)
    mu = np.nanmean(x, axis=axis).reshape(shapeNew)
    std = np.nanstd(x, axis=axis).reshape(shapeNew)
    return (x - mu) / std


def zscore_dim_ord(x, dimOrdSrc, dimOrdZ=None):
    if dimOrdZ is not None:
        axisZScore = tuple([i for i, e in enumerate(dimOrdSrc) if e in dimOrdZ])
        return zscore(x, axisZScore)
    else:
        return x


# ZScore list of arrays, computing mean and std from concatenated data
def zscore_list(lst):
    xFlat = np.hstack([data.flatten for data in lst])
    mu = np.nanmean(xFlat)
    std = np.nanstd(mu)
    return [(data - mu)/std for data in lst]


# Compute discretized exponential decay convolution
# Works with multidimensional arrays, as long as shapes are the same
def approx_decay_conv(data, tau, dt):
    dataShape = data.shape
    nTimesTmp = dataShape[0] + 1  # Temporary data 1 longer because recursive formula depends on past
    tmpShape = (nTimesTmp,) + dataShape[1:]

    alpha = dt / tau
    beta = 1 - alpha

    rez = np.zeros(tmpShape)
    for i in range(1, nTimesTmp):
        rez[i] = data[i - 1] * alpha + rez[i - 1] * beta

    return rez[1:]  # Remove first element, because it is zero and meaningless. Get same shape as original data
