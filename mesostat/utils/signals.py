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


# Get the approximation of y, fitted using x
def polyfit_transform(x, y, ord=1):
    param = np.polyfit(x, y, ord)
    poly = np.poly1d(param)
    return poly(x)


# Bin data by splitting it into N bins of equal number of datapoints
# For each datapoint, return bin index to which it belongs
def bin_data_1D(data, nBins):
    boundaries = np.quantile(data, np.linspace(0, 1, nBins + 1))
    boundaries[0] -= 1.0E-10
    condLeft = np.array([data <= b for b in boundaries])
    condRight = np.array([data > b for b in boundaries])
    condBoth = np.logical_and(condLeft[1:], condRight[:-1])
    return np.where(condBoth.T)[1]


def bin_data(data, nBins, axis=0):
    lenAxis = data.shape[axis]
    rezLst = []
    for iAx in range(lenAxis):
        dataThis = np.take(data, iAx, axis=axis)
        dataFlat = dataThis.flatten()
        dataBinned = bin_data_1D(dataFlat, nBins)
        rezLst += [dataBinned.reshape(dataThis.shape)]

    # Move the binned dimension back to where it was originally
    return numpy_move_dimension(np.array(rezLst), 0, axis)


# Compute discretized exponential decay convolution
# Works with multidimensional arrays, as long as shapes are the same
def approx_decay_conv(data, tau, dt):
    dataShape = data.shape
    nTimesTmp = dataShape[0] + 1   # Temporary data 1 longer because recursive formula depends on past
    tmpShape = (nTimesTmp, ) + dataShape[1:]

    alpha = dt / tau
    beta = 1-alpha
    
    rez = np.zeros(tmpShape)
    for i in range(1, nTimesTmp):
        rez[i] = data[i-1]*alpha + rez[i-1]*beta

    return rez[1:]  # Remove first element, because it is zero and meaningless. Get same shape as original data


# Downsample uniformly-spaced points by grouping them together and taking averages
# * Advantage is that there is no overlap between points
# * Disadvantage is that the options are limited to just a few values of nt
#
# - By convention, truncate tail if number of points is not divisible by nt.
#   It is preferential to lose the tail than to have non-uniform time spacing
#
# Can handle arbitrary dimension, as long as downsampling is done along the first dimension
def downsample_int(x1, y1, nt):
    nTimes1 = len(x1)
    nTimes2 = nTimes1 // nt
    assert y1.shape[0] == nTimes1, "Times array and selected axis of data array must match"
    shape2 = (nTimes2,) + y1.shape[1:]

    x2 = np.zeros(nTimes2)
    y2 = np.zeros(shape2)

    for i in range(nTimes2):
        l, r = i*nt, (i+1)*nt
        x2[i] = np.mean(x1[l:r], axis=0)
        y2[i] = np.mean(y1[l:r], axis=0)

    return x2, y2
    

# Kernel for gaussian downsampling
# Can later downsample any dataset with exactly the same sampling points simply multiplying it by the kernel
def resample_kernel(x1, x2, sig2):
    # Each downsampled val is average of all original val weighted by proximity kernel
    n1 = x1.shape[0]
    n2 = x2.shape[0]

    xx1 = np.outer(x2, np.ones(n1))
    xx2 = np.outer(np.ones(n2), x1)
    W = gaussian(xx2 - xx1, sig2)

    # Normalize weights, so they sum up to 1 for every target point
    for i in range(n2):
        W[i] /= np.sum(W[i])

    return W


# General resampling
# Switches between downsampling and upsampling
def resample(x1, y1, x2, param):
    N2 = len(x2)
    y2 = np.zeros(N2)
    DX2 = x2[1] - x2[0]   # step size for final distribution

    # Check that the new data range does not exceed the old one
    rangeX1 = [np.min(x1), np.max(x1)]
    rangeX2 = [np.min(x2), np.max(x2)]
    if (rangeX2[0] < rangeX1[0])or(rangeX2[1] > rangeX1[1]):
        raise ValueError("Requested range", rangeX2, "exceeds the original data range", rangeX1)
    
    # UpSampling: Use if original dataset has lower sampling rate than final
    if param["method"] == "interpolative":
        kind = param["kind"] if "kind" in param.keys() else "cubic"
        y2 = interpolate.interp1d(x1, y1, kind=kind)(x2)
        
    # Downsample uniformly-sampled data by kernel or bin-averaging
    # DownSampling: Use if original dataset has higher sampling rate than final
    else:
        kind = param["kind"] if "kind" in param.keys() else "window"
        # Window-average method
        if kind == "window":
            window_size = param["window_size"] if "window_size" in param.keys() else DX2

            for i2 in range(N2):
                # Find time-window to average
                w_l = x2[i2] - 0.5 * window_size
                w_r = x2[i2] + 0.5 * window_size

                # Find points of original dataset to average
                i1_l, i1_r = slice_sorted(x1, [w_l, w_r])
                # i1_l = np.max([int(np.ceil((w_l - x1[0]) / DX1)), 0])
                # i1_r = np.min([int(np.floor((w_r - x1[0]) / DX1)), N1])

                # Compute downsampled values by averaging
                y2[i2] = np.mean(y1[i1_l:i1_r])

        # Gaussian kernel method
        else:
            ker_sig2 = param["ker_sig2"] if "ker_sig2" in param.keys() else (DX2/2)**2
            WKer = param["ker_w"] if "ker_w" in param else resample_kernel(x1, x2, ker_sig2)
            y2 = WKer.dot(y1)

            # # Each downsampled val is average of all original val weighted by proximity kernel
            # w_ker = gaussian(x2[i2] - x1, ker_sig2)
            # w_ker /= np.sum(w_ker)
            # y2[i2] = w_ker.dot(y1)
        
    return y2


# Wrapper without x-coordinates
# Resamples dataset to the same interval but different discretization
def resample_stretch(y1, n2):
    n1 = len(y1)
    x1 = np.linspace(0, 1, n1)
    x2 = np.linspace(0, 1, n2)
    if n1 <= n2:
        return resample(x1, y1, x2, {"method" : "interpolative"})
    else:
        return resample(x1, y1, x2, {"method": "gaussian"})


# Resample all arrays to the overlapping range using piecewise-linear interpolation
def resample_shortest_linear(xLst2D, yLst2D, timestep=None, assume_same=True):
    '''
     Algorithm:
        1. Pick the shortest of all ranges, and use it for all other datasets
        2. For each dataset, construct piecewise-linear interpolator
        3. Sample all points for that shortest range
    '''

    # If we suspect that the arrays are same, we can just test that their lengths are same
    # and skip the resampling procedure, if it is not necessary
    if assume_same:
        nXLst = np.array([len(x) for x in xLst2D])
        if np.all(nXLst == nXLst[0]):
            return xLst2D[0], np.array(yLst2D)
        else:
            print("positions are not same, resampling to shortest overlap")

    # Guess timestep
    timestep = timestep if timestep is not None else xLst2D[0][1] - xLst2D[0][0]

    # Find range
    xMin = -np.inf
    xMax = np.inf
    for x in xLst2D:
        xMin = np.max([xMin, np.min(x)])
        xMax = np.min([xMax, np.max(x)])

    assert xMin < xMax, "The overlap is zero"

    # Generate target steps
    nX = int(np.round((xMax - xMin)/timestep)) + 1
    xTarget = xMin + timestep * np.arange(nX)

    # Perform linear interpolation
    rezLst2D = [np.interp(xTarget, xLst, yLst) for xLst, yLst in zip(xLst2D, yLst2D)]

    # Return results
    return xTarget, np.array(rezLst2D)


